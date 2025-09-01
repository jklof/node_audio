# === File: additional_plugins/visualization_spectrogram.py ===

import numpy as np
import time
import threading
import logging

# --- Node System Imports ---
from node_system import Node
from constants import SpectralFrame

# --- UI and Qt Imports ---
from ui_elements import NodeItem
from PySide6.QtWidgets import QWidget, QSizePolicy, QLabel, QVBoxLayout
from PySide6.QtGui import QPainter, QColor, QPen, QImage, QFont, QFontMetrics, QPaintEvent
from PySide6.QtCore import Qt, Signal, Slot, QObject, QPoint, QSize

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants for Visualization ---
UI_UPDATE_INTERVAL_S = 0.033
MIN_DB_DISPLAY = -70.0
MAX_DB_DISPLAY = 6.0
MIN_FREQ_DISPLAY = 20.0


# ==============================================================================
# 1. Custom Drawing Widget: SpectrogramWidget
# ==============================================================================
class SpectrogramWidget(QWidget):
    """A QWidget that renders a scrolling spectrogram from FFT data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._image: QImage | None = None
        self._colormap = self._create_colormap()
        self._font = QFont("Arial", 8)
        self._sample_rate = 44100
        self._fft_size = 1024
        self._freq_bins: np.ndarray | None = None
        self._db_range = MAX_DB_DISPLAY - MIN_DB_DISPLAY
        if self._db_range <= 0:
            self._db_range = 1.0
        self._recalculate_frequency_bins()

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def _create_colormap(self) -> np.ndarray:
        size = 256
        c = np.array(
            [[68, 1, 84, 255], [59, 82, 139, 255], [33, 145, 140, 255], [94, 201, 98, 255], [253, 231, 37, 255]]
        )
        indices = np.linspace(0, 1, len(c))
        map_indices = np.linspace(0, 1, size)
        r = np.interp(map_indices, indices, c[:, 0])
        g = np.interp(map_indices, indices, c[:, 1])
        b = np.interp(map_indices, indices, c[:, 2])
        return np.column_stack((r, g, b)).astype(np.uint8)

    def _recalculate_frequency_bins(self):
        if self._sample_rate > 0 and self._fft_size > 0:
            self._freq_bins = np.fft.rfftfreq(self._fft_size, d=1.0 / self._sample_rate)
        else:
            self._freq_bins = None

    @Slot(np.ndarray, int, int)
    def update_data(self, fft_frame: np.ndarray | None, sample_rate: int, fft_size: int):
        params_changed = False
        if self._sample_rate != sample_rate:
            self._sample_rate, params_changed = sample_rate, True
        if self._fft_size != fft_size:
            self._fft_size, params_changed = fft_size, True
        if params_changed:
            self._recalculate_frequency_bins()

        if self._image is None or self._image.size() != self.size():
            self._image = QImage(self.size(), QImage.Format.Format_RGB32)
            self._image.fill(Qt.GlobalColor.black)

        painter = QPainter(self._image)
        painter.drawImage(QPoint(-1, 0), self._image)
        painter.end()

        if fft_frame is not None and self._freq_bins is not None:
            magnitudes_db = 20 * np.log10(np.abs(fft_frame) + 1e-12)
            normalized_db = (np.clip(magnitudes_db, MIN_DB_DISPLAY, MAX_DB_DISPLAY) - MIN_DB_DISPLAY) / self._db_range
            color_indices = (normalized_db * (len(self._colormap) - 1)).astype(np.uint8)

            h = self.height()
            nyquist = self._sample_rate / 2.0
            y_coords = np.clip((h - 1) * (self._freq_bins / nyquist), 0, h - 1).astype(int)
            column_x = self.width() - 1

            if column_x >= 0:
                for i in range(len(self._freq_bins)):
                    y = (h - 1) - y_coords[i]
                    color_rgb = self._colormap[color_indices[i]]
                    self._image.setPixelColor(column_x, y, QColor(*color_rgb))
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        if self._image:
            painter.drawImage(0, 0, self._image)

        painter.setPen(QColor(200, 200, 200))
        painter.setFont(self._font)
        fm = QFontMetrics(self._font)
        h, nyquist = self.height(), self._sample_rate / 2.0
        if nyquist <= 0:
            return

        num_labels = max(2, h // 40)
        for i in range(num_labels + 1):
            freq_ratio = i / num_labels
            freq_hz = freq_ratio * nyquist
            if freq_hz < MIN_FREQ_DISPLAY:
                continue
            y = np.clip(h - int(freq_ratio * h), fm.height(), h - 2)
            text = f"{freq_hz / 1000:.1f}k" if freq_hz >= 1000 else f"{freq_hz:.0f}"
            painter.drawText(2, y, text)
        painter.end()


# ==============================================================================
# 2. Custom NodeItem for the Spectrogram Visualizer
# ==============================================================================
class SpectrogramVisualizerNodeItem(NodeItem):
    """Custom NodeItem that embeds the SpectrogramWidget."""

    # --- FIX 1: DEFINE A SUITABLE WIDTH FOR THIS NODE ---
    NODE_SPECIFIC_WIDTH = 270

    def __init__(self, node_logic: "SpectrogramVisualizerNode"):
        # --- FIX 2: PASS THE CUSTOM WIDTH TO THE SUPERCLASS CONSTRUCTOR ---
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.display_widget = SpectrogramWidget()
        self.setContentWidget(self.display_widget)
        node_logic.newDataReady.connect(self._update_visualization)

    @Slot(object, int, int)
    def _update_visualization(self, fft_frame: np.ndarray, sample_rate: int, fft_size: int):
        self.display_widget.update_data(fft_frame, sample_rate, fft_size)

    @Slot()
    def updateFromLogic(self):
        self.display_widget.update_data(None, 44100, 1024)
        super().updateFromLogic()


# ==============================================================================
# 3. Node Logic Class: SpectrogramVisualizerNode
# ==============================================================================
class SpectrogramVisualizerNode(Node):
    NODE_TYPE = "Spectrogram Visualizer"
    CATEGORY = "Visualization"
    DESCRIPTION = "Displays a scrolling spectrogram (waterfall) of a spectral frame stream."
    UI_CLASS = SpectrogramVisualizerNodeItem

    class WrappedSignal(QObject):
        _s = Signal(object, int, int)

        def emit(self, fft_frame: np.ndarray | None, sample_rate: int, fft_size: int):
            try:
                data_copy = fft_frame.copy() if fft_frame is not None else None
                self._s.emit(data_copy, sample_rate, fft_size)
            except RuntimeError as e:
                logger.debug(f"SpectrogramVisualizerNode WrappedSignal: Error emitting signal: {e}")

        def connect(self, slot_func: Slot):
            self._s.connect(slot_func)

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.newDataReady = self.WrappedSignal()
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self._next_ui_update_time = 0

    def process(self, input_data: dict) -> dict:
        current_time = time.monotonic()
        if current_time < self._next_ui_update_time:
            return {}
        self._next_ui_update_time = current_time + UI_UPDATE_INTERVAL_S

        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {}

        fft_data, sample_rate, fft_size = frame.data, frame.sample_rate, frame.fft_size
        if not np.iscomplexobj(fft_data) or fft_data.shape[1] == 0:
            return {}

        mono_fft_frame = np.mean(fft_data, axis=1)
        self.newDataReady.emit(mono_fft_frame, sample_rate, fft_size)
        return {}
