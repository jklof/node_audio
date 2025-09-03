import torch
import numpy as np
import time
import threading
import logging

# --- Node System Imports ---
from node_system import Node
from constants import SpectralFrame

# --- UI and Qt Imports ---
from ui_elements import NodeItem
from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtGui import QPainter, QColor, QImage, QFont, QFontMetrics, QPaintEvent
from PySide6.QtCore import Qt, Signal, Slot, QObject, QPoint, QSize, QRect

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
    """A QWidget that renders a scrolling spectrogram from pre-rendered pixel columns."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._image: QImage | None = None
        self._font = QFont("Arial", 8)
        self._sample_rate = 44100
        self._colormap = self._create_colormap()

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def _create_colormap(self) -> np.ndarray:
        size = 256
        c = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]])
        indices = np.linspace(0, 1, len(c))
        map_indices = np.linspace(0, 1, size)
        r = np.interp(map_indices, indices, c[:, 0])
        g = np.interp(map_indices, indices, c[:, 1])
        b = np.interp(map_indices, indices, c[:, 2])
        rgb_array = np.column_stack((b, g, r, np.zeros(size))).astype(np.uint8)
        return rgb_array.flatten().view(np.uint32)

    @Slot(np.ndarray, int)
    def update_column(self, new_column_rgb: np.ndarray, sample_rate: int):
        """Scrolls the spectrogram and draws a new vertical line of pixel data."""
        self._sample_rate = sample_rate

        if self._image is None or self._image.size() != self.size():
            self._image = QImage(self.size(), QImage.Format.Format_RGB32)
            self._image.fill(Qt.GlobalColor.black)

        painter = QPainter(self._image)
        w, h = self._image.width(), self._image.height()

        if w > 1:
            source_rect = QRect(1, 0, w - 1, h)
            dest_point = QPoint(0, 0)
            painter.drawImage(dest_point, self._image, source_rect)

        if len(new_column_rgb) == h:
            column_image = QImage(new_column_rgb.data, 1, h, QImage.Format.Format_RGB32)
            painter.drawImage(w - 1, 0, column_image)
        else:
            painter.fillRect(w - 1, 0, 1, h, Qt.GlobalColor.black)

        painter.end()
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
# 2. Custom NodeItem
# ==============================================================================
class SpectrogramVisualizerNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 270

    def __init__(self, node_logic: "SpectrogramVisualizerNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)
        self.display_widget = SpectrogramWidget()
        self.setContentWidget(self.display_widget)
        node_logic.newDataReady.connect(self._on_new_data_received)

    @Slot(object, int)
    def _on_new_data_received(self, new_column_rgb: np.ndarray | None, sample_rate: int):
        if new_column_rgb is not None:
            self.display_widget.update_column(new_column_rgb, sample_rate)

    @Slot()
    def updateFromLogic(self):
        """
        This method is called during graph resyncs (e.g., when moving a node).
        We explicitly DO NOT update the visual content here to prevent artifacts like
        black lines. The widget will hold its last drawn state until new data
        arrives from the processing thread via the newDataReady signal.
        """
        super().updateFromLogic()  # Call parent for any other essential updates


# ==============================================================================
# 3. Node Logic Class
# ==============================================================================
class SpectrogramVisualizerNode(Node):
    NODE_TYPE = "Spectrogram Visualizer"
    CATEGORY = "Visualization"
    DESCRIPTION = "Displays a scrolling spectrogram (waterfall) of a spectral frame stream."
    UI_CLASS = SpectrogramVisualizerNodeItem

    class WrappedSignal(QObject):
        _s = Signal(object, int)

        def emit(self, new_column_rgb: np.ndarray | None, sample_rate: int):
            try:
                self._s.emit(new_column_rgb, sample_rate)
            except RuntimeError as e:
                logger.debug(f"SpectrogramVisualizerNode WrappedSignal: Error emitting signal: {e}")

        def connect(self, slot_func: Slot):
            self._s.connect(slot_func)

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.newDataReady = self.WrappedSignal()
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self._next_ui_update_time = 0
        self._colormap = self._create_colormap()
        self._db_range = MAX_DB_DISPLAY - MIN_DB_DISPLAY
        self._cached_params_key = None
        self._cached_y_coords_indices: torch.Tensor | None = None

    def _create_colormap(self) -> np.ndarray:
        size = 256
        c = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]])
        indices = np.linspace(0, 1, len(c))
        map_indices = np.linspace(0, 1, size)
        r = np.interp(map_indices, indices, c[:, 0])
        g = np.interp(map_indices, indices, c[:, 1])
        b = np.interp(map_indices, indices, c[:, 2])
        rgb_array = np.column_stack((b, g, r, np.zeros(size))).astype(np.uint8)
        return rgb_array.flatten().view(np.uint32)

    def _prepare_render_cache(self, frame: SpectralFrame, display_height: int):
        self._cached_params_key = (frame.fft_size, frame.sample_rate, display_height)
        full_freqs = torch.fft.rfftfreq(frame.fft_size, d=1.0 / frame.sample_rate)
        nyquist = frame.sample_rate / 2.0
        y_coords_float = torch.clamp((display_height - 1) * (full_freqs / nyquist), 0, display_height - 1)
        self._cached_y_coords_indices = (display_height - 1) - y_coords_float.long()
        logger.debug(f"[{self.name}] Recalculated spectrogram render cache.")

    def process(self, input_data: dict) -> dict:
        current_time = time.monotonic()
        if current_time < self._next_ui_update_time:
            return {}
        self._next_ui_update_time = current_time + UI_UPDATE_INTERVAL_S
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame) or not torch.is_complex(frame.data):
            return {}
        try:
            display_height = 150
            params_key = (frame.fft_size, frame.sample_rate, display_height)
            if self._cached_params_key != params_key:
                self._prepare_render_cache(frame, display_height)

            mono_fft_frame = torch.mean(torch.abs(frame.data), dim=0)
            magnitudes_db = 20 * torch.log10(mono_fft_frame + 1e-12)
            normalized_db = (
                torch.clamp(magnitudes_db, MIN_DB_DISPLAY, MAX_DB_DISPLAY) - MIN_DB_DISPLAY
            ) / self._db_range
            column_tensor = torch.zeros(display_height, dtype=torch.float32)
            column_tensor.scatter_reduce_(
                0, self._cached_y_coords_indices, normalized_db, reduce="amax", include_self=False
            )
            final_color_indices = (column_tensor * (len(self._colormap) - 1)).long().numpy()
            pixel_column = self._colormap[final_color_indices]
            self.newDataReady.emit(pixel_column, frame.sample_rate)
        except Exception as e:
            logger.error(f"[{self.name}] Error processing spectrogram frame: {e}", exc_info=True)
        return {}
