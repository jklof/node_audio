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
from PySide6.QtGui import QPainter, QColor, QPen, QPaintEvent, QFont, QFontMetrics, QPainterPath
from PySide6.QtCore import Qt, Signal, Slot, QObject, QPointF, QSize

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
MIN_DB_DISPLAY = -70.0
MAX_DB_DISPLAY = 6.0
UI_UPDATE_INTERVAL_S = 0.033
MIN_FREQ_DISPLAY = 20.0
DISPLAY_BINS = 256


# ==============================================================================
# 1. Custom Drawing Widget: STFTSpectrumDisplayWidget
# ==============================================================================
class STFTSpectrumDisplayWidget(QWidget):
    """QWidget for rendering pre-computed spectrum plot points."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._plot_points: np.ndarray | None = None
        self._sample_rate = 44100

        self._grid_pen = QPen(QColor(40, 40, 40), 1)
        self._label_pen = QPen(QColor(150, 150, 150))
        self._spectrum_fill_brush = QColor(50, 150, 250, 150)
        self._spectrum_peak_pen = QPen(QColor(255, 255, 255, 180), 1.5)
        self._font = QFont("Arial", 8)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    @Slot(object, int)
    def update_data(self, plot_points: np.ndarray | None, sample_rate: int):
        self._plot_points = plot_points
        self._sample_rate = sample_rate if sample_rate > 0 else 44100
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        w, h = self.width(), self.height()
        padding = 5
        plot_width, plot_height = w - 2 * padding, h - 2 * padding
        if plot_width <= 0 or plot_height <= 0:
            return

        # --- Draw Grid & Labels ---
        painter.setPen(self._grid_pen)
        num_db_lines = 6
        for i in range(num_db_lines + 1):
            y = padding + plot_height * (i / num_db_lines)
            painter.drawLine(padding, int(y), w - padding, int(y))

        painter.setPen(self._label_pen)
        painter.setFont(self._font)
        fm = QFontMetrics(self._font)
        max_freq_grid = self._sample_rate / 2.0
        log_min_freq = np.log10(max(1.0, MIN_FREQ_DISPLAY))
        log_max_freq = np.log10(max(log_min_freq + 1e-6, max_freq_grid))
        log_freq_range = log_max_freq - log_min_freq
        if log_freq_range <= 0:
            log_freq_range = 1.0

        for freq in [100, 1000, 10000]:
            if MIN_FREQ_DISPLAY <= freq <= max_freq_grid:
                log_freq = np.log10(max(1.0, freq))
                norm_x = (log_freq - log_min_freq) / log_freq_range
                x = padding + plot_width * norm_x
                painter.setPen(self._grid_pen)
                painter.drawLine(int(x), padding, int(x), h - padding)
                painter.setPen(self._label_pen)
                text = f"{freq/1000:.0f}k" if freq >= 1000 else str(freq)
                painter.drawText(int(x - fm.horizontalAdvance(text) / 2), h - 2, text)

        # --- Draw Spectrum from pre-computed points ---
        if self._plot_points is None or len(self._plot_points) < 2:
            painter.end()
            return

        # --- Create a QPainterPath for the filled polygon ---
        fill_path = QPainterPath()

        # 1. Scale the first normalized point to pixel coordinates
        first_x_norm, first_y_norm = self._plot_points[0]
        start_px = padding + first_x_norm * plot_width
        start_py = padding + first_y_norm * plot_height

        # 2. Move to the bottom-left of the plot area to start the fill
        fill_path.moveTo(start_px, padding + plot_height)

        # 3. Add a line up to the first data point
        fill_path.lineTo(start_px, start_py)

        # 4. Iterate through the rest of the points and add lines
        for i in range(1, len(self._plot_points)):
            x_norm, y_norm = self._plot_points[i]
            px = padding + x_norm * plot_width
            py = padding + y_norm * plot_height
            fill_path.lineTo(px, py)

        # 5. Add a line from the last data point down to the bottom to close the shape
        last_x_norm, _ = self._plot_points[-1]
        last_px = padding + last_x_norm * plot_width
        fill_path.lineTo(last_px, padding + plot_height)

        # --- Draw the fill and the peak line ---
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._spectrum_fill_brush)
        painter.drawPath(fill_path)

        # Create a separate path for the polyline on top
        line_path = QPainterPath()
        line_path.moveTo(start_px, start_py)
        for i in range(1, len(self._plot_points)):
            x_norm, y_norm = self._plot_points[i]
            px = padding + x_norm * plot_width
            py = padding + y_norm * plot_height
            line_path.lineTo(px, py)

        painter.setPen(self._spectrum_peak_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(line_path)

        painter.end()


# ==============================================================================
# 2. Custom NodeItem for the Spectrum Visualizer
# ==============================================================================
class STFTSpectrumVisualizerNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 270

    def __init__(self, node_logic: "STFTSpectrumVisualizerNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)
        self.spectrum_widget = STFTSpectrumDisplayWidget()
        self.setContentWidget(self.spectrum_widget)
        node_logic.newDataReady.connect(self._update_visualization)

    @Slot(object, int)
    def _update_visualization(self, plot_points: np.ndarray | None, sample_rate: int):
        self.spectrum_widget.update_data(plot_points, sample_rate)

    @Slot()
    def updateFromLogic(self):
        self.spectrum_widget.update_data(None, 44100)
        super().updateFromLogic()


# ==============================================================================
# 3. Node Logic Class: STFTSpectrumVisualizerNode
# ==============================================================================
class STFTSpectrumVisualizerNode(Node):
    NODE_TYPE = "Spectrum Visualizer"
    CATEGORY = "Visualization"
    DESCRIPTION = "Displays the frequency spectrum of a spectral frame stream."
    UI_CLASS = STFTSpectrumVisualizerNodeItem

    class WrappedSignal(QObject):
        _s = Signal(object, int)

        def emit(self, plot_points, samplerate):
            try:
                self._s.emit(plot_points, samplerate)
            except RuntimeError as e:
                logger.debug(f"SpectrumVisualizerNode WrappedSignal: Error emitting signal: {e}")

        def connect(self, slot_func: Slot):
            self._s.connect(slot_func)

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.newDataReady = self.WrappedSignal()
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self._next_ui_update_time = 0

        self._cached_fft_size = -1
        self._cached_sample_rate = -1
        self._cached_log_freqs_x: torch.Tensor | None = None
        self._cached_indices: torch.Tensor | None = None

    def _prepare_display_bins(self, frame: SpectralFrame):
        self._cached_fft_size = frame.fft_size
        self._cached_sample_rate = frame.sample_rate

        full_freqs = torch.fft.rfftfreq(frame.fft_size, d=1.0 / frame.sample_rate)

        log_min_freq = torch.log10(torch.tensor(max(1.0, MIN_FREQ_DISPLAY)))
        log_max_freq = torch.log10(torch.tensor(max(log_min_freq + 1e-6, frame.sample_rate / 2.0)))
        log_freq_range = log_max_freq - log_min_freq
        if log_freq_range <= 0:
            log_freq_range = 1.0

        log_spaced_freqs = torch.logspace(log_min_freq, log_max_freq, DISPLAY_BINS)

        self._cached_indices = torch.searchsorted(full_freqs, log_spaced_freqs)
        self._cached_indices.clamp_(0, len(full_freqs) - 1)

        log_freqs = torch.log10(torch.clamp(full_freqs[self._cached_indices], min=1.0))
        self._cached_log_freqs_x = (log_freqs - log_min_freq) / log_freq_range

        logger.debug(f"[{self.name}] Recalculated display bins for FFT size {frame.fft_size}.")

    def process(self, input_data: dict) -> dict:
        current_time = time.monotonic()
        if current_time < self._next_ui_update_time:
            return {}
        self._next_ui_update_time = current_time + UI_UPDATE_INTERVAL_S

        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame) or not torch.is_complex(frame.data) or frame.data.shape[1] == 0:
            self.newDataReady.emit(None, 44100)
            return {}

        if self._cached_fft_size != frame.fft_size or self._cached_sample_rate != frame.sample_rate:
            self._prepare_display_bins(frame)

        try:
            magnitudes_raw = torch.abs(frame.data)
            mono_magnitudes = (
                torch.mean(magnitudes_raw, dim=0) if magnitudes_raw.shape[0] > 1 else magnitudes_raw.squeeze(0)
            )

            downsampled_mags = mono_magnitudes[self._cached_indices]

            window_sum = torch.sum(frame.analysis_window)
            db_ref = max(1e-9, window_sum.item() / 2.0)
            magnitudes_db = 20 * torch.log10(downsampled_mags / db_ref + 1e-12)

            db_range = MAX_DB_DISPLAY - MIN_DB_DISPLAY
            normalized_db = (torch.clamp(magnitudes_db, MIN_DB_DISPLAY, MAX_DB_DISPLAY) - MIN_DB_DISPLAY) / db_range

            y_coords = 1.0 - normalized_db

            plot_points = torch.stack((self._cached_log_freqs_x, y_coords), dim=1).numpy()

            self.newDataReady.emit(plot_points, frame.sample_rate)

        except Exception as e:
            logger.error(f"[{self.name}] Error calculating spectrum: {e}", exc_info=True)

        return {}
