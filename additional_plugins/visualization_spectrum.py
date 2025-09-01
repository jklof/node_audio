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
from PySide6.QtGui import QPainter, QColor, QPen, QPolygonF, QPaintEvent, QFont, QFontMetrics
from PySide6.QtCore import Qt, Signal, Slot, QObject, QPointF, QSize

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
MIN_DB_DISPLAY = -70.0
MAX_DB_DISPLAY = 6.0
UI_UPDATE_INTERVAL_S = 0.033
SPECTRUM_SMOOTHING_FACTOR = 0.4
MIN_FREQ_DISPLAY = 20.0


# ==============================================================================
# 1. Custom Drawing Widget: STFTSpectrumDisplayWidget
# ==============================================================================
class STFTSpectrumDisplayWidget(QWidget):
    """QWidget for rendering the spectrum, applying temporal smoothing internally."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._raw_magnitudes_db: np.ndarray | None = None
        self._smoothed_magnitudes_db: np.ndarray | None = None
        self._frequencies: np.ndarray | None = None
        self._sample_rate = 44100
        self._log_freq_axis = True
        self._min_db = MIN_DB_DISPLAY
        self._max_db = MAX_DB_DISPLAY

        self._grid_pen = QPen(QColor(40, 40, 40), 1)
        self._label_pen = QPen(QColor(150, 150, 150))
        self._spectrum_fill_brush = QColor(50, 150, 250, 150)
        self._spectrum_peak_pen = QPen(QColor(255, 255, 255, 180), 1.5)
        self._font = QFont("Arial", 8)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    @Slot(np.ndarray, np.ndarray, int)
    def update_data(self, magnitudes_db: np.ndarray | None, frequencies: np.ndarray | None, sample_rate: int):
        if sample_rate > 0:
            self._sample_rate = sample_rate
        if magnitudes_db is None or frequencies is None:
            if self._raw_magnitudes_db is not None:
                self._raw_magnitudes_db = None
                self._smoothed_magnitudes_db = None
                self.update()
            return

        self._raw_magnitudes_db = magnitudes_db
        if self._smoothed_magnitudes_db is None or self._smoothed_magnitudes_db.shape != self._raw_magnitudes_db.shape:
            self._smoothed_magnitudes_db = self._raw_magnitudes_db.copy()
        else:
            self._smoothed_magnitudes_db = (
                SPECTRUM_SMOOTHING_FACTOR * self._raw_magnitudes_db
                + (1 - SPECTRUM_SMOOTHING_FACTOR) * self._smoothed_magnitudes_db
            )
        self._frequencies = frequencies
        self.update()

    def _prepare_polygon(self) -> QPolygonF | None:
        mags = self._smoothed_magnitudes_db
        freqs = self._frequencies
        if mags is None or freqs is None or len(mags) == 0:
            return None

        w, h = self.width(), self.height()
        padding = 5
        plot_width, plot_height = w - 2 * padding, h - 2 * padding
        if plot_width <= 0 or plot_height <= 0:
            return None

        max_freq_display = self._sample_rate / 2.0
        valid_indices = freqs >= MIN_FREQ_DISPLAY
        freqs, mags = freqs[valid_indices], mags[valid_indices]
        if len(freqs) == 0:
            return None

        polygon = QPolygonF()
        db_range = self._max_db - self._min_db
        if db_range <= 0:
            db_range = 1.0

        def map_db_to_y(db):
            normalized_db = (np.clip(db, self._min_db, self._max_db) - self._min_db) / db_range
            return padding + plot_height * (1.0 - normalized_db)

        log_min_freq = np.log10(max(1.0, MIN_FREQ_DISPLAY))
        log_max_freq = np.log10(max(log_min_freq + 1e-6, max_freq_display))
        log_freq_range = log_max_freq - log_min_freq
        if log_freq_range <= 0:
            log_freq_range = 1.0

        def map_freq_to_x(freq):
            log_freq = np.log10(max(1.0, freq))
            normalized_log_freq = (log_freq - log_min_freq) / log_freq_range
            return padding + plot_width * normalized_log_freq

        start_x = map_freq_to_x(freqs[0])
        bottom_y = padding + plot_height
        polygon.append(QPointF(start_x, bottom_y))

        last_x = -1
        for freq, mag_db in zip(freqs, mags):
            x = map_freq_to_x(freq)
            if abs(x - last_x) > 0.5:
                y = map_db_to_y(mag_db)
                polygon.append(QPointF(x, y))
                last_x = x
        if last_x != -1:
            polygon.append(QPointF(last_x, bottom_y))
        return polygon

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
                text_y_pos = h - 2
                painter.drawText(int(x - fm.horizontalAdvance(text) / 2), text_y_pos, text)

        # --- Draw Spectrum ---
        polygon = self._prepare_polygon()
        if polygon and polygon.count() > 2:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._spectrum_fill_brush)
            painter.drawPolygon(polygon)
            painter.setPen(self._spectrum_peak_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            data_points = polygon.toList()[1:-1]
            if len(data_points) > 0:
                painter.drawPolyline(QPolygonF(data_points))
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

    @Slot(object, object, int)
    def _update_visualization(self, magnitudes_db: np.ndarray, frequencies: np.ndarray, sample_rate: int):
        self.spectrum_widget.update_data(magnitudes_db, frequencies, sample_rate)

    @Slot()
    def updateFromLogic(self):
        self.spectrum_widget.update_data(None, None, 44100)
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
        _s = Signal(object, object, int)

        def emit(self, magnitudes, frequencies, samplerate):
            try:
                m_copy = magnitudes.copy() if magnitudes is not None else None
                f_copy = frequencies.copy() if frequencies is not None else None
                self._s.emit(m_copy, f_copy, samplerate)
            except RuntimeError as e:
                logger.debug(f"SpectrumVisualizerNode WrappedSignal: Error emitting signal: {e}")

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
            self.newDataReady.emit(None, None, 44100)
            return {}

        fft_data = frame.data
        if not np.iscomplexobj(fft_data) or fft_data.shape[1] == 0:
            return {}

        try:
            magnitudes_raw = np.abs(fft_data)
            mono_magnitudes = np.mean(magnitudes_raw, axis=1) if fft_data.shape[1] > 1 else magnitudes_raw.flatten()

            window_sum = np.sum(frame.analysis_window)
            db_ref = max(1e-9, window_sum / 2.0)

            magnitudes_db = 20 * np.log10(mono_magnitudes / db_ref + 1e-12)
            frequencies = np.fft.rfftfreq(frame.fft_size, d=1.0 / frame.sample_rate)

            self.newDataReady.emit(magnitudes_db, frequencies, frame.sample_rate)
        except Exception as e:
            logger.error(f"[{self.name}] Error calculating spectrum: {e}", exc_info=True)

        return {}
