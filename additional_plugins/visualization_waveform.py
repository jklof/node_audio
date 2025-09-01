import numpy as np
import threading
import logging
import time
from collections import deque

from node_system import Node
from ui_elements import NodeItem

from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PySide6.QtGui import QPainter, QColor, QPen, QPaintEvent, QPainterPath
from PySide6.QtCore import Qt, Slot, Signal, QObject, QPointF, QRectF, QSize

from constants import DEFAULT_DTYPE

UI_UPDATE_INTERVAL = 0.033  # seconds Default: 0.033 for ~30 FPS

# Configure logging
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Custom Drawing Widget
# ==============================================================================


class WaveformWidget(QWidget):
    """A custom widget to draw multi-channel waveforms."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Stores waveform data as a 2D numpy array (samples, channels) or None
        self._waveform_data: np.ndarray | None = None
        self._background_color = QColor(30, 30, 30)
        self.channel_colors = [
            QColor("cyan"),
            QColor("magenta"),
            QColor("yellow"),
            QColor("lightgreen"),
            QColor("orange"),
            QColor("lightcoral"),
            QColor("skyblue"),
            QColor("pink"),
            QColor(200, 200, 200),  # A light gray for >8th channel
        ]
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(50)

    def sizeHint(self) -> QSize:
        """
        Provide a valid size hint to the layout system.
        The NodeItem container uses this to correctly calculate its total height.
        """
        # The width is expanding, so we can leave it to the parent.
        # The height is fixed, so we return the minimum height we've set.
        return QSize(super().sizeHint().width(), self.minimumHeight())

    def setData(self, data: np.ndarray | None):
        """
        Set the waveform data. Expects data to be None or a 2D numpy array (samples, channels).
        This method is called from the GUI thread.
        """
        if data is None:
            self._waveform_data = None
            self.update()
            return

        if not isinstance(data, np.ndarray) or data.size == 0:
            self._waveform_data = None
            self.update()
            return

        processed_data = None
        if data.ndim == 1:  # Single channel data
            if data.shape[0] > 0:  # Must have samples
                processed_data = data.astype(np.float32).reshape(-1, 1)
        elif data.ndim == 2:  # Potentially multi-channel
            if data.shape[0] > 0 and data.shape[1] > 0:  # Must have samples and channels
                processed_data = data.astype(np.float32)
        # Else, data is invalid (e.g. ndim > 2, or was empty after checks)

        self._waveform_data = processed_data  # Will be None if processing failed
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self._background_color)

        # Draw center line
        painter.setPen(QPen(QColor(80, 80, 80), 1, Qt.PenStyle.DotLine))
        center_y = self.height() / 2.0
        painter.drawLine(0, int(center_y), self.width(), int(center_y))

        if self._waveform_data is None:
            painter.end()
            return

        num_samples, num_data_channels = self._waveform_data.shape
        if num_samples == 0 or num_data_channels == 0:
            painter.end()
            return

        w = self.width()
        h = self.height()
        half_h = h / 2.0

        x_scale = w / float(max(1, num_samples - 1))
        y_scale = half_h

        for ch_idx in range(num_data_channels):
            channel_waveform = self._waveform_data[:, ch_idx]
            pen_color = self.channel_colors[ch_idx % len(self.channel_colors)]
            painter.setPen(QPen(pen_color, 1.5))

            path = QPainterPath()
            first_y_val = np.clip(channel_waveform[0], -1.0, 1.0)
            path.moveTo(0, center_y - first_y_val * y_scale)

            for i in range(1, num_samples):
                x = i * x_scale
                y_val = np.clip(channel_waveform[i], -1.0, 1.0)
                y = center_y - y_val * y_scale
                path.lineTo(x, y)

            painter.drawPath(path)

        painter.end()


# ==============================================================================
# 2. Custom NodeItem for the Visualizer
# ==============================================================================


class WaveformVisualizerNodeItem(NodeItem):
    """Custom NodeItem embedding the WaveformWidget."""

    def __init__(self, node_logic):
        super().__init__(node_logic)
        self.waveform_widget = WaveformWidget()
        self.setContentWidget(self.waveform_widget)
        self.node_logic.newDataReady.connect(self._update_waveform_display)

    @Slot(np.ndarray)
    def _update_waveform_display(self, data: np.ndarray | None):  # data can be None now
        if self.waveform_widget:
            self.waveform_widget.setData(data)

    @Slot()
    def updateFromLogic(self):
        """Initialize display on load/creation."""
        # Pass None to clear/initialize the widget
        self.waveform_widget.setData(None)
        super().updateFromLogic()


# ==============================================================================
# 3. Node Logic Class
# ==============================================================================


class WaveformVisualizerNode(Node):
    NODE_TYPE = "Waveform Visualizer"
    CATEGORY = "Visualization"
    DESCRIPTION = "Displays the waveform of all channels in the incoming audio signal."
    UI_CLASS = WaveformVisualizerNodeItem

    class WrappedSignal(QObject):
        # Allow emitting None or np.ndarray
        _s = Signal(object)  # Use object to allow None or np.ndarray

        def __init__(self):
            super().__init__()

        def emit(self, data: np.ndarray | None):
            data_to_emit = None
            if data is not None:
                # Ensure it's a copy if it's an ndarray
                data_to_emit = data.copy()

            try:
                self._s.emit(data_to_emit)
            except RuntimeError as e:
                # This can happen if the QObject (or its parent) is being deleted
                logger.debug(f"WaveformVisualizerNode WrappedSignal: Error emitting signal: {e}")

        def connect(self, x):
            self._s.connect(x)

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.newDataReady = self.WrappedSignal()
        self.add_input("in", data_type=np.ndarray)  # Accepts multi-channel
        self.add_output("out", data_type=np.ndarray)

        self.lock = threading.Lock()  # Not used in this version, but good practice
        self._next_ui_update_time = 0
        logger.debug(f"WaveformVisualizerNode [{self.name}] initialized.")

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")

        current_time = time.monotonic()
        if current_time < self._next_ui_update_time:
            return {"out": signal}

        self._next_ui_update_time = current_time + UI_UPDATE_INTERVAL

        signal_to_emit = None  # Default to None (clears display)

        if signal is not None and isinstance(signal, np.ndarray) and signal.size > 0:
            # At this point, signal is a non-empty numpy array
            MAX_DISPLAY_SAMPLES = 256
            current_samples = signal.shape[0]

            # Ensure signal is 2D [samples, channels]
            if signal.ndim == 1:
                _signal_2d = signal.reshape(-1, 1)
            elif signal.ndim == 2:
                _signal_2d = signal
            else:  # Invalid dimensions
                _signal_2d = None

            if _signal_2d is not None and _signal_2d.shape[1] > 0:  # Must have channels
                current_samples = _signal_2d.shape[0]  # Re-evaluate samples from 2D version
                if current_samples > MAX_DISPLAY_SAMPLES:
                    indices = np.linspace(0, current_samples - 1, MAX_DISPLAY_SAMPLES, dtype=np.int32)
                    signal_to_emit = _signal_2d[indices, :]
                else:
                    signal_to_emit = _signal_2d  # Already a copy will be made by emit()
            # If _signal_2d is None or has 0 channels, signal_to_emit remains None

        # self.newDataReady.emit makes a copy if signal_to_emit is an ndarray
        self.newDataReady.emit(signal_to_emit)
        return {"out": signal}

    def start(self):
        pass

    def stop(self):
        # Optionally clear display on stop
        self.newDataReady.emit(None)
        pass

    def remove(self):
        pass

    def serialize_extra(self) -> dict:
        return {}

    def deserialize_extra(self, data: dict):
        pass
