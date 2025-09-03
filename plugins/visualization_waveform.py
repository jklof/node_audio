import torch
import numpy as np
import threading
import logging
import time

from node_system import Node
from ui_elements import NodeItem

from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PySide6.QtGui import QPainter, QColor, QPen, QPaintEvent, QPainterPath
from PySide6.QtCore import Qt, Slot, Signal, QObject, QPointF, QSize

UI_UPDATE_INTERVAL = 0.033  # seconds Default: 0.033 for ~30 FPS

# Configure logging
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Custom Drawing Widget
# ==============================================================================


class WaveformWidget(QWidget):
    """A custom widget to draw multi-channel waveforms from torch.Tensors."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Stores waveform data as a 2D torch.Tensor (channels, samples) or None
        self._waveform_data: torch.Tensor | None = None
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
        """Provide a valid size hint to the layout system for correct height calculation."""
        return QSize(super().sizeHint().width(), self.minimumHeight())

    def setData(self, data: torch.Tensor | None):
        """
        Set the waveform data. Expects data to be None or a torch.Tensor with shape (channels, samples).
        This method is called from the GUI thread.
        """
        if data is None:
            self._waveform_data = None
            self.update()
            return

        if not isinstance(data, torch.Tensor) or data.numel() == 0:
            self._waveform_data = None
            self.update()
            return

        processed_data = None
        if data.ndim == 1:  # Mono signal, add channel dimension
            processed_data = data.to(torch.float32).unsqueeze(0)
        elif data.ndim == 2:  # Already (channels, samples)
            processed_data = data.to(torch.float32)

        self._waveform_data = processed_data
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

        num_data_channels, num_samples = self._waveform_data.shape
        if num_samples == 0 or num_data_channels == 0:
            painter.end()
            return

        w = self.width()
        h = self.height()
        half_h = h / 2.0

        x_scale = w / float(max(1, num_samples - 1))
        y_scale = half_h

        for ch_idx in range(num_data_channels):
            channel_waveform = self._waveform_data[ch_idx, :]
            pen_color = self.channel_colors[ch_idx % len(self.channel_colors)]
            painter.setPen(QPen(pen_color, 1.5))

            path = QPainterPath()
            first_y_val = torch.clamp(channel_waveform[0], -1.0, 1.0).item()
            path.moveTo(0, center_y - first_y_val * y_scale)

            for i in range(1, num_samples):
                x = i * x_scale
                y_val = torch.clamp(channel_waveform[i], -1.0, 1.0).item()
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

    @Slot(object)
    def _update_waveform_display(self, data: torch.Tensor | None):
        if self.waveform_widget:
            self.waveform_widget.setData(data)

    @Slot()
    def updateFromLogic(self):
        """Initialize display on load/creation."""
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
        # Allow emitting None or torch.Tensor
        _s = Signal(object)

        def __init__(self):
            super().__init__()

        def emit(self, data: torch.Tensor | None):
            data_to_emit = data.clone() if data is not None else None
            try:
                self._s.emit(data_to_emit)
            except RuntimeError as e:
                logger.debug(f"WaveformVisualizerNode WrappedSignal: Error emitting signal: {e}")

        def connect(self, x):
            self._s.connect(x)

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.newDataReady = self.WrappedSignal()
        self.add_input("in", data_type=torch.Tensor)
        self.add_output("out", data_type=torch.Tensor)

        self._next_ui_update_time = 0
        logger.debug(f"WaveformVisualizerNode [{self.name}] initialized.")

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")

        current_time = time.monotonic()
        if current_time < self._next_ui_update_time:
            return {"out": signal}

        self._next_ui_update_time = current_time + UI_UPDATE_INTERVAL
        signal_to_emit = None

        if isinstance(signal, torch.Tensor) and signal.numel() > 0 and signal.ndim == 2:
            _num_channels, current_samples = signal.shape
            MAX_DISPLAY_SAMPLES = 256

            if current_samples > MAX_DISPLAY_SAMPLES:
                # Create integer indices for downsampling
                indices = torch.from_numpy(np.linspace(0, current_samples - 1, MAX_DISPLAY_SAMPLES, dtype=np.int64))
                # Sample along the samples dimension (dim=1)
                signal_to_emit = signal[:, indices]
            else:
                signal_to_emit = signal

        # The emit method will clone the tensor if it's not None
        self.newDataReady.emit(signal_to_emit)
        return {"out": signal}

    def stop(self):
        # Clear display on stop
        self.newDataReady.emit(None)

    def serialize_extra(self) -> dict:
        return {}

    def deserialize_extra(self, data: dict):
        pass
