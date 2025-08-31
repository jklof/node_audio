import numpy as np
import threading
import logging
import time
from collections import deque
from typing import Deque, Optional, Any

from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING

from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QLabel
from PySide6.QtGui import QPainter, QColor, QPen, QPaintEvent, QPainterPath
from PySide6.QtCore import Qt, Slot, Signal, QObject, QPointF, QSize

# --- Constants for this plugin ---
UI_UPDATE_INTERVAL = 0.033  # In seconds, for ~30 FPS
PLOT_HISTORY_LENGTH = 256   # Number of float values to store and plot

# Configure logging for this module
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Custom Drawing Widget
# ==============================================================================

class ValuePlotWidget(QWidget):
    """A custom widget dedicated to drawing a time-series plot of values."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._plot_data: Optional[Deque[float]] = None
        self._background_color = QColor(30, 30, 30)
        self._line_color = QColor("orange")
        self._grid_color = QColor(80, 80, 80)
        self._label_color = QColor(180, 180, 180)
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(60)

    def sizeHint(self) -> QSize:
        """Provide a size hint for layout management within the NodeItem."""
        return QSize(super().sizeHint().width(), self.minimumHeight())

    def setData(self, data: Optional[Deque[float]]):
        """
        Sets the plot data. This method is called from the GUI thread.
        Expects a deque of floats or None to clear the plot.
        """
        self._plot_data = data
        self.update()  # Schedule a repaint

    def paintEvent(self, event: QPaintEvent):
        """Renders the plot."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self._background_color)

        if not self._plot_data or len(self._plot_data) < 2:
            painter.end()
            return

        w = self.width()
        h = self.height()
        
        # --- Determine Data Range for Y-Axis Scaling ---
        min_val = min(self._plot_data)
        max_val = max(self._plot_data)
        data_range = max_val - min_val
        if data_range < 1e-9:  # Avoid division by zero if data is flat
            data_range = 1.0
            min_val = 0.0
        
        # Add padding to the range for better visualization
        padding = data_range * 0.1
        plot_min_y = min_val - padding
        plot_max_y = max_val + padding
        plot_range_y = plot_max_y - plot_min_y
        if plot_range_y < 1e-9: plot_range_y = 1.0


        # --- Draw Grid and Labels ---
        font = painter.font()
        font.setPixelSize(10)
        painter.setFont(font)
        painter.setPen(QPen(self._label_color))
        painter.drawText(5, 12, f"{max_val:.3f}")
        painter.drawText(5, h - 5, f"{min_val:.3f}")
        painter.setPen(QPen(self._grid_color, 1, Qt.PenStyle.DotLine))
        painter.drawLine(0, h // 2, w, h // 2)

        # --- Prepare for Plotting ---
        path = QPainterPath()
        points = list(self._plot_data)
        num_points = len(points)
        x_scale = w / float(max(1, PLOT_HISTORY_LENGTH - 1))

        # --- Build the Path ---
        first_val = points[0]
        y_coord = h - ((first_val - plot_min_y) / plot_range_y * h)
        path.moveTo(0, y_coord)
        
        for i in range(1, num_points):
            x = i * x_scale
            y_val = points[i]
            y = h - ((y_val - plot_min_y) / plot_range_y * h)
            path.lineTo(x, y)

        # --- Draw the Path ---
        painter.setPen(QPen(self._line_color, 1.5))
        painter.drawPath(path)
        painter.end()

# ==============================================================================
# 2. Custom NodeItem for the Visualizer
# ==============================================================================

class ValuePlotterNodeItem(NodeItem):
    """Custom NodeItem that embeds the ValuePlotWidget."""

    def __init__(self, node_logic: "ValuePlotterNode"):
        super().__init__(node_logic)
        self.plot_widget = ValuePlotWidget()
        self.setContentWidget(self.plot_widget)
        # Connect the signal from the logic node to the UI update slot
        self.node_logic.newDataReady.connect(self._update_plot_display)

    @Slot(object)
    def _update_plot_display(self, data: Optional[Deque[float]]):
        """Receives data from the logic node and passes it to the widget."""
        if self.plot_widget:
            self.plot_widget.setData(data)

    @Slot()
    def updateFromLogic(self):
        """Initializes the display when the node is loaded or created."""
        self.plot_widget.setData(None)
        super().updateFromLogic()

# ==============================================================================
# 3. Node Logic Class
# ==============================================================================

class ValuePlotterNode(Node):
    NODE_TYPE = "Value Plotter"
    CATEGORY = "Visualization"
    DESCRIPTION = "Displays a time-series plot of single values."
    UI_CLASS = ValuePlotterNodeItem

    class WrappedSignal(QObject):
        """A QObject wrapper for thread-safe signal emission."""
        # Use `object` to allow emitting None or a deque
        _s = Signal(object)

        def __init__(self):
            super().__init__()

        def emit(self, data: Optional[Deque[float]]):
            # A copy is made before calling this to ensure thread safety.
            try:
                self._s.emit(data)
            except RuntimeError as e:
                logger.debug(f"ValuePlotterNode WrappedSignal: Error emitting signal: {e}")

        def connect(self, slot_func: Slot):
            self._s.connect(slot_func)

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.newDataReady = self.WrappedSignal()

        # Define sockets
        self.add_input("in", data_type=float)
        self.add_output("out", data_type=float) # Pass-through

        # Internal state
        self._history_deque = deque(maxlen=PLOT_HISTORY_LENGTH)
        self._next_ui_update_time = 0
        logger.debug(f"ValuePlotterNode [{self.name}] initialized.")

    def process(self, input_data: dict) -> dict:
        signal_value = input_data.get("in")
        
        # --- Data Collection ---
        # Ensure we have a valid float to plot, otherwise plot zero.
        if signal_value is not None and isinstance(signal_value, (float, int, np.number)):
            self._history_deque.append(float(signal_value))
        else:
            self._history_deque.append(0.0)

        # --- UI Update Throttling ---
        current_time = time.monotonic()
        if current_time >= self._next_ui_update_time:
            self._next_ui_update_time = current_time + UI_UPDATE_INTERVAL
            
            # Emit a copy of the deque for thread safety
            self.newDataReady.emit(self._history_deque.copy())
            
        # Pass the original signal through to the output
        return {"out": signal_value}

    def stop(self):
        # Clear the display when processing stops
        self.newDataReady.emit(None)