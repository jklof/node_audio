import logging
import math
from PySide6.QtWidgets import QGraphicsView, QGraphicsLineItem, QGraphicsProxyWidget
from PySide6.QtCore import Qt, QPointF, QRectF, Signal, Slot, QLineF, QByteArray
from PySide6.QtGui import QPainter, QColor, QPen, QCursor
from PySide6.QtSvg import QSvgRenderer
from typing import Any

from node_system import NodeGraph, Socket
from ui_elements import NodeItem, SocketItem, ConnectionItem
from graph_scene import NodeGraphScene
from ui_icons import LOGO_SVG_DATA

logger = logging.getLogger(__name__)


class NodeGraphWidget(QGraphicsView):
    """The main view widget for the node graph, handling user interactions."""

    connectionRequested = Signal(object, object)  # start_socket_logic, end_socket_logic
    nodeDeletionRequested = Signal(str)  # node_id
    connectionDeletionRequested = Signal(str)  # connection_id

    def __init__(self, graph_logic: NodeGraph, parent=None):
        super().__init__(parent)
        self.graph_scene = NodeGraphScene(graph_logic, self)
        self.setScene(self.graph_scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(40, 40, 40))

        # Interaction state
        self._start_socket_item: SocketItem | None = None
        self._temp_connection_line: QGraphicsLineItem | None = None
        self._is_panning = False
        self._last_pan_pos = QPointF()

        self._logo_renderer = None
        self._load_logo()

    def _load_logo(self):
        """Loads the SVG logo from an inline string for rendering."""
        svg_bytes = QByteArray(LOGO_SVG_DATA.encode("utf-8"))
        self._logo_renderer = QSvgRenderer(svg_bytes)
        if not self._logo_renderer.isValid():
            logger.error("Failed to load SVG logo for background.")
            self._logo_renderer = None

    def drawBackground(self, painter: QPainter, rect: QRectF):
        """
        Draws a two-level grid as the background. The `rect` is the exposed
        area in scene coordinates, provided by the QGraphicsView framework.
        """
        super().drawBackground(painter, rect)

        if self._logo_renderer and self._logo_renderer.isValid():
            scale = 0.2
            w = self._logo_renderer.defaultSize().width() * scale
            h = self._logo_renderer.defaultSize().height() * scale
            logo_rect = QRectF(-w / 2, -h / 2, w, h)
            if rect.intersects(logo_rect):
                painter.save()
                painter.setOpacity(0.05)
                self._logo_renderer.render(painter, logo_rect)
                painter.restore()

        grid_size_fine = 15
        grid_size_coarse = 150

        color_fine = QColor(50, 50, 50)
        color_coarse = QColor(60, 60, 60)

        pen_fine = QPen(color_fine, 1.0)
        pen_coarse = QPen(color_coarse, 1.5)

        # 'rect' is already the scene area to be drawn.
        # Use floor division to find the first grid line inside or to the left/top of the rect.
        left = int(math.floor(rect.left() / grid_size_fine)) * grid_size_fine
        top = int(math.floor(rect.top() / grid_size_fine)) * grid_size_fine

        lines_fine, lines_coarse = [], []

        # Draw vertical lines
        x = float(left)
        while x < rect.right():
            line = QLineF(x, rect.top(), x, rect.bottom())
            if x % grid_size_coarse == 0:
                lines_coarse.append(line)
            else:
                lines_fine.append(line)
            x += grid_size_fine

        # Draw horizontal lines
        y = float(top)
        while y < rect.bottom():
            line = QLineF(rect.left(), y, rect.right(), y)
            if y % grid_size_coarse == 0:
                lines_coarse.append(line)
            else:
                lines_fine.append(line)
            y += grid_size_fine

        painter.setPen(pen_fine)
        painter.drawLines(lines_fine)
        painter.setPen(pen_coarse)
        painter.drawLines(lines_coarse)

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if event.button() == Qt.MouseButton.LeftButton and isinstance(item, SocketItem):
            self._start_connection_draw(item)
            event.accept()
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning:
            delta = event.pos() - self._last_pan_pos
            self._last_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
        elif self._temp_connection_line:
            self._update_temp_connection_line(event.pos())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._is_panning and event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        elif self._temp_connection_line:
            self._finish_connection_draw(event.pos())
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        # Check if an input widget (hosted in a QGraphicsProxyWidget) has focus.
        focus_item = self.scene().focusItem()
        is_input_widget_focused = isinstance(focus_item, QGraphicsProxyWidget)

        # Only trigger graph item deletion if an input widget is NOT focused.
        if (event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace) and not is_input_widget_focused:
            self._delete_selected_items()
            event.accept()
        else:
            # For all other keys, OR if an input widget has focus,
            # pass the event to the default handler. This allows typing,
            # backspace, delete, etc., in the input widgets.
            super().keyPressEvent(event)

    def _delete_selected_items(self):
        """
        Collects IDs of selected items and emits signals to request their deletion.
        """
        selected = self.graph_scene.selectedItems()
        if not selected:
            return

        conn_ids_to_delete = []
        node_ids_to_delete = []

        for item in selected:
            if isinstance(item, ConnectionItem):
                conn_ids_to_delete.append(item.connection_logic.id)
            elif isinstance(item, NodeItem):
                node_ids_to_delete.append(item.node_logic.id)

        # Remove connections before the nodes they are attached to.
        for conn_id in conn_ids_to_delete:
            logger.info(f"View: Requesting deletion of connection {conn_id[:4]}")
            self.connectionDeletionRequested.emit(conn_id)

        for node_id in node_ids_to_delete:
            logger.info(f"View: Requesting deletion of node {node_id[:4]}")
            self.nodeDeletionRequested.emit(node_id)

    def _start_connection_draw(self, start_socket: SocketItem):
        self._start_socket_item = start_socket
        self._temp_connection_line = QGraphicsLineItem()
        pen = QPen(QColor("yellow"), 2, Qt.PenStyle.DashLine)
        self._temp_connection_line.setPen(pen)
        self.graph_scene.addItem(self._temp_connection_line)
        self._update_temp_connection_line(self.mapFromGlobal(self.cursor().pos()))

    def _update_temp_connection_line(self, view_pos):
        if self._start_socket_item:
            start_pos = self._start_socket_item.get_scene_position()
            end_pos = self.mapToScene(view_pos)
            self._temp_connection_line.setLine(start_pos.x(), start_pos.y(), end_pos.x(), end_pos.y())

    def _finish_connection_draw(self, view_pos):
        if self._temp_connection_line:
            self.graph_scene.removeItem(self._temp_connection_line)
            self._temp_connection_line = None

        end_item = self.itemAt(view_pos)
        start_item = self._start_socket_item
        self._start_socket_item = None

        if isinstance(end_item, SocketItem) and start_item is not None and end_item != start_item:
            start_logic = start_item.socket_logic
            end_logic = end_item.socket_logic

            # --- UI-Level Type Validation Logic ---
            start_type = start_logic.data_type if start_logic.data_type is not None else Any
            end_type = end_logic.data_type if end_logic.data_type is not None else Any

            is_compatible = start_type is Any or end_type is Any or start_type == end_type

            if not is_compatible:
                logger.warning(
                    f"Connection rejected: Type mismatch between {start_logic} ({start_type.__name__}) and {end_logic} ({end_type.__name__})."
                )
                return  # Abort the connection attempt

            if start_logic.is_input and not end_logic.is_input:
                self.connectionRequested.emit(end_logic, start_logic)
            elif not start_logic.is_input and end_logic.is_input:
                self.connectionRequested.emit(start_logic, end_logic)
            else:
                logger.warning("Connection must be between an input and an output.")

    def wheelEvent(self, event):
        """
        This is the new, correct implementation. It checks if the mouse is over
        an embedded UI widget. If so, it lets the base QGraphicsView handle
        the event, which correctly forwards it. Otherwise, it performs the
        custom graph zoom.
        """
        item_under_mouse = self.itemAt(event.position().toPoint())

        if isinstance(item_under_mouse, QGraphicsProxyWidget):
            # Let the default implementation handle event propagation to the widget.
            super().wheelEvent(event)
        else:
            # Perform custom graph-wide zoom.
            zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(zoom_factor, zoom_factor)

    # --- Public Slots for Toolbar ---
    @Slot()
    def zoom_in(self):
        """Zooms in the view."""
        self.scale(1.15, 1.15)

    @Slot()
    def zoom_out(self):
        """Zooms out the view."""
        self.scale(1 / 1.15, 1 / 1.15)

    @Slot()
    def zoom_to_fit(self):
        """
        Adjusts the view to fit all nodes in the scene.
        This version calculates the bounding box based only on NodeItems
        to avoid issues with large connection bounding boxes.
        """
        node_items = list(self.graph_scene.node_items.values())
        if not node_items:
            return

        # Manually compute the union of all node bounding rectangles. This is
        # more reliable than scene.itemsBoundingRect(), which can be skewed
        # by the large bounding boxes of Bezier curve connections.
        total_rect = QRectF()
        is_first = True
        for item in node_items:
            if is_first:
                total_rect = item.sceneBoundingRect()
                is_first = False
            else:
                total_rect = total_rect.united(item.sceneBoundingRect())

        if not total_rect.isValid():
            return

        # Add a margin for better visibility
        margin = 50
        total_rect.adjust(-margin, -margin, margin, margin)

        self.fitInView(total_rect, Qt.AspectRatioMode.KeepAspectRatio)
