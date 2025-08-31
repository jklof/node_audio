import logging
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsObject, QGraphicsTextItem, QGraphicsPathItem,
    QStyleOptionGraphicsItem, QWidget, QMenu, QGraphicsProxyWidget,
    QInputDialog, QLineEdit
)
from PySide6.QtGui import (
    QPainter, QPen, QColor, QBrush, QPainterPath, QPainterPathStroker, QAction
)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, Slot, QTimer

# Import the interface to check against
from node_system import IClockProvider

logger = logging.getLogger(__name__)

# --- Constants ---
SOCKET_SIZE = 12
NODE_WIDTH = 120  # Default minimum width
HEADER_HEIGHT = 20
SOCKET_Y_SPACING = 25
NODE_CONTENT_PADDING = 5

class SocketItem(QGraphicsObject):
    """Visual representation of a socket in the scene."""
    positionChanged = Signal(QPointF)

    def __init__(self, socket_logic, parent_item):
        super().__init__(parent_item)
        self.socket_logic = socket_logic
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
        self.setAcceptHoverEvents(True)
        self._brush = QBrush(QColor("lightblue") if socket_logic.is_input else QColor("lightgreen"))
        self._hover_brush = QBrush(QColor("cyan") if socket_logic.is_input else QColor("lime"))
        self._pen = QPen(Qt.GlobalColor.black, 1)
        self._is_hovered = False

    def boundingRect(self):
        return QRectF(-SOCKET_SIZE/2, -SOCKET_SIZE/2, SOCKET_SIZE, SOCKET_SIZE)

    def paint(self, painter: QPainter, option, widget):
        painter.setPen(self._pen)
        painter.setBrush(self._hover_brush if self._is_hovered else self._brush)
        painter.drawEllipse(self.boundingRect())

    def hoverEnterEvent(self, event):
        self._is_hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._is_hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged:
            self.positionChanged.emit(self.scenePos())
        return super().itemChange(change, value)

    def get_scene_position(self) -> QPointF:
        return self.scenePos()

class NodeItem(QGraphicsObject):
    """
    Visual representation of a node in the scene.
    Manages its geometry centrally and adapts to its content.
    """
    def __init__(self, node_logic, width=NODE_WIDTH):
        super().__init__()
        self.node_logic = node_logic

        # --- Initialize attributes ---
        self._width = width
        self._height = HEADER_HEIGHT
        self._socket_items = {}
        self._socket_labels = {}
        self._content_proxy = None
        self._is_updating_geometry = False # Re-entrancy guard for update_geometry
        self._processing_percentage = 0.0
        self._show_processing_bar = False

        # --- Set Flags ---
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

        # --- Create UI Children ---
        self.title_item = QGraphicsTextItem(self.node_logic.name, self)
        self.title_item.setDefaultTextColor(Qt.GlobalColor.white)
        
        #Create socket and label items ---
        for logic_socket in self.node_logic.inputs.values():
            self._socket_items[logic_socket] = SocketItem(logic_socket, self)
            label = QGraphicsTextItem(logic_socket.name, self)
            label.setDefaultTextColor(Qt.GlobalColor.lightGray)
            self._socket_labels[logic_socket] = label

        for logic_socket in self.node_logic.outputs.values():
            self._socket_items[logic_socket] = SocketItem(logic_socket, self)
            label = QGraphicsTextItem(logic_socket.name, self)
            label.setDefaultTextColor(Qt.GlobalColor.lightGray)
            self._socket_labels[logic_socket] = label
        
        self.update_geometry()
        self.setPos(QPointF(*node_logic.pos))

    def get_socket_item(self, logic_socket):
        return self._socket_items.get(logic_socket)

    def setContentWidget(self, widget: QWidget | None):
        """Embeds a QWidget, handling proxy management and dynamic resizing."""
        # Clean up the old proxy and its signal connection
        if self._content_proxy:
            try:
                # Disconnect the geometry update signal from the old proxy
                self._content_proxy.geometryChanged.disconnect(self._schedule_geometry_update)
            except (TypeError, RuntimeError):
                pass # Ignore if it was already disconnected
            if self._content_proxy.scene():
                self.scene().removeItem(self._content_proxy)
            self._content_proxy.deleteLater()
            self._content_proxy = None
        
        if widget:
            self._content_proxy = QGraphicsProxyWidget(self)
            self._content_proxy.setWidget(widget)
            # Connect the new proxy's signal to the scheduler slot
            self._content_proxy.geometryChanged.connect(self._schedule_geometry_update)
        
        # Trigger a full geometry update when the content widget is set/cleared
        self.update_geometry()

    @Slot(float)
    def set_processing_percentage(self, percentage: float):
        """Slot to receive processing load and trigger a repaint."""
        self._processing_percentage = max(0.0, min(100.0, percentage))
        self.update()

    @Slot(bool)
    def set_processing_bar_visible(self, visible: bool):
        """Controls whether the processing load bar is drawn."""
        if self._show_processing_bar != visible:
            self._show_processing_bar = visible
            self.update()

    @Slot()
    def _schedule_geometry_update(self):
        """
        Schedules a call to update_geometry using a zero-ms timer.
        This coalesces multiple rapid-fire geometryChanged signals (e.g., from
        a slider drag updating a label) into a single, deferred update,
        preventing UI lag and recursive update loops.
        """
        QTimer.singleShot(0, self.update_geometry)

    def update_geometry(self):
        """Calculates and applies the final geometry for the node and its children."""
        # Re-entrancy guard to prevent recursive calls
        if self._is_updating_geometry:
            return
        self._is_updating_geometry = True
        
        try: # Use a try...finally block to ensure the guard is always released
            self.prepareGeometryChange()

            # --- Calculate Required Width ---
            title_width = self.title_item.boundingRect().width()
            
            max_input_label_width = 0
            for socket in self.node_logic.inputs.values():
                label = self._socket_labels[socket]
                max_input_label_width = max(max_input_label_width, label.boundingRect().width())

            max_output_label_width = 0
            for socket in self.node_logic.outputs.values():
                label = self._socket_labels[socket]
                max_output_label_width = max(max_output_label_width, label.boundingRect().width())
            
            sockets_and_labels_width = (
                SOCKET_SIZE + max_input_label_width + max_output_label_width +
                (NODE_CONTENT_PADDING * 4)
            )

            content_width = 0
            if self._content_proxy and self._content_proxy.widget():
                content_width = self._content_proxy.widget().sizeHint().width()

            self._width = max(
                NODE_WIDTH,
                title_width + NODE_CONTENT_PADDING * 2,
                content_width + NODE_CONTENT_PADDING * 2,
                sockets_and_labels_width
            )

            # --- Position Sockets, Labels, and Title ---
            self.title_item.setPos(NODE_CONTENT_PADDING, 0)

            y_in = HEADER_HEIGHT + SOCKET_Y_SPACING / 2
            for logic_socket in self.node_logic.inputs.values():
                socket_item = self._socket_items[logic_socket]
                label_item = self._socket_labels[logic_socket]
                socket_item.setPos(0, y_in)
                label_item.setPos(SOCKET_SIZE, y_in - label_item.boundingRect().height() / 2)
                y_in += SOCKET_Y_SPACING
            
            y_out = HEADER_HEIGHT + SOCKET_Y_SPACING / 2
            for logic_socket in self.node_logic.outputs.values():
                socket_item = self._socket_items[logic_socket]
                label_item = self._socket_labels[logic_socket]
                socket_item.setPos(self._width, y_out)
                label_item.setPos(self._width - SOCKET_SIZE - label_item.boundingRect().width(), y_out - label_item.boundingRect().height() / 2)
                y_out += SOCKET_Y_SPACING

            # Determine height needed for sockets area
            sockets_area_height = max(y_in, y_out) - SOCKET_Y_SPACING / 2

            # --- Position Content and Determine Final Height ---
            content_height = 0
            if self._content_proxy and self._content_proxy.widget():
                content_y_start = sockets_area_height + NODE_CONTENT_PADDING
                content_height = self._content_proxy.widget().sizeHint().height()
                self._content_proxy.setPos(NODE_CONTENT_PADDING, content_y_start)
                self._content_proxy.resize(self._width - NODE_CONTENT_PADDING * 2, content_height)
                self._height = content_y_start + content_height + NODE_CONTENT_PADDING
            else:
                self._height = sockets_area_height + NODE_CONTENT_PADDING

            self.update() # Trigger a repaint
        finally:
            self._is_updating_geometry = False # Release the guard

    def boundingRect(self):
        # Adjust bounding rect to include sockets protruding from the sides
        return QRectF(0, 0, self._width, self._height).adjusted(-SOCKET_SIZE/2, 0, SOCKET_SIZE/2, 0)

    def paint(self, painter: QPainter, option, widget=None):
        body_rect = QRectF(0, 0, self._width, self._height)
        
        # Main body
        painter.setBrush(QColor(50, 50, 50, 200))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(body_rect, 5, 5)
        
        # Header with rounded top corners
        path = QPainterPath()
        path.moveTo(0, 5)
        path.arcTo(0, 0, 10, 10, 180, -90)
        path.lineTo(self._width - 5, 0)
        path.arcTo(self._width - 10, 0, 10, 10, 90, -90)
        path.lineTo(self._width, HEADER_HEIGHT)
        path.lineTo(0, HEADER_HEIGHT)
        path.closeSubpath()
        painter.setBrush(QColor(70, 70, 70, 220))
        painter.drawPath(path)
        
        # Processing time percentage bar
        if self._show_processing_bar and self._processing_percentage > 0:
            bar_width = self._width * (self._processing_percentage / 100.0)
            
            # Determine color based on percentage (green -> yellow -> red)
            if self._processing_percentage < 50:
                color = QColor(0, 255, 0, 100) # Green, semi-transparent
            elif self._processing_percentage < 85:
                color = QColor(255, 255, 0, 100) # Yellow, semi-transparent
            else:
                color = QColor(255, 0, 0, 120) # Red, more opaque
            
            bar_rect = QRectF(0, 0, bar_width, HEADER_HEIGHT)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(bar_rect)


        # Selection outline
        if self.isSelected():
            painter.setPen(QPen(QColor("orange"), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(body_rect.adjusted(1, 1, -1, -1), 4, 4)

        # Clock source indicator
        if isinstance(self.node_logic, IClockProvider):
            is_selected_clock = self.scene().current_clock_id == self.node_logic.id
            color = QColor("lime") if is_selected_clock else QColor("gray")
            painter.setPen(QPen(color, 1.5))
            indicator_rect = QRectF(self._width - HEADER_HEIGHT + 4, 4, HEADER_HEIGHT - 8, HEADER_HEIGHT - 8)
            center = indicator_rect.center()
            painter.drawEllipse(indicator_rect)
            painter.drawLine(center, center + QPointF(0, -indicator_rect.height() / 2))
            painter.drawLine(center, center + QPointF(-indicator_rect.width() / 2, 0))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if self.node_logic:
                self.node_logic.pos = (value.x(), value.y())
        return super().itemChange(change, value)
    
    @Slot()
    def updateFromLogic(self):
        # Explicitly check and update the title text from the logic object
        if self.title_item.toPlainText() != self.node_logic.name:
            self.title_item.setPlainText(self.node_logic.name)
        self.update_geometry()
        self.update()

    def contextMenuEvent(self, event):
        menu = QMenu()

        #Check for the interface and add clock source option
        if isinstance(self.node_logic, IClockProvider):
            action = menu.addAction("Set as Clock Source")
            action.setCheckable(True)
            action.setChecked(self.scene().current_clock_id == self.node_logic.id)
            action.triggered.connect(lambda: self.scene().clockSourceSetRequested.emit(self.node_logic.id))

        # rename action
        rename_action = menu.addAction("Rename Node")
        def request_rename():
            # Use the view as the parent for the dialog for proper modality
            view = self.scene().views()[0] if self.scene().views() else None
            current_name = self.node_logic.name
            new_name, ok = QInputDialog.getText(
                view,
                "Rename Node",
                "Enter new name:",
                QLineEdit.EchoMode.Normal,
                current_name
            )
            if ok and new_name and new_name != current_name:
                self.scene().nodeRenameRequested.emit(self.node_logic.id, new_name)
        rename_action.triggered.connect(request_rename)

        menu.addSeparator()
        #delete action
        delete_action = menu.addAction("Delete Node")
        delete_action.triggered.connect(lambda: self.scene().parent().nodeDeletionRequested.emit(self.node_logic.id))
    
        menu.exec(event.screenPos())


class ConnectionItem(QGraphicsPathItem):
    """Visual representation of a connection line."""
    def __init__(self, conn_logic, start_socket, end_socket, parent=None):
        super().__init__(parent)
        self.connection_logic = conn_logic
        self.start_socket = start_socket
        self.end_socket = end_socket
        self._pen = QPen(QColor("white"), 2)
        self._pen_hover = QPen(QColor("cyan"), 2.5)
        self._pen_selected = QPen(QColor("orange"), 2.5)
        self._is_hovered = False
        self.setPen(self._pen)
        self.setZValue(-1)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIsSelectable)
        self.start_socket.positionChanged.connect(self.update_path)
        self.end_socket.positionChanged.connect(self.update_path)
        self.update_path()
        
    def remove(self):
        """Disconnects signals to prevent errors on deletion."""
        if self.start_socket:
            try: self.start_socket.positionChanged.disconnect(self.update_path)
            except (TypeError, RuntimeError): pass
        if self.end_socket:
            try: self.end_socket.positionChanged.disconnect(self.update_path)
            except (TypeError, RuntimeError): pass
        
    @Slot()
    def update_path(self):
        if not self.start_socket or not self.end_socket:
            return
        start_pos = self.start_socket.get_scene_position()
        end_pos = self.end_socket.get_scene_position()
        path = QPainterPath(start_pos)
        dx = end_pos.x() - start_pos.x()
        ctrl1 = QPointF(start_pos.x() + abs(dx) * 0.5, start_pos.y())
        ctrl2 = QPointF(end_pos.x() - abs(dx) * 0.5, end_pos.y())
        path.cubicTo(ctrl1, ctrl2, end_pos)
        self.setPath(path)
        
    def paint(self, painter: QPainter, option, widget=None):
        if self.isSelected():
            painter.setPen(self._pen_selected)
        elif self._is_hovered:
            painter.setPen(self._pen_hover)
        else:
            painter.setPen(self._pen)
        painter.drawPath(self.path())
        
    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(10)
        return stroker.createStroke(self.path())
        
    def hoverEnterEvent(self, event):
        self._is_hovered = True
        self.update()
        
    def hoverLeaveEvent(self, event):
        self._is_hovered = False
        self.update()
        
    def contextMenuEvent(self, event):
        menu = QMenu()
        delete_action = menu.addAction("Delete Connection")
        view = self.scene().parent() if self.scene() else None
        if view and hasattr(view, 'connectionDeletionRequested'):
            delete_action.triggered.connect(lambda: view.connectionDeletionRequested.emit(self.connection_logic.id))
        menu.exec(event.screenPos())
        event.accept()