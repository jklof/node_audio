import logging
from typing import Dict, Optional, Tuple

from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING, HEADER_HEIGHT, SOCKET_Y_SPACING, SOCKET_SIZE
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit
from PySide6.QtCore import Slot, QSignalBlocker, Qt, QPointF, QRectF
from PySide6.QtGui import QPen, QColor

logger = logging.getLogger(__name__)


class SimpleTextNodeItem(NodeItem):
    """
    A UI for the SimpleTextNode that provides a resizable text editing area.
    It synchronizes its content with the logic node in a thread-safe manner.
    """

    MIN_NODE_WIDTH = 250
    MIN_NODE_HEIGHT = 150

    def __init__(self, node_logic: "SimpleTextNode"):
        initial_width = node_logic.ui_size[0] if node_logic.ui_size else 300
        super().__init__(node_logic, width=initial_width)

        self._height = node_logic.ui_size[1] if node_logic.ui_size else 250
        self._is_resizing = False
        self._resize_handle_size = 15
        self._initial_mouse_pos = QPointF()
        self._initial_size = QPointF()

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Enter text or connect a text source...")
        main_layout.addWidget(self.editor)

        self.setContentWidget(self.container_widget)

        self.editor.textChanged.connect(self._on_text_changed)
        self.setAcceptHoverEvents(True)
        self.updateFromLogic()

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        handle_rect = self._get_resize_handle_rect()
        painter.setPen(QPen(QColor(120, 120, 120), 1.5))
        painter.drawLine(
            handle_rect.topLeft() + QPointF(self._resize_handle_size * 0.2, self._resize_handle_size),
            handle_rect.topLeft() + QPointF(self._resize_handle_size, self._resize_handle_size * 0.2),
        )
        painter.drawLine(
            handle_rect.topLeft() + QPointF(self._resize_handle_size * 0.6, self._resize_handle_size),
            handle_rect.topLeft() + QPointF(self._resize_handle_size, self._resize_handle_size * 0.6),
        )

    def _get_resize_handle_rect(self) -> QRectF:
        return QRectF(
            self._width - self._resize_handle_size,
            self._height - self._resize_handle_size,
            self._resize_handle_size,
            self._resize_handle_size,
        )

    def hoverMoveEvent(self, event):
        on_handle = self._get_resize_handle_rect().contains(event.pos())
        self.setCursor(Qt.CursorShape.SizeFDiagCursor if on_handle else Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._get_resize_handle_rect().contains(event.pos()):
            self._is_resizing = True
            self._initial_mouse_pos = event.scenePos()
            self._initial_size = QPointF(self._width, self._height)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_resizing:
            delta = event.scenePos() - self._initial_mouse_pos
            self._width = max(self.MIN_NODE_WIDTH, self._initial_size.x() + delta.x())
            self._height = max(self.MIN_NODE_HEIGHT, self._initial_size.y() + delta.y())
            self.node_logic.ui_size = (self._width, self._height)
            self.update_geometry()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._is_resizing and event.button() == Qt.MouseButton.LeftButton:
            self._is_resizing = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def update_geometry(self):
        if self._is_updating_geometry:
            return
        self._is_updating_geometry = True
        try:
            self.prepareGeometryChange()
            self.title_item.setPos(NODE_CONTENT_PADDING, 0)
            y_in = HEADER_HEIGHT + SOCKET_Y_SPACING / 2
            for logic_socket in self.node_logic.inputs.values():
                self._socket_items[logic_socket].setPos(0, y_in)
                self._socket_labels[logic_socket].setPos(
                    SOCKET_SIZE, y_in - self._socket_labels[logic_socket].boundingRect().height() / 2
                )
                y_in += SOCKET_Y_SPACING
            y_out = HEADER_HEIGHT + SOCKET_Y_SPACING / 2
            for logic_socket in self.node_logic.outputs.values():
                self._socket_items[logic_socket].setPos(self._width, y_out)
                label_width = self._socket_labels[logic_socket].boundingRect().width()
                self._socket_labels[logic_socket].setPos(
                    self._width - SOCKET_SIZE - label_width,
                    y_out - self._socket_labels[logic_socket].boundingRect().height() / 2,
                )
                y_out += SOCKET_Y_SPACING
            sockets_area_height = max(y_in, y_out) - SOCKET_Y_SPACING / 2
            if self._content_proxy and self._content_proxy.widget():
                content_y_start = sockets_area_height + NODE_CONTENT_PADDING
                available_width = self._width - NODE_CONTENT_PADDING * 2
                content_height = self._height - content_y_start - NODE_CONTENT_PADDING
                self._content_proxy.setPos(NODE_CONTENT_PADDING, content_y_start)
                self._content_proxy.widget().setFixedSize(max(1, available_width), max(1, content_height))
            self.update()
        finally:
            self._is_updating_geometry = False

    @Slot()
    def _on_text_changed(self):
        self.node_logic.set_text(self.editor.toPlainText())

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: Dict):
        super()._on_state_updated_from_logic(state)
        new_text = state.get("text", "")
        with QSignalBlocker(self.editor):
            if self.editor.toPlainText() != new_text:
                self.editor.setPlainText(new_text)


class SimpleTextNode(Node):
    NODE_TYPE = "Simple Text"
    UI_CLASS = SimpleTextNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "A simple text box for editing, inputting, and outputting string data."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        # --- Manually define state attributes ---
        self._text: str = ""
        self.ui_size: Optional[Tuple[float, float]] = None
        
        self.add_input("text_in", data_type=str)
        self.add_output("text_out", data_type=str)

    @Slot(str)
    def set_text(self, new_text: str):
        """A thread-safe Qt Slot to update the text from the UI."""
        state_to_emit = None
        with self._lock:
            if self._text != new_text:
                self._text = new_text
                state_to_emit = self._get_state_snapshot_locked()
        
        if state_to_emit:
            # This callback is thread-safe and marshals the call to the main thread
            self.ui_update_callback(state_to_emit)

    def _get_state_snapshot_locked(self) -> Dict:
        """Returns a snapshot of the current state while the lock is held."""
        return {"text": self._text}

    def process(self, input_data: dict) -> dict:
        incoming_text = input_data.get("text_in")
        state_to_emit = None
        
        with self._lock:
            if incoming_text is not None and self._text != incoming_text:
                self._text = str(incoming_text)
                state_to_emit = self._get_state_snapshot_locked()
        
        if state_to_emit:
            self.ui_update_callback(state_to_emit)
            
        with self._lock:
            return {"text_out": self._text}

    def serialize_extra(self) -> dict:
        """Manually save both text and UI size."""
        with self._lock:
            return {"text": self._text, "ui_size": self.ui_size}

    def deserialize_extra(self, data: dict):
        """Manually load both text and UI size."""
        with self._lock:
            self._text = data.get("text", "")
            self.ui_size = data.get("ui_size")