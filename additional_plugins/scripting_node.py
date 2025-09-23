import torch
import numpy as np
import math
import threading
import logging
import ast
import re
import traceback
from types import CodeType
from typing import Dict, Optional, Any, Tuple

from node_system import Node
from constants import DEFAULT_DTYPE
from ui_elements import (
    NodeItem,
    NODE_CONTENT_PADDING,
    HEADER_HEIGHT,
    SOCKET_Y_SPACING,
    SOCKET_SIZE,
    SocketItem,
)
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QPlainTextEdit,
    QSizePolicy,
    QGraphicsTextItem,
    QTextEdit,
)
from PySide6.QtCore import Qt, Slot, QSignalBlocker, QPointF, QRectF
from PySide6.QtGui import QFont, QCursor, QPen, QColor, QSyntaxHighlighter, QTextCharFormat, QTextCursor

logger = logging.getLogger(__name__)

TYPE_MAP = {
    "torch.Tensor": torch.Tensor,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "object": object,
    "Any": Any,
    str(None): Any,
}

DEFAULT_CODE = """# Define inputs and outputs as dictionaries.
# Click "Apply" to update the node's sockets.
inputs = {
    'value_in': 'float'
}

outputs = {
    'value_out': 'float'
}

# The 'state' dictionary persists between processing ticks.
# This initialization runs once when the script is applied.
if 'counter' not in state:
    state['counter'] = 0

# --- Main processing logic ---
# Input variables are available by their dictionary key names.
if value_in is not None:
    # Your code here...
    value_out = value_in * state['counter']
    state['counter'] += 1
else:
    value_out = 0.0

# Output variables must be assigned with their dictionary key names.
"""


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """A syntax highlighter for basic Python syntax."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#ff79c6"))  # Pink
        keywords = [
            "\\bif\\b",
            "\\belif\\b",
            "\\belse\\b",
            "\\bfor\\b",
            "\\bwhile\\b",
            "\\bin\\b",
            "\\bis\\b",
            "\\bnot\\b",
            "\\band\\b",
            "\\bor\\b",
            "\\bdef\\b",
            "\\bclass\\b",
            "\\breturn\\b",
            "\\bpass\\b",
            "\\bcontinue\\b",
            "\\bbreak\\b",
            "\\btry\\b",
            "\\bexcept\\b",
            "\\bfinally\\b",
            "\\braise\\b",
            "\\bimport\\b",
            "\\bfrom\\b",
            "\\bas\\b",
            "\\bwith\\b",
        ]
        for word in keywords:
            self.highlighting_rules.append((re.compile(word), keyword_format))

        # Built-ins and common variables
        builtin_format = QTextCharFormat()
        builtin_format.setForeground(QColor("#8be9fd"))  # Cyan
        builtins = ["torch", "np", "math", "state", "inputs", "outputs"]
        for word in builtins:
            self.highlighting_rules.append((re.compile(f"\\b{word}\\b"), builtin_format))

        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#bd93f9"))  # Purple
        self.highlighting_rules.append((re.compile(r"\b[0-9]+\.?[0-9]*\b"), number_format))

        # Strings
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#f1fa8c"))  # Yellow
        self.highlighting_rules.append((re.compile(r"'.*?'"), string_format))
        self.highlighting_rules.append((re.compile(r'".*?"'), string_format))

        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#6272a4"))  # Gray/Blue
        self.highlighting_rules.append((re.compile(r"#[^\n]*"), comment_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            for match in pattern.finditer(text):
                start, end = match.span()
                self.setFormat(start, end - start, format)


class CodeNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 450
    MIN_NODE_WIDTH = 250
    MIN_NODE_HEIGHT = 150

    def __init__(self, node_logic: "CodeNode"):
        super().__init__(node_logic, width=node_logic.ui_size[0] if node_logic.ui_size else self.NODE_SPECIFIC_WIDTH)

        self._height = node_logic.ui_size[1] if node_logic.ui_size else 400
        self._is_resizing = False
        self._resize_handle_size = 15
        self._initial_mouse_pos = QPointF()
        self._initial_size = QPointF()

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(5)

        self.code_editor = QPlainTextEdit()
        self.code_editor.setFont(QFont("Courier New", 9))
        self.code_editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        main_layout.addWidget(self.code_editor)

        self.highlighter = PythonSyntaxHighlighter(self.code_editor.document())

        self.apply_button = QPushButton("Apply")
        main_layout.addWidget(self.apply_button)

        self.status_label = QLabel("Status: OK")
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)

        self.setContentWidget(self.container_widget)

        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.code_editor.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
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
        self.setCursor(
            Qt.CursorShape.SizeFDiagCursor
            if self._get_resize_handle_rect().contains(event.pos())
            else Qt.CursorShape.ArrowCursor
        )
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
            title_width = self.title_item.boundingRect().width()
            max_input_label_width = max(
                (self._socket_labels[s].boundingRect().width() for s in self.node_logic.inputs.values()), default=0
            )
            max_output_label_width = max(
                (self._socket_labels[s].boundingRect().width() for s in self.node_logic.outputs.values()), default=0
            )
            sockets_and_labels_width = max_input_label_width + max_output_label_width + (NODE_CONTENT_PADDING * 4)
            min_required_width = max(
                self.MIN_NODE_WIDTH, title_width + NODE_CONTENT_PADDING * 2, sockets_and_labels_width
            )
            self._width = max(self._width, min_required_width)
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
    def _on_apply_clicked(self):
        self.node_logic.apply_code(self.code_editor.toPlainText())

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):

        # 1. First, check if the sockets have changed. If so, reconcile them.
        if state.get("sockets_changed", False):
            # Get the current set of UI sockets and the desired set from the logic
            current_ui_sockets = set(self._socket_items.keys())
            desired_logic_sockets = set(self.node_logic.inputs.values()) | set(self.node_logic.outputs.values())

            sockets_to_remove = current_ui_sockets - desired_logic_sockets
            sockets_to_add = desired_logic_sockets - current_ui_sockets

            # Remove UI items for sockets that no longer exist in the logic
            for logic_socket in sockets_to_remove:
                socket_item = self._socket_items.pop(logic_socket, None)
                label_item = self._socket_labels.pop(logic_socket, None)
                if socket_item and socket_item.scene():
                    self.scene().removeItem(socket_item)
                if label_item and label_item.scene():
                    self.scene().removeItem(label_item)

            # Create new UI items for sockets that were added to the logic
            for logic_socket in sockets_to_add:
                self._socket_items[logic_socket] = SocketItem(logic_socket, self)
                label = QGraphicsTextItem(logic_socket.name, self)
                label.setDefaultTextColor(Qt.GlobalColor.lightGray)
                self._socket_labels[logic_socket] = label

        status, error = state.get("status", "OK"), state.get("error", "")
        error_lineno = state.get("error_lineno", -1)

        self.code_editor.setExtraSelections([])

        if status == "Error":
            self.status_label.setText(f"Error: {error}")
            self.status_label.setStyleSheet("color: red;")
            if error_lineno > 0:
                selection = QTextEdit.ExtraSelection()
                selection.format.setBackground(QColor(100, 0, 0, 150))
                selection.format.setProperty(QTextCharFormat.Property.FullWidthSelection, True)
                cursor = self.code_editor.textCursor()
                cursor.setPosition(0)
                cursor.movePosition(
                    QTextCursor.MoveOperation.NextBlock, QTextCursor.MoveMode.MoveAnchor, error_lineno - 1
                )
                selection.cursor = cursor
                self.code_editor.setExtraSelections([selection])
                self.code_editor.setTextCursor(cursor)
            self.set_error_display_state(error)
        else:
            self.status_label.setText("Status: OK")
            self.status_label.setStyleSheet("color: lightgreen;")
            self.set_error_display_state(None)

        conn_ids_to_delete = state.get("connections_to_delete", [])
        if conn_ids_to_delete:
            view = self.scene().views()[0] if self.scene() and self.scene().views() else None
            if view:
                for conn_id in conn_ids_to_delete:
                    view.connectionDeletionRequested.emit(conn_id)

        # If sockets were changed, we MUST refresh the geometry now that the
        # child items have been updated.
        if state.get("sockets_changed", False):
            self.update_geometry()

    @Slot()
    def updateFromLogic(self):
        # Handle code node specific updates
        state = self.node_logic.get_current_state_snapshot()
        with QSignalBlocker(self.code_editor):
            if self.code_editor.toPlainText() != state.get("code", ""):
                self.code_editor.setPlainText(state.get("code", ""))
        self._on_state_updated_from_logic(state)

        # Call the parent implementation
        super().updateFromLogic()


class CodeNode(Node):
    NODE_TYPE = "Programmable Code"
    UI_CLASS = CodeNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Executes custom Python code. Sockets are defined in the code."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self._lock = threading.Lock()
        self._code: str = DEFAULT_CODE
        self._compiled_code: Optional[CodeType] = None
        self._state: Dict[str, Any] = {}
        self._last_error: str = ""
        self._last_error_lineno: int = -1
        self._status: str = "OK"
        self.ui_size: Optional[Tuple[float, float]] = None
        self._execution_globals = {"torch": torch, "np": np, "math": math, "print": logger.info}
        self.apply_code(self._code, is_init=True)

    def _str_to_type(self, type_str: str) -> type:
        return TYPE_MAP.get(type_str, object)

    def apply_code(self, code: str, is_init: bool = False):
        sockets_changed = False
        connections_to_delete_ids = []

        with self._lock:
            self._code = code
            try:
                parsed_tree = ast.parse(self._code)
                parsed_inputs, parsed_outputs = {}, {}
                for node in parsed_tree.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                if target.id == "inputs":
                                    parsed_inputs = ast.literal_eval(node.value)
                                elif target.id == "outputs":
                                    parsed_outputs = ast.literal_eval(node.value)

                current_input_names = set(self.inputs.keys())
                desired_input_names = set(parsed_inputs.keys())
                if current_input_names != desired_input_names:
                    sockets_changed = True
                    inputs_to_remove = current_input_names - desired_input_names
                    for name in inputs_to_remove:
                        for conn in self.inputs[name].connections:
                            connections_to_delete_ids.append(conn.id)
                        self.inputs.pop(name)

                current_output_names = set(self.outputs.keys())
                desired_output_names = set(parsed_outputs.keys())
                if current_output_names != desired_output_names:
                    sockets_changed = True
                    outputs_to_remove = current_output_names - desired_output_names
                    for name in outputs_to_remove:
                        for conn in self.outputs[name].connections:
                            connections_to_delete_ids.append(conn.id)
                        self.outputs.pop(name)

                for name, type_str in parsed_inputs.items():
                    dtype = self._str_to_type(type_str)
                    if name not in self.inputs or self.inputs[name].data_type != dtype:
                        self.add_input(name, dtype)
                        sockets_changed = True
                for name, type_str in parsed_outputs.items():
                    dtype = self._str_to_type(type_str)
                    if name not in self.outputs or self.outputs[name].data_type != dtype:
                        self.add_output(name, dtype)
                        sockets_changed = True

                self._compiled_code = compile(self._code, f"<{self.name}>", "exec")
                self.clear_error_state()
                self._status, self._last_error, self._last_error_lineno = "OK", "", -1
                if not is_init:
                    self._state.clear()

            except SyntaxError as e:
                self._compiled_code = None
                self._status = "Error"
                self._last_error = f"Syntax: {e.msg}"
                self._last_error_lineno = e.lineno
                logger.error(f"[{self.name}] Code syntax error: {e}", exc_info=True)
            except Exception as e:
                self._compiled_code = None
                self._status = "Error"
                self._last_error = str(e).replace("\n", " ")
                self._last_error_lineno = -1
                logger.error(f"[{self.name}] Code apply failed: {e}", exc_info=True)

            state_to_emit = self._get_current_state_snapshot_locked()
            state_to_emit["sockets_changed"] = sockets_changed
            state_to_emit["connections_to_delete"] = connections_to_delete_ids

        self.ui_update_callback(state_to_emit)

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {
            "code": self._code,
            "status": self._status,
            "error": self._last_error,
            "error_lineno": self._last_error_lineno,
        }

    def process(self, input_data: dict) -> dict:
        if self.error_state is not None:
            return {name: None for name in self.outputs}

        with self._lock:
            if self._compiled_code is None:
                return {name: None for name in self.outputs}
            execution_scope = {**input_data, "state": self._state}
        try:
            exec(self._compiled_code, self._execution_globals, execution_scope)
            with self._lock:
                self._state = execution_scope.get("state", self._state)
                if self._status == "Error":
                    self._status, self._last_error, self._last_error_lineno = "OK", "", -1
                    self.ui_update_callback(self._get_current_state_snapshot_locked())
            return {name: execution_scope.get(name) for name in self.outputs}
        except Exception as e:
            lineno = -1
            try:
                tb = traceback.extract_tb(e.__traceback__)
                if tb:
                    lineno = tb[-1].lineno
            except Exception:
                pass

            with self._lock:
                self._status = "Error"
                self._last_error = f"Runtime: {str(e).replace(chr(10), ' ')}"
                self._last_error_lineno = lineno
                self.ui_update_callback(self._get_current_state_snapshot_locked())

            raise e

    def serialize_extra(self) -> dict:
        with self._lock:
            return {"code": self._code, "persistent_state": self._state, "ui_size": self.ui_size}

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._state = data.get("persistent_state", {})
            self.ui_size = data.get("ui_size")

        code = data.get("code", DEFAULT_CODE)
        self.apply_code(code, is_init=True)
