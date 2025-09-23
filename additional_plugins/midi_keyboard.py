import mido
import threading
import logging
from collections import deque
from typing import Dict, Optional, List

# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING

# -- import MIDIPacket type ---
from constants import MIDIPacket

# --- Qt Imports ---
from PySide6.QtWidgets import (
    QWidget,
    QSizePolicy,
    QSpinBox,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QVBoxLayout,
)
from PySide6.QtCore import Qt, Slot, Signal, QRectF, QSize, QSignalBlocker
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QPaintEvent

# Configure logging
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Custom Piano Widget (UI Rendering and Interaction)
# ==============================================================================
class PianoWidget(QWidget):
    """A custom widget that draws and handles interactions for a piano keyboard."""

    noteOn = Signal(int, int)  # note, velocity
    noteOff = Signal(int)  # note

    def __init__(self, start_note=48, num_octaves=2, parent=None):
        super().__init__(parent)
        self.start_note = start_note
        self.num_octaves = num_octaves
        self.num_keys = num_octaves * 12
        self._key_rects = {}
        self._active_notes = set()
        self._last_mouse_note = -1
        self.white_key_brush = QBrush(QColor("white"))
        self.black_key_brush = QBrush(QColor("black"))
        self.pressed_key_brush = QBrush(QColor("orange"))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(80)

    def sizeHint(self) -> QSize:
        return QSize(300 + self.num_octaves * 70, 80)

    @Slot(int, int)
    def set_keyboard_range(self, start_note: int, num_octaves: int):
        needs_update = False
        if self.start_note != start_note:
            self.start_note = start_note
            needs_update = True
        if self.num_octaves != num_octaves:
            self.num_octaves = num_octaves
            self.num_keys = num_octaves * 12
            needs_update = True
            self.updateGeometry()
        if needs_update:
            self.update()

    @Slot(list)
    def set_active_notes(self, notes: List[int]):
        new_notes = set(notes)
        if self._active_notes != new_notes:
            self._active_notes = new_notes
            self.update()

    def _calculate_key_rects(self):
        self._key_rects = {}
        num_white_keys = self.num_octaves * 7
        if num_white_keys == 0:
            return
        white_key_width = self.width() / num_white_keys
        white_key_height = self.height()
        black_key_width = white_key_width * 0.6
        black_key_height = white_key_height * 0.6
        white_key_notes = [0, 2, 4, 5, 7, 9, 11]
        black_key_notes = [1, 3, 6, 8, 10]
        white_key_index = 0
        for i in range(self.num_keys + 1):
            note = self.start_note + i
            pitch_class = i % 12
            if pitch_class in white_key_notes:
                x = white_key_index * white_key_width
                self._key_rects[note] = QRectF(x, 0, white_key_width, white_key_height)
                white_key_index += 1
        white_key_index = 0
        for i in range(self.num_keys + 1):
            note = self.start_note + i
            pitch_class = i % 12
            if pitch_class in white_key_notes:
                white_key_index += 1
            elif pitch_class in black_key_notes:
                x = white_key_index * white_key_width - (black_key_width / 2)
                self._key_rects[note] = QRectF(x, 0, black_key_width, black_key_height)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._calculate_key_rects()
        for note, rect in self._key_rects.items():
            if (note - self.start_note) % 12 not in [1, 3, 6, 8, 10]:
                painter.setBrush(self.pressed_key_brush if note in self._active_notes else self.white_key_brush)
                painter.setPen(QPen(QColor("black"), 1))
                painter.drawRect(rect)
        for note, rect in self._key_rects.items():
            if (note - self.start_note) % 12 in [1, 3, 6, 8, 10]:
                painter.setBrush(self.pressed_key_brush if note in self._active_notes else self.black_key_brush)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(rect)

    def _get_note_at_pos(self, pos):
        for note, rect in self._key_rects.items():
            if (note - self.start_note) % 12 in [1, 3, 6, 8, 10] and rect.contains(pos):
                return note
        for note, rect in self._key_rects.items():
            if (note - self.start_note) % 12 not in [1, 3, 6, 8, 10] and rect.contains(pos):
                return note
        return -1

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            note = self._get_note_at_pos(event.position())
            if note != -1:
                self._last_mouse_note = note
                self.noteOn.emit(note, 100)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            note = self._get_note_at_pos(event.position())
            if note != -1 and note != self._last_mouse_note:
                if self._last_mouse_note != -1:
                    self.noteOff.emit(self._last_mouse_note)
                self._last_mouse_note = note
                self.noteOn.emit(note, 100)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._last_mouse_note != -1:
            self.noteOff.emit(self._last_mouse_note)
            self._last_mouse_note = -1


# ==============================================================================
# 2. Node UI Class (NodeItem)
# ==============================================================================
class MIDIKeyboardNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = None

    def __init__(self, node_logic: "MIDIKeyboardNode"):
        super().__init__(node_logic)
        self.container = QWidget()
        main_layout = QVBoxLayout(self.container)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(4)
        control_layout = QHBoxLayout()
        self.shift_left_button = QPushButton("<")
        self.shift_right_button = QPushButton(">")
        self.octave_label = QLabel("C3 - B4")
        self.octave_spinbox = QSpinBox()
        self.octave_spinbox.setRange(2, 8)
        self.octave_spinbox.setSuffix(" oct")
        control_layout.addWidget(self.shift_left_button)
        control_layout.addWidget(self.shift_right_button)
        control_layout.addWidget(self.octave_label, 1, Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(QLabel("Size:"))
        control_layout.addWidget(self.octave_spinbox)
        main_layout.addLayout(control_layout)
        self.piano_widget = PianoWidget()
        main_layout.addWidget(self.piano_widget)
        self.setContentWidget(self.container)
        self.piano_widget.noteOn.connect(self.node_logic.play_note)
        self.piano_widget.noteOff.connect(self.node_logic.stop_note)
        self.shift_right_button.clicked.connect(lambda: self.node_logic.shift_octave(1))
        self.octave_spinbox.valueChanged.connect(self.node_logic.set_num_octaves)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        active_notes = state.get("active_notes", [])
        self.piano_widget.set_active_notes(active_notes)
        start_note = state.get("start_note", 48)
        num_octaves = state.get("num_octaves", 2)
        self.piano_widget.set_keyboard_range(start_note, num_octaves)
        with QSignalBlocker(self.octave_spinbox):
            self.octave_spinbox.setValue(num_octaves)
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        start_note_name = f"{note_names[start_note % 12]}{start_note // 12 - 1}"
        end_note = start_note + (num_octaves * 12) - 1
        end_note_name = f"{note_names[end_note % 12]}{end_note // 12 - 1}"
        self.octave_label.setText(f"{start_note_name} – {end_note_name}")
        self.shift_left_button.setEnabled(start_note > 0)
        self.shift_right_button.setEnabled(start_note < 128 - (num_octaves * 12))
        self.update_geometry()

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated_from_logic(state)
        super().updateFromLogic()


# ==============================================================================
# 3. Node Logic Class
# ==============================================================================
class MIDIKeyboardNode(Node):
    NODE_TYPE = "MIDI Keyboard"
    UI_CLASS = MIDIKeyboardNodeItem
    CATEGORY = "MIDI"
    DESCRIPTION = "An on-screen keyboard that also visualizes and passes through incoming MIDI."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("msg_in", data_type=MIDIPacket)
        self.add_output("msg_out", data_type=MIDIPacket)
        self._lock = threading.Lock()
        self._message_queue = deque(maxlen=100)
        self._active_notes = set()
        self._start_note = 48
        self._num_octaves = 2

    def _get_current_state_snapshot_locked(self) -> Dict:
        """Helper to get a snapshot. ASSUMES LOCK IS HELD."""
        return {
            "active_notes": list(self._active_notes),
            "start_note": self._start_note,
            "num_octaves": self._num_octaves,
        }

    def get_current_state_snapshot(self) -> Dict:
        """Thread-safely gets a copy of the current state."""
        with self._lock:
            return self._get_current_state_snapshot_locked()

    @Slot(int)
    def set_start_note(self, start_note: int):
        state_to_emit = None
        with self._lock:
            max_start_note = 128 - (self._num_octaves * 12)
            new_start_note = max(0, min(start_note, max_start_note))
            if self._start_note != new_start_note:
                self._start_note = new_start_note
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(int)
    def set_num_octaves(self, num_octaves: int):
        state_to_emit = None
        with self._lock:
            new_num_octaves = max(2, min(num_octaves, 8))
            if self._num_octaves != new_num_octaves:
                self._num_octaves = new_num_octaves
                max_start_note = 128 - (self._num_octaves * 12)
                self._start_note = max(0, min(self._start_note, max_start_note))
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def shift_octave(self, direction: int):
        with self._lock:
            current_start = self._start_note
        new_start_note = current_start + (12 * direction)
        self.set_start_note(new_start_note)

    @Slot(int, int)
    def play_note(self, note: int, velocity: int):
        state_to_emit = None
        with self._lock:
            msg = mido.Message("note_on", note=note, velocity=velocity)
            self._message_queue.append(msg)
            if note not in self._active_notes:
                self._active_notes.add(note)
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(int)
    def stop_note(self, note: int):
        state_to_emit = None
        with self._lock:
            msg = mido.Message("note_off", note=note)
            self._message_queue.append(msg)
            if note in self._active_notes:
                self._active_notes.discard(note)
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def process(self, input_data: Dict) -> Dict:
        external_packet = input_data.get("msg_in")
        state_changed_by_external = False
        merged_messages = []
        state_to_emit = None

        if isinstance(external_packet, MIDIPacket):
            with self._lock:
                for _, msg in external_packet.messages:
                    if msg.type == "note_on" and msg.velocity > 0:
                        if msg.note not in self._active_notes:
                            self._active_notes.add(msg.note)
                            state_changed_by_external = True
                    elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                        if msg.note in self._active_notes:
                            self._active_notes.discard(msg.note)
                            state_changed_by_external = True
            merged_messages.extend(external_packet.messages)

        with self._lock:
            while self._message_queue:
                internal_msg = self._message_queue.popleft()
                merged_messages.append((0, internal_msg))

        if state_changed_by_external:
            with self._lock:
                state_to_emit = self._get_current_state_snapshot_locked()

        if state_to_emit:
            self.ui_update_callback(state_to_emit)

        if merged_messages:
            return {"msg_out": MIDIPacket(messages=merged_messages)}
        return {"msg_out": None}

    def _reset_and_notify(self):
        state_to_emit = None
        with self._lock:
            self._message_queue.clear()
            self._active_notes.clear()
            state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def start(self):
        self._reset_and_notify()

    def stop(self):
        self._reset_and_notify()

    def serialize_extra(self) -> Dict:
        with self._lock:
            return {
                "start_note": self._start_note,
                "num_octaves": self._num_octaves,
            }

    def deserialize_extra(self, data: Dict):
        self.set_num_octaves(data.get("num_octaves", 2))
        self.set_start_note(data.get("start_note", 48))
