import mido
import threading
import logging
from collections import deque
from typing import Dict, Optional, List

# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING

# --- Qt Imports ---
from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, Slot, Signal, QRectF, QSize
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QPaintEvent

# Configure logging
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Custom Piano Widget (UI Rendering and Interaction)
# ==============================================================================
class PianoWidget(QWidget):
    """A custom widget that draws and handles interactions for a piano keyboard."""

    noteOn = Signal(int, int)  # note, velocity
    noteOff = Signal(int)      # note

    def __init__(self, start_note=48, num_octaves=2, parent=None):
        super().__init__(parent)
        self.start_note = start_note
        self.num_octaves = num_octaves
        self.num_keys = num_octaves * 12

        self._key_rects = {}
        self._active_notes = set()  # This is the source of truth for DRAWING ONLY.
        self._last_mouse_note = -1

        self.white_key_brush = QBrush(QColor("white"))
        self.black_key_brush = QBrush(QColor("black"))
        self.pressed_key_brush = QBrush(QColor("orange"))

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(80)

    def sizeHint(self) -> QSize:
        """Provides a reasonable default size for the widget."""
        return QSize(300, 80)

    @Slot(list)
    def set_active_notes(self, notes: List[int]):
        """Receives the authoritative list of active notes from the logic node."""
        new_notes = set(notes)
        if self._active_notes != new_notes:
            self._active_notes = new_notes
            self.update()  # Schedule a repaint to reflect the new state

    def _calculate_key_rects(self):
        """Calculates the QRectF for each key based on widget size."""
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

        # Draw white keys first
        for note, rect in self._key_rects.items():
            if (note - self.start_note) % 12 not in [1, 3, 6, 8, 10]:
                painter.setBrush(self.pressed_key_brush if note in self._active_notes else self.white_key_brush)
                painter.setPen(QPen(QColor("black"), 1))
                painter.drawRect(rect)

        # Draw black keys on top
        for note, rect in self._key_rects.items():
            if (note - self.start_note) % 12 in [1, 3, 6, 8, 10]:
                painter.setBrush(self.pressed_key_brush if note in self._active_notes else self.black_key_brush)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(rect)

    def _get_note_at_pos(self, pos):
        # Check black keys first as they are on top
        for note, rect in self._key_rects.items():
            if (note - self.start_note) % 12 in [1, 3, 6, 8, 10] and rect.contains(pos):
                return note
        # Then check white keys
        for note, rect in self._key_rects.items():
            if (note - self.start_note) % 12 not in [1, 3, 6, 8, 10] and rect.contains(pos):
                return note
        return -1

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            note = self._get_note_at_pos(event.position())
            if note != -1:
                self._last_mouse_note = note
                self.noteOn.emit(note, 100)  # Emit signal to logic node

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
    NODE_SPECIFIC_WIDTH = 320

    def __init__(self, node_logic: "MIDIKeyboardNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.piano_widget = PianoWidget()
        self.setContentWidget(self.piano_widget)

        # Connect UI actions (View -> Logic)
        self.piano_widget.noteOn.connect(self.node_logic.play_note)
        self.piano_widget.noteOff.connect(self.node_logic.stop_note)

        # Connect state updates (Logic -> View)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        """Receives state from the logic and tells the UI widget what to draw."""
        active_notes = state.get("active_notes", [])
        self.piano_widget.set_active_notes(active_notes)


# ==============================================================================
# 3. Node Logic Class (Refactored)
# ==============================================================================
class MIDIKeyboardNode(Node):
    NODE_TYPE = "MIDI Keyboard"
    UI_CLASS = MIDIKeyboardNodeItem
    CATEGORY = "MIDI"
    DESCRIPTION = "An on-screen keyboard for generating MIDI note messages."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self.add_output("msg_out", data_type=object)

        self._lock = threading.Lock()
        self._message_queue = deque(maxlen=100)
        self._active_notes = set()  # This is the single source of truth for the note state.

    def get_current_state_snapshot(self) -> Dict:
        """Thread-safely gets a copy of the current state for UI updates or serialization."""
        with self._lock:
            return {"active_notes": list(self._active_notes)}

    @Slot(int, int)
    def play_note(self, note: int, velocity: int):
        """Slot to handle note-on events from the UI thread."""
        state_to_emit = None
        with self._lock:
            # 1. Update the internal state (queue a message and update active notes).
            msg = mido.Message("note_on", note=note, velocity=velocity)
            self._message_queue.append(msg)
            
            # 2. Check if a state change occurred that requires a UI update.
            if note not in self._active_notes:
                self._active_notes.add(note)
                # 3. If so, capture the new state for later emission.
                state_to_emit = {"active_notes": list(self._active_notes)}
        
        # 4. After releasing the lock, emit the signal to the UI thread.
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(int)
    def stop_note(self, note: int):
        """Slot to handle note-off events from the UI thread."""
        state_to_emit = None
        with self._lock:
            # 1. Update the internal state.
            msg = mido.Message("note_off", note=note)
            self._message_queue.append(msg)

            # 2. Check for a state change.
            if note in self._active_notes:
                self._active_notes.discard(note)
                # 3. Capture the new state.
                state_to_emit = {"active_notes": list(self._active_notes)}

        # 4. After releasing the lock, emit the signal.
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def process(self, input_data: Dict) -> Dict:
        """Called by the processing thread to get the next MIDI message."""
        with self._lock:
            if self._message_queue:
                return {"msg_out": self._message_queue.popleft()}
        return {"msg_out": None}

    def _reset_and_notify(self):
        """Helper to clear all state and notify the UI."""
        with self._lock:
            self._message_queue.clear()
            self._active_notes.clear()
            state_to_emit = {"active_notes": []}
        self.emitter.stateUpdated.emit(state_to_emit)

    def start(self):
        """Called when graph processing starts. Resets the node's state."""
        self._reset_and_notify()

    def stop(self):
        """Called when graph processing stops. Resets state."""
        self._reset_and_notify()