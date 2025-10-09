import mido
import numpy as np
import threading
import logging
import time
from collections import deque
from typing import Dict, Optional, List, Tuple, Any

# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING

from constants import MIDIPacket

# --- Qt Imports ---
from PySide6.QtWidgets import (
    QWidget,
    QComboBox,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QSizePolicy,
    QSpinBox,
    QInputDialog,
)
from PySide6.QtCore import Qt, Slot, QSignalBlocker, Signal, QObject, QTimer

# --- Dependency Check ---
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("MIDI nodes: 'librosa' library not found. MIDI note to frequency conversion will be disabled.")

# Configure logging
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. MIDI Device Manager
# ==============================================================================
class MIDIDeviceManager:
    """Manages MIDI device discovery."""

    @staticmethod
    def get_input_devices() -> List[str]:
        """Returns a list of available MIDI input device names."""
        try:
            return mido.get_input_names()
        except Exception as e:
            logger.error(f"MIDIDeviceManager: Error getting input devices: {e}")
            return []

    @staticmethod
    def get_output_devices() -> List[str]:
        """Returns a list of available MIDI output device names."""
        try:
            return mido.get_output_names()
        except Exception as e:
            logger.error(f"MIDIDeviceManager: Error getting output devices: {e}")
            return []


# ==============================================================================
# 2. MIDI Input Node (Listens for all MIDI messages)
# ==============================================================================
class MIDIInputNodeItem(NodeItem):
    NODE_WIDTH = 250

    def __init__(self, node_logic: "MIDIInputNode"):
        super().__init__(node_logic)
        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(self.NODE_WIDTH - 40)
        device_row.addWidget(self.device_combo)
        self.refresh_button = QPushButton("🔄")
        self.refresh_button.setFixedSize(24, 24)
        device_row.addWidget(self.refresh_button)
        layout.addLayout(device_row)
        self.status_label = QLabel("Status: Initializing...")
        layout.addWidget(self.status_label)
        self.setContentWidget(self.container_widget)

        self.device_combo.currentIndexChanged.connect(self._on_device_selection_changed)
        self.refresh_button.clicked.connect(self._populate_device_combobox)

        # Initial population and sync
        self._populate_device_combobox()
        self.updateFromLogic()

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        super()._on_state_updated_from_logic(state)
        """Updates the status label and its color based on the state dictionary."""
        status = state.get("status", "")
        self.status_label.setText(status)
        if "Error" in status:
            self.status_label.setStyleSheet("color: red;")
        elif "Active" in status:
            self.status_label.setStyleSheet("color: lightgreen;")
        else:
            self.status_label.setStyleSheet("color: lightgray;")

        # Update combo box selection without re-populating the whole list
        with QSignalBlocker(self.device_combo):
            port_name = state.get("port_name")
            index = self.device_combo.findData(port_name)
            if index != -1:
                self.device_combo.setCurrentIndex(index)
            else:
                self.device_combo.setCurrentIndex(0)  # "No Device"

    @Slot()
    def _populate_device_combobox(self):
        with self.node_logic._lock:
            current_selection = self.node_logic._port_name

        with QSignalBlocker(self.device_combo):
            self.device_combo.clear()
            self.device_combo.addItem("No Device", userData=None)
            devices = MIDIDeviceManager.get_input_devices()
            for name in devices:
                self.device_combo.addItem(name, userData=name)
            index = self.device_combo.findData(current_selection)
            if index != -1:
                self.device_combo.setCurrentIndex(index)
            else:
                if self.node_logic._port_name is not None:
                    # This will trigger a state update and correctly set the UI
                    self.node_logic.set_device(None)

    @Slot(int)
    def _on_device_selection_changed(self, index: int):
        port_name = self.device_combo.itemData(index)
        self.node_logic.set_device(port_name)


class MIDIInputNode(Node):
    NODE_TYPE = "MIDI Device Input"
    CATEGORY = "MIDI"
    DESCRIPTION = "Receives messages from a MIDI input device."
    UI_CLASS = MIDIInputNodeItem

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_output("msg_out", data_type=MIDIPacket)

        self._port_name: Optional[str] = None
        self._port: Optional[mido.ports.BaseInput] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._message_queue = deque(maxlen=100)
        self._status = "No Device"

    def _get_state_snapshot_locked(self) -> Dict:
        return {"status": self._status, "port_name": self._port_name}

    def _midi_input_loop(self):
        port_to_close = None
        try:
            with self._lock:
                port_name = self._port_name

            try:
                port = mido.open_input(port_name)
                port_to_close = port
                with self._lock:
                    self._port = port
            except Exception as e:
                error_str = str(e)
                logger.error(f"[{self.name}] Failed to open MIDI port '{port_name}': {error_str}", exc_info=True)
                user_message = "Error: Port is busy or unavailable."
                with self._lock:
                    self._status = user_message
                self.ui_update_callback(self._get_state_snapshot_locked())
                return

            with self._lock:
                self._status = f"Active: {port_name.split(':')[0]}"
            self.ui_update_callback(self._get_state_snapshot_locked())

            try:
                # This blocking iterator will raise an exception if the device is disconnected.
                for msg in port:
                    if self._stop_event.is_set():
                        break
                    with self._lock:
                        self._message_queue.append(msg)

            except IOError as e:
                # Handle unexpected device disconnection
                if not self._stop_event.is_set():
                    logger.warning(f"[{self.name}] MIDI device '{self._port_name}' was disconnected: {e}")
                    with self._lock:
                        self._status = "Error: Device disconnected."
                    self.ui_update_callback(self._get_state_snapshot_locked())

        except Exception as e:
            # This is expected when the port is closed by stop(), but if not, it's an error.
            if not self._stop_event.is_set():
                logger.error(f"[{self.name}] Unhandled error in MIDI input thread: {e}", exc_info=True)
                with self._lock:
                    self._status = f"Error: {e}"
                self.ui_update_callback(self._get_state_snapshot_locked())
        finally:
            if port_to_close:
                port_to_close.close()
            logger.info(f"[{self.name}] MIDI worker thread finished for port '{self._port_name}'.")

    @Slot(str)
    def set_device(self, port_name: Optional[str]):
        self.stop()
        with self._lock:
            self._port_name = port_name
        self.start()

    def process(self, input_data: Dict) -> Dict:
        messages_this_tick = []
        with self._lock:
            while self._message_queue:
                # For now, assign all messages an offset of 0.
                # A more advanced system could calculate a more precise offset.
                messages_this_tick.append((0, self._message_queue.popleft()))

        if messages_this_tick:
            return {"msg_out": MIDIPacket(messages=messages_this_tick)}
        return {"msg_out": None}

    def start(self):
        with self._lock:
            if not self._port_name:
                self._status = "No Device Selected"
                self.ui_update_callback(self._get_state_snapshot_locked())
                return
            if self._worker_thread is not None:
                return

            self._stop_event.clear()
            self._message_queue.clear()
            self._worker_thread = threading.Thread(target=self._midi_input_loop, daemon=True)
            self._worker_thread.start()
            self._status = "Connecting..."
        self.ui_update_callback(self._get_state_snapshot_locked())

    def stop(self):
        self._stop_event.set()
        port_to_close = None
        worker_to_join = None
        with self._lock:
            if self._port:
                port_to_close = self._port
                self._port = None
            if self._worker_thread:
                worker_to_join = self._worker_thread
                self._worker_thread = None
        if port_to_close:
            try:
                port_to_close.close()
            except Exception as e:
                logger.warning(f"[{self.name}] Error while closing port during stop: {e}")
        if worker_to_join:
            worker_to_join.join(timeout=1.0)
            if worker_to_join.is_alive():
                logger.warning(f"[{self.name}] MIDI worker thread did not terminate cleanly.")
        with self._lock:
            self._status = "Stopped"
        self.ui_update_callback(self._get_state_snapshot_locked())

    def remove(self):
        self.stop()
        super().remove()

    def serialize_extra(self) -> Dict:
        with self._lock:
            return {"port_name": self._port_name}

    def deserialize_extra(self, data: Dict):
        port_name = data.get("port_name")
        QTimer.singleShot(0, lambda: self.set_device(port_name))


# ==============================================================================
# 3. MIDI Note to Gate/Pitch Node (FINAL REVISION)
# ==============================================================================
class MIDINoteToGatePitchNode(Node):
    NODE_TYPE = "MIDI Note to Gate/Pitch"
    CATEGORY = "MIDI"
    DESCRIPTION = "Converts MIDI notes to mono (last-note) and a polyphonic list of note data tuples."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("msg_in", data_type=MIDIPacket)

        # --- Monophonic outputs for backward compatibility ---
        self.add_output("gate_out", data_type=bool)
        self.add_output("pitch_out", data_type=float)
        self.add_output("velocity_out", data_type=float)

        # --- NEW: Single polyphonic output with structured data ---
        self.add_output("notes_data_out", data_type=list)

        # Internal state for tracking all active notes and their press order
        self._active_notes: Dict[int, Dict[str, float]] = {}
        self._note_priority_stack: List[int] = []

    def process(self, input_data: Dict) -> Dict:
        packet = input_data.get("msg_in")

        if isinstance(packet, MIDIPacket):
            for _, msg in packet.messages:
                # --- NOTE ON ---
                if msg.type == "note_on" and msg.velocity > 0:
                    if msg.note in self._note_priority_stack:
                        self._note_priority_stack.remove(msg.note)

                    self._note_priority_stack.append(msg.note)

                    self._active_notes[msg.note] = {
                        "velocity": msg.velocity / 127.0,
                        "pitch_hz": librosa.midi_to_hz(msg.note) if LIBROSA_AVAILABLE else 0.0,
                    }

                # --- NOTE OFF ---
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    if msg.note in self._active_notes:
                        del self._active_notes[msg.note]
                    if msg.note in self._note_priority_stack:
                        self._note_priority_stack.remove(msg.note)

        # --- Monophonic Outputs (Last-Note Priority) ---
        mono_gate = False
        mono_pitch = None
        mono_velocity = 0.0

        if self._note_priority_stack:
            last_note = self._note_priority_stack[-1]
            if last_note in self._active_notes:
                mono_gate = True
                mono_pitch = self._active_notes[last_note]["pitch_hz"]
                mono_velocity = self._active_notes[last_note]["velocity"]

        # --- Polyphonic Output (List of Tuples) ---
        # A list comprehension creates the structured list directly from the dictionary
        poly_notes_data = [(note, data["velocity"], data["pitch_hz"]) for note, data in self._active_notes.items()]
        if len(poly_notes_data) == 0:
            poly_notes_data = None

        # Return all outputs
        return {
            "gate_out": mono_gate,
            "pitch_out": mono_pitch,
            "velocity_out": mono_velocity,
            "notes_data_out": poly_notes_data,
        }

    def start(self):
        """Reset state when processing starts."""
        self._active_notes.clear()
        self._note_priority_stack.clear()
        super().start()

    def stop(self):
        """Reset state when processing stops."""
        self._active_notes.clear()
        self._note_priority_stack.clear()
        super().stop()


# ==============================================================================
# 4. MIDI Control Change (CC) Node
# ==============================================================================
class MIDIControlChangeNodeItem(NodeItem):
    NODE_WIDTH = 150

    def __init__(self, node_logic: "MIDIControlChangeNode"):
        super().__init__(node_logic, width=self.NODE_WIDTH)
        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(4)
        layout.addWidget(QLabel("CC Number:"))
        self.cc_spinbox = QSpinBox()
        self.cc_spinbox.setRange(0, 127)
        layout.addWidget(self.cc_spinbox)
        self.value_label = QLabel("Value: 0.00")
        layout.addWidget(self.value_label)
        self.setContentWidget(self.container_widget)

        self.cc_spinbox.valueChanged.connect(self.node_logic.set_cc_number)
        self.updateFromLogic()

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: Dict):
        super()._on_state_updated_from_logic(state)
        with QSignalBlocker(self.cc_spinbox):
            self.cc_spinbox.setValue(state.get("cc_number", 1))
        self.value_label.setText(f'Value: {state.get("value", 0.0):.2f}')


class MIDIControlChangeNode(Node):
    NODE_TYPE = "MIDI Control Change"
    CATEGORY = "MIDI"
    DESCRIPTION = "Outputs the value of a specific MIDI CC controller."
    UI_CLASS = MIDIControlChangeNodeItem

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("msg_in", data_type=MIDIPacket)
        self.add_output("value_out", data_type=float)

        self._cc_number = 1
        self._last_value = 0.0

    @Slot(int)
    def set_cc_number(self, cc_number: int):
        state_to_emit = None
        with self._lock:
            if self._cc_number != cc_number:
                self._cc_number = cc_number
                self._last_value = 0.0
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def _get_state_snapshot_locked(self) -> Dict:
        return {"cc_number": self._cc_number, "value": self._last_value}

    def process(self, input_data: Dict) -> Dict:
        packet = input_data.get("msg_in")
        if isinstance(packet, MIDIPacket):
            with self._lock:
                target_cc = self._cc_number

            # Find the last matching CC message in the packet
            last_matching_value = None
            for _, msg in reversed(packet.messages):
                if msg.type == "control_change" and msg.control == target_cc:
                    last_matching_value = msg.value
                    break

            if last_matching_value is not None:
                new_value = last_matching_value / 127.0
                state_to_emit = None
                with self._lock:
                    if self._last_value != new_value:
                        self._last_value = new_value
                        state_to_emit = self._get_state_snapshot_locked()
                if state_to_emit:
                    self.ui_update_callback(state_to_emit)

        with self._lock:
            output_value = self._last_value
        return {"value_out": output_value}

    def serialize_extra(self) -> Dict:
        with self._lock:
            return {"cc_number": self._cc_number}

    def deserialize_extra(self, data: Dict):
        self.set_cc_number(data.get("cc_number", 1))


# ==============================================================================
# 5. MIDI Pitch Wheel Node
# ==============================================================================
class MIDIPitchWheelNodeItem(NodeItem):
    NODE_WIDTH = 150

    def __init__(self, node_logic: "MIDIPitchWheelNode"):
        super().__init__(node_logic, width=self.NODE_WIDTH)
        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(4)
        self.value_label = QLabel("Value: 0.00")
        layout.addWidget(self.value_label)
        self.setContentWidget(self.container_widget)
        self.updateFromLogic()

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: Dict):
        super()._on_state_updated_from_logic(state)
        self.value_label.setText(f'Value: {state.get("value", 0.0):.2f}')


class MIDIPitchWheelNode(Node):
    NODE_TYPE = "MIDI Pitch Wheel"
    CATEGORY = "MIDI"
    DESCRIPTION = "Outputs the value of the MIDI pitch wheel (-1.0 to 1.0)."
    UI_CLASS = MIDIPitchWheelNodeItem

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("msg_in", data_type=MIDIPacket)
        self.add_output("value_out", data_type=float)

        self._last_value = 0.0

    def _get_state_snapshot_locked(self) -> Dict:
        return {"value": self._last_value}

    def process(self, input_data: Dict) -> Dict:
        packet = input_data.get("msg_in")
        if isinstance(packet, MIDIPacket):
            last_pitch_value = None
            for _, msg in reversed(packet.messages):
                if msg.type == "pitchwheel":
                    last_pitch_value = msg.pitch
                    break

            if last_pitch_value is not None:
                new_value = last_pitch_value / 8191.0
                state_to_emit = None
                with self._lock:
                    if abs(self._last_value - new_value) > 1e-6:
                        self._last_value = new_value
                        state_to_emit = self._get_state_snapshot_locked()
                if state_to_emit:
                    self.ui_update_callback(state_to_emit)

        with self._lock:
            output_value = self._last_value
        return {"value_out": output_value}

    def serialize_extra(self) -> Dict:
        return {}

    def deserialize_extra(self, data: Dict):
        pass


# ==============================================================================
# 6. MIDI Output Node
# ==============================================================================
class MIDIOutputNodeItem(NodeItem):
    NODE_WIDTH = 250

    def __init__(self, node_logic: "MIDIOutputNode"):
        super().__init__(node_logic)
        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        device_row.addWidget(self.device_combo)

        self.refresh_button = QPushButton("🔄")
        self.refresh_button.setFixedSize(24, 24)
        self.refresh_button.setToolTip("Refresh device list")
        device_row.addWidget(self.refresh_button)

        self.new_virtual_button = QPushButton("New...")
        self.new_virtual_button.setToolTip("Create a new virtual MIDI output port")
        device_row.addWidget(self.new_virtual_button)

        layout.addLayout(device_row)
        self.status_label = QLabel("Status: Initializing...")
        layout.addWidget(self.status_label)
        self.setContentWidget(self.container_widget)

        self.device_combo.currentIndexChanged.connect(self._on_device_selection_changed)
        self.refresh_button.clicked.connect(self._populate_device_combobox)
        self.new_virtual_button.clicked.connect(self._on_create_virtual_port_clicked)
        self._populate_device_combobox()
        self.updateFromLogic()

    @Slot()
    def _on_create_virtual_port_clicked(self):
        parent_widget = self.scene().views()[0] if self.scene() and self.scene().views() else None
        port_name, ok = QInputDialog.getText(
            parent_widget, "Create Virtual MIDI Port", "Enter a name for the new port:"
        )
        if ok and port_name:
            self.node_logic.create_and_set_virtual_port(port_name)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        super()._on_state_updated_from_logic(state)
        status = state.get("status", "")
        self.status_label.setText(status)
        if "Error" in status:
            self.status_label.setStyleSheet("color: red;")
        elif "Active" in status:
            self.status_label.setStyleSheet("color: lightgreen;")
        else:
            self.status_label.setStyleSheet("color: lightgray;")

        if state.get("device_list_refreshed", False):
            logger.info(f"[{self.node_logic.name}] UI: Refreshing device list due to state update.")
            self._populate_device_combobox()

        with QSignalBlocker(self.device_combo):
            port_name = state.get("port_name")
            index = self.device_combo.findData(port_name)
            if index != -1:
                self.device_combo.setCurrentIndex(index)
            else:
                self.device_combo.setCurrentIndex(0)

    @Slot()
    def _populate_device_combobox(self):
        with self.node_logic._lock:  # FIX: Correctly acquire the lock from the logic object
            current_selection = self.node_logic._port_name

        with QSignalBlocker(self.device_combo):
            self.device_combo.clear()
            self.device_combo.addItem("No Device", userData=None)
            devices = MIDIDeviceManager.get_output_devices()
            for name in devices:
                self.device_combo.addItem(name, userData=name)
            index = self.device_combo.findData(current_selection)
            if index != -1:
                self.device_combo.setCurrentIndex(index)
            elif self.node_logic._port_name is not None:
                pass

    @Slot(int)
    def _on_device_selection_changed(self, index: int):
        if index < 0:
            return
        port_name = self.device_combo.itemData(index)
        self.node_logic.set_device(port_name)


class MIDIOutputNode(Node):
    NODE_TYPE = "MIDI Device Output"
    CATEGORY = "MIDI"
    DESCRIPTION = "Sends MIDI messages to an external device or application."
    UI_CLASS = MIDIOutputNodeItem

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("msg_in", data_type=MIDIPacket)

        self._port_name: Optional[str] = None
        self._port: Optional[mido.ports.BaseOutput] = None
        self._status = "No Device"
        self._is_virtual = False

    def _get_state_snapshot_locked(self) -> Dict:
        """Helper to get a snapshot of the current state under lock."""
        return {"status": self._status, "port_name": self._port_name, "is_virtual": self._is_virtual}

    @Slot(str)
    def create_and_set_virtual_port(self, port_name: str):
        """Creates a new virtual port and sets it as the active device."""
        self.stop()
        with self._lock:
            self._port_name = port_name
            self._is_virtual = True
        self.start()
        state = self._get_state_snapshot_locked()
        state["device_list_refreshed"] = True
        self.ui_update_callback(state)

    @Slot(str)
    def set_device(self, port_name: Optional[str]):
        """Sets an existing device as the active output."""
        self.stop()
        with self._lock:
            self._port_name = port_name
            self._is_virtual = False
        self.start()

    def process(self, input_data: Dict) -> Dict:
        packet = input_data.get("msg_in")
        port_to_close = None
        error_status = None

        if isinstance(packet, MIDIPacket):
            with self._lock:
                port = self._port
                if port and not port.closed:
                    for _, msg in packet.messages:
                        try:
                            port.send(msg)
                        except IOError as e:
                            logger.error(f"[{self.name}] Device disconnected or error sending MIDI message: {e}")
                            self._status = "Error: Device disconnected."
                            error_status = self._status
                            port_to_close = port
                            self._port = None
                            break  # Stop trying to send more messages on the broken port
                        except Exception as e:
                            logger.error(f"[{self.name}] Failed to send MIDI message: {e}")
                            self._status = f"Error: {e}"
                            error_status = self._status

        if port_to_close:
            try:
                port_to_close.close()
            except Exception as e:
                logger.warning(f"[{self.name}] Error closing disconnected port: {e}")

        if error_status:
            with self._lock:
                state_to_emit = self._get_state_snapshot_locked()
            self.ui_update_callback(state_to_emit)

        return {}

    def start(self):
        with self._lock:
            if not self._port_name:
                self._status = "No Device Selected"
                self.ui_update_callback(self._get_state_snapshot_locked())
                return
            if self._port and not self._port.closed:
                return

            try:
                if self._is_virtual:
                    self._port = mido.open_output(self._port_name, virtual=True)
                    logger.info(f"[{self.name}] Created and opened virtual MIDI output port: '{self._port_name}'")
                else:
                    self._port = mido.open_output(self._port_name)
                    logger.info(f"[{self.name}] Opened MIDI output port: '{self._port_name}'")

                self._status = f"Active: {self._port_name.split(':')[0]}"
            except Exception as e:
                self._status = f"Error: {e}"
                logger.error(f"[{self.name}] Failed to open MIDI output port: {e}", exc_info=True)
        self.ui_update_callback(self._get_state_snapshot_locked())

    def stop(self):
        port_to_close = None
        with self._lock:
            if self._port and not self._port.closed:
                port_to_close = self._port
                self._port = None

        if port_to_close:
            try:
                port_to_close.close()
                logger.info(f"[{self.name}] Closed MIDI output port: '{self._port_name}'")
            except Exception as e:
                logger.warning(f"[{self.name}] Error closing MIDI port: {e}")

        with self._lock:
            self._status = "Inactive"
        self.ui_update_callback(self._get_state_snapshot_locked())

    def remove(self):
        self.stop()
        super().remove()

    def serialize_extra(self) -> Dict:
        with self._lock:
            return {"port_name": self._port_name, "is_virtual": self._is_virtual}

    def _set_device_on_load(self, port_name: Optional[str], is_virtual: bool):
        """Internal method to configure the device when loading a graph."""
        self.stop()
        with self._lock:
            self._port_name = port_name
            self._is_virtual = is_virtual
        self.start()

    def deserialize_extra(self, data: Dict):
        port_name = data.get("port_name")
        is_virtual = data.get("is_virtual", False)
        QTimer.singleShot(0, lambda: self._set_device_on_load(port_name, is_virtual))


# ==============================================================================
# 7. MIDI Pitch Wheel Output Node
# ==============================================================================
class MIDIPitchWheelOutNodeItem(NodeItem):
    NODE_WIDTH = 150

    def __init__(self, node_logic: "MIDIPitchWheelOutNode"):
        super().__init__(node_logic, width=self.NODE_WIDTH)
        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(4)
        self.value_label = QLabel("Value: 0.00")
        layout.addWidget(self.value_label)
        self.setContentWidget(self.container_widget)
        self.updateFromLogic()

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: Dict):
        super()._on_state_updated_from_logic(state)
        self.value_label.setText(f'Value: {state.get("value", 0.0):.2f}')


class MIDIPitchWheelOutNode(Node):
    NODE_TYPE = "MIDI Pitch Wheel Out"
    CATEGORY = "MIDI"
    DESCRIPTION = "Converts a float value (-1.0 to 1.0) to a MIDI Pitch Wheel message."
    UI_CLASS = MIDIPitchWheelOutNodeItem

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("value_in", data_type=float)
        self.add_output("msg_out", data_type=MIDIPacket)
        self._last_sent_value = 0.0

    def _get_state_snapshot_locked(self) -> Dict:
        return {"value": self._last_sent_value}

    def process(self, input_data: Dict) -> Dict:
        value_in = input_data.get("value_in")
        if value_in is None:
            return {"msg_out": None}

        try:
            # Clamp value to the expected range
            clamped_value = max(-1.0, min(1.0, float(value_in)))
            value_changed = False
            packet = None
            state_to_emit = None

            with self._lock:
                if abs(self._last_sent_value - clamped_value) > 1e-4:
                    self._last_sent_value = clamped_value
                    value_changed = True
                    state_to_emit = self._get_state_snapshot_locked()

            if value_changed and state_to_emit:
                self.ui_update_callback(state_to_emit)
                pitch_value = int(round(clamped_value * 8191))
                msg = mido.Message("pitchwheel", pitch=pitch_value)
                packet = MIDIPacket(messages=[(0, msg)])

            return {"msg_out": packet}

        except (TypeError, ValueError) as e:
            logger.warning(f"[{self.name}] Invalid input for pitch wheel: {value_in}. Error: {e}")
            return {"msg_out": None}


# ==============================================================================
# 8. MIDI Control Change (CC) Output Node
# ==============================================================================
class MIDIControlChangeOutNodeItem(NodeItem):
    NODE_WIDTH = 150

    def __init__(self, node_logic: "MIDIControlChangeOutNode"):
        super().__init__(node_logic, width=self.NODE_WIDTH)
        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(4)
        layout.addWidget(QLabel("CC Number:"))
        self.cc_spinbox = QSpinBox()
        self.cc_spinbox.setRange(0, 127)
        layout.addWidget(self.cc_spinbox)
        self.value_label = QLabel("Value: 0.00")
        layout.addWidget(self.value_label)
        self.setContentWidget(self.container_widget)

        self.cc_spinbox.valueChanged.connect(self.node_logic.set_cc_number)
        self.updateFromLogic()

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: Dict):
        super()._on_state_updated_from_logic(state)
        with QSignalBlocker(self.cc_spinbox):
            self.cc_spinbox.setValue(state.get("cc_number", 1))
        self.value_label.setText(f'Value: {state.get("value", 0.0):.2f}')


class MIDIControlChangeOutNode(Node):
    NODE_TYPE = "MIDI Control Change Out"
    CATEGORY = "MIDI"
    DESCRIPTION = "Converts a float value (0.0 to 1.0) to a MIDI CC message."
    UI_CLASS = MIDIControlChangeOutNodeItem

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("value_in", data_type=float)
        self.add_output("msg_out", data_type=MIDIPacket)

        self._cc_number = 1
        self._last_sent_value = 0.0

    @Slot(int)
    def set_cc_number(self, cc_number: int):
        with self._lock:
            if self._cc_number != cc_number:
                self._cc_number = cc_number
        self.ui_update_callback(self._get_state_snapshot_locked())

    def _get_state_snapshot_locked(self) -> Dict:
        return {"cc_number": self._cc_number, "value": self._last_sent_value}

    def process(self, input_data: Dict) -> Dict:
        value_in = input_data.get("value_in")
        if value_in is None:
            return {"msg_out": None}

        try:
            # Clamp value to the expected range
            clamped_value = max(0.0, min(1.0, float(value_in)))
            # Convert float (0-1) to MIDI CC value (0-127)
            cc_value = int(round(clamped_value * 127))

            state_to_emit = None
            with self._lock:
                cc_num = self._cc_number
                if abs(self._last_sent_value - clamped_value) > 1e-4:
                    self._last_sent_value = clamped_value
                    state_to_emit = self._get_state_snapshot_locked()

            if state_to_emit:
                self.ui_update_callback(state_to_emit)

            msg = mido.Message("control_change", control=cc_num, value=cc_value)
            packet = MIDIPacket(messages=[(0, msg)])
            return {"msg_out": packet}

        except (TypeError, ValueError) as e:
            logger.warning(f"[{self.name}] Invalid input for CC value: {value_in}. Error: {e}")
            return {"msg_out": None}

    def serialize_extra(self) -> Dict:
        with self._lock:
            return {"cc_number": self._cc_number}

    def deserialize_extra(self, data: Dict):
        self.set_cc_number(data.get("cc_number", 1))


# ==============================================================================
# 9. --- NEW: MIDI Merge (2-to-1) Node ---
# ==============================================================================
class MIDIMergeNode(Node):
    NODE_TYPE = "MIDI Merge (2-to-1)"
    CATEGORY = "MIDI"
    DESCRIPTION = "Merges two MIDI streams into one. Input A has priority."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("msg_in_A", data_type=MIDIPacket)
        self.add_input("msg_in_B", data_type=MIDIPacket)
        self.add_output("msg_out", data_type=MIDIPacket)

    def process(self, input_data: Dict) -> Dict:
        """
        Merges messages from two MIDIPackets, prioritizing all messages from packet A.
        """
        packet_a = input_data.get("msg_in_A")
        packet_b = input_data.get("msg_in_B")

        merged_messages = []
        if isinstance(packet_a, MIDIPacket):
            merged_messages.extend(packet_a.messages)
        if isinstance(packet_b, MIDIPacket):
            merged_messages.extend(packet_b.messages)

        if merged_messages:
            # Note: For simplicity, this doesn't sort by sample offset.
            # A more advanced merge might do that.
            return {"msg_out": MIDIPacket(messages=merged_messages)}

        return {"msg_out": None}
