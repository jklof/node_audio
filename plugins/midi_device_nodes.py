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

# --- Qt Imports ---
from PySide6.QtWidgets import QWidget, QComboBox, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QSizePolicy, QSpinBox
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
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        self._populate_device_combobox()

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        """Updates the status label and its color based on the state dictionary."""
        status = state.get("status", "")
        self.status_label.setText(status)
        if "Error" in status:
            self.status_label.setStyleSheet("color: red;")
        elif "Active" in status:
            self.status_label.setStyleSheet("color: lightgreen;")
        else:
            self.status_label.setStyleSheet("color: lightgray;")

    @Slot()
    def _populate_device_combobox(self):
        with QSignalBlocker(self.device_combo):
            current_selection = self.node_logic._port_name
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

    class Emitter(QObject):
        deviceListChanged = Signal()
        stateUpdated = Signal(dict)

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = self.Emitter()
        self.add_output("msg_out", data_type=object)

        self._lock = threading.Lock()
        self._port_name: Optional[str] = None
        self._port: Optional[mido.ports.BaseInput] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._message_queue = deque(maxlen=100)
        self._status = "No Device"

    def _midi_input_loop(self):
        port_to_close = None
        try:
            with self._lock:
                port_name = self._port_name

            try:
                # This is the line that can fail if the port is busy
                port = mido.open_input(port_name)
                port_to_close = port
                with self._lock:
                    self._port = port  # Store the active port
            except Exception as e:
                error_str = str(e)
                logger.error(f"[{self.name}] Failed to open MIDI port '{port_name}': {error_str}", exc_info=True)
                user_message = "Error: Port is busy or unavailable."
                self.emitter.stateUpdated.emit({"status": user_message})
                return

            self.emitter.stateUpdated.emit({"status": f"Active: {port_name.split(':')[0]}"})
            # The blocking iterator will raise an exception when the port is closed from another thread
            for msg in port:
                if self._stop_event.is_set():
                    break
                with self._lock:
                    self._message_queue.append(msg)

        except Exception as e:
            # This is expected when the port is closed externally by our stop() method
            if not self._stop_event.is_set():
                logger.error(f"[{self.name}] Unhandled error in MIDI input thread: {e}", exc_info=True)
                self.emitter.stateUpdated.emit({"status": f"Error: {e}"})
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
        with self._lock:
            if self._message_queue:
                msg = self._message_queue.popleft()
                return {"msg_out": msg}
        return {"msg_out": None}

    def start(self):
        with self._lock:
            if not self._port_name:
                self.emitter.stateUpdated.emit({"status": "No Device Selected"})
                return
            if self._worker_thread is not None:
                return

            self._stop_event.clear()
            self._message_queue.clear()
            self._worker_thread = threading.Thread(target=self._midi_input_loop, daemon=True)
            self._worker_thread.start()
            self._status = "Connecting..."
        self.emitter.stateUpdated.emit({"status": self._status})

    def stop(self):
        """--- FIX: Robustly stops the MIDI thread ---"""
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

        # Close the port from this main thread. This will interrupt the
        # blocking `for msg in port:` loop in the worker thread.
        if port_to_close:
            try:
                port_to_close.close()
            except Exception as e:
                logger.warning(f"[{self.name}] Error while closing port during stop: {e}")

        # Now, join the worker thread, which should exit quickly.
        if worker_to_join:
            worker_to_join.join(timeout=1.0)
            if worker_to_join.is_alive():
                logger.warning(f"[{self.name}] MIDI worker thread did not terminate cleanly.")

        self._status = "Stopped"
        self.emitter.stateUpdated.emit({"status": self._status})

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
# 3. MIDI Note to Gate/Pitch Node
# ==============================================================================
class MIDINoteToGatePitchNode(Node):
    NODE_TYPE = "MIDI Note to Gate/Pitch"
    CATEGORY = "MIDI"
    DESCRIPTION = "Converts MIDI note messages into gate, pitch, and velocity signals."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("msg_in", data_type=object)
        self.add_output("gate_out", data_type=bool)
        self.add_output("pitch_out", data_type=float)
        self.add_output("velocity_out", data_type=float)

        self._active_note = None
        self._gate = False

    def process(self, input_data: Dict) -> Dict:
        msg = input_data.get("msg_in")
        pitch_hz = None
        velocity = None

        if not isinstance(msg, mido.Message):
            return {"gate_out": self._gate, "pitch_out": None, "velocity_out": None}

        if msg.type == "note_on" and msg.velocity > 0:
            self._active_note = msg.note
            self._gate = True
            velocity = msg.velocity / 127.0
            if LIBROSA_AVAILABLE:
                pitch_hz = librosa.midi_to_hz(msg.note)

        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note == self._active_note:
                self._gate = False
                self._active_note = None

        return {"gate_out": self._gate, "pitch_out": pitch_hz, "velocity_out": velocity}


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
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

    @Slot()
    def updateFromLogic(self):
        """
        Pulls the current state from the logic node to initialize the UI.
        """
        state = self.node_logic.get_current_state()
        self._on_state_updated(state)
        super().updateFromLogic()

    @Slot(dict)
    def _on_state_updated(self, state: Dict):
        with QSignalBlocker(self.cc_spinbox):
            self.cc_spinbox.setValue(state.get("cc_number", 1))
        self.value_label.setText(f'Value: {state.get("value", 0.0):.2f}')


class MIDIControlChangeNode(Node):
    NODE_TYPE = "MIDI Control Change"
    CATEGORY = "MIDI"
    DESCRIPTION = "Outputs the value of a specific MIDI CC controller."
    UI_CLASS = MIDIControlChangeNodeItem

    class Emitter(QObject):
        stateUpdated = Signal(dict)

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = self.Emitter()
        self.add_input("msg_in", data_type=object)
        self.add_output("value_out", data_type=float)

        self._lock = threading.Lock()
        self._cc_number = 1
        self._last_value = 0.0

    @Slot(int)
    def set_cc_number(self, cc_number: int):
        state_to_emit = None
        with self._lock:
            if self._cc_number != cc_number:
                self._cc_number = cc_number
                self._last_value = 0.0
                state_to_emit = {"cc_number": self._cc_number, "value": self._last_value}
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def get_current_state(self) -> Dict:
        with self._lock:
            return {"cc_number": self._cc_number, "value": self._last_value}

    def process(self, input_data: Dict) -> Dict:
        msg = input_data.get("msg_in")
        if isinstance(msg, mido.Message) and msg.type == "control_change":
            with self._lock:
                target_cc = self._cc_number
            if msg.control == target_cc:
                new_value = msg.value / 127.0
                with self._lock:
                    if self._last_value != new_value:
                        self._last_value = new_value
                        self.emitter.stateUpdated.emit({"cc_number": target_cc, "value": new_value})

        return {"value_out": self._last_value}

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

        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

    @Slot()
    def updateFromLogic(self):
        """
        Pulls the current state from the logic node to initialize the UI.
        """
        state = self.node_logic.get_current_state()
        self._on_state_updated(state)
        super().updateFromLogic()

    @Slot(dict)
    def _on_state_updated(self, state: Dict):
        self.value_label.setText(f'Value: {state.get("value", 0.0):.2f}')


class MIDIPitchWheelNode(Node):
    NODE_TYPE = "MIDI Pitch Wheel"
    CATEGORY = "MIDI"
    DESCRIPTION = "Outputs the value of the MIDI pitch wheel (-1.0 to 1.0)."
    UI_CLASS = MIDIPitchWheelNodeItem

    class Emitter(QObject):
        stateUpdated = Signal(dict)

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = self.Emitter()
        self.add_input("msg_in", data_type=object)
        self.add_output("value_out", data_type=float)

        self._lock = threading.Lock()
        self._last_value = 0.0

    def get_current_state(self) -> Dict:
        with self._lock:
            return {"value": self._last_value}

    def process(self, input_data: Dict) -> Dict:
        msg = input_data.get("msg_in")
        if isinstance(msg, mido.Message) and msg.type == "pitchwheel":
            # Pitch value ranges from -8192 to 8191.
            # We divide by 8191 to get a range of approx -1.0 to 1.0.
            new_value = msg.pitch / 8191.0
            with self._lock:
                # Use a small tolerance for float comparison
                if abs(self._last_value - new_value) > 1e-6:
                    self._last_value = new_value
                    self.emitter.stateUpdated.emit({"value": new_value})

        # Always return the last known value
        return {"value_out": self._last_value}

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
        self.device_combo.setMinimumWidth(self.NODE_WIDTH - 40)
        device_row.addWidget(self.device_combo)
        self.refresh_button = QPushButton("🔄")
        self.refresh_button.setFixedSize(24, 24)
        device_row.addWidget(self.device_combo)
        layout.addLayout(device_row)
        self.status_label = QLabel("Status: Initializing...")
        layout.addWidget(self.status_label)
        self.setContentWidget(self.container_widget)

        self.device_combo.currentIndexChanged.connect(self._on_device_selection_changed)
        self.refresh_button.clicked.connect(self._populate_device_combobox)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        self._populate_device_combobox()

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        status = state.get("status", "")
        self.status_label.setText(status)
        if "Error" in status:
            self.status_label.setStyleSheet("color: red;")
        elif "Active" in status:
            self.status_label.setStyleSheet("color: lightgreen;")
        else:
            self.status_label.setStyleSheet("color: lightgray;")

    @Slot()
    def _populate_device_combobox(self):
        with QSignalBlocker(self.device_combo):
            current_selection = self.node_logic._port_name
            self.device_combo.clear()
            self.device_combo.addItem("No Device", userData=None)
            devices = MIDIDeviceManager.get_output_devices()
            for name in devices:
                self.device_combo.addItem(name, userData=name)
            index = self.device_combo.findData(current_selection)
            if index != -1:
                self.device_combo.setCurrentIndex(index)
            elif self.node_logic._port_name is not None:
                self.node_logic.set_device(None)

    @Slot(int)
    def _on_device_selection_changed(self, index: int):
        port_name = self.device_combo.itemData(index)
        self.node_logic.set_device(port_name)


class MIDIOutputNode(Node):
    NODE_TYPE = "MIDI Device Output"
    CATEGORY = "MIDI"
    DESCRIPTION = "Sends MIDI messages to an external device or application."
    UI_CLASS = MIDIOutputNodeItem

    class Emitter(QObject):
        stateUpdated = Signal(dict)

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = self.Emitter()
        self.add_input("msg_in", data_type=object)

        self._lock = threading.Lock()
        self._port_name: Optional[str] = None
        self._port: Optional[mido.ports.BaseOutput] = None
        self._status = "No Device"

    @Slot(str)
    def set_device(self, port_name: Optional[str]):
        self.stop()
        with self._lock:
            self._port_name = port_name
        self.start()

    def process(self, input_data: Dict) -> Dict:
        msg = input_data.get("msg_in")
        with self._lock:
            port = self._port
        if isinstance(msg, mido.Message) and port and not port.closed:
            try:
                port.send(msg)
            except Exception as e:
                logger.error(f"[{self.name}] Failed to send MIDI message: {e}")
                self.emitter.stateUpdated.emit({"status": f"Error: {e}"})
        return {}

    def start(self):
        with self._lock:
            if not self._port_name:
                self.emitter.stateUpdated.emit({"status": "No Device Selected"})
                return
            if self._port and not self._port.closed:
                return

            try:
                self._port = mido.open_output(self._port_name)
                self._status = f"Active: {self._port_name.split(':')[0]}"
                logger.info(f"[{self.name}] Opened MIDI output port: '{self._port_name}'")
            except Exception as e:
                self._status = f"Error: {e}"
                logger.error(f"[{self.name}] Failed to open MIDI output port: {e}", exc_info=True)
        self.emitter.stateUpdated.emit({"status": self._status})

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

        self._status = "Inactive"
        self.emitter.stateUpdated.emit({"status": self._status})

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

        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

    @Slot()
    def updateFromLogic(self):
        """
        Pulls the current state from the logic node to initialize the UI.
        """
        state = self.node_logic.get_current_state()
        self._on_state_updated(state)
        super().updateFromLogic()

    @Slot(dict)
    def _on_state_updated(self, state: Dict):
        self.value_label.setText(f'Value: {state.get("value", 0.0):.2f}')


class MIDIPitchWheelOutNode(Node):
    NODE_TYPE = "MIDI Pitch Wheel Out"
    CATEGORY = "MIDI"
    DESCRIPTION = "Converts a float value (-1.0 to 1.0) to a MIDI Pitch Wheel message."
    UI_CLASS = MIDIPitchWheelOutNodeItem

    class Emitter(QObject):
        stateUpdated = Signal(dict)

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = self.Emitter()
        self.add_input("value_in", data_type=float)
        self.add_output("msg_out", data_type=object)
        self._lock = threading.Lock()
        self._last_sent_value = 0.0

    def get_current_state(self) -> Dict:
        with self._lock:
            return {"value": self._last_sent_value}

    def process(self, input_data: Dict) -> Dict:
        value_in = input_data.get("value_in")
        if value_in is None:
            return {"msg_out": None}

        try:
            # Clamp value to the expected range
            clamped_value = max(-1.0, min(1.0, float(value_in)))
            # Convert float (-1 to 1) to MIDI pitch value (-8192 to 8191)
            pitch_value = int(round(clamped_value * 8191))
            msg = mido.Message("pitchwheel", pitch=pitch_value)

            with self._lock:
                if abs(self._last_sent_value - clamped_value) > 1e-4:
                    self._last_sent_value = clamped_value
                    self.emitter.stateUpdated.emit({"value": clamped_value})

            return {"msg_out": msg}
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
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

    @Slot()
    def updateFromLogic(self):
        """
        Pulls the current state from the logic node to initialize the UI.
        """
        state = self.node_logic.get_current_state()
        self._on_state_updated(state)
        super().updateFromLogic()

    @Slot(dict)
    def _on_state_updated(self, state: Dict):
        with QSignalBlocker(self.cc_spinbox):
            self.cc_spinbox.setValue(state.get("cc_number", 1))
        self.value_label.setText(f'Value: {state.get("value", 0.0):.2f}')


class MIDIControlChangeOutNode(Node):
    NODE_TYPE = "MIDI Control Change Out"
    CATEGORY = "MIDI"
    DESCRIPTION = "Converts a float value (0.0 to 1.0) to a MIDI CC message."
    UI_CLASS = MIDIControlChangeOutNodeItem

    class Emitter(QObject):
        stateUpdated = Signal(dict)

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = self.Emitter()
        self.add_input("value_in", data_type=float)
        self.add_output("msg_out", data_type=object)

        self._lock = threading.Lock()
        self._cc_number = 1
        self._last_sent_value = 0.0

    @Slot(int)
    def set_cc_number(self, cc_number: int):
        with self._lock:
            self._cc_number = cc_number
        self.emitter.stateUpdated.emit({"cc_number": self._cc_number, "value": self._last_sent_value})

    def get_current_state(self) -> Dict:
        with self._lock:
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

            with self._lock:
                cc_num = self._cc_number
                if abs(self._last_sent_value - clamped_value) > 1e-4:
                    self._last_sent_value = clamped_value
                    self.emitter.stateUpdated.emit({"cc_number": cc_num, "value": clamped_value})

            msg = mido.Message("control_change", control=cc_num, value=cc_value)
            return {"msg_out": msg}

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
        self.add_input("msg_in_A", data_type=object)
        self.add_input("msg_in_B", data_type=object)
        self.add_output("msg_out", data_type=object)
        # A small queue to handle simultaneous messages without loss.
        self._queue = deque(maxlen=50)

    def process(self, input_data: Dict) -> Dict:
        """
        Queues incoming messages and outputs one per tick, ensuring no data loss
        and prioritizing input A.
        """
        msg_a = input_data.get("msg_in_A")
        if isinstance(msg_a, mido.Message):
            # Add to the front of the queue to give it priority
            self._queue.appendleft(msg_a)

        msg_b = input_data.get("msg_in_B")
        if isinstance(msg_b, mido.Message):
            self._queue.append(msg_b)

        if self._queue:
            return {"msg_out": self._queue.popleft()}

        # Otherwise, send nothing
        return {"msg_out": None}

    def start(self):
        self._queue.clear()

    def stop(self):
        self._queue.clear()
