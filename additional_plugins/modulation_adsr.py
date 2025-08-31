import numpy as np
import threading
import logging
from typing import Dict, Optional

# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE

# --- Qt Imports ---
from PySide6.QtCore import Qt, Signal, Slot, QObject, QSignalBlocker
from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget


# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
# A small value to prevent division by zero for instant transitions
MIN_TIME = 0.001
# A small value to add to denominators to prevent division by zero
EPSILON = 1e-9


# ==============================================================================
# 1. State Emitter for UI Communication
# ==============================================================================
class ADSREmitter(QObject):
    """A dedicated QObject to safely emit signals from the logic to the UI thread."""
    stateUpdated = Signal(dict)

# ==============================================================================
# 2. Custom UI Class (ADSRNodeItem)
# ==============================================================================
class ADSRNodeItem(NodeItem):
    """Provides a user interface with sliders to control the ADSR parameters."""
    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "ADSRNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING)
        main_layout.setSpacing(4)

        # Create slider controls for each parameter
        self.attack_slider, self.attack_label = self._create_slider_control("Attack", 0.0, 5.0, "{:.2f} s")
        self.decay_slider, self.decay_label = self._create_slider_control("Decay", 0.0, 5.0, "{:.2f} s")
        self.sustain_slider, self.sustain_label = self._create_slider_control("Sustain", 0.0, 1.0, "{:.1%}")
        self.release_slider, self.release_label = self._create_slider_control("Release", 0.0, 5.0, "{:.2f} s")

        for label, slider in [
            (self.attack_label, self.attack_slider),
            (self.decay_label, self.decay_slider),
            (self.sustain_label, self.sustain_slider),
            (self.release_label, self.release_slider),
        ]:
            main_layout.addWidget(label)
            main_layout.addWidget(slider)
        
        self.setContentWidget(self.container_widget)

        # Connect UI interactions to the logic node
        self.attack_slider.valueChanged.connect(self._on_attack_changed)
        self.decay_slider.valueChanged.connect(self._on_decay_changed)
        self.sustain_slider.valueChanged.connect(self._on_sustain_changed)
        self.release_slider.valueChanged.connect(self._on_release_changed)

        # Connect the logic node's state updates back to the UI
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        
        # Initial synchronization
        self.updateFromLogic()

    def _create_slider_control(self, name: str, min_val: float, max_val: float, fmt: str) -> tuple[QSlider, QLabel]:
        """Helper function to create a labeled slider."""
        label = QLabel(f"{name}: ...")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)
        # Store metadata directly on the slider widget for easy access
        slider.setProperty("min_val", min_val)
        slider.setProperty("max_val", max_val)
        slider.setProperty("name", name)
        slider.setProperty("format", fmt)
        return slider, label

    def _map_slider_to_logical(self, slider: QSlider) -> float:
        """Converts an integer slider position (0-1000) to its logical float value."""
        min_val = slider.property("min_val")
        max_val = slider.property("max_val")
        normalized = slider.value() / 1000.0
        return min_val + normalized * (max_val - min_val)

    def _map_logical_to_slider(self, slider: QSlider, logical_value: float) -> int:
        """Converts a logical float value to the corresponding integer slider position."""
        min_val = slider.property("min_val")
        max_val = slider.property("max_val")
        range_val = max_val - min_val
        if range_val == 0: return 0
        normalized = (logical_value - min_val) / range_val
        return int(np.clip(normalized, 0.0, 1.0) * 1000.0)

    @Slot(int)
    def _on_attack_changed(self):
        val = self._map_slider_to_logical(self.attack_slider)
        self.node_logic.set_attack(val)
        self.attack_label.setText(f"Attack: {self.attack_slider.property('format').format(val)}")

    @Slot(int)
    def _on_decay_changed(self):
        val = self._map_slider_to_logical(self.decay_slider)
        self.node_logic.set_decay(val)
        self.decay_label.setText(f"Decay: {self.decay_slider.property('format').format(val)}")
        
    @Slot(int)
    def _on_sustain_changed(self):
        val = self._map_slider_to_logical(self.sustain_slider)
        self.node_logic.set_sustain(val)
        self.sustain_label.setText(f"Sustain: {self.sustain_slider.property('format').format(val)}")
        
    @Slot(int)
    def _on_release_changed(self):
        val = self._map_slider_to_logical(self.release_slider)
        self.node_logic.set_release(val)
        self.release_label.setText(f"Release: {self.release_slider.property('format').format(val)}")

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        """Updates all UI controls based on a state dictionary from the logic node."""
        sliders_map = {
            "attack": (self.attack_slider, self.attack_label),
            "decay": (self.decay_slider, self.decay_label),
            "sustain": (self.sustain_slider, self.sustain_label),
            "release": (self.release_slider, self.release_label),
        }
        
        for key, (slider, label) in sliders_map.items():
            value = state.get(key, slider.property("min_val"))
            # Disable the slider if its corresponding input socket is connected
            is_connected = key in self.node_logic.inputs and self.node_logic.inputs[key].connections
            slider.setEnabled(not is_connected)
            
            # Block signals to prevent feedback loops while setting the value
            with QSignalBlocker(slider):
                slider.setValue(self._map_logical_to_slider(slider, value))
            
            label_text = f"{slider.property('name')}: {slider.property('format').format(value)}"
            if is_connected:
                label_text += " (ext)" # Indicate external control
            label.setText(label_text)
    
    @Slot()
    def updateFromLogic(self):
        """Requests a full state snapshot from the logic and updates the UI."""
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ==============================================================================
# 3. Node Logic Class (ADSRNode)
# ==============================================================================
class ADSRNode(Node):
    NODE_TYPE = "ADSR Envelope"
    UI_CLASS = ADSRNodeItem
    CATEGORY = "Modulation"
    DESCRIPTION = "Generates a control signal based on a gate input."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = ADSREmitter()

        # --- Define Sockets ---
        self.add_input("gate", data_type=bool)
        self.add_input("attack", data_type=float)
        self.add_input("decay", data_type=float)
        self.add_input("sustain", data_type=float)
        self.add_input("release", data_type=float)
        self.add_output("out", data_type=float)

        # --- Internal State ---
        self._lock = threading.Lock()
        
        # User-configurable parameters (defaults)
        self._attack_s: float = 0.01
        self._decay_s: float = 0.2
        self._sustain_level: float = 0.7
        self._release_s: float = 0.5
        
        # Envelope state machine
        self._state: str = 'idle'
        self._current_level: float = 0.0
        self._previous_gate: bool = False

    # --- Thread-safe setters for UI interaction ---
    @Slot(float)
    def set_attack(self, value: float):
        with self._lock: self._attack_s = float(value)
    @Slot(float)
    def set_decay(self, value: float):
        with self._lock: self._decay_s = float(value)
    @Slot(float)
    def set_sustain(self, value: float):
        with self._lock: self._sustain_level = float(value)
    @Slot(float)
    def set_release(self, value: float):
        with self._lock: self._release_s = float(value)

    def get_current_state_snapshot(self, locked: bool = False) -> Dict:
        """Returns a copy of the current parameters for UI synchronization."""
        if locked:
            return {
                "attack": self._attack_s, "decay": self._decay_s,
                "sustain": self._sustain_level, "release": self._release_s,
            }
        with self._lock:
            return {
                "attack": self._attack_s, "decay": self._decay_s,
                "sustain": self._sustain_level, "release": self._release_s,
            }

    def process(self, input_data: dict) -> dict:
        state_snapshot_to_emit = None
        with self._lock:
            ui_update_needed = False

            # --- Prioritize socket inputs over internal state ---
            attack_socket_val = input_data.get("attack")
            attack_s = float(attack_socket_val) if attack_socket_val is not None else self._attack_s

            decay_socket_val = input_data.get("decay")
            decay_s = float(decay_socket_val) if decay_socket_val is not None else self._decay_s

            sustain_socket_val = input_data.get("sustain")
            sustain_level = float(sustain_socket_val) if sustain_socket_val is not None else self._sustain_level

            release_socket_val = input_data.get("release")
            release_s = float(release_socket_val) if release_socket_val is not None else self._release_s

            # --- Check for changes from sockets to update UI ---
            if self._attack_s != attack_s: self._attack_s = attack_s; ui_update_needed = True
            if self._decay_s != decay_s: self._decay_s = decay_s; ui_update_needed = True
            if self._sustain_level != sustain_level: self._sustain_level = sustain_level; ui_update_needed = True
            if self._release_s != release_s: self._release_s = release_s; ui_update_needed = True
            
            if ui_update_needed:
                state_snapshot_to_emit = self.get_current_state_snapshot(locked=True)

            gate = bool(input_data.get("gate", False))
            
            # --- State Machine Triggering ---
            if gate and not self._previous_gate:  # Rising edge (Note-On)
                self._state = 'attack'
            elif not gate and self._previous_gate:  # Falling edge (Note-Off)
                self._state = 'release'
            
            self._previous_gate = gate

            # --- Block-based Envelope Calculation ---
            block_duration_s = DEFAULT_BLOCKSIZE / DEFAULT_SAMPLERATE
            
            if self._state == 'attack':
                time_to_peak = (1.0 - self._current_level) * max(MIN_TIME, attack_s)
                
                if block_duration_s < time_to_peak:
                    self._current_level += block_duration_s / max(MIN_TIME, attack_s)
                else:
                    time_in_decay = block_duration_s - time_to_peak
                    level_after_decay = 1.0 - (1.0 - sustain_level) * (time_in_decay / max(MIN_TIME, decay_s))
                    
                    if level_after_decay <= sustain_level:
                        self._current_level = sustain_level
                        self._state = 'sustain'
                    else:
                        self._current_level = level_after_decay
                        self._state = 'decay'

            elif self._state == 'decay':
                level_range_to_decay = self._current_level - sustain_level
                if level_range_to_decay > 0:
                    time_to_sustain = (level_range_to_decay / (1.0 - sustain_level + EPSILON)) * max(MIN_TIME, decay_s)
                    
                    if block_duration_s < time_to_sustain:
                        self._current_level -= (block_duration_s / max(MIN_TIME, decay_s)) * (1.0 - sustain_level)
                    else:
                        self._current_level = sustain_level
                        self._state = 'sustain'
                else:
                    self._current_level = sustain_level
                    self._state = 'sustain'

            elif self._state == 'sustain':
                self._current_level = sustain_level

            elif self._state == 'release':
                if release_s > MIN_TIME:
                    samples_in_release = DEFAULT_SAMPLERATE * release_s
                    decay_factor = np.exp(-DEFAULT_BLOCKSIZE / samples_in_release)
                    self._current_level *= decay_factor
                else:
                    self._current_level = 0.0
                
                if self._current_level < 1e-5:
                    self._current_level = 0.0
                    self._state = 'idle'

            elif self._state == 'idle':
                self._current_level = 0.0
                
            output_value = self._current_level

        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)
            
        return {"out": float(output_value)}


    def start(self):
        super().start()
        # Reset state when processing starts
        with self._lock:
            self._state = 'idle'
            self._current_level = 0.0
            self._previous_gate = False
    
    def serialize_extra(self) -> dict:
        """Save the node's user-configured parameters."""
        return self.get_current_state_snapshot()

    def deserialize_extra(self, data: dict):
        """Load the node's parameters from a file."""
        with self._lock:
            self._attack_s = data.get("attack", 0.01)
            self._decay_s = data.get("decay", 0.2)
            self._sustain_level = data.get("sustain", 0.7)
            self._release_s = data.get("release", 0.5)


# ==============================================================================
# UI Class for the Gate Button Node
# ==============================================================================
class GateButtonNodeItem(NodeItem):
    """A simple UI with a press-and-hold button."""

    def __init__(self, node_logic: "GateButtonNode"):
        super().__init__(node_logic)

        # Create a container and a button
        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING)
        
        self.button = QPushButton("Gate")
        self.button.setCheckable(False) # It's a momentary button
        
        layout.addWidget(self.button)
        self.setContentWidget(self.container_widget)

        # Connect the button's pressed and released signals to the logic
        self.button.pressed.connect(self.node_logic.set_gate_true)
        self.button.released.connect(self.node_logic.set_gate_false)
    
    # No updateFromLogic is needed as the UI only sends state to the logic

# ==============================================================================
# Logic Class for the Gate Button Node
# ==============================================================================
class GateButtonNode(Node):
    NODE_TYPE = "Gate Button"
    UI_CLASS = GateButtonNodeItem
    CATEGORY = "Modulation"
    DESCRIPTION = "Outputs a True signal while the button is pressed."

    def __init__(self, name: str, node_id: str = None):
        super().__init__(name, node_id)
        self.add_output("out", data_type=bool)

        self._lock = threading.Lock()
        self._gate_state = False

    @Slot()
    def set_gate_true(self):
        """Called by the UI when the button is pressed."""
        with self._lock:
            self._gate_state = True

    @Slot()
    def set_gate_false(self):
        """Called by the UI when the button is released."""
        with self._lock:
            self._gate_state = False

    def process(self, input_data: dict) -> dict:
        """Outputs the current state of the button."""
        with self._lock:
            return {"out": self._gate_state}

    def stop(self):
        """Ensure the gate is false when processing stops."""
        with self._lock:
            self._gate_state = False
        super().stop()