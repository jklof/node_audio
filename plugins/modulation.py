import numpy as np
import threading
import logging
from typing import Dict, Optional, Deque

# --- Node System Imports ---
from node_system import Node
from ui_elements import ParameterNodeItem, NodeItem, NODE_CONTENT_PADDING
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE

# --- Qt Imports ---
from PySide6.QtCore import Qt, Signal, Slot, QObject, QSignalBlocker
from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout, QPushButton, QSizePolicy

# --- Helper Imports ---
from node_helpers import with_parameters, Parameter

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
# A small value to prevent division by zero for instant transitions
MIN_TIME = 0.001
# A small value to add to denominators to prevent division by zero
EPSILON = 1e-9


# ==============================================================================
# 2. ADSR Node UI Class
# ==============================================================================
class ADSRNodeItem(ParameterNodeItem):
    """Provides a user interface with sliders to control the ADSR parameters."""

    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "ADSRNode"):
        # Define the parameters for this node
        parameters = [
            {
                "key": "attack",
                "name": "Attack",
                "min": 0.0,
                "max": 5.0,
                "format": "{:.2f} s",
                "is_log": False,
            },
            {
                "key": "decay",
                "name": "Decay",
                "min": 0.0,
                "max": 5.0,
                "format": "{:.2f} s",
                "is_log": False,
            },
            {
                "key": "sustain",
                "name": "Sustain",
                "min": 0.0,
                "max": 1.0,
                "format": "{:.1%}",
                "is_log": False,
            },
            {
                "key": "release",
                "name": "Release",
                "min": 0.0,
                "max": 5.0,
                "format": "{:.2f} s",
                "is_log": False,
            },
        ]

        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 3. ADSR Node Logic Class
# ==============================================================================
@with_parameters
class ADSRNode(Node):
    NODE_TYPE = "ADSR Envelope"
    UI_CLASS = ADSRNodeItem
    CATEGORY = "Modulation"
    DESCRIPTION = "Generates a control signal based on a gate input."

    # --- Declarative managed parameters ---
    attack = Parameter(default=0.01, clip=(0.0, 5.0))
    decay = Parameter(default=0.2, clip=(0.0, 5.0))
    sustain = Parameter(default=0.7, clip=(0.0, 1.0))
    release = Parameter(default=0.5, clip=(0.0, 5.0))

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self._init_parameters()

        # --- Define Sockets ---
        self.add_input("gate", data_type=bool)
        self.add_input("attack", data_type=float)
        self.add_input("decay", data_type=float)
        self.add_input("sustain", data_type=float)
        self.add_input("release", data_type=float)
        self.add_output("out", data_type=float)

        # --- Internal State ---
        self._state: str = "idle"
        self._current_level: float = 0.0
        self._previous_gate: bool = False

    def _get_state_snapshot_locked(self) -> dict:
        return self._get_parameters_state()

    def serialize_extra(self) -> dict:
        return self._serialize_parameters()

    def deserialize_extra(self, data: dict):
        self._deserialize_parameters(data)

    def process(self, input_data: dict) -> dict:
        # --- Update parameters from sockets. This also handles UI updates. ---
        self._update_parameters_from_sockets(input_data)

        with self._lock:
            # --- Get a consistent snapshot of the managed parameters ---
            attack_s = self._attack
            decay_s = self._decay
            sustain_level = self._sustain
            release_s = self._release

            gate = bool(input_data.get("gate", False))

            # --- State Machine Triggering ---
            if gate and not self._previous_gate:  # Rising edge (Note-On)
                self._state = "attack"
            elif not gate and self._previous_gate:  # Falling edge (Note-Off)
                self._state = "release"

            self._previous_gate = gate

            # --- Block-based Envelope Calculation ---
            block_duration_s = DEFAULT_BLOCKSIZE / DEFAULT_SAMPLERATE

            if self._state == "attack":
                time_to_peak = (1.0 - self._current_level) * max(MIN_TIME, attack_s)

                if block_duration_s < time_to_peak:
                    self._current_level += block_duration_s / max(MIN_TIME, attack_s)
                else:
                    time_in_decay = block_duration_s - time_to_peak
                    level_after_decay = 1.0 - (1.0 - sustain_level) * (time_in_decay / max(MIN_TIME, decay_s))

                    if level_after_decay <= sustain_level:
                        self._current_level = sustain_level
                        self._state = "sustain"
                    else:
                        self._current_level = level_after_decay
                        self._state = "decay"

            elif self._state == "decay":
                level_range_to_decay = self._current_level - sustain_level
                if level_range_to_decay > 0:
                    time_to_sustain = (level_range_to_decay / (1.0 - sustain_level + EPSILON)) * max(MIN_TIME, decay_s)

                    if block_duration_s < time_to_sustain:
                        self._current_level -= (block_duration_s / max(MIN_TIME, decay_s)) * (1.0 - sustain_level)
                    else:
                        self._current_level = sustain_level
                        self._state = "sustain"
                else:
                    self._current_level = sustain_level
                    self._state = "sustain"

            elif self._state == "sustain":
                self._current_level = sustain_level

            elif self._state == "release":
                if release_s > MIN_TIME:
                    samples_in_release = DEFAULT_SAMPLERATE * release_s
                    decay_factor = np.exp(-DEFAULT_BLOCKSIZE / samples_in_release)
                    self._current_level *= decay_factor
                else:
                    self._current_level = 0.0

                if self._current_level < 1e-5:
                    self._current_level = 0.0
                    self._state = "idle"

            elif self._state == "idle":
                self._current_level = 0.0

            output_value = self._current_level

        return {"out": float(output_value)}

    def start(self):
        super().start()
        # Reset state when processing starts
        with self._lock:
            self._state = "idle"
            self._current_level = 0.0
            self._previous_gate = False


# ==============================================================================
# 4. Gate Button Node UI Class
# ==============================================================================
class GateButtonNodeItem(NodeItem):
    """A simple UI with a press-and-hold button."""

    def __init__(self, node_logic: "GateButtonNode"):
        super().__init__(node_logic)

        # Create a container and a button
        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )

        self.button = QPushButton("Gate")
        self.button.setCheckable(False)  # It's a momentary button

        layout.addWidget(self.button)
        self.setContentWidget(self.container_widget)

        # Connect the button's pressed and released signals to the logic
        self.button.pressed.connect(self.node_logic.set_gate_true)
        self.button.released.connect(self.node_logic.set_gate_false)

    # No updateFromLogic is needed as the UI only sends state to the logic


# ==============================================================================
# 5. Gate Button Node Logic Class
# ==============================================================================
class GateButtonNode(Node):
    NODE_TYPE = "Gate Button"
    UI_CLASS = GateButtonNodeItem
    CATEGORY = "Modulation"
    DESCRIPTION = "Outputs a True signal while the button is pressed."

    def __init__(self, name: str, node_id: str = None):
        super().__init__(name, node_id)
        self.add_output("out", data_type=bool)

        self._gate_state = False

        # Initial state emission is handled by graph_scene

    def _get_state_snapshot_locked(self) -> Dict:
        return {"gate_state": self._gate_state}

    @Slot()
    def set_gate_true(self):
        """Called by the UI when the button is pressed."""
        with self._lock:
            self._gate_state = True
            state = self._get_state_snapshot_locked()
        self.ui_update_callback(state)

    @Slot()
    def set_gate_false(self):
        """Called by the UI when the button is released."""
        with self._lock:
            self._gate_state = False
            state = self._get_state_snapshot_locked()
        self.ui_update_callback(state)

    def process(self, input_data: dict) -> dict:
        """Outputs the current state of the button."""
        with self._lock:
            return {"out": self._gate_state}

    def stop(self):
        """Ensure the gate is false when processing stops."""
        with self._lock:
            self._gate_state = False
        super().stop()


# ==============================================================================
# 6. LFO Node UI Class
# ==============================================================================
class LFONodeItem(ParameterNodeItem):
    """UI for LFO node using ParameterNodeItem base class."""

    def __init__(self, node_logic: "LFONode"):
        # Define the parameters for this node
        parameters = [
            {
                "key": "frequency",
                "name": "Frequency",
                "min": 0.01,
                "max": 20.0,
                "format": "{:.2f} Hz",
                "is_log": False,
            },
        ]

        super().__init__(node_logic, parameters, width=180)


# ==============================================================================
# 7. LFO Node Logic Class
# ==============================================================================
@with_parameters
class LFONode(Node):
    NODE_TYPE = "LFO"
    CATEGORY = "Modulation"
    DESCRIPTION = "Low-frequency oscillator for modulation (sine, square, saw)."
    UI_CLASS = LFONodeItem
    IS_CLOCK_SOURCE = False  # Driven by graph ticks

    # --- Declarative managed parameters ---
    frequency = Parameter(default=1.0, clip=(0.01, 20.0))

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self._init_parameters()

        self.add_input("sync_control", data_type=bool)
        self.add_input("frequency", data_type=float)
        self.add_output("sine_out", data_type=float)
        self.add_output("square_out", data_type=float)
        self.add_output("saw_out", data_type=float)

        self.samplerate = DEFAULT_SAMPLERATE
        self.blocksize = DEFAULT_BLOCKSIZE

        self._phase = 0.0  # [0, 1) range

        logger.debug(f"LFO [{self.name}] initialized.")

    def _get_state_snapshot_locked(self) -> dict:
        return self._get_parameters_state()

    def serialize_extra(self) -> dict:
        return self._serialize_parameters()

    def deserialize_extra(self, data: dict):
        self._deserialize_parameters(data)

    def process(self, input_data: dict) -> dict:
        # --- Update parameters from sockets. This also handles UI updates. ---
        self._update_parameters_from_sockets(input_data)

        with self._lock:
            # Get a consistent snapshot of the managed parameter
            freq = self._frequency

        # phase increment calculation.
        # The phase must advance by the number of samples in one processing block (tick).
        phase_increment = (freq / self.samplerate) * self.blocksize

        # --- sync trigger ---
        sync_trigger = input_data.get("sync_control")
        if sync_trigger is not None and sync_trigger:
            self._phase = 0.0
        else:
            self._phase = (self._phase + phase_increment) % 1.0

        phase = self._phase

        # Calculate waveforms
        sine_val = float(np.sin(2 * np.pi * phase))
        square_val = 1.0 if phase < 0.5 else -1.0
        saw_val = (2.0 * phase) - 1.0  # Ramps from -1.0 to 1.0

        return {"sine_out": sine_val, "square_out": square_val, "saw_out": saw_val}
