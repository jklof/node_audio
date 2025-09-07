import numpy as np
import threading
import logging
from typing import Dict, Optional

# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE

# --- Qt Imports ---
from PySide6.QtCore import Qt, Signal, Slot, QObject, QSignalBlocker
from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout, QPushButton, QSizePolicy

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
class ADSRNodeItem(NodeItem):
    """Provides a user interface with sliders to control the ADSR parameters."""

    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "ADSRNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
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

    @Slot()
    def updateFromLogic(self):
        """
        Pulls the current state from the logic node to initialize the UI.
        """
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()

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
        if range_val == 0:
            return 0
        normalized = (logical_value - min_val) / range_val
        return int(np.clip(normalized, 0.0, 1.0) * 1000.0)

    @Slot(int)
    def _on_attack_changed(self):
        val = self._map_slider_to_logical(self.attack_slider)
        self.node_logic.set_attack(val)

    @Slot(int)
    def _on_decay_changed(self):
        val = self._map_slider_to_logical(self.decay_slider)
        self.node_logic.set_decay(val)

    @Slot(int)
    def _on_sustain_changed(self):
        val = self._map_slider_to_logical(self.sustain_slider)
        self.node_logic.set_sustain(val)

    @Slot(int)
    def _on_release_changed(self):
        val = self._map_slider_to_logical(self.release_slider)
        self.node_logic.set_release(val)

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
                label_text += " (ext)"  # Indicate external control
            label.setText(label_text)

        # Initial state emission will be triggered by graph_scene.py


# ==============================================================================
# 3. ADSR Node Logic Class
# ==============================================================================
class ADSRNode(Node):
    NODE_TYPE = "ADSR Envelope"
    UI_CLASS = ADSRNodeItem
    CATEGORY = "Modulation"
    DESCRIPTION = "Generates a control signal based on a gate input."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()

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
        self._state: str = "idle"
        self._current_level: float = 0.0
        self._previous_gate: bool = False

    # --- Thread-safe setters for UI interaction ---
    @Slot(float)
    def set_attack(self, value: float):
        state_to_emit = None
        with self._lock:
            self._attack_s = float(value)
            state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_decay(self, value: float):
        state_to_emit = None
        with self._lock:
            self._decay_s = float(value)
            state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_sustain(self, value: float):
        state_to_emit = None
        with self._lock:
            self._sustain_level = float(value)
            state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_release(self, value: float):
        state_to_emit = None
        with self._lock:
            self._release_s = float(value)
            state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {
            "attack": self._attack_s,
            "decay": self._decay_s,
            "sustain": self._sustain_level,
            "release": self._release_s,
        }

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

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
            if self._attack_s != attack_s:
                self._attack_s = attack_s
                ui_update_needed = True
            if self._decay_s != decay_s:
                self._decay_s = decay_s
                ui_update_needed = True
            if self._sustain_level != sustain_level:
                self._sustain_level = sustain_level
                ui_update_needed = True
            if self._release_s != release_s:
                self._release_s = release_s
                ui_update_needed = True

            if ui_update_needed:
                state_snapshot_to_emit = self._get_current_state_snapshot_locked()

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

        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

        return {"out": float(output_value)}

    def start(self):
        super().start()
        # Reset state when processing starts
        with self._lock:
            self._state = "idle"
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

        # Connect the logic node's state updates back to the UI (for standardization)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        # The button's visual state doesn't need to be updated as user controls it
        # This is primarily for consistency with the unidirectional data flow pattern
        pass

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
        self.emitter = NodeStateEmitter()
        self.add_output("out", data_type=bool)

        self._lock = threading.Lock()
        self._gate_state = False

        # Initial state emission is handled by graph_scene

    @Slot()
    def set_gate_true(self):
        """Called by the UI when the button is pressed."""
        with self._lock:
            self._gate_state = True
            state = {"gate_state": self._gate_state}
        self.emitter.stateUpdated.emit(state)

    @Slot()
    def set_gate_false(self):
        """Called by the UI when the button is released."""
        with self._lock:
            self._gate_state = False
            state = {"gate_state": self._gate_state}
        self.emitter.stateUpdated.emit(state)

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
class LFONodeItem(NodeItem):
    """UI for LFO node: just one slider for frequency control."""

    def __init__(self, node_logic: "LFONode"):
        super().__init__(node_logic)

        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(4)

        self.freq_label = QLabel("Frequency: ...")
        self.freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.freq_label)

        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.slider_min_int, self.slider_max_int = 1, 2000  # integer steps
        self.logical_min_freq, self.logical_max_freq = 0.01, 20.0  # Hz range
        self.freq_slider.setRange(self.slider_min_int, self.slider_max_int)
        self.freq_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.freq_slider)

        self.container_widget.setLayout(layout)
        self.setContentWidget(self.container_widget)

        # Connect slider and signal
        self.freq_slider.valueChanged.connect(self._on_slider_change)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

        # Initial state emission will be triggered by graph_scene.py

    @Slot()
    def updateFromLogic(self):
        """
        Pulls the current state from the logic node to initialize the UI.
        """
        state = {"frequency": self.node_logic.get_frequency_hz()}
        self._on_state_updated(state)
        super().updateFromLogic()

    def _map_slider_to_logical(self, slider_value: int) -> float:
        # Map integer slider to float freq
        norm = (slider_value - self.slider_min_int) / (self.slider_max_int - self.slider_min_int)
        return self.logical_min_freq + norm * (self.logical_max_freq - self.logical_min_freq)

    def _map_logical_to_slider(self, logical_value: float) -> int:
        # Clamp logical value to ensure it's within the expected range before mapping
        clamped_logical = max(self.logical_min_freq, min(logical_value, self.logical_max_freq))
        norm = (clamped_logical - self.logical_min_freq) / (self.logical_max_freq - self.logical_min_freq)
        return int(round(self.slider_min_int + norm * (self.slider_max_int - self.slider_min_int)))

    @Slot(int)
    def _on_slider_change(self, slider_value: int):
        freq = self._map_slider_to_logical(slider_value)
        self.node_logic.set_frequency_hz(freq)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        """Updates the UI from a state dictionary."""
        freq = state.get("frequency", 1.0)
        self.freq_label.setText(f"Frequency: {freq:.2f} Hz")
        with QSignalBlocker(self.freq_slider):
            self.freq_slider.setValue(self._map_logical_to_slider(freq))


# ==============================================================================
# 7. LFO Node Logic Class
# ==============================================================================
class LFONode(Node):
    NODE_TYPE = "LFO"
    CATEGORY = "Modulation"
    DESCRIPTION = "Low-frequency oscillator for modulation (sine, square, saw)."
    UI_CLASS = LFONodeItem
    IS_CLOCK_SOURCE = False  # Driven by graph ticks

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()

        self.add_input("sync_control", data_type=bool)
        self.add_output("sine_out", data_type=float)
        self.add_output("square_out", data_type=float)
        self.add_output("saw_out", data_type=float)

        self.samplerate = DEFAULT_SAMPLERATE
        self.blocksize = DEFAULT_BLOCKSIZE
        self.lock = threading.Lock()

        self._frequency_hz = 1.0
        self._phase = 0.0  # [0, 1) range

        logger.debug(f"LFO [{self.name}] initialized at {self._frequency_hz} Hz")

    # -----------------
    # UI thread methods (thread-safe)
    # -----------------
    @Slot(float)
    def set_frequency_hz(self, freq: float):
        state_to_emit = None
        with self.lock:
            new_freq = max(0.001, float(freq))  # Avoid 0 Hz
            if self._frequency_hz != new_freq:
                self._frequency_hz = new_freq
                state_to_emit = {"frequency": self._frequency_hz}
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def get_frequency_hz(self) -> float:
        with self.lock:
            return self._frequency_hz

    # -----------------
    # Worker thread method
    # -----------------
    def process(self, input_data: dict) -> dict:
        sync_trigger = input_data.get("sync_control")
        with self.lock:
            freq = self._frequency_hz

        # phase increment calculation.
        # The phase must advance by the number of samples in one processing block (tick).
        phase_increment = (freq / self.samplerate) * self.blocksize

        # --- sync trigger ---
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

    def serialize_extra(self):
        with self.lock:
            return {"frequency_hz": self._frequency_hz, "phase": self._phase}

    def deserialize_extra(self, data):
        with self.lock:
            self._frequency_hz = float(data.get("frequency_hz", 1.0))
            self._phase = float(data.get("phase", 0.0))
