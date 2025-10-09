import torch
import numpy as np
import threading
import logging
from enum import Enum
from typing import Dict

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS

# --- UI and Qt Imports ---
from ui_elements import ParameterNodeItem
from PySide6.QtCore import Slot

# Configure logging
logger = logging.getLogger(__name__)


# ==============================================================================
# Enum for Waveform Types
# ==============================================================================
class Waveform(Enum):
    SINE = "Sine"
    SQUARE = "Square"
    SAWTOOTH = "Sawtooth"
    TRIANGLE = "Triangle"


# ==============================================================================
# UI Class for the Oscillator Node
# ==============================================================================
class OscillatorNodeItem(ParameterNodeItem):
    """
    Refactored UI for the OscillatorNode using the declarative ParameterNodeItem base class.
    """

    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "OscillatorNode"):
        # Define the UI controls declaratively
        parameters = [
            {
                "key": "waveform",
                "name": "Waveform",
                "type": "combobox",
                "items": [(wf.value, wf) for wf in Waveform],  # (Display Text, Enum Member)
            },
            {
                "key": "frequency",
                "name": "Frequency",
                "type": "dial",
                "min": 20.0,
                "max": 20000.0,
                "format": "{:.1f} Hz",
                "is_log": True,  # Use a logarithmic scale for frequency
            },
            {
                "key": "pulse_width",
                "name": "Pulse Width",
                "type": "dial",
                "min": 0.01,
                "max": 0.99,
                "format": "{:.2f}",
                "is_log": False,
            },
        ]

        # The superclass constructor creates all the widgets and connects the signals
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        """
        Overrides the base class method to add custom UI logic.
        """
        # First, call the parent method to handle all standard updates
        # (e.g., setting values, enabling/disabling based on connections).
        super()._on_state_updated_from_logic(state)

        # Add custom logic to show/hide the Pulse Width control
        waveform = state.get("waveform")
        pw_control_info = self._controls.get("pulse_width")

        if pw_control_info:
            # The _controls dictionary holds references to the generated widgets
            pw_widget = pw_control_info["widget"]
            pw_label = pw_control_info["label"]
            is_visible = waveform == Waveform.SQUARE

            pw_widget.setVisible(is_visible)
            pw_label.setVisible(is_visible)

            # This is crucial to make the node resize correctly when controls are hidden/shown.
            self.container_widget.adjustSize()
            self.update_geometry()


# ==============================================================================
# Oscillator Logic Node
# ==============================================================================
class OscillatorNode(Node):
    NODE_TYPE = "Oscillator"
    UI_CLASS = OscillatorNodeItem
    CATEGORY = "Generators"
    DESCRIPTION = "Generates classic synthesizer waveforms (Sine, Square, Saw, Triangle)."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("freq", data_type=float)
        self.add_input("pulse_width", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self.samplerate = DEFAULT_SAMPLERATE
        self.blocksize = DEFAULT_BLOCKSIZE
        self.channels = DEFAULT_CHANNELS

        # --- Internal State ---
        self._phase = 0.0
        self._waveform: Waveform = Waveform.SINE
        self._frequency: float = 440.0
        self._pulse_width: float = 0.5

        # --- Pre-computed constants for performance ---
        self._two_pi = torch.tensor(2 * np.pi, dtype=DEFAULT_DTYPE)
        self._inv_two_pi = torch.tensor(1.0 / (2 * np.pi), dtype=DEFAULT_DTYPE)
        self._sr_reciprocal = 1.0 / self.samplerate
        self._half = torch.tensor(0.5, dtype=DEFAULT_DTYPE)
        self._neg_half = torch.tensor(-0.5, dtype=DEFAULT_DTYPE)
        self._two = torch.tensor(2.0, dtype=DEFAULT_DTYPE)

        # --- Pre-allocate buffers to avoid allocation in process() ---
        self._phase_ramp = None
        self._phases_buffer = None
        self._norm_phases_buffer = None
        self._output_1d_buffer = None
        self._resize_buffers()  # Initialize buffers on creation

    def _resize_buffers(self):
        """Initializes or re-initializes all processing buffers."""
        logger.debug(f"[{self.name}] Resizing internal buffers for blocksize {self.blocksize}.")
        self._phase_ramp = torch.arange(self.blocksize, dtype=DEFAULT_DTYPE)
        self._phases_buffer = torch.empty(self.blocksize, dtype=DEFAULT_DTYPE)
        self._norm_phases_buffer = torch.empty(self.blocksize, dtype=DEFAULT_DTYPE)
        self._output_1d_buffer = torch.empty(self.blocksize, dtype=DEFAULT_DTYPE)

    def _get_state_snapshot_locked(self) -> Dict:
        """Returns a copy of the current parameters for UI synchronization. Assumes lock is held."""
        return {"waveform": self._waveform, "frequency": self._frequency, "pulse_width": self._pulse_width}

    @Slot(Waveform)
    def set_waveform(self, waveform: Waveform):
        state_to_emit = None
        with self._lock:
            if self._waveform != waveform:
                self._waveform = waveform
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_frequency(self, frequency: float):
        state_to_emit = None
        with self._lock:
            new_freq = np.clip(float(frequency), 20.0, 20000.0)
            if self._frequency != new_freq:
                self._frequency = new_freq
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_pulse_width(self, pulse_width: float):
        state_to_emit = None
        with self._lock:
            new_pw = np.clip(float(pulse_width), 0.01, 0.99)  # Avoid extremes
            if self._pulse_width != new_pw:
                self._pulse_width = new_pw
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def process(self, input_data: dict) -> dict:
        state_snapshot_to_emit = None
        with self._lock:
            # Check for external modulation via input sockets
            freq_socket = input_data.get("freq")
            if freq_socket is not None:
                new_freq = np.clip(float(freq_socket), 20.0, 20000.0)
                if abs(self._frequency - new_freq) > 1e-6:
                    self._frequency = new_freq
                    state_snapshot_to_emit = self._get_state_snapshot_locked()

            pw_socket = input_data.get("pulse_width")
            if pw_socket is not None:
                new_pw = np.clip(float(pw_socket), 0.01, 0.99)
                if abs(self._pulse_width - new_pw) > 1e-6:
                    self._pulse_width = new_pw
                    # If freq also changed, we don't need to get the snapshot again
                    if not state_snapshot_to_emit:
                        state_snapshot_to_emit = self._get_state_snapshot_locked()

            # Copy state to local variables for processing
            frequency = self._frequency
            pulse_width = self._pulse_width
            waveform = self._waveform

        # Emit signal after releasing the lock to avoid deadlocks
        if state_snapshot_to_emit:
            self.ui_update_callback(state_snapshot_to_emit)

        # --- Generate Waveform using pre-allocated buffers and constants ---
        phase_increment = frequency * self._two_pi * self._sr_reciprocal

        # Calculate phases for the entire block in-place
        torch.mul(self._phase_ramp, phase_increment, out=self._phases_buffer)
        self._phases_buffer.add_(self._phase)

        # Normalize phases to [0, 2*pi) for periodic functions
        torch.fmod(self._phases_buffer, self._two_pi, out=self._norm_phases_buffer)

        # --- Waveform generation into pre-allocated 1D buffer ---
        if waveform == Waveform.SINE:
            torch.sin(self._norm_phases_buffer, out=self._output_1d_buffer)
            self._output_1d_buffer.mul_(self._half)
        elif waveform == Waveform.SQUARE:
            pw_threshold = pulse_width * self._two_pi
            torch.where(self._norm_phases_buffer < pw_threshold, self._half, self._neg_half, out=self._output_1d_buffer)
        elif waveform == Waveform.SAWTOOTH:
            torch.mul(self._norm_phases_buffer, self._inv_two_pi, out=self._output_1d_buffer)
            self._output_1d_buffer.sub_(self._half)
        elif waveform == Waveform.TRIANGLE:
            torch.mul(self._norm_phases_buffer, self._inv_two_pi, out=self._output_1d_buffer)
            self._output_1d_buffer.sub_(self._half)
            torch.abs(self._output_1d_buffer, out=self._output_1d_buffer)
            self._output_1d_buffer.mul_(self._two)
            self._output_1d_buffer.sub_(self._half)

        # Update phase for the next block
        self._phase = torch.fmod(self._phases_buffer[-1] + phase_increment, self._two_pi).item()

        # Create the final output tensor by creating a view and then cloning it for safety.
        # This is more efficient than writing to an intermediate multi-channel buffer first.
        output_signal = self._output_1d_buffer.unsqueeze(0).expand(self.channels, -1).clone()

        return {"out": output_signal}

    def serialize_extra(self) -> dict:
        with self._lock:
            state = self._get_state_snapshot_locked()
            # Convert the enum to its string name for JSON serialization
            state["waveform"] = state["waveform"].name
            return state

    def deserialize_extra(self, data: dict):
        with self._lock:
            waveform_name = data.get("waveform", Waveform.SINE.name)
            try:
                self._waveform = Waveform[waveform_name]
            except KeyError:
                self._waveform = Waveform.SINE
            self._frequency = float(data.get("frequency", 440.0))
            self._pulse_width = float(data.get("pulse_width", 0.5))
