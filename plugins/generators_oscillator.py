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
# UI Class for the Oscillator Node (REFACTORED)
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
        super()._on_state_updated(state)

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
# Oscillator Logic Node (Unchanged)
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

        self._lock = threading.Lock()
        self.samplerate = DEFAULT_SAMPLERATE
        self.blocksize = DEFAULT_BLOCKSIZE
        self.channels = DEFAULT_CHANNELS

        # --- Internal State ---
        self._phase = 0.0
        self._waveform: Waveform = Waveform.SINE
        self._frequency: float = 440.0
        self._pulse_width: float = 0.5

    def _get_current_state_snapshot_locked(self) -> Dict:
        """Returns a copy of the current parameters for UI synchronization. Assumes lock is held."""
        return {"waveform": self._waveform, "frequency": self._frequency, "pulse_width": self._pulse_width}

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    @Slot(Waveform)
    def set_waveform(self, waveform: Waveform):
        state_to_emit = None
        with self._lock:
            if self._waveform != waveform:
                self._waveform = waveform
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_frequency(self, frequency: float):
        state_to_emit = None
        with self._lock:
            new_freq = np.clip(float(frequency), 20.0, 20000.0)
            if self._frequency != new_freq:
                self._frequency = new_freq
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_pulse_width(self, pulse_width: float):
        state_to_emit = None
        with self._lock:
            new_pw = np.clip(float(pulse_width), 0.01, 0.99)  # Avoid extremes
            if self._pulse_width != new_pw:
                self._pulse_width = new_pw
                state_to_emit = self._get_current_state_snapshot_locked()
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
                    state_snapshot_to_emit = self._get_current_state_snapshot_locked()

            pw_socket = input_data.get("pulse_width")
            if pw_socket is not None:
                new_pw = np.clip(float(pw_socket), 0.01, 0.99)
                if abs(self._pulse_width - new_pw) > 1e-6:
                    self._pulse_width = new_pw
                    # If freq also changed, we don't need to get the snapshot again
                    if not state_snapshot_to_emit:
                        state_snapshot_to_emit = self._get_current_state_snapshot_locked()

            # Copy state to local variables for processing
            frequency = self._frequency
            pulse_width = self._pulse_width
            waveform = self._waveform

        # Emit signal after releasing the lock to avoid deadlocks
        if state_snapshot_to_emit:
            self.ui_update_callback(state_snapshot_to_emit)

        # --- Generate Waveform using PyTorch ---
        phase_increment = (2 * torch.pi * frequency) / self.samplerate
        phases = self._phase + torch.arange(self.blocksize, dtype=DEFAULT_DTYPE) * phase_increment

        output_1d = None
        # Normalize phase to [0, 2*pi) for periodic functions
        norm_phases = torch.fmod(phases, 2 * torch.pi)

        if waveform == Waveform.SINE:
            output_1d = 0.5 * torch.sin(norm_phases)
        elif waveform == Waveform.SQUARE:
            # Create square wave from -0.5 to 0.5
            output_1d = torch.where(norm_phases < (2 * torch.pi * pulse_width), 0.5, -0.5)
        elif waveform == Waveform.SAWTOOTH:
            # Create sawtooth from -0.5 to 0.5
            output_1d = (norm_phases / (2 * torch.pi)) - 0.5
        elif waveform == Waveform.TRIANGLE:
            # Create triangle from -0.5 to 0.5
            output_1d = 2 * torch.abs((norm_phases / (2 * torch.pi)) - 0.5) - 0.5

        # Update phase for the next block
        self._phase = torch.fmod(phases[-1] + phase_increment, 2 * torch.pi).item()

        # Tile to match channel count, creating (channels, samples) shape
        output_2d = output_1d.unsqueeze(0).expand(self.channels, -1)

        return {"out": output_2d}

    def serialize_extra(self) -> dict:
        with self._lock:
            return {
                "waveform": self._waveform.name,
                "frequency": self._frequency,
                "pulse_width": self._pulse_width,
            }

    def deserialize_extra(self, data: dict):
        with self._lock:
            waveform_name = data.get("waveform", Waveform.SINE.name)
            try:
                self._waveform = Waveform[waveform_name]
            except KeyError:
                self._waveform = Waveform.SINE

            self._frequency = float(data.get("frequency", 440.0))
            self._pulse_width = float(data.get("pulse_width", 0.5))
