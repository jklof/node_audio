import torch
import numpy as np
import threading
import logging
from typing import Dict

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, TICK_DURATION_S

# --- UI and Qt Imports ---
from ui_elements import ParameterNodeItem
from PySide6.QtCore import Slot

# --- Helper Imports ---
from node_helpers import with_parameters, Parameter

# Configure logging
logger = logging.getLogger(__name__)

# A small value to add to denominators to prevent division by zero
EPSILON = 1e-9


# ==============================================================================
# 1. UI Class for the Envelope Follower Node (Unchanged)
# ==============================================================================
class EnvelopeFollowerNodeItem(ParameterNodeItem):
    """
    UI for the EnvelopeFollowerNode, providing sliders for attack and release times.
    This class uses the ParameterNodeItem base to automatically generate its UI.
    """

    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "EnvelopeFollowerNode"):
        # Define the parameters declaratively. The base class will create the controls.
        parameters = [
            {
                "key": "attack_ms",
                "name": "Attack",
                "min": 1.0,
                "max": 500.0,
                "format": "{:.1f} ms",
                "is_log": True,  # Logarithmic scale feels more natural for time
            },
            {
                "key": "release_ms",
                "name": "Release",
                "min": 1.0,
                "max": 5000.0,
                "format": "{:.1f} ms",
                "is_log": True,
            },
        ]

        # The superclass constructor handles all the heavy lifting of UI creation.
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 2. Logic Class for the Envelope Follower Node
# ==============================================================================
@with_parameters
class EnvelopeFollowerNode(Node):
    NODE_TYPE = "Envelope Follower"
    UI_CLASS = EnvelopeFollowerNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Outputs a control signal based on the amplitude of an audio signal."

    # --- Declarative managed parameters ---
    attack_ms = Parameter(default=10.0, clip=(1.0, 500.0))
    release_ms = Parameter(default=200.0, clip=(1.0, 5000.0))

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)

        self._init_parameters()

        self.add_input("in", data_type=torch.Tensor)
        # Sockets match parameter names for automatic modulation
        self.add_input("attack_ms", data_type=float)
        self.add_input("release_ms", data_type=float)
        self.add_output("out", data_type=float)

        # --- DSP State (not a parameter) ---
        self._envelope: float = 0.0  # The current value of the envelope

    def _get_state_snapshot_locked(self) -> dict:
        return self._get_parameters_state()

    def serialize_extra(self) -> dict:
        return self._serialize_parameters()

    def deserialize_extra(self, data: dict):
        self._deserialize_parameters(data)

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")

        # If no signal is present, the envelope should decay to zero.
        peak_value = 0.0
        if isinstance(signal, torch.Tensor) and signal.numel() > 0:
            # 1. Get the peak absolute amplitude of the current block.
            mono_signal = torch.mean(torch.abs(signal), dim=0)
            peak_value = torch.max(mono_signal).item()

        # REFACTORED: Update parameters from sockets with a single helper method call.
        self._update_parameters_from_sockets(input_data)

        # The lock is no longer needed here as _update_params_from_sockets is thread-safe
        # and we read the managed parameters into local variables.
        with self._lock:
            effective_attack_ms = self._attack_ms
            effective_release_ms = self._release_ms

        # 2. Calculate time-based smoothing coefficients (alpha).
        attack_time_s = max(0.001, effective_attack_ms / 1000.0)
        release_time_s = max(0.001, effective_release_ms / 1000.0)

        alpha_attack = 1.0 - np.exp(-TICK_DURATION_S / attack_time_s)
        alpha_release = 1.0 - np.exp(-TICK_DURATION_S / release_time_s)

        # 3. Choose the appropriate coefficient based on the direction of movement.
        alpha = alpha_attack if peak_value > self._envelope else alpha_release

        # 4. Apply the one-pole smoothing filter to the envelope.
        self._envelope += alpha * (peak_value - self._envelope)

        return {"out": float(self._envelope)}

    def start(self):
        """Reset DSP state when processing starts."""
        with self._lock:
            self._envelope = 0.0
        super().start()
