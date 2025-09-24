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

# Configure logging
logger = logging.getLogger(__name__)

# A small value to add to denominators to prevent division by zero
EPSILON = 1e-9


# ==============================================================================
# 1. UI Class for the Envelope Follower Node
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
class EnvelopeFollowerNode(Node):
    NODE_TYPE = "Envelope Follower"
    UI_CLASS = EnvelopeFollowerNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Outputs a control signal based on the amplitude of an audio signal."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("attack_ms", data_type=float)
        self.add_input("release_ms", data_type=float)
        self.add_output("out", data_type=float)

        # --- Internal State Parameters ---
        self._attack_ms: float = 10.0
        self._release_ms: float = 200.0

        # --- DSP State ---
        self._envelope: float = 0.0  # The current value of the envelope

    def _get_state_snapshot_locked(self) -> Dict:
        """Returns a copy of the current parameters for UI or serialization."""
        return {"attack_ms": self._attack_ms, "release_ms": self._release_ms}

    # --- Thread-safe setters for UI interaction ---
    @Slot(float)
    def set_attack_ms(self, value: float):
        state_to_emit = None
        with self._lock:
            # Ensure value is a positive float
            clipped_value = max(1.0, float(value))
            if self._attack_ms != clipped_value:
                self._attack_ms = clipped_value
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_release_ms(self, value: float):
        state_to_emit = None
        with self._lock:
            # Ensure value is a positive float
            clipped_value = max(1.0, float(value))
            if self._release_ms != clipped_value:
                self._release_ms = clipped_value
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")

        # If no signal is present, the envelope should decay to zero.
        # We can simulate this by setting the target peak to 0.
        peak_value = 0.0
        if isinstance(signal, torch.Tensor) and signal.numel() > 0:
            # 1. Get the peak absolute amplitude of the current block.
            # This is our target for the envelope to move towards.
            mono_signal = torch.mean(torch.abs(signal), dim=0)
            peak_value = torch.max(mono_signal).item()

        state_to_emit = None
        with self._lock:
            ui_update_needed = False
            # Update parameters from sockets, overriding internal state if connected.
            attack_socket = input_data.get("attack_ms")
            effective_attack_ms = float(attack_socket) if attack_socket is not None else self._attack_ms
            if self._attack_ms != effective_attack_ms:
                self._attack_ms = effective_attack_ms
                ui_update_needed = True

            release_socket = input_data.get("release_ms")
            effective_release_ms = float(release_socket) if release_socket is not None else self._release_ms
            if self._release_ms != effective_release_ms:
                self._release_ms = effective_release_ms
                ui_update_needed = True

            if ui_update_needed:
                state_to_emit = self._get_state_snapshot_locked()

            # 2. Calculate time-based smoothing coefficients (alpha).
            # This converts attack/release times in ms to a per-block smoothing factor.
            attack_time_s = max(0.001, effective_attack_ms / 1000.0)
            release_time_s = max(0.001, effective_release_ms / 1000.0)

            alpha_attack = 1.0 - np.exp(-TICK_DURATION_S / attack_time_s)
            alpha_release = 1.0 - np.exp(-TICK_DURATION_S / release_time_s)

            # 3. Choose the appropriate coefficient based on the direction of movement.
            alpha = alpha_attack if peak_value > self._envelope else alpha_release

            # 4. Apply the one-pole smoothing filter to the envelope.
            self._envelope += alpha * (peak_value - self._envelope)

            output_value = self._envelope

        # Emit signal to UI AFTER the lock is released
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

        return {"out": float(output_value)}

    def start(self):
        """Reset DSP state when processing starts."""
        with self._lock:
            self._envelope = 0.0
        super().start()

    def serialize_extra(self) -> dict:
        with self._lock:
            return self._get_state_snapshot_locked()

    def deserialize_extra(self, data: dict):
        # Use the public setters to ensure UI is updated correctly on load.
        self.set_attack_ms(data.get("attack_ms", 10.0))
        self.set_release_ms(data.get("release_ms", 200.0))
