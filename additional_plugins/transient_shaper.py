import torch
import numpy as np
import threading
import logging
from typing import Dict, Optional

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE, TICK_DURATION_S

# --- UI and Qt Imports ---
from ui_elements import ParameterNodeItem
from PySide6.QtCore import Slot

# Configure logging
logger = logging.getLogger(__name__)

# A small value to prevent division by zero in calculations
EPSILON = 1e-9


# ==============================================================================
# 1. UI Class for the Transient Shaper Node
# ==============================================================================
class TransientShaperNodeItem(ParameterNodeItem):
    """
    UI for the TransientShaperNode, providing dials for Attack and Sustain.
    This class uses the ParameterNodeItem base to automatically generate its UI.
    """

    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "TransientShaperNode"):
        # Define the parameters declaratively. The base class will create the controls.
        parameters = [
            {
                "key": "attack",
                "name": "Attack",
                "type": "dial",
                "min": -1.0,  # -100% (cut)
                "max": 2.0,  # +200% (boost)
                "format": "{:+.0%}",
                "is_log": False,
            },
            {
                "key": "sustain",
                "name": "Sustain",
                "type": "dial",
                "min": -1.0,  # -100% (cut)
                "max": 1.0,  # +100% (boost)
                "format": "{:+.0%}",
                "is_log": False,
            },
        ]

        # The superclass constructor handles all the heavy lifting of UI creation.
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 2. Logic Class for the Transient Shaper Node
# ==============================================================================
class TransientShaperNode(Node):
    NODE_TYPE = "Transient Shaper"
    UI_CLASS = TransientShaperNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Emphasizes or de-emphasizes the attack and sustain portions of a signal."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("attack", data_type=float)
        self.add_input("sustain", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        # --- Internal State Parameters ---
        self._attack_gain: float = 0.0  # Range -1.0 to 2.0
        self._sustain_gain: float = 0.0  # Range -1.0 to 1.0

        # --- DSP State ---
        # Fast envelope to detect transients
        self._fast_envelope: float = 0.0
        # Slow envelope to detect the body/sustain of the sound
        self._slow_envelope: float = 0.0

    def _get_state_snapshot_locked(self) -> Dict:
        """Returns a copy of the current parameters for UI or serialization."""
        return {"attack": self._attack_gain, "sustain": self._sustain_gain}

    # --- Thread-safe setters for UI interaction ---
    @Slot(float)
    def set_attack(self, value: float):
        state_to_emit = None
        with self._lock:
            # Clip value to the allowed range
            clipped_value = np.clip(float(value), -1.0, 2.0).item()
            if self._attack_gain != clipped_value:
                self._attack_gain = clipped_value
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_sustain(self, value: float):
        state_to_emit = None
        with self._lock:
            # Clip value to the allowed range
            clipped_value = np.clip(float(value), -1.0, 1.0).item()
            if self._sustain_gain != clipped_value:
                self._sustain_gain = clipped_value
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}

        state_to_emit = None
        with self._lock:
            ui_update_needed = False
            # Update parameters from sockets, overriding internal state if connected.
            attack_socket = input_data.get("attack")
            if attack_socket is not None:
                clipped_val = np.clip(float(attack_socket), -1.0, 2.0).item()
                if self._attack_gain != clipped_val:
                    self._attack_gain = clipped_val
                    ui_update_needed = True

            sustain_socket = input_data.get("sustain")
            if sustain_socket is not None:
                clipped_val = np.clip(float(sustain_socket), -1.0, 1.0).item()
                if self._sustain_gain != clipped_val:
                    self._sustain_gain = clipped_val
                    ui_update_needed = True

            if ui_update_needed:
                state_to_emit = self._get_state_snapshot_locked()

            # Copy parameters to local variables for processing
            attack_gain = self._attack_gain
            sustain_gain = self._sustain_gain

        # Emit signal to UI AFTER the lock is released
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

        # --- Core DSP Processing (All in PyTorch) ---

        # 1. Prepare mono signal and get its absolute value (rectified)
        mono_signal = torch.mean(signal, dim=0) if signal.shape[0] > 1 else signal.squeeze(0)
        rectified_signal = torch.abs(mono_signal)

        # 2. Calculate smoothing coefficients for the two envelope followers
        # These times are fixed for simplicity, but could be exposed as parameters.
        fast_attack_s, fast_release_s = 0.001, 0.020
        slow_attack_s, slow_release_s = 0.050, 0.200

        alpha_fast_attack = 1.0 - torch.exp(torch.tensor(-TICK_DURATION_S / fast_attack_s))
        alpha_fast_release = 1.0 - torch.exp(torch.tensor(-TICK_DURATION_S / fast_release_s))
        alpha_slow_attack = 1.0 - torch.exp(torch.tensor(-TICK_DURATION_S / slow_attack_s))
        alpha_slow_release = 1.0 - torch.exp(torch.tensor(-TICK_DURATION_S / slow_release_s))

        # 3. Process envelopes at a block level (more efficient than sample-by-sample)
        peak_fast = torch.max(rectified_signal).item()
        peak_slow = peak_fast  # Use the same peak for both

        alpha_fast = alpha_fast_attack if peak_fast > self._fast_envelope else alpha_fast_release
        self._fast_envelope += alpha_fast * (peak_fast - self._fast_envelope)

        alpha_slow = alpha_slow_attack if peak_slow > self._slow_envelope else alpha_slow_release
        self._slow_envelope += alpha_slow * (peak_slow - self._slow_envelope)

        # 4. Calculate the transient signal
        # This is the core of the algorithm: the difference between the fast and slow envelopes.
        transient_signal = self._fast_envelope - self._slow_envelope

        # 5. Calculate the final gain envelope
        # We apply the user's gain settings to the appropriate components.
        # Note: We add 1.0 to the gains because the user input is a modulation amount.
        gain_envelope = ((transient_signal * (1.0 + attack_gain)) + (self._slow_envelope * (1.0 + sustain_gain))) / (
            self._fast_envelope + EPSILON
        )

        # Clamp the gain to prevent extreme values, especially during silence
        gain_envelope = torch.clamp(gain_envelope, 0.0, 4.0)

        # 6. Apply the computed gain to the original input signal
        # The gain is applied to all channels of the original signal (broadcasting).
        output_signal = signal * gain_envelope

        return {"out": output_signal}

    def start(self):
        """Reset DSP state when processing starts."""
        with self._lock:
            self._fast_envelope = 0.0
            self._slow_envelope = 0.0
        super().start()

    def serialize_extra(self) -> dict:
        with self._lock:
            return self._get_state_snapshot_locked()

    def deserialize_extra(self, data: dict):
        # Use the public setters to ensure UI is updated correctly on load.
        self.set_attack(data.get("attack", 0.0))
        self.set_sustain(data.get("sustain", 0.0))
