import torch
import numpy as np
import threading
import logging
from collections import deque
from typing import Dict

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE
from ui_elements import ParameterNodeItem
from node_helpers import with_parameters, Parameter

# --- UI and Qt Imports ---
from PySide6.QtCore import Qt, Slot, QSignalBlocker

EPSILON = 1e-9

# Configure logging for this plugin
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. UI Class for the AutoGain Node (No changes needed)
# ==============================================================================
class AutoGainNodeItem(ParameterNodeItem):
    """
    UI for the AutoGainNode with intuitive controls for professional leveling.
    This class requires no changes as it already uses the declarative ParameterNodeItem.
    """

    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "AutoGainNode"):
        # Define the parameters for this node
        parameters = [
            {
                "key": "target_db",
                "name": "Target Level",
                "min": -40.0,
                "max": 0.0,
                "format": "{:.1f} dB",
                "is_log": False,
            },
            {
                "key": "averaging_time_s",
                "name": "Averaging Time",
                "min": 0.5,
                "max": 10.0,
                "format": "{:.1f} s",
                "is_log": False,
            },
            {
                "key": "gain_smoothing_ms",
                "name": "Gain Smoothing",
                "min": 50.0,
                "max": 2000.0,
                "format": "{:.0f} ms",
                "is_log": True,
            },
        ]

        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 2. Logic Class for the AutoGain Node
# ==============================================================================
@with_parameters
class AutoGainNode(Node):
    NODE_TYPE = "Auto Gain"
    UI_CLASS = AutoGainNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Automatically calculates gain to match a target signal level (RMS)."

    # --- Declarative managed parameters ---
    # The decorator automatically creates thread-safe setters (e.g., set_target_db),
    # serialization, deserialization, and the UI update callback mechanism.
    target_db = Parameter(default=-14.0)
    averaging_time_s = Parameter(default=3.0, on_change="_recalculate_deque_size")
    gain_smoothing_ms = Parameter(default=500.0)

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        # Sockets that match parameter names will be automatically updated by the helper
        self.add_input("target_db", data_type=float)
        self.add_input("averaging_time_s", data_type=float)
        self.add_input("gain_smoothing_ms", data_type=float)
        self.add_output("gain_out", data_type=float)

        # --- DSP State (not a managed parameter) ---
        self._current_gain_db: float = -70.0  # Start silent
        self._rms_history: deque = deque(maxlen=1)
        # Initial deque sizing based on the default parameter value
        self._recalculate_deque_size()

    def _recalculate_deque_size(self):
        """
        Calculates the required size of the history buffer.
        This is now called by the on_change hook of the averaging_time_s parameter.
        The decorator ensures this is called within a lock.
        """
        num_blocks = int((self._averaging_time_s * DEFAULT_SAMPLERATE) / (DEFAULT_BLOCKSIZE + EPSILON))
        new_maxlen = max(1, num_blocks)
        if self._rms_history.maxlen != new_maxlen:
            # Recreate the deque with the new size, preserving existing data
            self._rms_history = deque(self._rms_history, maxlen=new_maxlen)

    def _get_state_snapshot_locked(self) -> dict:
        return self._get_parameters_state()

    def serialize_extra(self) -> dict:
        return self._serialize_parameters()

    def deserialize_extra(self, data: dict):
        self._deserialize_parameters(data)

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            # Return a gain of 1.0 (0 dB) if there is no valid input signal.
            return {"gain_out": 1.0}

        # --- The decorator provides this helper method ---
        # It handles updating parameters from sockets and emitting a single UI update if needed.
        self._update_parameters_from_sockets(input_data)

        # Acquire the lock once to get a consistent snapshot of parameters for this tick's DSP.
        with self._lock:
            target_db = self._target_db
            gain_smoothing_ms = self._gain_smoothing_ms

        # --- STAGE 1: Long-Term Loudness Measurement ---
        mono_signal = torch.mean(signal, dim=0)
        rms_linear = torch.sqrt(torch.mean(torch.square(mono_signal)) + EPSILON)
        self._rms_history.append(rms_linear.item())

        long_term_rms_linear = np.mean(list(self._rms_history))
        long_term_rms_db = 20 * np.log10(long_term_rms_linear + EPSILON)

        # --- STAGE 2: Gain Correction ---
        gain_needed_db = target_db - long_term_rms_db

        # --- STAGE 3: Gain Smoothing ---
        samples_per_block = signal.shape[1]
        smoothing_samples = (gain_smoothing_ms / 1000.0) * DEFAULT_SAMPLERATE
        alpha = 1 - torch.exp(torch.tensor(-samples_per_block / (smoothing_samples + EPSILON)))
        self._current_gain_db += alpha * (gain_needed_db - self._current_gain_db)

        output_gain_linear = 10.0 ** (self._current_gain_db / 20.0)

        return {"gain_out": float(output_gain_linear)}

    def start(self):
        """Reset DSP state when processing starts."""
        with self._lock:
            # --- Reset gain to a silent default on every start ---
            self._current_gain_db = -70.0

            # Also clear the history to not use stale loudness data
            self._recalculate_deque_size()
            self._rms_history.clear()
        super().start()
