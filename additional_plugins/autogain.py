import torch
import numpy as np
import threading
import logging
from collections import deque
from typing import Dict

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE
from ui_elements import ParameterNodeItem, NodeItem, NODE_CONTENT_PADDING

# --- UI and Qt Imports ---
from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout
from PySide6.QtCore import Qt, Slot, QSignalBlocker

EPSILON = 1e-9

# Configure logging for this plugin
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. UI Class for the AutoGain Node
# ==============================================================================
class AutoGainNodeItem(ParameterNodeItem):
    """
    UI for the AutoGainNode with intuitive controls for professional leveling.
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
# 2. Logic Class for the AutoGain Node (Professional Leveler Algorithm)
# ==============================================================================
class AutoGainNode(Node):
    NODE_TYPE = "Auto Gain"
    UI_CLASS = AutoGainNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Automatically calculates gain to match a target signal level (RMS)."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("target_db", data_type=float)
        self.add_input("averaging_time_s", data_type=float)
        self.add_input("gain_smoothing_ms", data_type=float)
        self.add_output("gain_out", data_type=float)

        self._lock = threading.Lock()

        # --- Internal state parameters ---
        self._target_db: float = -14.0
        self._averaging_time_s: float = 3.0
        self._gain_smoothing_ms: float = 500.0

        # --- DSP State ---
        self._current_gain_db: float = -70.0  # Start silent
        self._rms_history: deque = deque(maxlen=1)
        self._recalculate_deque_size()

    def _recalculate_deque_size(self):
        """Calculates the required size of the history buffer based on averaging time."""
        num_blocks = int((self._averaging_time_s * DEFAULT_SAMPLERATE) / (DEFAULT_BLOCKSIZE + EPSILON))
        new_maxlen = max(1, num_blocks)
        if self._rms_history.maxlen != new_maxlen:
            self._rms_history = deque(self._rms_history, maxlen=new_maxlen)

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {
            "target_db": self._target_db,
            "averaging_time_s": self._averaging_time_s,
            "gain_smoothing_ms": self._gain_smoothing_ms,
        }

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    # --- Thread-safe setters ---
    @Slot(float)
    def set_target_db(self, value: float):
        state_to_emit = None
        with self._lock:
            if self._target_db != value:
                self._target_db = value
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_averaging_time_s(self, value: float):
        state_to_emit = None
        with self._lock:
            if self._averaging_time_s != value:
                self._averaging_time_s = value
                self._recalculate_deque_size()
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_gain_smoothing_ms(self, value: float):
        state_to_emit = None
        with self._lock:
            if self._gain_smoothing_ms != value:
                self._gain_smoothing_ms = value
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            # Return a gain of 1.0 (0 dB) if there is no valid input signal.
            return {"gain_out": 1.0}

        state_snapshot_to_emit = None
        with self._lock:
            ui_update_needed = False
            # Update parameters from sockets
            target_db_socket = input_data.get("target_db")
            if target_db_socket is not None and self._target_db != float(target_db_socket):
                self._target_db = float(target_db_socket)
                ui_update_needed = True
            avg_time_socket = input_data.get("averaging_time_s")
            if avg_time_socket is not None and self._averaging_time_s != float(avg_time_socket):
                self._averaging_time_s = float(avg_time_socket)
                self._recalculate_deque_size()
                ui_update_needed = True
            gain_smooth_socket = input_data.get("gain_smoothing_ms")
            if gain_smooth_socket is not None and self._gain_smoothing_ms != float(gain_smooth_socket):
                self._gain_smoothing_ms = float(gain_smooth_socket)
                ui_update_needed = True

            if ui_update_needed:
                state_snapshot_to_emit = self._get_current_state_snapshot_locked()

            target_db = self._target_db
            gain_smoothing_ms = self._gain_smoothing_ms

        if state_snapshot_to_emit:
            self.ui_update_callback(state_snapshot_to_emit)

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

    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()

    def deserialize_extra(self, data: dict):
        self.set_target_db(data.get("target_db", -14.0))
        self.set_averaging_time_s(data.get("averaging_time_s", 3.0))
        self.set_gain_smoothing_ms(data.get("gain_smoothing_ms", 500.0))
