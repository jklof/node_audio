import torch
import torch.nn.functional as F
import numpy as np
import threading
import logging
from typing import Dict, Optional, Tuple

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS

# --- UI and Qt Imports ---
from ui_elements import ParameterNodeItem, NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING
from PySide6.QtWidgets import QWidget, QSlider, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, Signal, Slot, QObject, QSignalBlocker

# --- Configure logging ---
logger = logging.getLogger(__name__)

# --- Node-Specific Constants ---
MIN_THRESHOLD_DB = -60.0
MAX_THRESHOLD_DB = 0.0
MIN_RATIO = 1.0
MAX_RATIO = 20.0
MIN_ATTACK_MS = 0.1
MAX_ATTACK_MS = 100.0
MIN_RELEASE_MS = 1.0
MAX_RELEASE_MS = 2000.0
MIN_KNEE_DB = 0.0
MAX_KNEE_DB = 24.0
EPSILON = 1e-9
SIDECHAIN_DOWNSAMPLE_FACTOR = 4


# ==============================================================================
# 1. UI Class for the Compressor Node (Unchanged)
# ==============================================================================
class CompressorNodeItem(ParameterNodeItem):
    """Custom UI for the CompressorNode with slider controls."""

    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "CompressorNode"):
        # Define the parameters for this node
        parameters = [
            {
                "key": "threshold_db",
                "name": "Threshold",
                "min": MIN_THRESHOLD_DB,
                "max": MAX_THRESHOLD_DB,
                "format": "{:.1f} dB",
                "is_log": False,
            },
            {
                "key": "ratio",
                "name": "Ratio",
                "min": MIN_RATIO,
                "max": MAX_RATIO,
                "format": "{:.1f}:1",
                "is_log": False,
            },
            {
                "key": "attack_ms",
                "name": "Attack",
                "min": MIN_ATTACK_MS,
                "max": MAX_ATTACK_MS,
                "format": "{:.1f} ms",
                "is_log": True,
            },
            {
                "key": "release_ms",
                "name": "Release",
                "min": MIN_RELEASE_MS,
                "max": MAX_RELEASE_MS,
                "format": "{:.0f} ms",
                "is_log": True,
            },
            {
                "key": "knee_db",
                "name": "Knee",
                "min": MIN_KNEE_DB,
                "max": MAX_KNEE_DB,
                "format": "{:.1f} dB",
                "is_log": False,
            },
        ]

        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 2. Logic Class for the Compressor Node (MODIFIED)
# ==============================================================================
class CompressorNode(Node):
    NODE_TYPE = "Compressor"
    UI_CLASS = CompressorNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Reduces the dynamic range of a signal."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("threshold_db", data_type=float)
        self.add_input("ratio", data_type=float)
        self.add_input("attack_ms", data_type=float)
        self.add_input("release_ms", data_type=float)
        self.add_input("knee_db", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._lock = threading.Lock()
        self._threshold_db = -20.0
        self._ratio = 4.0
        self._attack_ms = 5.0
        self._release_ms = 100.0
        self._knee_db = 6.0
        self._samplerate = DEFAULT_SAMPLERATE
        self._envelope = 0.0

        # --- NEW: Delay buffer for latency compensation ---
        self._delay_samples = SIDECHAIN_DOWNSAMPLE_FACTOR // 2
        self._delay_buffer = torch.zeros((DEFAULT_CHANNELS, self._delay_samples), dtype=DEFAULT_DTYPE)

    # --- Parameter Setters (Unchanged) ---
    def set_threshold_db(self, value: float):
        state_to_emit = None
        with self._lock:
            clipped_value = np.clip(float(value), MIN_THRESHOLD_DB, MAX_THRESHOLD_DB)
            if self._threshold_db != clipped_value:
                self._threshold_db = clipped_value
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def set_ratio(self, value: float):
        state_to_emit = None
        with self._lock:
            clipped_value = np.clip(float(value), MIN_RATIO, MAX_RATIO)
            if self._ratio != clipped_value:
                self._ratio = clipped_value
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def set_attack_ms(self, value: float):
        state_to_emit = None
        with self._lock:
            clipped_value = np.clip(float(value), MIN_ATTACK_MS, MAX_ATTACK_MS)
            if self._attack_ms != clipped_value:
                self._attack_ms = clipped_value
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def set_release_ms(self, value: float):
        state_to_emit = None
        with self._lock:
            clipped_value = np.clip(float(value), MIN_RELEASE_MS, MAX_RELEASE_MS)
            if self._release_ms != clipped_value:
                self._release_ms = clipped_value
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def set_knee_db(self, value: float):
        state_to_emit = None
        with self._lock:
            clipped_value = np.clip(float(value), MIN_KNEE_DB, MAX_KNEE_DB)
            if self._knee_db != clipped_value:
                self._knee_db = clipped_value
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def get_current_state_snapshot(self, locked: bool = False) -> Dict:
        state = {
            "threshold_db": self._threshold_db,
            "ratio": self._ratio,
            "attack_ms": self._attack_ms,
            "release_ms": self._release_ms,
            "knee_db": self._knee_db,
        }
        if locked:
            return state
        with self._lock:
            return state

    def start(self):
        with self._lock:
            self._envelope = 0.0
            self._delay_buffer.zero_()

    @staticmethod
    @torch.jit.script
    def _jit_envelope_loop(
        sidechain: torch.Tensor, initial_envelope: float, attack_coeff: float, release_coeff: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        num_samples = sidechain.shape[0]
        envelope_out = torch.zeros_like(sidechain)
        env = torch.tensor(initial_envelope, dtype=sidechain.dtype, device=sidechain.device)

        for i in range(num_samples):
            target = sidechain[i]
            coeff = attack_coeff if target > env else release_coeff
            env = target + coeff * (env - target)
            envelope_out[i] = env

        return envelope_out, env

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}

        # --- State update logic (Unchanged) ---
        state_to_emit = None
        ui_update_needed = False
        with self._lock:
            # This section remains the same...
            threshold_socket_val = input_data.get("threshold_db")
            if threshold_socket_val is not None:
                clipped_val = np.clip(float(threshold_socket_val), MIN_THRESHOLD_DB, MAX_THRESHOLD_DB)
                if self._threshold_db != clipped_val:
                    self._threshold_db = clipped_val
                    ui_update_needed = True
            ratio_socket_val = input_data.get("ratio")
            if ratio_socket_val is not None:
                clipped_val = np.clip(float(ratio_socket_val), MIN_RATIO, MAX_RATIO)
                if self._ratio != clipped_val:
                    self._ratio = clipped_val
                    ui_update_needed = True
            attack_socket_val = input_data.get("attack_ms")
            if attack_socket_val is not None:
                clipped_val = np.clip(float(attack_socket_val), MIN_ATTACK_MS, MAX_ATTACK_MS)
                if self._attack_ms != clipped_val:
                    self._attack_ms = clipped_val
                    ui_update_needed = True
            release_socket_val = input_data.get("release_ms")
            if release_socket_val is not None:
                clipped_val = np.clip(float(release_socket_val), MIN_RELEASE_MS, MAX_RELEASE_MS)
                if self._release_ms != clipped_val:
                    self._release_ms = clipped_val
                    ui_update_needed = True
            knee_socket_val = input_data.get("knee_db")
            if knee_socket_val is not None:
                clipped_val = np.clip(float(knee_socket_val), MIN_KNEE_DB, MAX_KNEE_DB)
                if self._knee_db != clipped_val:
                    self._knee_db = clipped_val
                    ui_update_needed = True

            threshold_db, ratio, attack_ms, release_ms, knee_db, initial_envelope = (
                self._threshold_db,
                self._ratio,
                self._attack_ms,
                self._release_ms,
                self._knee_db,
                self._envelope,
            )

            if ui_update_needed:
                state_to_emit = self.get_current_state_snapshot(locked=True)

        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

        # --- DSP Processing (MODIFIED FOR DELAY COMPENSATION) ---

        # 1. Level Detection & Gain Computation (on original signal)
        level_db = 20 * torch.log10(torch.abs(signal) + EPSILON)

        slope = 1.0 / ratio - 1.0
        knee_start = threshold_db - knee_db / 2.0
        knee_end = threshold_db + knee_db / 2.0
        is_below = level_db < knee_start
        is_inside = (level_db >= knee_start) & (level_db <= knee_end)

        gain_below = torch.zeros_like(level_db)
        gain_inside = slope * (((level_db - knee_start) ** 2) / (2.0 * max(EPSILON, knee_db)))
        gain_above = slope * (level_db - threshold_db)

        gain_reduction_db = torch.where(is_below, gain_below, torch.where(is_inside, gain_inside, gain_above))

        # 2. Sidechain Preparation
        sidechain_target, _ = torch.max(torch.abs(gain_reduction_db), dim=0)

        # 3. Downsample with stateless avg pooling
        downsampled_sidechain = F.avg_pool1d(
            sidechain_target.unsqueeze(0), kernel_size=SIDECHAIN_DOWNSAMPLE_FACTOR, stride=SIDECHAIN_DOWNSAMPLE_FACTOR
        ).squeeze(0)

        # 4. Run the JIT envelope follower
        downsampled_samplerate = self._samplerate / SIDECHAIN_DOWNSAMPLE_FACTOR
        attack_coeff = np.exp(-1.0 / (downsampled_samplerate * (attack_ms / 1000.0))).item()
        release_coeff = np.exp(-1.0 / (downsampled_samplerate * (release_ms / 1000.0))).item()

        envelope_out_down, final_env_tensor = self._jit_envelope_loop(
            downsampled_sidechain, initial_envelope, attack_coeff, release_coeff
        )
        self._envelope = final_env_tensor.item()

        # 5. Upsample with stateless linear interpolation
        envelope_out_mono = (
            F.interpolate(
                envelope_out_down.unsqueeze(0).unsqueeze(0), size=signal.shape[1], mode="linear", align_corners=False
            )
            .squeeze(0)
            .squeeze(0)
        )

        # 6. DELAY COMPENSATION
        # The resampling introduces a delay. We apply an equal delay to the audio signal.
        if self._delay_samples > 0:
            # Create a combined signal of the buffer and the current input
            combined_signal = torch.cat((self._delay_buffer, signal), dim=1)
            # The signal to be processed is the first part of this combined signal
            delayed_signal = combined_signal[:, : signal.shape[1]]
            # The new delay buffer is the last part of the combined signal
            self._delay_buffer = combined_signal[:, signal.shape[1] :]
        else:
            delayed_signal = signal

        # 7. Apply Gain to the DELAYED signal
        gain_reduction_linear = 10 ** (-envelope_out_mono / 20.0)
        output_signal = delayed_signal * gain_reduction_linear

        return {"out": output_signal}

    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._threshold_db = data.get("threshold_db", -20.0)
            self._ratio = data.get("ratio", 4.0)
            self._attack_ms = data.get("attack_ms", 5.0)
            self._release_ms = data.get("release_ms", 100.0)
            self._knee_db = data.get("knee_db", 6.0)
