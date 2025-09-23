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
from ui_elements import ParameterNodeItem, NodeItem, NODE_CONTENT_PADDING
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
SIDECHAIN_DOWNSAMPLE_FACTOR = 16


# ==============================================================================
# 1. UI Class for the Compressor Node (Unchanged)
# ==============================================================================
class CompressorNodeItem(ParameterNodeItem):
    """Custom UI for the CompressorNode with slider controls."""

    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "CompressorNode"):
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
# 2. Logic Class for the Compressor Node
# ==============================================================================
class CompressorNode(Node):
    NODE_TYPE = "Compressor"
    UI_CLASS = CompressorNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Reduces the dynamic range of a signal."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("threshold_db", data_type=float)
        self.add_input("ratio", data_type=float)
        self.add_input("attack_ms", data_type=float)
        self.add_input("release_ms", data_type=float)
        self.add_input("knee_db", data_type=float)
        self.add_input("sidechain_in", data_type=torch.Tensor)  # Can be None if not connected
        self.add_output("out", data_type=torch.Tensor)

        self._threshold_db = -20.0
        self._ratio = 4.0
        self._attack_ms = 5.0
        self._release_ms = 100.0
        self._knee_db = 6.0
        self._samplerate = DEFAULT_SAMPLERATE
        self._envelope = 0.0

        self._delay_samples = SIDECHAIN_DOWNSAMPLE_FACTOR // 2
        self._delay_buffer = torch.zeros((DEFAULT_CHANNELS, self._delay_samples), dtype=DEFAULT_DTYPE)

        # --- Performance Optimization Attributes ---
        self._params_dirty = True
        self._attack_coeff = 0.0
        self._release_coeff = 0.0
        self._last_signal_shape = None

        # Pre-allocated buffers
        self._power_buffer = None
        self._sidechain_power = None
        self._indices_buffer = None
        self._downsampled_sidechain = None
        self._envelope_out_down = None
        self._envelope_out_mono_power = None
        self._envelope_db = None
        self._gain_reduction_db = None
        self._gain_reduction_linear = None
        self._gain_below = None
        self._gain_inside = None
        self._gain_above = None

    def _update_coefficients(self):
        """Recalculates time-based coefficients only when parameters change."""
        downsampled_samplerate = self._samplerate / SIDECHAIN_DOWNSAMPLE_FACTOR
        self._attack_coeff = torch.exp(
            torch.tensor(-1.0 / (downsampled_samplerate * (self._attack_ms / 1000.0)))
        ).item()
        self._release_coeff = torch.exp(
            torch.tensor(-1.0 / (downsampled_samplerate * (self._release_ms / 1000.0)))
        ).item()
        self._params_dirty = False

    def _resize_buffers(self, signal_shape: torch.Size):
        """Re-allocates all intermediate buffers if the input signal shape changes."""
        _num_channels, block_size = signal_shape
        downsampled_size = block_size // SIDECHAIN_DOWNSAMPLE_FACTOR

        self._power_buffer = torch.empty(signal_shape, dtype=DEFAULT_DTYPE)
        self._sidechain_power = torch.empty(block_size, dtype=DEFAULT_DTYPE)
        self._indices_buffer = torch.empty(block_size, dtype=torch.long)

        self._downsampled_sidechain = torch.empty(downsampled_size, dtype=DEFAULT_DTYPE)
        self._envelope_out_down = torch.empty(downsampled_size, dtype=DEFAULT_DTYPE)
        self._envelope_out_mono_power = torch.empty(block_size, dtype=DEFAULT_DTYPE)
        self._envelope_db = torch.empty(block_size, dtype=DEFAULT_DTYPE)
        self._gain_reduction_db = torch.empty(block_size, dtype=DEFAULT_DTYPE)
        self._gain_reduction_linear = torch.empty(block_size, dtype=DEFAULT_DTYPE)

        # Buffers for the torch.where calculation
        self._gain_below = torch.zeros(block_size, dtype=DEFAULT_DTYPE)
        self._gain_inside = torch.empty(block_size, dtype=DEFAULT_DTYPE)
        self._gain_above = torch.empty(block_size, dtype=DEFAULT_DTYPE)

        # Also resize the delay buffer if the number of channels has changed.
        if self._delay_buffer.shape[0] != _num_channels:
            logger.debug(f"[{self.name}] Resizing delay buffer for {_num_channels} channels.")
            self._delay_buffer = torch.zeros((_num_channels, self._delay_samples), dtype=DEFAULT_DTYPE)

        self._last_signal_shape = signal_shape
        logger.debug(f"[{self.name}] Resized internal buffers for shape {signal_shape}")

    # --- Parameter Setters ---
    def set_threshold_db(self, value: float):
        with self._lock:
            clipped_value = np.clip(float(value), MIN_THRESHOLD_DB, MAX_THRESHOLD_DB)
            if self._threshold_db != clipped_value:
                self._threshold_db = clipped_value
        self.ui_update_callback(self.get_current_state_snapshot())

    def set_ratio(self, value: float):
        with self._lock:
            clipped_value = np.clip(float(value), MIN_RATIO, MAX_RATIO)
            if self._ratio != clipped_value:
                self._ratio = clipped_value
        self.ui_update_callback(self.get_current_state_snapshot())

    def set_attack_ms(self, value: float):
        with self._lock:
            clipped_value = np.clip(float(value), MIN_ATTACK_MS, MAX_ATTACK_MS)
            if self._attack_ms != clipped_value:
                self._attack_ms = clipped_value
                self._params_dirty = True
        self.ui_update_callback(self.get_current_state_snapshot())

    def set_release_ms(self, value: float):
        with self._lock:
            clipped_value = np.clip(float(value), MIN_RELEASE_MS, MAX_RELEASE_MS)
            if self._release_ms != clipped_value:
                self._release_ms = clipped_value
                self._params_dirty = True
        self.ui_update_callback(self.get_current_state_snapshot())

    def set_knee_db(self, value: float):
        with self._lock:
            clipped_value = np.clip(float(value), MIN_KNEE_DB, MAX_KNEE_DB)
            if self._knee_db != clipped_value:
                self._knee_db = clipped_value
        self.ui_update_callback(self.get_current_state_snapshot())

    def _get_state_snapshot_locked(self) -> Dict:
        return {
            "threshold_db": self._threshold_db,
            "ratio": self._ratio,
            "attack_ms": self._attack_ms,
            "release_ms": self._release_ms,
            "knee_db": self._knee_db,
        }

    def start(self):
        with self._lock:
            self._envelope = 0.0
            self._delay_buffer.zero_()
            self._last_signal_shape = None  # Force buffer reallocation on first run
            self._params_dirty = True

    @staticmethod
    @torch.jit.script
    def _jit_envelope_loop(
        sidechain: torch.Tensor,
        envelope_out: torch.Tensor,
        initial_envelope: float,
        attack_coeff: float,
        release_coeff: float,
    ) -> torch.Tensor:
        num_samples = sidechain.shape[0]
        env = torch.tensor(initial_envelope, dtype=sidechain.dtype, device=sidechain.device)
        for i in range(num_samples):
            target = sidechain[i]
            coeff = attack_coeff if target > env else release_coeff
            env = target + coeff * (env - target)
            envelope_out[i] = env
        return env  # Return only the final envelope state

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}

        sidechain_signal = input_data.get("sidechain_in")

        with torch.no_grad():
            # --- State update logic ---
            with self._lock:
                # Socket updates can still happen here. We read them and update the internal state.
                ui_update_needed = False
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
                        self._params_dirty = True
                        ui_update_needed = True
                release_socket_val = input_data.get("release_ms")
                if release_socket_val is not None:
                    clipped_val = np.clip(float(release_socket_val), MIN_RELEASE_MS, MAX_RELEASE_MS)
                    if self._release_ms != clipped_val:
                        self._release_ms = clipped_val
                        self._params_dirty = True
                        ui_update_needed = True
                knee_socket_val = input_data.get("knee_db")
                if knee_socket_val is not None:
                    clipped_val = np.clip(float(knee_socket_val), MIN_KNEE_DB, MAX_KNEE_DB)
                    if self._knee_db != clipped_val:
                        self._knee_db = clipped_val
                        ui_update_needed = True

                # If any socket caused a change, emit the new state to the UI
                if ui_update_needed:
                    self.ui_update_callback(self._get_state_snapshot_locked())

                # Copy locked parameters to local variables for this tick's processing
                threshold_db, ratio, knee_db, initial_envelope = (
                    self._threshold_db,
                    self._ratio,
                    self._knee_db,
                    self._envelope,
                )

                if self._params_dirty:
                    self._update_coefficients()

                # Check if we need to reallocate buffers
                if signal.shape != self._last_signal_shape:
                    self._resize_buffers(signal.shape)

            # --- DSP Processing ---

            # Determine which signal to use for level detection.
            # If a valid sidechain signal is connected, use it. Otherwise, fall back to the main input.
            detection_signal = sidechain_signal if isinstance(sidechain_signal, torch.Tensor) else signal

            # 1. Level Detection using squared amplitude (power) into a pre-allocated buffer.
            torch.square(detection_signal, out=self._power_buffer)
            # --- Provide the pre-allocated indices buffer to the out= argument ---
            torch.max(self._power_buffer, dim=0, out=(self._sidechain_power, self._indices_buffer))

            # 2. Downsample the power sidechain.
            self._downsampled_sidechain = F.avg_pool1d(
                self._sidechain_power.unsqueeze(0), kernel_size=SIDECHAIN_DOWNSAMPLE_FACTOR
            ).squeeze(0)

            # 3. Run the JIT envelope follower.
            final_env = self._jit_envelope_loop(
                self._downsampled_sidechain,
                self._envelope_out_down,
                initial_envelope,
                self._attack_coeff,
                self._release_coeff,
            )
            self._envelope = final_env.item()

            # 4. Upsample using efficient repeat_interleave.
            self._envelope_out_mono_power = torch.repeat_interleave(
                self._envelope_out_down, SIDECHAIN_DOWNSAMPLE_FACTOR
            )

            # 5. Convert smoothed power to dB.
            torch.log10(self._envelope_out_mono_power + EPSILON, out=self._envelope_db)
            self._envelope_db.mul_(10.0)

            # 6. Calculate gain reduction in dB (fully vectorized).
            slope = 1.0 / ratio - 1.0
            knee_start = threshold_db - knee_db / 2.0
            knee_end = threshold_db + knee_db / 2.0

            torch.sub(self._envelope_db, threshold_db, out=self._gain_above).mul_(slope)
            torch.sub(self._envelope_db, knee_start, out=self._gain_inside).pow_(2).mul_(
                slope / (2.0 * max(EPSILON, knee_db))
            )

            is_below = self._envelope_db < knee_start
            is_inside = (self._envelope_db >= knee_start) & (self._envelope_db <= knee_end)

            torch.where(
                is_below,
                self._gain_below,
                torch.where(is_inside, self._gain_inside, self._gain_above),
                out=self._gain_reduction_db,
            )

            # 7. DELAY COMPENSATION
            if self._delay_samples > 0:
                combined_signal = torch.cat((self._delay_buffer, signal), dim=1)
                delayed_signal = combined_signal[:, : signal.shape[1]]
                self._delay_buffer = combined_signal[:, signal.shape[1] :]
            else:
                delayed_signal = signal

            # 8. Apply Gain.
            torch.div(self._gain_reduction_db, 20.0, out=self._gain_reduction_db)
            torch.pow(10.0, self._gain_reduction_db, out=self._gain_reduction_linear)

            output_signal = delayed_signal * self._gain_reduction_linear

            return {"out": output_signal}

    def serialize_extra(self) -> dict:
        with self._lock:
            return self._get_state_snapshot_locked()

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._threshold_db = data.get("threshold_db", -20.0)
            self._ratio = data.get("ratio", 4.0)
            self._attack_ms = data.get("attack_ms", 5.0)
            self._release_ms = data.get("release_ms", 100.0)
            self._knee_db = data.get("knee_db", 6.0)
            self._params_dirty = True
