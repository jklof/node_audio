# File: additional_plugins/limiter.py

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Optional, Tuple

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE

# --- UI and Qt Imports ---
from ui_elements import ParameterNodeItem
from PySide6.QtCore import Slot

# Configure logging
logger = logging.getLogger(__name__)

# --- Node-Specific Constants ---
LOOKAHEAD_MS = 5.0
MIN_THRESHOLD_DB = -40.0
MAX_THRESHOLD_DB = 0.0
MIN_RELEASE_MS = 1.0
MAX_RELEASE_MS = 2000.0
EPSILON = 1e-9

# Process the sidechain at a lower sample rate for a huge performance gain.
DOWNSAMPLE_FACTOR = 16


# ==============================================================================
# 1. JIT-Compiled Gain Computer
# ==============================================================================
@torch.jit.script
def _jit_downsampled_gain_loop(
    downsampled_peaks: torch.Tensor,
    gain_envelope_out: torch.Tensor,
    initial_gain: float,
    threshold_lin: float,
    release_coeff: float,
    epsilon: float,
) -> torch.Tensor:
    """
    Calculates a gain envelope on a downsampled peak signal.
    This loop runs far fewer times than a sample-by-sample loop, making it much faster.
    """
    num_frames = downsampled_peaks.shape[0]
    current_gain = torch.tensor(initial_gain, dtype=downsampled_peaks.dtype, device=downsampled_peaks.device)
    one_tensor = torch.tensor(1.0, dtype=downsampled_peaks.dtype, device=downsampled_peaks.device)

    for i in range(num_frames):
        peak_val = downsampled_peaks[i]
        gain_needed = threshold_lin / (peak_val + epsilon) if peak_val > threshold_lin else one_tensor

        if gain_needed < current_gain:
            current_gain = gain_needed
        else:
            current_gain = current_gain + release_coeff * (one_tensor - current_gain)
            if current_gain > 1.0:
                current_gain = one_tensor

        gain_envelope_out[i] = current_gain

    return current_gain


# ==============================================================================
# 2. UI Class for the Brickwall Limiter Node
# ==============================================================================
class BrickwallLimiterNodeItem(ParameterNodeItem):
    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "BrickwallLimiterNode"):
        parameters = [
            {
                "key": "threshold_db",
                "name": "Threshold",
                "min": MIN_THRESHOLD_DB,
                "max": MAX_THRESHOLD_DB,
                "format": "{:.1f} dB",
            },
            {
                "key": "release_ms",
                "name": "Release",
                "min": MIN_RELEASE_MS,
                "max": MAX_RELEASE_MS,
                "format": "{:.0f} ms",
                "is_log": True,
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 3. Logic Class for the Brickwall Limiter Node (Heavily Optimized)
# ==============================================================================
class BrickwallLimiterNode(Node):
    NODE_TYPE = "Brickwall Limiter"
    UI_CLASS = BrickwallLimiterNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Prevents a signal from exceeding a threshold using lookahead."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("threshold_db", data_type=float)
        self.add_input("release_ms", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._threshold_db: float = -0.1
        self._release_ms: float = 100.0

        self._current_gain: float = 1.0
        self._delay_samples: int = int((LOOKAHEAD_MS / 1000.0) * DEFAULT_SAMPLERATE)
        self._delay_buffer: Optional[torch.Tensor] = None

        self._params_dirty: bool = True
        self._release_coeff: float = 0.0
        self._last_signal_shape: Optional[torch.Size] = None

        # Buffers for downsampled processing
        self._abs_mono_buffer: Optional[torch.Tensor] = None
        self._downsampled_peaks_buffer: Optional[torch.Tensor] = None
        self._downsampled_gain_buffer: Optional[torch.Tensor] = None
        self._gain_envelope_buffer: Optional[torch.Tensor] = None

    def _update_coefficients(self):
        # Calculate release coefficient based on the downsampled rate
        downsampled_samplerate = DEFAULT_SAMPLERATE / DOWNSAMPLE_FACTOR
        release_samples = (self._release_ms / 1000.0) * downsampled_samplerate
        self._release_coeff = 1.0 - torch.exp(torch.tensor(-1.0 / (release_samples + EPSILON))).item()
        self._params_dirty = False
        logger.debug(f"[{self.name}] Recalculated limiter coefficients for downsampled rate.")

    def _resize_buffers_if_needed(self, signal_shape: torch.Size):
        if self._last_signal_shape == signal_shape:
            return

        num_channels, block_size = signal_shape
        downsampled_size = block_size // DOWNSAMPLE_FACTOR

        self._delay_buffer = torch.zeros((num_channels, self._delay_samples), dtype=DEFAULT_DTYPE)

        # Resize all new and existing buffers
        self._abs_mono_buffer = torch.empty((1, 1, block_size), dtype=DEFAULT_DTYPE)  # Reshaped for avg_pool1d
        self._downsampled_peaks_buffer = torch.empty(downsampled_size, dtype=DEFAULT_DTYPE)
        self._downsampled_gain_buffer = torch.empty(downsampled_size, dtype=DEFAULT_DTYPE)
        self._gain_envelope_buffer = torch.empty(block_size, dtype=DEFAULT_DTYPE)

        self._last_signal_shape = signal_shape
        logger.info(f"[{self.name}] Resized internal buffers for shape {signal_shape}")

    @Slot(float)
    def set_threshold_db(self, value: float):
        state_to_emit = None
        with self._lock:
            clipped_value = np.clip(float(value), MIN_THRESHOLD_DB, MAX_THRESHOLD_DB)
            if self._threshold_db != clipped_value:
                self._threshold_db = clipped_value
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_release_ms(self, value: float):
        state_to_emit = None
        with self._lock:
            clipped_value = np.clip(float(value), MIN_RELEASE_MS, MAX_RELEASE_MS)
            if self._release_ms != clipped_value:
                self._release_ms = clipped_value
                self._params_dirty = True
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def _get_state_snapshot_locked(self) -> Dict:
        return {"threshold_db": self._threshold_db, "release_ms": self._release_ms}

    def start(self):
        with self._lock:
            self._current_gain = 1.0
            if self._delay_buffer is not None:
                self._delay_buffer.zero_()
            self._last_signal_shape = None
            self._params_dirty = True
        super().start()

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}

        with torch.no_grad():
            state_to_emit = None
            with self._lock:
                # (Parameter update logic remains the same)
                ui_update_needed = False
                threshold_socket = input_data.get("threshold_db")
                if threshold_socket is not None:
                    clipped_val = np.clip(float(threshold_socket), MIN_THRESHOLD_DB, MAX_THRESHOLD_DB)
                    if self._threshold_db != clipped_val:
                        self._threshold_db = clipped_val
                        ui_update_needed = True
                release_socket = input_data.get("release_ms")
                if release_socket is not None:
                    clipped_val = np.clip(float(release_socket), MIN_RELEASE_MS, MAX_RELEASE_MS)
                    if self._release_ms != clipped_val:
                        self._release_ms = clipped_val
                        self._params_dirty = True
                        ui_update_needed = True
                if ui_update_needed:
                    state_to_emit = self._get_state_snapshot_locked()

                self._resize_buffers_if_needed(signal.shape)
                if self._params_dirty:
                    self._update_coefficients()

                threshold_lin = 10.0 ** (self._threshold_db / 20.0)
                initial_gain = self._current_gain
                release_coeff = self._release_coeff

            if state_to_emit:
                self.ui_update_callback(state_to_emit)

            # --- DSP Processing ---

            # 1. LOOKAHEAD: Delay the main audio path
            combined_signal = torch.cat((self._delay_buffer, signal), dim=1)
            delayed_signal = combined_signal[:, : signal.shape[1]]
            self._delay_buffer.copy_(combined_signal[:, signal.shape[1] :])

            # 2. LEVEL DETECTION: Create a mono, absolute signal
            if signal.shape[0] > 1:
                torch.max(
                    torch.abs(signal),
                    dim=0,
                    out=(self._abs_mono_buffer.squeeze(), torch.empty(signal.shape[1], dtype=torch.long)),
                )
            else:
                torch.abs(signal[0], out=self._abs_mono_buffer.squeeze())

            # 3. Downsample the peak-detected sidechain signal
            self._downsampled_peaks_buffer = F.avg_pool1d(
                self._abs_mono_buffer, kernel_size=DOWNSAMPLE_FACTOR
            ).squeeze()

            # 4. Run the fast JIT loop on the SMALL downsampled signal
            final_gain = _jit_downsampled_gain_loop(
                self._downsampled_peaks_buffer,
                self._downsampled_gain_buffer,
                initial_gain,
                threshold_lin,
                release_coeff,
                EPSILON,
            )
            self._current_gain = final_gain.item()

            # 5. Upsample the low-resolution gain envelope to full block size
            self._gain_envelope_buffer = torch.repeat_interleave(self._downsampled_gain_buffer, DOWNSAMPLE_FACTOR)

            # 6. APPLY GAIN: Multiply the delayed signal by the final, upsampled gain envelope
            output_signal = delayed_signal * self._gain_envelope_buffer

            return {"out": output_signal}

    def serialize_extra(self) -> dict:
        with self._lock:
            return self._get_state_snapshot_locked()

    def deserialize_extra(self, data: dict):
        self.set_threshold_db(data.get("threshold_db", -0.1))
        self.set_release_ms(data.get("release_ms", 100.0))
