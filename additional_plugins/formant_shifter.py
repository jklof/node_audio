# additional_plugins/formant_shifter.py

import torch
import torch.nn.functional as F
import threading
import logging
from typing import Dict, Optional

from PySide6.QtCore import Slot

from node_system import Node
from constants import SpectralFrame, DEFAULT_COMPLEX_DTYPE
from ui_elements import ParameterNodeItem

logger = logging.getLogger(__name__)

# A small value to prevent division by zero in calculations
EPSILON = 1e-12
# Minimum F0 to be considered valid for adaptive kernel calculation
MIN_VALID_F0 = 20.0


# --- UI Class (Unchanged) ---
class FormantShifterNodeItem(ParameterNodeItem):
    """UI for the pure Formant Shifter node."""

    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "FormantShifterNode"):
        parameters = [
            {
                "key": "formant_shift_st",
                "name": "Formant Shift",
                "min": -24.0,
                "max": 24.0,
                "format": "{:+.1f} st",
                "is_log": False,
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# --- Logic Class (UPGRADED) ---
class FormantShifterNode(Node):
    NODE_TYPE = "Formant Shifter"
    UI_CLASS = FormantShifterNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "Shifts spectral formants. Uses F0 for better quality and smooths parameter changes."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)

        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("f0_hz_in", data_type=float)  # <-- IMPROVEMENT 1: F0 input
        self.add_input("formant_shift_st", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        self._lock = threading.Lock()
        self._formant_shift_st: float = 0.0
        # --- IMPROVEMENT 2: State for parameter smoothing ---
        self._last_formant_ratio: float = 1.0

    def start(self):
        """Called when processing starts to reset the node's state."""
        with self._lock:
            # Reset the smoothing parameter to ensure no stale values
            self._last_formant_ratio = 1.0

    @Slot(float)
    def set_formant_shift_st(self, value: float):
        state_to_emit = None
        with self._lock:
            if self._formant_shift_st != value:
                self._formant_shift_st = float(value)
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {"formant_shift_st": self._formant_shift_st}

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    def _resample_magnitudes(self, magnitudes: torch.Tensor, ratio: float) -> torch.Tensor:
        num_channels, num_bins = magnitudes.shape
        if abs(ratio - 1.0) < 1e-6:
            return magnitudes

        resampled = F.interpolate(
            magnitudes.unsqueeze(0), scale_factor=ratio, mode="linear", align_corners=False, recompute_scale_factor=True
        ).squeeze(0)

        resampled_bins = resampled.shape[1]
        if resampled_bins < num_bins:
            padding = num_bins - resampled_bins
            resampled = F.pad(resampled, (0, padding), "constant", 0)
        elif resampled_bins > num_bins:
            resampled = resampled[:, :num_bins]
        return resampled

    def _get_spectral_envelope(
        self, magnitudes: torch.Tensor, sample_rate: float, fft_size: int, f0_hz: Optional[float]
    ) -> torch.Tensor:
        """
        Calculates the spectral envelope using a moving average filter.
        IMPROVEMENT 1: Kernel size is now adaptive to the fundamental frequency (F0).
        """
        # --- Adaptive Kernel Logic ---
        if f0_hz is not None and f0_hz > MIN_VALID_F0:
            # Set kernel width to be 1.5x the fundamental frequency for good harmonic smoothing
            kernel_size_hz = f0_hz * 1.5
        else:
            # Fallback to a fixed default if no valid F0 is provided
            kernel_size_hz = 150.0

        freq_per_bin = sample_rate / fft_size
        kernel_size = int(kernel_size_hz / freq_per_bin)
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size

        padded_mags = F.pad(magnitudes.unsqueeze(1), (kernel_size // 2, kernel_size // 2), mode="reflect")
        envelope = F.avg_pool1d(padded_mags, kernel_size=kernel_size, stride=1, padding=0)
        return envelope.squeeze(1)

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        state_snapshot_to_emit = None
        with self._lock:
            ui_update_needed = False
            formant_socket_val = input_data.get("formant_shift_st")
            f0_hz = input_data.get("f0_hz_in")  # Read F0 input

            effective_formant_shift = (
                float(formant_socket_val) if formant_socket_val is not None else self._formant_shift_st
            )

            if self._formant_shift_st != effective_formant_shift:
                self._formant_shift_st = effective_formant_shift
                ui_update_needed = True

            if ui_update_needed:
                state_snapshot_to_emit = self._get_current_state_snapshot_locked()

            formant_ratio = 2 ** (effective_formant_shift / 12.0)

            # --- IMPROVEMENT 2: Parameter Smoothing Ramp ---
            # Create a ramp from the last ratio to the current one for smooth transitions.
            ramp = torch.linspace(
                self._last_formant_ratio, formant_ratio, steps=frame.data.shape[1], device=frame.data.device
            )
            self._last_formant_ratio = formant_ratio  # Store the new target ratio for the next frame

        if state_snapshot_to_emit:
            self.ui_update_callback(state_snapshot_to_emit)

        # Optimization: Pass through if no shift and ramp is flat
        if abs(formant_ratio - 1.0) < 1e-6 and abs(ramp[0] - 1.0) < 1e-6:
            return {"spectral_frame_out": frame}

        input_magnitudes = torch.abs(frame.data)
        input_phases = torch.angle(frame.data)

        # 1. Get original envelope, passing F0 for adaptive quality
        original_envelope = self._get_spectral_envelope(input_magnitudes, frame.sample_rate, frame.fft_size, f0_hz)

        # 2. Create the new, shifted envelope
        shifted_envelope = self._resample_magnitudes(original_envelope, formant_ratio)

        # 3. Create a smoothed correction factor using the ramp
        correction_factor = shifted_envelope / (original_envelope + EPSILON)
        # Apply the ramp to the *change* from a neutral gain of 1.0
        smoothed_correction = 1.0 + (correction_factor - 1.0) * ramp

        # 4. Apply to original magnitudes and reconstruct
        final_magnitudes = input_magnitudes * smoothed_correction
        shifted_fft_data = torch.polar(final_magnitudes, input_phases)

        output_frame = SpectralFrame(
            data=shifted_fft_data.to(DEFAULT_COMPLEX_DTYPE),
            fft_size=frame.fft_size,
            hop_size=frame.hop_size,
            window_size=frame.window_size,
            sample_rate=frame.sample_rate,
            analysis_window=frame.analysis_window,
        )
        return {"spectral_frame_out": output_frame}

    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()

    def deserialize_extra(self, data: dict):
        self.set_formant_shift_st(data.get("formant_shift_st", 0.0))
