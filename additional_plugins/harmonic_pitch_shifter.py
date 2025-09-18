# additional_plugins/harmonic_pitch_shifter.py

import torch
import torch.nn.functional as F
import numpy as np
import threading
import logging
from typing import Dict, Optional

from PySide6.QtCore import Slot

from node_system import Node
from constants import SpectralFrame, DEFAULT_COMPLEX_DTYPE
from ui_elements import ParameterNodeItem, NodeStateEmitter

logger = logging.getLogger(__name__)

# A small value to prevent division by zero or log(0)
EPSILON = 1e-12
# Minimum F0 to be considered "voiced"
MIN_VOICED_F0 = 40.0


# --- UI Class ---
class HarmonicPitchShifterNodeItem(ParameterNodeItem):
    """UI for the Harmonic Pitch & Formant Shifter node."""

    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "HarmonicPitchShifterNode"):
        parameters = [
            {
                "key": "pitch_shift_st",
                "name": "Pitch Shift",
                "min": -24.0,
                "max": 24.0,
                "format": "{:+.1f} st",
                "is_log": False,
            },
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


# --- Logic Class ---
class HarmonicPitchShifterNode(Node):
    NODE_TYPE = "Harmonic Pitch Shifter"
    UI_CLASS = HarmonicPitchShifterNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "High-quality pitch/formant shifter using F0 to separate harmonics from noise."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("f0_hz_in", data_type=float)
        self.add_input("pitch_shift_st", data_type=float)
        self.add_input("formant_shift_st", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        self._lock = threading.Lock()
        self._pitch_shift_st: float = 0.0
        self._formant_shift_st: float = 0.0

        self._last_input_phases: Optional[torch.Tensor] = None
        self._last_output_phases: Optional[torch.Tensor] = None
        self._last_frame_params: tuple = (0, 0, 0, 0)

        self._freqs_buf: Optional[torch.Tensor] = None
        self._input_mags_buf: Optional[torch.Tensor] = None
        self._current_input_phases_buf: Optional[torch.Tensor] = None
        self._original_envelope_buf: Optional[torch.Tensor] = None
        self._source_mags_buf: Optional[torch.Tensor] = None
        self._shifted_source_mags_buf: Optional[torch.Tensor] = None
        self._shifted_envelope_buf: Optional[torch.Tensor] = None
        self._final_mags_buf: Optional[torch.Tensor] = None
        self._propagated_phases_buf: Optional[torch.Tensor] = None

    def _reset_state_locked(self):
        self._last_input_phases = None
        self._last_output_phases = None
        self._last_frame_params = (0, 0, 0, 0)
        logger.debug(f"[{self.name}] State reset.")

    def start(self):
        with self._lock:
            self._reset_state_locked()

    @Slot(float)
    def set_pitch_shift_st(self, value: float):
        state_to_emit = None
        with self._lock:
            if self._pitch_shift_st != value:
                self._pitch_shift_st = float(value)
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_formant_shift_st(self, value: float):
        state_to_emit = None
        with self._lock:
            if self._formant_shift_st != value:
                self._formant_shift_st = float(value)
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {"pitch_shift_st": self._pitch_shift_st, "formant_shift_st": self._formant_shift_st}

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    def _resize_buffers_if_needed(self, frame: SpectralFrame) -> bool:
        """
        Checks if frame parameters have changed. If so, resizes all buffers and stateful tensors.
        Returns True if a reset occurred, False otherwise.
        """
        num_channels, num_bins = frame.data.shape
        current_params = (frame.fft_size, frame.sample_rate, num_channels, num_bins)
        if self._last_frame_params == current_params:
            return False

        logger.info(f"[{self.name}] Frame format changed. Re-allocating buffers and resetting state.")
        self._freqs_buf = torch.fft.rfftfreq(frame.fft_size, d=1.0 / frame.sample_rate)

        # --- Pre-allocated scratch buffers ---
        self._input_mags_buf = torch.empty((num_channels, num_bins), dtype=torch.float32)
        self._current_input_phases_buf = torch.empty((num_channels, num_bins), dtype=torch.float32)
        self._original_envelope_buf = torch.empty((num_channels, num_bins), dtype=torch.float32)
        self._source_mags_buf = torch.empty((num_channels, num_bins), dtype=torch.float32)
        self._shifted_source_mags_buf = torch.empty((num_channels, num_bins), dtype=torch.float32)
        self._shifted_envelope_buf = torch.empty((num_channels, num_bins), dtype=torch.float32)
        self._final_mags_buf = torch.empty((num_channels, num_bins), dtype=torch.float32)
        self._propagated_phases_buf = torch.empty((num_channels, num_bins), dtype=torch.float32)

        # --- Stateful phase accumulators ---
        self._last_input_phases = torch.empty_like(self._current_input_phases_buf)
        self._last_output_phases = torch.empty_like(self._current_input_phases_buf)

        self._last_frame_params = current_params
        return True  # Signal that a reset happened

    def _resample_magnitudes(self, magnitudes: torch.Tensor, ratio: float, out_buf: torch.Tensor) -> torch.Tensor:
        if abs(ratio - 1.0) < 1e-6:
            out_buf.copy_(magnitudes)
            return out_buf

        resampled = F.interpolate(
            magnitudes.unsqueeze(0), scale_factor=ratio, mode="linear", align_corners=False, recompute_scale_factor=True
        )

        num_bins = magnitudes.shape[1]
        diff = num_bins - resampled.shape[2]
        if diff > 0:
            resampled = F.pad(resampled, (0, diff))
        elif diff < 0:
            resampled = resampled[:, :, :num_bins]

        out_buf.copy_(resampled.squeeze(0))
        return out_buf

    def _get_spectral_envelope(
        self, magnitudes: torch.Tensor, frame: SpectralFrame, out_buf: torch.Tensor
    ) -> torch.Tensor:
        freq_per_bin = frame.sample_rate / frame.fft_size
        kernel_size = int(150 / freq_per_bin)
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1

        padded_mags = F.pad(magnitudes.unsqueeze(1), (kernel_size // 2, kernel_size // 2), mode="reflect")
        envelope = F.avg_pool1d(padded_mags, kernel_size=kernel_size, stride=1, padding=0)
        out_buf.copy_(envelope.squeeze(1))
        return out_buf

    def _process_frame_harmonic(
        self, frame: SpectralFrame, f0_hz: float, pitch_ratio: float, formant_ratio: float
    ) -> torch.Tensor:
        torch.abs(frame.data, out=self._input_mags_buf)
        torch.angle(frame.data, out=self._current_input_phases_buf)

        self._get_spectral_envelope(self._input_mags_buf, frame, out_buf=self._original_envelope_buf)
        torch.div(self._input_mags_buf, self._original_envelope_buf + EPSILON, out=self._source_mags_buf)

        self._resample_magnitudes(self._source_mags_buf, pitch_ratio, out_buf=self._shifted_source_mags_buf)
        self._resample_magnitudes(self._original_envelope_buf, formant_ratio, out_buf=self._shifted_envelope_buf)

        torch.mul(self._shifted_source_mags_buf, self._shifted_envelope_buf, out=self._final_mags_buf)

        expected_phase_advance = 2 * torch.pi * frame.hop_size * self._freqs_buf
        phase_deviation = self._current_input_phases_buf - self._last_input_phases - expected_phase_advance
        torch.fmod(phase_deviation + torch.pi, 2 * torch.pi, out=phase_deviation)
        phase_deviation -= torch.pi

        true_freq_hz = (expected_phase_advance + phase_deviation) / (2 * torch.pi * frame.hop_size / frame.sample_rate)
        shifted_freq_hz = true_freq_hz * pitch_ratio
        new_phase_advance = shifted_freq_hz * (2 * torch.pi * frame.hop_size / frame.sample_rate)

        torch.add(self._last_output_phases, new_phase_advance, out=self._propagated_phases_buf)

        is_voiced = f0_hz is not None and f0_hz > MIN_VOICED_F0
        if is_voiced:
            harmonic_indices = self._freqs_buf / f0_hz
            deviation = torch.abs(harmonic_indices - torch.round(harmonic_indices))
            harmonic_strength = 1.0 - 2.0 * deviation
            harmonic_strength.clamp_(0.0, 1.0)

            final_phases = torch.lerp(self._current_input_phases_buf, self._propagated_phases_buf, harmonic_strength)
        else:
            final_phases = self._current_input_phases_buf

        self._last_input_phases.copy_(self._current_input_phases_buf)
        torch.fmod(self._propagated_phases_buf, 2 * torch.pi, out=self._last_output_phases)

        return torch.polar(self._final_mags_buf, final_phases)

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        was_reset = self._resize_buffers_if_needed(frame)

        # On the very first frame, or after a resize, prime the state and pass original frame through
        if self._last_input_phases is None or was_reset:
            # Angle calculation is cheap, safe to do here.
            self._last_input_phases.copy_(torch.angle(frame.data))
            self._last_output_phases.copy_(torch.angle(frame.data))
            return {"spectral_frame_out": frame}

        state_snapshot_to_emit = None
        effective_pitch, effective_formant = 0.0, 0.0
        with self._lock:
            ui_update_needed = False
            pitch_socket_val = input_data.get("pitch_shift_st")
            formant_socket_val = input_data.get("formant_shift_st")

            effective_pitch = float(pitch_socket_val) if pitch_socket_val is not None else self._pitch_shift_st
            effective_formant = float(formant_socket_val) if formant_socket_val is not None else self._formant_shift_st

            if self._pitch_shift_st != effective_pitch:
                self._pitch_shift_st = effective_pitch
                ui_update_needed = True
            if self._formant_shift_st != effective_formant:
                self._formant_shift_st = effective_formant
                ui_update_needed = True
            if ui_update_needed:
                state_snapshot_to_emit = self._get_current_state_snapshot_locked()

        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

        pitch_ratio = 2 ** (effective_pitch / 12.0)
        formant_ratio = 2 ** (effective_formant / 12.0)
        f0_hz = input_data.get("f0_hz_in")

        shifted_fft_data = self._process_frame_harmonic(frame, f0_hz, pitch_ratio, formant_ratio)

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
        self.set_pitch_shift_st(data.get("pitch_shift_st", 0.0))
        self.set_formant_shift_st(data.get("formant_shift_st", 0.0))
