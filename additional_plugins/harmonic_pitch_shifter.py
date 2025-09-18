# additional_plugins/harmonic_pitch_shifter.py

import torch
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
        return {
            "pitch_shift_st": self._pitch_shift_st,
            "formant_shift_st": self._formant_shift_st,
        }

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    def _resample_magnitudes(self, magnitudes: torch.Tensor, ratio: float) -> torch.Tensor:
        num_channels, num_bins = magnitudes.shape
        if abs(ratio - 1.0) < 1e-6:
            return magnitudes
        original_indices = torch.arange(num_bins, dtype=torch.float32, device=magnitudes.device)
        resampled_indices = original_indices / ratio
        lower_indices = torch.floor(resampled_indices).long()
        upper_indices = lower_indices + 1
        weights = (resampled_indices - lower_indices).unsqueeze(0)
        lower_indices.clamp_(0, num_bins - 1)
        upper_indices.clamp_(0, num_bins - 1)
        mags_at_lower = torch.gather(magnitudes, 1, lower_indices.expand(num_channels, -1))
        mags_at_upper = torch.gather(magnitudes, 1, upper_indices.expand(num_channels, -1))
        return mags_at_lower * (1.0 - weights) + mags_at_upper * weights

    def _get_spectral_envelope(self, magnitudes: torch.Tensor, kernel_size: int = 31) -> torch.Tensor:
        if kernel_size % 2 == 0:
            kernel_size += 1
        padded_mags = torch.nn.functional.pad(
            magnitudes.unsqueeze(1), (kernel_size // 2, kernel_size // 2), mode="reflect"
        )
        envelope = torch.nn.functional.avg_pool1d(padded_mags, kernel_size=kernel_size, stride=1, padding=0)
        return envelope.squeeze(1)

    def _process_frame_harmonic(
        self, frame: SpectralFrame, f0_hz: float, pitch_ratio: float, formant_ratio: float
    ) -> torch.Tensor:
        freqs = torch.fft.rfftfreq(frame.fft_size, d=1.0 / frame.sample_rate, device=frame.data.device)
        input_magnitudes = torch.abs(frame.data)
        current_input_phases = torch.angle(frame.data)

        # --- CORRECTED LOGIC ---
        # 1. Get the source signal (harmonics) and the filter (envelope) from the ORIGINAL input
        original_envelope = self._get_spectral_envelope(input_magnitudes)
        source_magnitudes = input_magnitudes / (original_envelope + EPSILON)

        # 2. Shift source and filter INDEPENDENTLY
        shifted_source_magnitudes = self._resample_magnitudes(source_magnitudes, pitch_ratio)
        shifted_envelope = self._resample_magnitudes(original_envelope, formant_ratio)

        # 3. Recombine them to get the final magnitudes
        final_magnitudes = shifted_source_magnitudes * shifted_envelope

        # 4. Determine phase logic based on F0
        is_voiced = f0_hz is not None and f0_hz > MIN_VOICED_F0

        if is_voiced:
            bin_width_hz = frame.sample_rate / frame.fft_size
            harmonic_indices = freqs / f0_hz
            deviation = torch.abs(harmonic_indices - torch.round(harmonic_indices))
            harmonic_mask = deviation < (bin_width_hz / (f0_hz + EPSILON)) * 2.5

            expected_phase_advance = 2 * np.pi * frame.hop_size * freqs
            phase_deviation = current_input_phases - self._last_input_phases - expected_phase_advance
            phase_deviation = torch.fmod(phase_deviation + np.pi, 2 * np.pi) - np.pi
            true_freq_hz = (expected_phase_advance + phase_deviation) * frame.sample_rate / (2 * np.pi * frame.hop_size)
            shifted_freq_hz = true_freq_hz * pitch_ratio
            new_phase_advance = shifted_freq_hz * (2 * np.pi * frame.hop_size / frame.sample_rate)
            propagated_phases = self._last_output_phases + new_phase_advance

            final_phases = torch.where(harmonic_mask, propagated_phases, current_input_phases)
        else:
            final_phases = current_input_phases

        self._last_input_phases = current_input_phases
        self._last_output_phases = torch.fmod(final_phases, 2 * np.pi)

        return torch.polar(final_magnitudes, final_phases)

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        num_channels, _ = frame.data.shape
        current_frame_params = (frame.fft_size, frame.hop_size, frame.sample_rate, num_channels)

        state_snapshot_to_emit = None
        with self._lock:
            if self._last_frame_params != current_frame_params:
                self._reset_state_locked()
                self._last_frame_params = current_frame_params
                logger.info(f"[{self.name}] Frame format changed. State has been reset.")

            if self._last_input_phases is None:
                self._last_input_phases = torch.angle(frame.data)
                self._last_output_phases = torch.angle(frame.data)
                return {"spectral_frame_out": frame}

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

            pitch_ratio = 2 ** (effective_pitch / 12.0)
            formant_ratio = 2 ** (effective_formant / 12.0)
            f0_hz = input_data.get("f0_hz_in")

            shifted_fft_data = self._process_frame_harmonic(frame, f0_hz, pitch_ratio, formant_ratio)

        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

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
