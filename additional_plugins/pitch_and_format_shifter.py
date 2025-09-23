# additional_plugins/pitch_and_formant_shifter.py

import torch
import numpy as np
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


# --- UI Class (Inherits from the generic ParameterNodeItem) ---
class PitchAndFormantShifterNodeItem(ParameterNodeItem):
    """UI for the new Pitch & Formant Shifter node."""

    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "PitchAndFormantShifterNode"):
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
            {
                "key": "phase_jitter",
                "name": "Phase Jitter",
                "min": 0.0,
                "max": 1.0,
                "format": "{:.0%}",
                "is_log": False,
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# --- Logic Class ---
class PitchAndFormantShifterNode(Node):
    NODE_TYPE = "Pitch & Formant Shifter"
    UI_CLASS = PitchAndFormantShifterNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "Changes pitch and formants independently using a phase vocoder."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)

        # --- Sockets ---
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("pitch_shift_st", data_type=float)
        self.add_input("formant_shift_st", data_type=float)
        self.add_input("phase_jitter", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        self._lock = threading.Lock()
        # --- Internal State ---
        self._pitch_shift_st: float = 0.0
        self._formant_shift_st: float = 0.0
        self._phase_jitter: float = 0.1

        # --- Phase Vocoder State ---
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
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_formant_shift_st(self, value: float):
        state_to_emit = None
        with self._lock:
            if self._formant_shift_st != value:
                self._formant_shift_st = float(value)
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_phase_jitter(self, value: float):
        state_to_emit = None
        with self._lock:
            if self._phase_jitter != value:
                self._phase_jitter = float(value)
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {
            "pitch_shift_st": self._pitch_shift_st,
            "formant_shift_st": self._formant_shift_st,
            "phase_jitter": self._phase_jitter,
        }

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    def _resample_magnitudes(self, magnitudes: torch.Tensor, ratio: float) -> torch.Tensor:
        num_channels, num_bins = magnitudes.shape
        if ratio == 1.0:
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

    def _get_spectral_envelope(self, magnitudes: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
        if kernel_size % 2 == 0:
            kernel_size += 1
        padded_mags = torch.nn.functional.pad(
            magnitudes.unsqueeze(1), (kernel_size // 2, kernel_size // 2), mode="reflect"
        )
        envelope = torch.nn.functional.avg_pool1d(padded_mags, kernel_size=kernel_size, stride=1, padding=0)
        return envelope.squeeze(1)

    def _process_frame(
        self, frame: SpectralFrame, pitch_ratio: float, formant_ratio: float, phase_jitter: float
    ) -> torch.Tensor:
        freqs = torch.fft.rfftfreq(frame.fft_size, d=1.0 / frame.sample_rate, device=frame.data.device)
        expected_phase_advance = 2 * np.pi * frame.hop_size * freqs
        current_input_phases = torch.angle(frame.data)
        phase_deviation = current_input_phases - self._last_input_phases - expected_phase_advance
        phase_deviation = torch.fmod(phase_deviation + np.pi, 2 * np.pi) - np.pi

        true_freq_hz = (expected_phase_advance + phase_deviation) * frame.sample_rate / (2 * np.pi * frame.hop_size)

        input_magnitudes = torch.abs(frame.data)
        shifted_magnitudes = self._resample_magnitudes(input_magnitudes, pitch_ratio)
        shifted_freq_hz = true_freq_hz * pitch_ratio

        if formant_ratio != 1.0:
            spectral_envelope = self._get_spectral_envelope(input_magnitudes)
            shifted_envelope = self._resample_magnitudes(spectral_envelope, formant_ratio)
            final_magnitudes = shifted_magnitudes * (shifted_envelope / (spectral_envelope + EPSILON))
        else:
            final_magnitudes = shifted_magnitudes

        new_phase_advance = shifted_freq_hz * (2 * np.pi * frame.hop_size / frame.sample_rate)
        new_output_phases = self._last_output_phases + new_phase_advance

        if phase_jitter > 0:
            random_phase_offset = (torch.rand_like(new_output_phases) * 2.0 - 1.0) * np.pi * phase_jitter
            final_output_phases = new_output_phases + random_phase_offset
        else:
            final_output_phases = new_output_phases

        self._last_input_phases = current_input_phases
        self._last_output_phases = torch.fmod(new_output_phases, 2 * np.pi)

        return torch.polar(final_magnitudes, final_output_phases)

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
            pitch_socket = input_data.get("pitch_shift_st")
            if pitch_socket is not None and self._pitch_shift_st != float(pitch_socket):
                self._pitch_shift_st = float(pitch_socket)
                ui_update_needed = True

            formant_socket = input_data.get("formant_shift_st")
            if formant_socket is not None and self._formant_shift_st != float(formant_socket):
                self._formant_shift_st = float(formant_socket)
                ui_update_needed = True

            jitter_socket = input_data.get("phase_jitter")
            if jitter_socket is not None and self._phase_jitter != float(jitter_socket):
                self._phase_jitter = float(jitter_socket)
                ui_update_needed = True

            if ui_update_needed:
                state_snapshot_to_emit = self._get_current_state_snapshot_locked()

            pitch_ratio = 2 ** (self._pitch_shift_st / 12.0)
            formant_ratio = 2 ** (self._formant_shift_st / 12.0)
            phase_jitter = self._phase_jitter

            shifted_fft_data = self._process_frame(frame, pitch_ratio, formant_ratio, phase_jitter)

        if state_snapshot_to_emit:
            self.ui_update_callback(state_snapshot_to_emit)

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
        self.set_phase_jitter(data.get("phase_jitter", 0.1))
