# additional_plugins/formant_shifter_advanced.py

import torch
import torch.nn.functional as F
import threading
import logging
from typing import Dict, Optional

from PySide6.QtCore import Slot

from node_system import Node
from constants import SpectralFrame, DEFAULT_COMPLEX_DTYPE
from ui_elements import ParameterNodeItem, NodeStateEmitter

logger = logging.getLogger(__name__)
EPSILON = 1e-12


# --- UI Class (Unchanged) ---
class FormantShifterAdvancedNodeItem(ParameterNodeItem):
    """UI for the Advanced Formant Shifter node."""

    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "FormantShifterAdvancedNode"):
        parameters = [
            {
                "key": "formant_shift_st",
                "name": "Formant Shift",
                "min": -24.0,
                "max": 24.0,
                "format": "{:+.1f} st",
                "is_log": False,
            },
            {
                "key": "cepstral_cutoff",
                "name": "Envelope Smoothness",
                "min": 10,
                "max": 80,
                "format": "{:d}",
                "is_log": False,
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# --- Logic Class (CORRECTED) ---
class FormantShifterAdvancedNode(Node):
    NODE_TYPE = "Formant Shifter (Advanced)"
    UI_CLASS = FormantShifterAdvancedNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "High-quality formant shifting using cepstral analysis and parameter smoothing."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()

        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("formant_shift_st", data_type=float)
        self.add_input("cepstral_cutoff", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        self._lock = threading.Lock()
        self._formant_shift_st: float = 0.0
        self._cepstral_cutoff: int = 40
        self._last_formant_ratio: float = 1.0

    def start(self):
        with self._lock:
            self._last_formant_ratio = 1.0

    @Slot(float)
    def set_formant_shift_st(self, value: float):
        self._update_parameter("_formant_shift_st", value)

    @Slot(float)
    def set_cepstral_cutoff(self, value: float):
        self._update_parameter("_cepstral_cutoff", int(value))

    def _update_parameter(self, attr_name: str, value):
        state_to_emit = None
        with self._lock:
            if getattr(self, attr_name) != value:
                setattr(self, attr_name, value)
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {
            "formant_shift_st": self._formant_shift_st,
            "cepstral_cutoff": self._cepstral_cutoff,
        }

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    def _get_spectral_envelope_cepstral(self, magnitudes: torch.Tensor, cutoff: int) -> torch.Tensor:
        log_mags = torch.log(magnitudes + EPSILON)
        cepstrum = torch.fft.rfft(log_mags, dim=-1)
        cepstrum[:, cutoff:] = 0
        log_envelope = torch.fft.irfft(cepstrum, n=magnitudes.shape[-1], dim=-1)
        envelope = torch.exp(log_envelope)
        return envelope

    def _resample_magnitudes(self, magnitudes: torch.Tensor, ratio: float) -> torch.Tensor:
        num_channels, num_bins = magnitudes.shape
        if abs(ratio - 1.0) < 1e-6:
            return magnitudes
        resampled = F.interpolate(
            magnitudes.unsqueeze(0), scale_factor=ratio, mode="linear", align_corners=False, recompute_scale_factor=True
        ).squeeze(0)
        diff = num_bins - resampled.shape[1]
        if diff > 0:
            return F.pad(resampled, (0, diff))
        return resampled[:, :num_bins]

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        state_to_emit = None
        with self._lock:
            ui_update_needed = False
            formant_socket = input_data.get("formant_shift_st")
            effective_formant_shift = float(formant_socket) if formant_socket is not None else self._formant_shift_st
            if self._formant_shift_st != effective_formant_shift:
                self._formant_shift_st = effective_formant_shift
                ui_update_needed = True

            cutoff_socket = input_data.get("cepstral_cutoff")
            effective_cutoff = int(cutoff_socket) if cutoff_socket is not None else self._cepstral_cutoff
            if self._cepstral_cutoff != effective_cutoff:
                self._cepstral_cutoff = effective_cutoff
                ui_update_needed = True

            if ui_update_needed:
                state_to_emit = self._get_current_state_snapshot_locked()

            formant_ratio = 2 ** (effective_formant_shift / 12.0)
            cutoff = effective_cutoff

            ramp = torch.linspace(
                self._last_formant_ratio, formant_ratio, steps=frame.data.shape[1], device=frame.data.device
            )
            self._last_formant_ratio = formant_ratio

        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

        if abs(formant_ratio - 1.0) < 1e-6 and abs(ramp[0] - 1.0) < 1e-6:
            return {"spectral_frame_out": frame}

        input_magnitudes = torch.abs(frame.data)
        input_phases = torch.angle(frame.data)

        original_envelope = self._get_spectral_envelope_cepstral(input_magnitudes, cutoff)
        shifted_envelope = self._resample_magnitudes(original_envelope, formant_ratio)
        correction_factor = shifted_envelope / (original_envelope + EPSILON)
        smoothed_correction = 1.0 + (correction_factor - 1.0) * ramp
        final_magnitudes = input_magnitudes * smoothed_correction
        shifted_fft_data = torch.polar(final_magnitudes, input_phases)

        output_frame = SpectralFrame(
            data=shifted_fft_data.to(DEFAULT_COMPLEX_DTYPE),
            fft_size=frame.fft_size,
            hop_size=frame.hop_size,
            window_size=frame.window_size,
            sample_rate=frame.sample_rate,
            analysis_window=frame.analysis_window)

        return {"spectral_frame_out": output_frame}

    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()

    def deserialize_extra(self, data: dict):
        self.set_formant_shift_st(data.get("formant_shift_st", 0.0))
        self.set_cepstral_cutoff(data.get("cepstral_cutoff", 40))
