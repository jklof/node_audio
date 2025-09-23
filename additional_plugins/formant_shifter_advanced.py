# additional_plugins/formant_shifter_advanced.py

import torch
import torch.nn.functional as F
import threading
import logging
from typing import Dict, Optional, Tuple
import math

from PySide6.QtCore import Slot

from node_system import Node
from constants import SpectralFrame, DEFAULT_COMPLEX_DTYPE
from ui_elements import ParameterNodeItem

logger = logging.getLogger(__name__)
EPSILON = 1e-12


# --- UI Class (Unchanged) ---
class FormantShifterAdvancedNodeItem(ParameterNodeItem):
    """UI for the Advanced Formant Shifter node."""

    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "FormantShifterAdvancedNode"):
        parameters = [
            {"key": "formant_shift_st", "name": "Formant Shift", "min": -24.0, "max": 24.0, "format": "{:+.1f} st"},
            {"key": "cepstral_cutoff", "name": "Envelope Smoothness", "min": 10, "max": 80, "format": "{:d}"},
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# --- Logic Class (ULTRA-OPTIMIZED) ---
class FormantShifterAdvancedNode(Node):
    NODE_TYPE = "Formant Shifter (Advanced)"
    UI_CLASS = FormantShifterAdvancedNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "Ultra high-performance formant shifting using cepstral analysis and parameter smoothing."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("formant_shift_st", data_type=float)
        self.add_input("cepstral_cutoff", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        self._lock = threading.Lock()
        self._formant_shift_st: float = 0.0
        self._cepstral_cutoff: int = 40
        self._last_formant_ratio: float = 1.0

        # --- Performance Optimization: State and Buffer Management ---
        self._last_frame_params: Tuple[int, int, int] = (0, 0, 0)  # (fft_size, num_channels, num_bins)

        # --- Pre-allocated scratch buffers ---
        self._input_mags_buf: Optional[torch.Tensor] = None
        self._input_phases_buf: Optional[torch.Tensor] = None
        self._log_mags_buf: Optional[torch.Tensor] = None
        self._cepstrum_buf: Optional[torch.Tensor] = None
        self._log_envelope_buf: Optional[torch.Tensor] = None
        self._original_envelope_buf: Optional[torch.Tensor] = None
        self._shifted_envelope_buf: Optional[torch.Tensor] = None
        self._correction_factor_buf: Optional[torch.Tensor] = None
        self._final_mags_buf: Optional[torch.Tensor] = None
        self._shifted_fft_buf: Optional[torch.Tensor] = None
        self._ramp_buf: Optional[torch.Tensor] = None

        # --- Additional optimization buffers ---
        self._temp_interpolation_buf: Optional[torch.Tensor] = None  # For F.interpolate input reshaping

        # --- Precomputed constants ---
        self._device: Optional[torch.device] = None
        self._dtype: torch.dtype = torch.float32
        self._epsilon_tensor: Optional[torch.Tensor] = None

        # --- Caching for expensive operations ---
        self._last_cutoff: int = -1
        self._cached_cutoff_mask: Optional[torch.Tensor] = None

        # --- Early exit optimization ---
        self._bypass_threshold: float = 1e-6
        self._last_was_bypassed: bool = False

    def _resize_buffers_if_needed(self, frame: SpectralFrame) -> bool:
        """
        Optimized buffer allocation with device/dtype awareness.
        Returns True if a reset occurred, False otherwise.
        """
        num_channels, num_bins = frame.data.shape
        current_params = (frame.fft_size, num_channels, num_bins)
        device = frame.data.device

        # Check if we need to reallocate
        needs_realloc = self._last_frame_params != current_params or self._device != device

        if not needs_realloc:
            return False

        logger.info(f"[{self.name}] Reallocating buffers: {current_params}, device: {device}")

        # Store device and dtype info
        self._device = device
        self._dtype = torch.float32

        # Buffer shapes
        real_shape = (num_channels, num_bins)
        complex_shape = (num_channels, num_bins)
        # FFT output shape is different (rfft produces num_bins//2 + 1 frequency bins)
        fft_complex_shape = (num_channels, num_bins // 2 + 1)

        # Allocate all buffers on the correct device with optimal memory layout
        tensor_kwargs = {"device": device, "dtype": self._dtype}
        complex_kwargs = {"device": device, "dtype": DEFAULT_COMPLEX_DTYPE}

        # Real buffers with contiguous memory layout
        self._input_mags_buf = torch.empty(real_shape, **tensor_kwargs).contiguous()
        self._input_phases_buf = torch.empty(real_shape, **tensor_kwargs).contiguous()
        self._log_mags_buf = torch.empty(real_shape, **tensor_kwargs).contiguous()
        self._log_envelope_buf = torch.empty(real_shape, **tensor_kwargs).contiguous()
        self._original_envelope_buf = torch.empty(real_shape, **tensor_kwargs).contiguous()
        self._shifted_envelope_buf = torch.empty(real_shape, **tensor_kwargs).contiguous()
        self._correction_factor_buf = torch.empty(real_shape, **tensor_kwargs).contiguous()
        self._final_mags_buf = torch.empty(real_shape, **tensor_kwargs).contiguous()

        # Complex buffers (different sizes for FFT vs output)
        self._cepstrum_buf = torch.empty(fft_complex_shape, **complex_kwargs).contiguous()  # FFT output size
        self._shifted_fft_buf = torch.empty(complex_shape, **complex_kwargs).contiguous()  # Full spectrum size

        # Specialized buffers
        self._ramp_buf = torch.empty(num_bins, **tensor_kwargs).contiguous()
        self._temp_interpolation_buf = torch.empty((1, num_channels, num_bins), **tensor_kwargs).contiguous()

        # Precomputed constants
        self._epsilon_tensor = torch.tensor(EPSILON, device=device, dtype=self._dtype)

        # Reset cached values
        self._last_cutoff = -1
        self._cached_cutoff_mask = None

        self._last_frame_params = current_params
        self.start()  # Reset processing state
        return True

    def start(self):
        with self._lock:
            self._last_formant_ratio = 1.0
            self._last_was_bypassed = False

    @Slot(float)
    def set_formant_shift_st(self, value: float):
        self._update_parameter("_formant_shift_st", value)

    @Slot(float)
    def set_cepstral_cutoff(self, value: float):
        self._update_parameter("_cepstral_cutoff", int(value))

    def _update_parameter(self, attr_name: str, value):
        state_to_emit = None
        with self._lock:
            old_value = getattr(self, attr_name)
            if old_value != value:
                setattr(self, attr_name, value)
                state_to_emit = self._get_current_state_snapshot_locked()

                # Invalidate cutoff cache if needed
                if attr_name == "_cepstral_cutoff":
                    self._last_cutoff = -1

        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {"formant_shift_st": self._formant_shift_st, "cepstral_cutoff": self._cepstral_cutoff}

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    def _get_spectral_envelope_cepstral(self, magnitudes: torch.Tensor, cutoff: int):
        """Ultra-optimized cepstral envelope extraction with caching."""
        # Step 1: Log magnitudes (in-place)
        torch.add(magnitudes, self._epsilon_tensor, out=self._log_mags_buf)
        torch.log_(self._log_mags_buf)

        # Step 2: Forward FFT
        torch.fft.rfft(self._log_mags_buf, dim=-1, out=self._cepstrum_buf)

        # Step 3: Cepstral liftering with cached mask
        cepstrum_bins = self._cepstrum_buf.shape[-1]
        if cutoff < cepstrum_bins:
            if self._last_cutoff != cutoff or self._cached_cutoff_mask is None:
                # Create cutoff mask once and cache it
                self._cached_cutoff_mask = torch.arange(cepstrum_bins, device=self._device) >= cutoff
                self._last_cutoff = cutoff

            # Apply cutoff mask
            self._cepstrum_buf[..., self._cached_cutoff_mask] = 0

        # Step 4: Inverse FFT
        torch.fft.irfft(self._cepstrum_buf, n=magnitudes.shape[-1], dim=-1, out=self._log_envelope_buf)

        # Step 5: Exponential (in-place into original_envelope_buf)
        torch.exp(self._log_envelope_buf, out=self._original_envelope_buf)

    def _resample_magnitudes_optimized(self, magnitudes: torch.Tensor, ratio: float):
        """Ultra-optimized magnitude resampling with minimal allocations."""
        if abs(ratio - 1.0) < self._bypass_threshold:
            self._shifted_envelope_buf.copy_(magnitudes)
            return

        # Prepare input for F.interpolate (reuse temp buffer)
        self._temp_interpolation_buf.copy_(magnitudes.unsqueeze(0))

        # Interpolate
        resampled = F.interpolate(
            self._temp_interpolation_buf,
            scale_factor=ratio,
            mode="linear",
            align_corners=False,
            recompute_scale_factor=True,
        )

        # Handle size differences efficiently
        resampled_squeezed = resampled.squeeze(0)
        target_bins = self._shifted_envelope_buf.shape[1]
        source_bins = resampled_squeezed.shape[1]

        if source_bins >= target_bins:
            # Truncate
            self._shifted_envelope_buf.copy_(resampled_squeezed[:, :target_bins])
        else:
            # Pad with zeros
            self._shifted_envelope_buf.zero_()
            self._shifted_envelope_buf[:, :source_bins].copy_(resampled_squeezed)

    def _compute_ramp_vectorized(self, start: float, end: float):
        """Compute smooth parameter ramp across frequency bins."""
        if abs(start - end) < self._bypass_threshold:
            self._ramp_buf.fill_(end)
        else:
            torch.linspace(start, end, steps=len(self._ramp_buf), out=self._ramp_buf)

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        # Early buffer check
        if self._resize_buffers_if_needed(frame):
            return {"spectral_frame_out": frame}  # Pass through on first frame

        # Lock and read parameters
        state_to_emit = None
        with self._lock:
            ui_update_needed = False

            # Handle socket inputs with fallback to internal state
            formant_socket = input_data.get("formant_shift_st")
            effective_formant_shift = float(formant_socket) if formant_socket is not None else self._formant_shift_st
            if abs(self._formant_shift_st - effective_formant_shift) > 1e-10:
                self._formant_shift_st = effective_formant_shift
                ui_update_needed = True

            cutoff_socket = input_data.get("cepstral_cutoff")
            effective_cutoff = int(cutoff_socket) if cutoff_socket is not None else self._cepstral_cutoff
            if self._cepstral_cutoff != effective_cutoff:
                self._cepstral_cutoff = effective_cutoff
                self._last_cutoff = -1  # Invalidate cache
                ui_update_needed = True

            if ui_update_needed:
                state_to_emit = self._get_current_state_snapshot_locked()

            # Compute processing parameters
            formant_ratio = 2.0 ** (effective_formant_shift / 12.0)
            cutoff = effective_cutoff
            ramp_start = self._last_formant_ratio
            ramp_end = formant_ratio
            self._last_formant_ratio = formant_ratio

        if state_to_emit:
            self.ui_update_callback(state_to_emit)

        # Early exit: check if processing can be bypassed
        bypass_condition = (
            abs(ramp_end - 1.0) < self._bypass_threshold and abs(ramp_start - 1.0) < self._bypass_threshold
        )

        if bypass_condition:
            if not self._last_was_bypassed:
                logger.debug(f"[{self.name}] Bypassing processing (no shift needed)")
                self._last_was_bypassed = True
            return {"spectral_frame_out": frame}

        self._last_was_bypassed = False

        # === CORE DSP PROCESSING (ULTRA-OPTIMIZED) ===

        # Extract magnitude and phase (optimized)
        torch.abs(frame.data, out=self._input_mags_buf)
        torch.angle(frame.data, out=self._input_phases_buf)

        # Compute spectral envelope
        self._get_spectral_envelope_cepstral(self._input_mags_buf, cutoff)

        # Resample envelope according to formant ratio
        self._resample_magnitudes_optimized(self._original_envelope_buf, formant_ratio)

        # Compute correction factor: shifted / original
        torch.add(self._original_envelope_buf, self._epsilon_tensor, out=self._correction_factor_buf)
        torch.div(self._shifted_envelope_buf, self._correction_factor_buf, out=self._correction_factor_buf)

        # Apply smooth ramp to correction factor
        self._compute_ramp_vectorized(ramp_start, ramp_end)

        # Smooth correction: 1.0 + (correction - 1.0) * ramp
        torch.sub(self._correction_factor_buf, 1.0, out=self._correction_factor_buf)
        torch.mul(self._correction_factor_buf, self._ramp_buf.unsqueeze(0), out=self._correction_factor_buf)
        torch.add(self._correction_factor_buf, 1.0, out=self._correction_factor_buf)

        # Apply correction to magnitudes
        torch.mul(self._input_mags_buf, self._correction_factor_buf, out=self._final_mags_buf)

        # Reconstruct complex spectrum using optimized torch.polar
        torch.polar(self._final_mags_buf, self._input_phases_buf, out=self._shifted_fft_buf)

        # Create output frame (minimal allocation)
        output_frame = SpectralFrame(
            data=self._shifted_fft_buf,
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
        self.set_cepstral_cutoff(data.get("cepstral_cutoff", 40))
