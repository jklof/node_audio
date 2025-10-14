import torch
import numpy as np
import threading
import logging
from typing import Dict, Optional
from collections import deque

# --- Node System Imports ---
from node_system import Node
from constants import SpectralFrame, DEFAULT_DTYPE, DEFAULT_COMPLEX_DTYPE
from ui_elements import ParameterNodeItem, NodeItem, NODE_CONTENT_PADDING
from node_helpers import with_parameters, Parameter

# --- Qt Imports ---
from PySide6.QtCore import Qt, Signal, Slot, QObject, QSignalBlocker
from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout

# --- Logging ---
logger = logging.getLogger(__name__)

# A small value to prevent log(0) errors or division by zero
EPSILON = 1e-9


# ==============================================================================
# SPECTRAL SHIMMER
# ==============================================================================


# ==============================================================================
# 2. Custom UI Class (SpectralShimmerNodeItem)
# ==============================================================================
class SpectralShimmerNodeItem(ParameterNodeItem):
    """UI for SpectralShimmerNode using ParameterNodeItem base class."""

    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "SpectralShimmerNode"):
        # Define the parameters for this node
        parameters = [
            {
                "key": "pitch_shift",
                "name": "Pitch Shift",
                "min": -12.0,
                "max": 24.0,
                "format": "{:+.1f} st",
                "is_log": False,
            },
            {
                "key": "feedback",
                "name": "Feedback",
                "min": 0.0,
                "max": 1.0,
                "format": "{:.0%}",
                "is_log": False,
            },
            {
                "key": "mix",
                "name": "Mix",
                "min": 0.0,
                "max": 1.0,
                "format": "{:.0%}",
                "is_log": False,
            },
        ]

        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 3. Node Logic Class (SpectralShimmerNode)
# ==============================================================================
@with_parameters
class SpectralShimmerNode(Node):
    NODE_TYPE = "Spectral Shimmer"
    UI_CLASS = SpectralShimmerNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "A pitch-shifting feedback effect for creating ethereal textures."

    # --- Declarative managed parameters ---
    pitch_shift = Parameter(default=12.0)
    feedback = Parameter(default=0.75, clip=(0.0, 1.0))
    mix = Parameter(default=0.5, clip=(0.0, 1.0))

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self._init_parameters()

        # --- Setup Sockets (matching parameter keys) ---
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("pitch_shift", data_type=float)
        self.add_input("feedback", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        # --- DSP Buffers & State ---
        self._shimmer_buffer: Optional[torch.Tensor] = None
        self._last_frame_params: tuple = (0, 0, 0)

    def _get_state_snapshot_locked(self) -> dict:
        return self._get_parameters_state()

    def serialize_extra(self) -> dict:
        return self._serialize_parameters()

    def deserialize_extra(self, data: dict):
        self._deserialize_parameters(data)

    def _pitch_shift_frame(self, frame_data: torch.Tensor, ratio: float) -> torch.Tensor:
        """Performs pitch shifting on a spectral frame using linear interpolation."""
        num_channels, num_bins = frame_data.shape
        if ratio == 1.0:
            return frame_data

        # Create tensors for original and shifted indices
        original_indices = torch.arange(num_bins, dtype=torch.float32)
        shifted_indices = original_indices / ratio

        # Get magnitudes and phases
        magnitudes = torch.abs(frame_data)
        phases = torch.angle(frame_data)

        # --- Linear Interpolation of Magnitudes ---
        lower_indices = torch.floor(shifted_indices).long()
        upper_indices = lower_indices + 1
        weights = (shifted_indices - lower_indices).unsqueeze(0)  # Shape (1, bins) for broadcasting

        # Clamp indices to the valid range [0, num_bins - 1]
        lower_indices.clamp_(0, num_bins - 1)
        upper_indices.clamp_(0, num_bins - 1)

        # Gather magnitudes at the corresponding indices for each channel
        mags_at_lower = torch.gather(magnitudes, 1, lower_indices.expand(num_channels, -1))
        mags_at_upper = torch.gather(magnitudes, 1, upper_indices.expand(num_channels, -1))

        # Perform linear interpolation
        shifted_magnitudes = mags_at_lower * (1.0 - weights) + mags_at_upper * weights

        # Reconstruct the complex tensor with original phases (vocoder-style shift)
        shifted_frame = torch.polar(shifted_magnitudes, phases)
        return shifted_frame.to(DEFAULT_COMPLEX_DTYPE)

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        # Update parameters from input sockets and notify UI if changed.
        self._update_parameters_from_sockets(input_data)

        with self._lock:
            # Read the managed parameters for this processing block
            pitch_shift_st = self._pitch_shift
            feedback_val = self._feedback
            mix_val = self._mix

            num_channels, num_bins = frame.data.shape
            current_frame_params = (frame.fft_size, frame.hop_size, num_channels)
            if self._last_frame_params != current_frame_params:
                self._last_frame_params = current_frame_params
                self._shimmer_buffer = torch.zeros((num_channels, num_bins), dtype=DEFAULT_COMPLEX_DTYPE)

            pitch_ratio = 2 ** (pitch_shift_st / 12.0)
            shifted_tail = self._pitch_shift_frame(self._shimmer_buffer, pitch_ratio)
            wet_signal = shifted_tail * feedback_val
            self._shimmer_buffer = frame.data + wet_signal
            output_fft = (frame.data * (1.0 - mix_val)) + (wet_signal * mix_val)

        output_frame = SpectralFrame(
            data=output_fft,
            fft_size=frame.fft_size,
            hop_size=frame.hop_size,
            window_size=frame.window_size,
            sample_rate=frame.sample_rate,
            analysis_window=frame.analysis_window,
        )
        return {"spectral_frame_out": output_frame}

    def start(self):
        with self._lock:
            self._shimmer_buffer = None
            self._last_frame_params = (0, 0, 0)


# ==============================================================================
# SPECTRAL MODULATOR
# ==============================================================================


# ==============================================================================
# 1. Custom UI Class (SpectralModulatorNodeItem)
# ==============================================================================
class SpectralModulatorNodeItem(ParameterNodeItem):
    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "SpectralModulatorNode"):
        parameters = [
            {
                "key": "rate",
                "name": "Rate",
                "min": 0.1,
                "max": 10.0,
                "format": "{:.2f} Hz",
            },
            {
                "key": "depth",
                "name": "Depth",
                "min": 0.0,
                "max": 20.0,
                "format": "{:.1f} ms",
            },
            {
                "key": "mix",
                "name": "Mix",
                "min": 0.0,
                "max": 1.0,
                "format": "{:.0%}",
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)

    def _on_state_updated_from_logic(self, state: dict):
        super()._on_state_updated_from_logic(state)
        # Custom logic to disable rate control if mod_in is connected
        is_mod_ext = self.node_logic.inputs["mod_in"].connections
        self._controls["rate"]["widget"].setEnabled(not is_mod_ext)
        label_text = self._controls["rate"]["label"].text()
        if is_mod_ext and "(ext)" not in label_text:
            self._controls["rate"]["label"].setText(label_text + " (ext)")
        elif not is_mod_ext and "(ext)" in label_text:
            self._controls["rate"]["label"].setText(label_text.replace(" (ext)", ""))


# ==============================================================================
# 3. Node Logic Class (SpectralModulatorNode)
# ==============================================================================
@with_parameters
class SpectralModulatorNode(Node):
    NODE_TYPE = "Spectral Modulator"
    UI_CLASS = SpectralModulatorNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "Applies phase modulation to a spectral frame, with internal or external LFO."

    # --- Declarative managed parameters ---
    rate = Parameter(default=1.5)
    depth = Parameter(default=5.0)
    mix = Parameter(default=0.5, clip=(0.0, 1.0))

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self._init_parameters()

        # --- Setup Sockets (matching parameter keys) ---
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("mod_in", data_type=float)
        self.add_input("rate", data_type=float)
        self.add_input("depth", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        # --- DSP State ---
        self._lfo_phase: float = 0.0

    def _get_state_snapshot_locked(self) -> dict:
        return self._get_parameters_state()

    def serialize_extra(self) -> dict:
        return self._serialize_parameters()

    def deserialize_extra(self, data: dict):
        self._deserialize_parameters(data)

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        self._update_parameters_from_sockets(input_data)

        with self._lock:
            # Read the managed parameters for this processing block
            rate_hz = self._rate
            depth_ms = self._depth
            mix_val = self._mix

            lfo_value = 0.0
            lfo_mod_input = input_data.get("mod_in")

            if lfo_mod_input is not None:
                lfo_value = float(lfo_mod_input)
            else:
                frame_duration_s = frame.hop_size / frame.sample_rate
                phase_increment = 2 * torch.pi * rate_hz * frame_duration_s
                self._lfo_phase = (self._lfo_phase + phase_increment) % (2 * torch.pi)
                lfo_value = torch.sin(torch.tensor(self._lfo_phase)).item()

            delay_s = (depth_ms / 1000.0) * lfo_value
            freqs = torch.fft.rfftfreq(frame.fft_size, d=1.0 / frame.sample_rate)
            phase_shifts_rad = 2 * torch.pi * freqs * delay_s

            # Create the complex phasor using torch.polar
            phasor = torch.polar(torch.ones_like(phase_shifts_rad), phase_shifts_rad).to(DEFAULT_COMPLEX_DTYPE)

            # Broadcasting will apply the 1D phasor to each channel in the 2D frame data
            wet_signal = frame.data * phasor
            output_fft = (frame.data * (1.0 - mix_val)) + (wet_signal * mix_val)

        output_frame = SpectralFrame(
            data=output_fft,
            fft_size=frame.fft_size,
            hop_size=frame.hop_size,
            window_size=frame.window_size,
            sample_rate=frame.sample_rate,
            analysis_window=frame.analysis_window,
        )
        return {"spectral_frame_out": output_frame}

    def start(self):
        with self._lock:
            self._lfo_phase = 0.0


# ==============================================================================
# SPECTRAL REVERB
# ==============================================================================


# ==============================================================================
# 2. Custom UI Class (SpectralReverbNodeItem)
# ==============================================================================
class SpectralReverbNodeItem(ParameterNodeItem):
    """UI for SpectralReverbNode using ParameterNodeItem base class."""

    NODE_SPECIFIC_WIDTH = 240

    def __init__(self, node_logic: "SpectralReverbNode"):
        # Define the parameters for this node
        parameters = [
            {
                "key": "pre_delay_ms",
                "name": "Pre-delay",
                "min": 0.0,
                "max": 250.0,
                "format": "{:.0f} ms",
                "is_log": False,
            },
            {
                "key": "decay_time",
                "name": "Decay Time",
                "min": 0.1,
                "max": 15.0,
                "format": "{:.1f} s",
                "is_log": False,
            },
            {
                "key": "hf_damp",
                "name": "HF Damp",
                "min": 500.0,
                "max": 20000.0,
                "format": "{:.1f} Hz",
                "is_log": True,
            },
            {
                "key": "lf_damp",
                "name": "LF Damp",
                "min": 20.0,
                "max": 2000.0,
                "format": "{:.1f} Hz",
                "is_log": True,
            },
            {
                "key": "diffusion",
                "name": "Diffusion",
                "min": 0.0,
                "max": 1.0,
                "format": "{:.0%}",
                "is_log": False,
            },
            {
                "key": "width",
                "name": "Stereo Width",
                "min": 0.0,
                "max": 1.0,
                "format": "{:.0%}",
                "is_log": False,
            },
            {
                "key": "mix",
                "name": "Mix",
                "min": 0.0,
                "max": 1.0,
                "format": "{:.0%}",
                "is_log": False,
            },
        ]

        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 3. Node Logic Class (SpectralReverbNode)
# ==============================================================================
@with_parameters
class SpectralReverbNode(Node):
    NODE_TYPE = "Spectral Reverb"
    UI_CLASS = SpectralReverbNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "Algorithmic reverb that applies frequency-dependent decay to spectral frames."

    # --- Declarative managed parameters with on_change hook ---
    pre_delay_ms = Parameter(default=20.0, on_change="_mark_params_dirty")
    decay_time = Parameter(default=2.5, on_change="_mark_params_dirty")
    hf_damp = Parameter(default=4000.0, on_change="_mark_params_dirty")
    lf_damp = Parameter(default=150.0, on_change="_mark_params_dirty")
    diffusion = Parameter(default=1.0, clip=(0.0, 1.0), on_change="_mark_params_dirty")
    width = Parameter(default=1.0, clip=(0.0, 1.0), on_change="_mark_params_dirty")
    mix = Parameter(default=0.5, clip=(0.0, 1.0))

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self._init_parameters()

        # --- Setup Sockets ---
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("pre_delay_ms", data_type=float)
        self.add_input("decay_time", data_type=float)
        self.add_input("hf_damp", data_type=float)
        self.add_input("lf_damp", data_type=float)
        self.add_input("diffusion", data_type=float)
        self.add_input("width", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        # --- DSP Buffers & State (Using torch.Tensor) ---
        self._reverb_fft_buffer: Optional[torch.Tensor] = None
        self._decay_factors: Optional[torch.Tensor] = None
        self._pre_delay_buffer: deque = deque()
        self._pre_delay_frames: int = 0
        self._last_frame_params: tuple = (0, 0)
        self._params_dirty: bool = True

    def _get_state_snapshot_locked(self) -> dict:
        return self._get_parameters_state()

    def serialize_extra(self) -> dict:
        return self._serialize_parameters()

    def deserialize_extra(self, data: dict):
        self._deserialize_parameters(data)

    def _mark_params_dirty(self):
        """
        Callback method for the decorator. Called when a parameter with
        on_change="_mark_params_dirty" is modified. The lock is held.
        """
        self._params_dirty = True

    def _recalculate_params(self, frame: SpectralFrame):
        # Read parameters via their internal names
        t60 = self._decay_time
        hf_damp_hz = self._hf_damp
        lf_damp_hz = self._lf_damp
        diffusion_val = self._diffusion
        width_val = self._width
        pre_delay_ms_val = self._pre_delay_ms

        freqs = torch.fft.rfftfreq(frame.fft_size, d=1.0 / frame.sample_rate)
        num_bins = freqs.shape[0]

        # --- Damping Logic ---
        hf_damp_factor = torch.clamp((hf_damp_hz - freqs) / hf_damp_hz, 0.1, 1.0)
        hf_damp_factor[freqs < hf_damp_hz] = 1.0
        lf_damp_factor = torch.clamp(freqs / lf_damp_hz, 0.1, 1.0)
        lf_damp_factor[freqs > lf_damp_hz] = 1.0
        damped_t60 = t60 * hf_damp_factor * lf_damp_factor

        frames_per_second = frame.sample_rate / frame.hop_size
        magnitudes = 10.0 ** (-3.0 / (damped_t60 * frames_per_second + EPSILON))

        # --- Stereo Width & Diffusion Logic ---
        phase_shift_amount = diffusion_val * torch.pi
        random_phases_1 = (torch.rand(num_bins) * 2 - 1) * phase_shift_amount
        random_phases_2 = (torch.rand(num_bins) * 2 - 1) * phase_shift_amount
        phases_L = random_phases_1
        phases_R = random_phases_1 * (1.0 - width_val) + random_phases_2 * width_val

        decay_L = torch.polar(magnitudes, phases_L).to(DEFAULT_COMPLEX_DTYPE)
        decay_R = torch.polar(magnitudes, phases_R).to(DEFAULT_COMPLEX_DTYPE)
        self._decay_factors = torch.stack([decay_L, decay_R], dim=0)

        # --- Recalculate Pre-delay Buffer ---
        pre_delay_s = pre_delay_ms_val / 1000.0
        frame_duration_s = frame.hop_size / frame.sample_rate
        self._pre_delay_frames = int(round(pre_delay_s / (frame_duration_s + EPSILON)))
        if self._pre_delay_buffer.maxlen != self._pre_delay_frames:
            self._pre_delay_buffer = deque(self._pre_delay_buffer, maxlen=self._pre_delay_frames)

        self._params_dirty = False
        logger.info(f"[{self.name}] Recalculated reverb params: Pre-delay={self._pre_delay_frames} frames.")

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        # The decorator's helper handles socket updates and marks params dirty via on_change
        self._update_parameters_from_sockets(input_data)

        with self._lock:
            # --- Initialize/Reset buffers on format change ---
            num_channels, num_bins = frame.data.shape
            current_frame_params = (frame.fft_size, frame.hop_size)
            if self._last_frame_params != current_frame_params:
                self._last_frame_params = current_frame_params
                self._params_dirty = True
                self._reverb_fft_buffer = torch.zeros((2, num_bins), dtype=DEFAULT_COMPLEX_DTYPE)
                self._pre_delay_buffer.clear()

            if self._params_dirty:
                self._recalculate_params(frame)

            mix_val = self._mix  # Read the mix parameter

            # --- Pre-delay Logic ---
            self._pre_delay_buffer.append(frame.data)
            if len(self._pre_delay_buffer) < self._pre_delay_frames or self._pre_delay_frames == 0:
                delayed_frame_data = torch.zeros_like(frame.data)
            else:
                delayed_frame_data = self._pre_delay_buffer[0]

            # --- Handle Mono Input and ensure stereo processing ---
            dry_signal = frame.data
            if num_channels == 1:
                delayed_frame_data = delayed_frame_data.repeat(2, 1)
                dry_signal = dry_signal.repeat(2, 1)

            # --- Core DSP Algorithm ---
            wet_signal = self._reverb_fft_buffer * self._decay_factors
            self._reverb_fft_buffer = delayed_frame_data + wet_signal
            output_fft = (dry_signal * (1.0 - mix_val)) + (wet_signal * mix_val)

        output_frame = SpectralFrame(
            data=output_fft,
            fft_size=frame.fft_size,
            hop_size=frame.hop_size,
            window_size=frame.window_size,
            sample_rate=frame.sample_rate,
            analysis_window=frame.analysis_window,
        )
        return {"spectral_frame_out": output_frame}

    def start(self):
        with self._lock:
            self._reverb_fft_buffer = None
            self._last_frame_params = (0, 0)
            self._pre_delay_buffer.clear()
            self._params_dirty = True
