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
class SpectralShimmerNode(Node):
    NODE_TYPE = "Spectral Shimmer"
    UI_CLASS = SpectralShimmerNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "A pitch-shifting feedback effect for creating ethereal textures."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)

        # --- Setup Sockets ---
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("pitch_shift", data_type=float)
        self.add_input("feedback", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        # --- Internal State ---
        self._pitch_shift_st: float = 12.0
        self._feedback: float = 0.75
        self._mix: float = 0.5

        # --- DSP Buffers & State ---
        self._shimmer_buffer: Optional[torch.Tensor] = None
        self._last_frame_params: tuple = (0, 0, 0)

    # --- Parameter Setter Slots ---
    @Slot(float)
    def set_pitch_shift(self, value: float):
        state_to_emit = None
        with self._lock:
            if self._pitch_shift_st != value:
                self._pitch_shift_st = value
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_feedback(self, value: float):
        state_to_emit = None
        with self._lock:
            new_feedback = max(0.0, min(float(value), 1.0))
            if self._feedback != new_feedback:
                self._feedback = new_feedback
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_mix(self, value: float):
        state_to_emit = None
        with self._lock:
            new_mix = max(0.0, min(float(value), 1.0))
            if self._mix != new_mix:
                self._mix = new_mix
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def _get_state_snapshot_locked(self) -> Dict:
        return {"pitch_shift": self._pitch_shift_st, "feedback": self._feedback, "mix": self._mix}

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

        state_snapshot_to_emit = None
        with self._lock:
            ui_update_needed = False
            # --- Update state from sockets if connected ---
            pitch_socket = input_data.get("pitch_shift")
            if pitch_socket is not None and self._pitch_shift_st != float(pitch_socket):
                self._pitch_shift_st = float(pitch_socket)
                ui_update_needed = True

            feedback_socket = input_data.get("feedback")
            if feedback_socket is not None:
                new_feedback = max(0.0, min(float(feedback_socket), 1.0))
                if self._feedback != new_feedback:
                    self._feedback = new_feedback
                    ui_update_needed = True

            mix_socket = input_data.get("mix")
            if mix_socket is not None:
                new_mix = max(0.0, min(float(mix_socket), 1.0))
                if self._mix != new_mix:
                    self._mix = new_mix
                    ui_update_needed = True

            if ui_update_needed:
                state_snapshot_to_emit = self._get_state_snapshot_locked()

            num_channels, num_bins = frame.data.shape
            current_frame_params = (frame.fft_size, frame.hop_size, num_channels)
            if self._last_frame_params != current_frame_params:
                self._last_frame_params = current_frame_params
                self._shimmer_buffer = torch.zeros((num_channels, num_bins), dtype=DEFAULT_COMPLEX_DTYPE)

            pitch_ratio = 2 ** (self._pitch_shift_st / 12.0)
            shifted_tail = self._pitch_shift_frame(self._shimmer_buffer, pitch_ratio)
            wet_signal = shifted_tail * self._feedback
            self._shimmer_buffer = frame.data + wet_signal
            output_fft = (frame.data * (1.0 - self._mix)) + (wet_signal * self._mix)

        if state_snapshot_to_emit:
            self.ui_update_callback(state_snapshot_to_emit)

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

    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._pitch_shift_st = data.get("pitch_shift", 12.0)
            self._feedback = data.get("feedback", 0.75)
            self._mix = data.get("mix", 0.5)


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

    def updateFromLogic(self):
        super().updateFromLogic()
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
class SpectralModulatorNode(Node):
    NODE_TYPE = "Spectral Modulator"
    UI_CLASS = SpectralModulatorNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "Applies phase modulation to a spectral frame, with internal or external LFO."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)

        # --- Setup Sockets ---
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("mod_in", data_type=float)
        self.add_input("rate", data_type=float)
        self.add_input("depth", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        # --- Internal State ---
        self._rate_hz: float = 1.5
        self._depth_ms: float = 5.0
        self._mix: float = 0.5

        # --- DSP State ---
        self._lfo_phase: float = 0.0

    # --- Parameter Setter Slots ---
    @Slot(float)
    def set_rate(self, value: float):
        state_to_emit = None
        with self._lock:
            if self._rate_hz != value:
                self._rate_hz = value
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_depth(self, value: float):
        state_to_emit = None
        with self._lock:
            if self._depth_ms != value:
                self._depth_ms = value
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_mix(self, value: float):
        state_to_emit = None
        with self._lock:
            new_mix = max(0.0, min(float(value), 1.0))
            if self._mix != new_mix:
                self._mix = new_mix
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def _get_state_snapshot_locked(self) -> Dict:
        return {"rate": self._rate_hz, "depth": self._depth_ms, "mix": self._mix}

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        state_snapshot_to_emit = None
        with self._lock:
            ui_update_needed = False
            # --- Update state from parameter sockets if connected ---
            rate_socket = input_data.get("rate")
            if rate_socket is not None and self._rate_hz != float(rate_socket):
                self._rate_hz = float(rate_socket)
                ui_update_needed = True

            depth_socket = input_data.get("depth")
            if depth_socket is not None and self._depth_ms != float(depth_socket):
                self._depth_ms = float(depth_socket)
                ui_update_needed = True

            mix_socket = input_data.get("mix")
            if mix_socket is not None:
                new_mix = max(0.0, min(float(mix_socket), 1.0))
                if self._mix != new_mix:
                    self._mix = new_mix
                    ui_update_needed = True

            if ui_update_needed:
                state_snapshot_to_emit = self._get_state_snapshot_locked()

            lfo_value = 0.0
            lfo_mod_input = input_data.get("mod_in")

            if lfo_mod_input is not None:
                lfo_value = float(lfo_mod_input)
            else:
                frame_duration_s = frame.hop_size / frame.sample_rate
                phase_increment = 2 * torch.pi * self._rate_hz * frame_duration_s
                self._lfo_phase = (self._lfo_phase + phase_increment) % (2 * torch.pi)
                lfo_value = torch.sin(torch.tensor(self._lfo_phase)).item()

            delay_s = (self._depth_ms / 1000.0) * lfo_value
            freqs = torch.fft.rfftfreq(frame.fft_size, d=1.0 / frame.sample_rate)
            phase_shifts_rad = 2 * torch.pi * freqs * delay_s

            # Create the complex phasor using torch.polar
            phasor = torch.polar(torch.ones_like(phase_shifts_rad), phase_shifts_rad).to(DEFAULT_COMPLEX_DTYPE)

            # Broadcasting will apply the 1D phasor to each channel in the 2D frame data
            wet_signal = frame.data * phasor
            output_fft = (frame.data * (1.0 - self._mix)) + (wet_signal * self._mix)

        # --- Emit state update AFTER releasing lock ---
        if state_snapshot_to_emit:
            self.ui_update_callback(state_snapshot_to_emit)

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

    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._rate_hz = data.get("rate", 1.5)
            self._depth_ms = data.get("depth", 5.0)
            self._mix = data.get("mix", 0.5)


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
class SpectralReverbNode(Node):
    NODE_TYPE = "Spectral Reverb"
    UI_CLASS = SpectralReverbNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "Algorithmic reverb that applies frequency-dependent decay to spectral frames."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)

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

        # --- Internal State ---
        self._pre_delay_ms: float = 20.0
        self._decay_time_s: float = 2.5
        self._hf_damp_hz: float = 4000.0
        self._lf_damp_hz: float = 150.0
        self._diffusion: float = 1.0
        self._width: float = 1.0
        self._mix: float = 0.5

        # --- DSP Buffers & State (Using torch.Tensor) ---
        self._reverb_fft_buffer: Optional[torch.Tensor] = None
        self._decay_factors: Optional[torch.Tensor] = None
        self._pre_delay_buffer: deque = deque()
        self._pre_delay_frames: int = 0
        self._last_frame_params: tuple = (0, 0)
        self._params_dirty: bool = True

    # --- Parameter Setter Slots ---
    @Slot(float)
    def set_pre_delay_ms(self, value: float):
        state = None
        with self._lock:
            if value != self._pre_delay_ms:
                self._pre_delay_ms = value
                self._params_dirty = True
                state = self._get_state_snapshot_locked()
        if state:
            self.ui_update_callback(state)

    @Slot(float)
    def set_decay_time(self, value: float):
        state = None
        with self._lock:
            if value != self._decay_time_s:
                self._decay_time_s = value
                self._params_dirty = True
                state = self._get_state_snapshot_locked()
        if state:
            self.ui_update_callback(state)

    @Slot(float)
    def set_hf_damp(self, value: float):
        state = None
        with self._lock:
            if value != self._hf_damp_hz:
                self._hf_damp_hz = value
                self._params_dirty = True
                state = self._get_state_snapshot_locked()
        if state:
            self.ui_update_callback(state)

    @Slot(float)
    def set_lf_damp(self, value: float):
        state = None
        with self._lock:
            if value != self._lf_damp_hz:
                self._lf_damp_hz = value
                self._params_dirty = True
                state = self._get_state_snapshot_locked()
        if state:
            self.ui_update_callback(state)

    @Slot(float)
    def set_diffusion(self, value: float):
        state = None
        with self._lock:
            value = max(0.0, min(float(value), 1.0))
            if value != self._diffusion:
                self._diffusion = value
                self._params_dirty = True
                state = self._get_state_snapshot_locked()
        if state:
            self.ui_update_callback(state)

    @Slot(float)
    def set_width(self, value: float):
        state = None
        with self._lock:
            value = max(0.0, min(float(value), 1.0))
            if value != self._width:
                self._width = value
                self._params_dirty = True
                state = self._get_state_snapshot_locked()
        if state:
            self.ui_update_callback(state)

    @Slot(float)
    def set_mix(self, value: float):
        state = None
        with self._lock:
            value = max(0.0, min(float(value), 1.0))
            if value != self._mix:
                self._mix = value
                state = self._get_state_snapshot_locked()
        if state:
            self.ui_update_callback(state)

    def _get_state_snapshot_locked(self) -> Dict:
        return {
            "pre_delay_ms": self._pre_delay_ms,
            "decay_time": self._decay_time_s,
            "hf_damp": self._hf_damp_hz,
            "lf_damp": self._lf_damp_hz,
            "diffusion": self._diffusion,
            "width": self._width,
            "mix": self._mix,
        }

    def _recalculate_params(self, frame: SpectralFrame):
        t60 = self._decay_time_s
        freqs = torch.fft.rfftfreq(frame.fft_size, d=1.0 / frame.sample_rate)
        num_bins = freqs.shape[0]

        # --- Damping Logic ---
        hf_damp_factor = torch.clamp((self._hf_damp_hz - freqs) / self._hf_damp_hz, 0.1, 1.0)
        hf_damp_factor[freqs < self._hf_damp_hz] = 1.0
        lf_damp_factor = torch.clamp(freqs / self._lf_damp_hz, 0.1, 1.0)
        lf_damp_factor[freqs > self._lf_damp_hz] = 1.0
        damped_t60 = t60 * hf_damp_factor * lf_damp_factor

        frames_per_second = frame.sample_rate / frame.hop_size
        magnitudes = 10.0 ** (-3.0 / (damped_t60 * frames_per_second + EPSILON))

        # --- Stereo Width & Diffusion Logic ---
        phase_shift_amount = self._diffusion * torch.pi
        random_phases_1 = (torch.rand(num_bins) * 2 - 1) * phase_shift_amount
        random_phases_2 = (torch.rand(num_bins) * 2 - 1) * phase_shift_amount
        phases_L = random_phases_1
        phases_R = random_phases_1 * (1.0 - self._width) + random_phases_2 * self._width

        decay_L = torch.polar(magnitudes, phases_L).to(DEFAULT_COMPLEX_DTYPE)
        decay_R = torch.polar(magnitudes, phases_R).to(DEFAULT_COMPLEX_DTYPE)
        # Stack to create (2, num_bins) tensor
        self._decay_factors = torch.stack([decay_L, decay_R], dim=0)

        # --- Recalculate Pre-delay Buffer ---
        pre_delay_s = self._pre_delay_ms / 1000.0
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

        state_snapshot_to_emit = None
        with self._lock:
            # --- Update parameters from input sockets ---
            param_map = {
                "pre_delay_ms": "_pre_delay_ms",
                "decay_time": "_decay_time_s",
                "hf_damp": "_hf_damp_hz",
                "lf_damp": "_lf_damp_hz",
                "diffusion": "_diffusion",
                "width": "_width",
                "mix": "_mix",
            }
            for socket_name, attr_name in param_map.items():
                socket_val = input_data.get(socket_name)
                if socket_val is not None and getattr(self, attr_name) != socket_val:
                    setattr(self, attr_name, float(socket_val))
                    self._params_dirty = True

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
                state_snapshot_to_emit = self._get_state_snapshot_locked()

            # --- Pre-delay Logic ---
            self._pre_delay_buffer.append(frame.data)
            if len(self._pre_delay_buffer) < self._pre_delay_frames or self._pre_delay_frames == 0:
                delayed_frame_data = torch.zeros_like(frame.data)
            else:
                delayed_frame_data = self._pre_delay_buffer[0]

            # --- Handle Mono Input and ensure stereo processing ---
            dry_signal = frame.data
            if num_channels == 1:
                delayed_frame_data = delayed_frame_data.repeat(2, 1)  # (1, bins) -> (2, bins)
                dry_signal = dry_signal.repeat(2, 1)

            # --- Core DSP Algorithm ---
            wet_signal = self._reverb_fft_buffer * self._decay_factors
            self._reverb_fft_buffer = delayed_frame_data + wet_signal
            # Broadcasting handles mono dry signal against stereo wet signal
            output_fft = (dry_signal * (1.0 - self._mix)) + (wet_signal * self._mix)

        if state_snapshot_to_emit:
            self.ui_update_callback(state_snapshot_to_emit)

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

    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._pre_delay_ms = data.get("pre_delay_ms", 20.0)
            self._decay_time_s = data.get("decay_time", 2.5)
            self._hf_damp_hz = data.get("hf_damp", 4000.0)
            self._lf_damp_hz = data.get("lf_damp", 150.0)
            self._diffusion = data.get("diffusion", 1.0)
            self._width = data.get("width", 1.0)
            self._mix = data.get("mix", 0.5)
            self._params_dirty = True
