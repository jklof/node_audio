import torch
import torch.nn.functional as F
import numpy as np
import threading
import logging
from typing import Dict, Optional, Tuple

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE

# --- UI and Qt Imports ---
from ui_elements import ParameterNodeItem
from PySide6.QtCore import Slot

# Configure logging
logger = logging.getLogger(__name__)

# --- Node-Specific Constants ---
MIN_FREQ = 20.0
MAX_FREQ = 8000.0
DOWNSAMPLE_FACTOR = 4  # Process at 1/4 the sample rate for a huge performance boost
INTERNAL_SAMPLERATE = DEFAULT_SAMPLERATE // DOWNSAMPLE_FACTOR
INTERNAL_BLOCKSIZE = DEFAULT_BLOCKSIZE // DOWNSAMPLE_FACTOR


# ==============================================================================
# 1. JIT-Compiled Feedback Loop (Unchanged, already optimal)
# ==============================================================================
@torch.jit.script
def _jit_feedback_loop(
    input_signal: torch.Tensor,
    initial_state: torch.Tensor,
    damping_coeff: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_samples = input_signal.shape[0]
    output_signal = torch.empty_like(input_signal)
    last_sample = initial_state
    for i in range(num_samples):
        last_sample = (1.0 - damping_coeff) * input_signal[i] + damping_coeff * last_sample
        output_signal[i] = last_sample
    return output_signal, last_sample


# ==============================================================================
# 2. UI Class for the Karplus-Strong Node (Unchanged)
# ==============================================================================
class KarplusStrongNodeItem(ParameterNodeItem):
    NODE_SPECIFIC_WIDTH = 200
    def __init__(self, node_logic: "KarplusStrongNode"):
        parameters = [
            {"key": "frequency", "name": "Frequency", "type": "dial", "min": MIN_FREQ, "max": MAX_FREQ, "format": "{:.1f} Hz", "is_log": True},
            {"key": "damping", "name": "Damping", "type": "slider", "min": 0.0, "max": 0.999, "format": "{:.1%}", "is_log": False},
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 3. Logic Class for the Karplus-Strong Node (HEAVILY OPTIMIZED)
# ==============================================================================
class KarplusStrongNode(Node):
    NODE_TYPE = "Karplus-Strong String"
    UI_CLASS = KarplusStrongNodeItem
    CATEGORY = "Generators"
    DESCRIPTION = "Simulates a plucked string using a highly efficient hybrid-rate physical model."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("trigger", data_type=bool)
        self.add_input("frequency", data_type=float)
        self.add_input("damping", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._frequency: float = 440.0
        self._damping: float = 0.5
        self._previous_trigger: bool = False

        # --- OPTIMIZATION: Hybrid-Rate DSP State ---
        self._buffer_size_samples = int(INTERNAL_SAMPLERATE / MIN_FREQ) + 1
        self._delay_buffer: torch.Tensor = torch.zeros(self._buffer_size_samples, dtype=DEFAULT_DTYPE)
        self._write_head: int = 0
        self._last_sample_state: torch.Tensor = torch.tensor(0.0, dtype=DEFAULT_DTYPE)
        
        # --- OPTIMIZATION: Pre-computed ramp for delay line reads ---
        self._ramp_low_rate = torch.arange(INTERNAL_BLOCKSIZE, dtype=DEFAULT_DTYPE)
        
        # --- OPTIMIZATION: State for cheap high-pass filter on pluck noise ---
        self._pluck_filter_state = torch.tensor(0.0, dtype=DEFAULT_DTYPE)

    def _get_state_snapshot_locked(self) -> Dict:
        return {"frequency": self._frequency, "damping": self._damping}

    @Slot(float)
    def set_frequency(self, frequency: float):
        state_to_emit = None
        with self._lock:
            new_freq = np.clip(float(frequency), MIN_FREQ, MAX_FREQ)
            if self._frequency != new_freq: self._frequency = new_freq; state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit: self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_damping(self, damping: float):
        state_to_emit = None
        with self._lock:
            new_damping = np.clip(float(damping), 0.0, 0.999)
            if self._damping != new_damping: self._damping = new_damping; state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit: self.ui_update_callback(state_to_emit)

    def _pluck(self, delay_len_low_rate: int):
        """
        OPTIMIZED: Generates a brightened noise burst directly at the low
        internal sample rate, avoiding expensive resampling.
        """
        # 1. Generate noise directly at the low internal sample rate.
        noise_low_rate = (torch.rand(delay_len_low_rate, dtype=DEFAULT_DTYPE) * 2.0) - 1.0

        # 2. Brighten the noise with a cheap first-order high-pass filter (diff).
        noise_with_state = torch.cat((self._pluck_filter_state.view(1), noise_low_rate))
        bright_noise = torch.diff(noise_with_state)
        self._pluck_filter_state = noise_low_rate[-1].view(1)

        # 3. Inject the brightened noise into the low-rate circular buffer.
        if self._write_head + delay_len_low_rate > self._buffer_size_samples:
            part1_len = self._buffer_size_samples - self._write_head
            self._delay_buffer[self._write_head:] = bright_noise[:part1_len]
            self._delay_buffer[:delay_len_low_rate - part1_len] = bright_noise[part1_len:]
        else:
            self._delay_buffer[self._write_head : self._write_head + delay_len_low_rate] = bright_noise
        
        self._last_sample_state = torch.tensor(0.0, dtype=DEFAULT_DTYPE)

    def process(self, input_data: dict) -> dict:
        state_snapshot_to_emit = None
        with self._lock:
            ui_update_needed = False
            freq_socket = input_data.get("frequency")
            if freq_socket is not None:
                new_freq = np.clip(float(freq_socket), MIN_FREQ, MAX_FREQ)
                if abs(self._frequency - new_freq) > 1e-6: self._frequency = new_freq; ui_update_needed = True
            damping_socket = input_data.get("damping")
            if damping_socket is not None:
                new_damping = np.clip(float(damping_socket), 0.0, 0.999)
                if abs(self._damping - new_damping) > 1e-6: self._damping = new_damping; ui_update_needed = True
            if ui_update_needed: state_snapshot_to_emit = self._get_state_snapshot_locked()
            
            frequency = self._frequency; damping = self._damping
            trigger = bool(input_data.get("trigger", False))
            is_triggered = trigger and not self._previous_trigger
            self._previous_trigger = trigger
            
            delay_samples = INTERNAL_SAMPLERATE / frequency

        if state_snapshot_to_emit: self.ui_update_callback(state_snapshot_to_emit)

        if is_triggered:
            self._pluck(int(np.floor(delay_samples)))

        # --- STEP 1: Process one small block at the LOW sample rate ---
        read_head_float = self._write_head - delay_samples
        
        # OPTIMIZATION: Use pre-computed ramp to avoid re-allocating `arange`
        indices = self._ramp_low_rate + read_head_float
        
        indices_wrapped = torch.fmod(indices, self._buffer_size_samples)
        indices_floor = indices_wrapped.long()
        indices_ceil = torch.fmod(indices_floor + 1, self._buffer_size_samples).long()
        fraction = indices_wrapped - indices_floor
        
        sample1 = self._delay_buffer[indices_floor]
        sample2 = self._delay_buffer[indices_ceil]
        output_signal_low_rate = sample1 * (1.0 - fraction) + sample2 * fraction

        feedback_signal, new_state = _jit_feedback_loop(output_signal_low_rate, self._last_sample_state, damping)
        self._last_sample_state = new_state
        
        write_indices_start = self._write_head
        if write_indices_start + INTERNAL_BLOCKSIZE > self._buffer_size_samples:
            part1_len = self._buffer_size_samples - write_indices_start
            self._delay_buffer[write_indices_start:] = feedback_signal[:part1_len]
            self._delay_buffer[:INTERNAL_BLOCKSIZE - part1_len] = feedback_signal[part1_len:]
        else:
            self._delay_buffer[write_indices_start : write_indices_start + INTERNAL_BLOCKSIZE] = feedback_signal

        self._write_head = (self._write_head + INTERNAL_BLOCKSIZE) % self._buffer_size_samples
        
        # --- STEP 2: Upsample the low-rate block to the final output size ---
        output_signal_high_rate = F.interpolate(
            output_signal_low_rate.view(1, 1, -1),
            size=DEFAULT_BLOCKSIZE,
            mode='linear',
            align_corners=False
        ).squeeze()

        return {"out": output_signal_high_rate.unsqueeze(0)}

    def start(self):
        with self._lock:
            self._delay_buffer.zero_()
            self._write_head = 0
            self._last_sample_state = torch.tensor(0.0, dtype=DEFAULT_DTYPE)
            self._pluck_filter_state = torch.tensor(0.0, dtype=DEFAULT_DTYPE)
            self._previous_trigger = False
        super().start()

    def serialize_extra(self) -> dict:
        with self._lock: return self._get_state_snapshot_locked()

    def deserialize_extra(self, data: dict):
        self.set_frequency(data.get("frequency", 440.0))
        self.set_damping(data.get("damping", 0.5))