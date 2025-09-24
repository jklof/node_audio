import torch
import numpy as np
import threading
import logging
from enum import Enum
from typing import Dict, Optional, Tuple

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE

# --- UI and Qt Imports ---
from ui_elements import ParameterNodeItem
from PySide6.QtCore import Slot

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants for this Node ---
MAX_DELAY_S = 0.05  # 50ms, sufficient for chorus
PHASER_STAGES = 6  # Kept at 6 stages for a rich sound, now highly performant
EPSILON = 1e-9


# ==============================================================================
# 1. JIT-Compiled Helper Function for Phaser (FINAL & FASTEST VERSION)
# ==============================================================================
@torch.jit.script
def _jit_phaser_pipeline(
    input_block: torch.Tensor,
    z_state_initial: torch.Tensor,  # Shape: (PHASER_STAGES, num_channels)
    d_block: torch.Tensor,
    feedback_amount: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Processes an entire audio block through the complete multi-stage phaser pipeline.
    This version uses an algebraically simplified all-pass filter form (Fused Multiply-Add)
    for maximum performance within the JIT compiler.
    """
    num_channels, num_samples = input_block.shape
    num_stages = z_state_initial.shape[0]
    output_block = torch.empty_like(input_block)

    z = z_state_initial.clone()

    # --- OPTIMIZATION: Pre-calculate (1 - d) for the whole block ---
    # This removes a subtraction from the innermost loop.
    d1_block = 1.0 - d_block

    for i in range(num_samples):
        feedback_input = z[-1] * feedback_amount
        stage_input = input_block[:, i] + feedback_input

        d = d_block[i]
        d1 = d1_block[i]

        # Loop through all phaser stages using the optimized form
        for stage_idx in range(num_stages):
            # --- OPTIMIZATION: Algebraically simplified FMA form ---
            # output = d * input + (1-d) * state
            output_sample = (d * stage_input) + (d1 * z[stage_idx])

            # Update the state for this stage
            z[stage_idx] = stage_input
            stage_input = output_sample

        output_block[:, i] = stage_input

    return output_block, z


# ==============================================================================
# 2. Enum for Effect Types
# ==============================================================================
class ModulationEffectType(Enum):
    CHORUS = "Chorus"
    FLANGER = "Flanger"
    PHASER = "Phaser"


# ==============================================================================
# 3. UI Class for the Modulation Effect Node
# ==============================================================================
class ModulationEffectNodeItem(ParameterNodeItem):
    """
    A declarative UI for the combined modulation effects node.
    """

    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "ModulationEffectNode"):
        parameters = [
            {"key": "mode", "name": "Mode", "type": "combobox", "items": [(m.value, m) for m in ModulationEffectType]},
            {
                "key": "rate",
                "name": "Rate",
                "type": "dial",
                "min": 0.05,
                "max": 20.0,
                "format": "{:.2f} Hz",
                "is_log": True,
            },
            {"key": "depth", "name": "Depth", "type": "slider", "min": 0.0, "max": 1.0, "format": "{:.0%}"},
            {"key": "feedback", "name": "Feedback", "type": "slider", "min": 0.0, "max": 0.98, "format": "{:.0%}"},
            {"key": "mix", "name": "Mix", "type": "slider", "min": 0.0, "max": 1.0, "format": "{:.0%}"},
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        super()._on_state_updated_from_logic(state)
        mode = state.get("mode")
        is_feedback_visible = mode in [ModulationEffectType.FLANGER, ModulationEffectType.PHASER]
        feedback_control = self._controls.get("feedback")
        if feedback_control:
            feedback_control["widget"].setVisible(is_feedback_visible)
            feedback_control["label"].setVisible(is_feedback_visible)
            self.container_widget.adjustSize()
            self.update_geometry()


# ==============================================================================
# 4. Logic Class for the Modulation Effect Node (FINAL OPTIMIZATION & BUGFIX)
# ==============================================================================
class ModulationEffectNode(Node):
    NODE_TYPE = "Modulation FX"
    UI_CLASS = ModulationEffectNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Provides Chorus, Flanger, and Phaser effects in one node."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("rate", data_type=float)
        self.add_input("depth", data_type=float)
        self.add_input("feedback", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("out", data_type=torch.Tensor)
        self._mode: ModulationEffectType = ModulationEffectType.CHORUS
        self._rate_hz: float = 0.5
        self._depth: float = 0.5
        self._feedback: float = 0.0
        self._mix: float = 0.5
        self._samplerate = DEFAULT_SAMPLERATE
        self._buffer_size_samples = int(MAX_DELAY_S * self._samplerate)
        self._delay_buffer: Optional[torch.Tensor] = None
        self._write_head: int = 0
        self._lfo_phase: float = 0.0
        self._phaser_z: Optional[torch.Tensor] = None
        self._expected_channels: Optional[int] = None
        self._block_indices_float: Optional[torch.Tensor] = None
        self._lfo_phases: Optional[torch.Tensor] = None

    def _initialize_buffers(self, num_channels: int):
        self._delay_buffer = torch.zeros((num_channels, self._buffer_size_samples), dtype=DEFAULT_DTYPE)
        self._phaser_z = torch.zeros((PHASER_STAGES, num_channels), dtype=DEFAULT_DTYPE)
        self._write_head = 0
        self._lfo_phase = 0.0
        self._expected_channels = num_channels
        self._block_indices_float = torch.arange(DEFAULT_BLOCKSIZE, dtype=torch.float32)
        self._lfo_phases = torch.empty(DEFAULT_BLOCKSIZE, dtype=DEFAULT_DTYPE)
        logger.info(f"[{self.name}] Buffers initialized for {num_channels} channels.")

    # --- Parameter Setters ---
    @Slot(ModulationEffectType)
    def set_mode(self, value: ModulationEffectType):
        self._update_parameter("_mode", value)

    @Slot(float)
    def set_rate(self, value: float):
        self._update_parameter("_rate_hz", float(value))

    @Slot(float)
    def set_depth(self, value: float):
        self._update_parameter("_depth", float(value))

    @Slot(float)
    def set_feedback(self, value: float):
        self._update_parameter("_feedback", float(value))

    @Slot(float)
    def set_mix(self, value: float):
        self._update_parameter("_mix", float(value))

    def _update_parameter(self, attr_name: str, value):
        state_to_emit = None
        with self._lock:
            if getattr(self, attr_name) != value:
                setattr(self, attr_name, value)
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def _get_state_snapshot_locked(self) -> Dict:
        return {
            "mode": self._mode,
            "rate": self._rate_hz,
            "depth": self._depth,
            "feedback": self._feedback,
            "mix": self._mix,
        }

    def start(self):
        with self._lock:
            self._write_head = 0
            if self._delay_buffer is not None:
                self._delay_buffer.zero_()
            if self._phaser_z is not None:
                self._phaser_z.zero_()
            self._lfo_phase = 0.0
            self._expected_channels = None

    def process(self, input_data: dict) -> dict:
        dry_signal = input_data.get("in")
        if not isinstance(dry_signal, torch.Tensor):
            return {"out": None}
        num_channels, block_size = dry_signal.shape
        if block_size != DEFAULT_BLOCKSIZE:
            return {"out": dry_signal}

        state_to_emit = None
        ui_update_needed = False
        with self._lock:
            if self._expected_channels != num_channels:
                self._initialize_buffers(num_channels)

            # --- ROBUST PARAMETER HANDLING (BUG FIX) ---
            rate_socket_val = input_data.get("rate")
            rate = float(rate_socket_val) if rate_socket_val is not None else self._rate_hz
            if self._rate_hz != rate:
                self._rate_hz, ui_update_needed = rate, True

            depth_socket_val = input_data.get("depth")
            depth = float(depth_socket_val) if depth_socket_val is not None else self._depth
            if self._depth != depth:
                self._depth, ui_update_needed = depth, True

            feedback_socket_val = input_data.get("feedback")
            feedback = float(feedback_socket_val) if feedback_socket_val is not None else self._feedback
            if self._feedback != feedback:
                self._feedback, ui_update_needed = feedback, True

            mix_socket_val = input_data.get("mix")
            mix = float(mix_socket_val) if mix_socket_val is not None else self._mix
            if self._mix != mix:
                self._mix, ui_update_needed = mix, True

            mode = self._mode

            if ui_update_needed:
                state_to_emit = self._get_state_snapshot_locked()

        if state_to_emit:
            self.ui_update_callback(state_to_emit)

        if mix < EPSILON:
            return {"out": dry_signal}

        phase_inc_per_sample = rate / self._samplerate
        torch.add(self._block_indices_float * phase_inc_per_sample, self._lfo_phase, out=self._lfo_phases)
        self._lfo_phase = self._lfo_phases[-1].item() % 1.0
        lfo_out = torch.sin(2 * torch.pi * self._lfo_phases)

        wet_signal = torch.zeros_like(dry_signal)

        if mode == ModulationEffectType.CHORUS or mode == ModulationEffectType.FLANGER:
            center_delay_ms = 25.0 if mode == ModulationEffectType.CHORUS else 5.0
            depth_ms = (20.0 if mode == ModulationEffectType.CHORUS else 4.9) * depth
            center_delay_samples = center_delay_ms / 1000.0 * self._samplerate
            depth_samples = depth_ms / 1000.0 * self._samplerate
            delay_samples = center_delay_samples + lfo_out * depth_samples
            read_head_float = self._write_head - delay_samples
            indices = self._block_indices_float + read_head_float
            indices_wrapped = torch.fmod(indices, self._buffer_size_samples)
            indices_floor = indices_wrapped.long()
            indices_ceil = torch.fmod(indices_floor + 1, self._buffer_size_samples).long()
            fraction = (indices_wrapped - indices_floor).unsqueeze(0)
            sample1 = self._delay_buffer[:, indices_floor]
            sample2 = self._delay_buffer[:, indices_ceil]
            wet_signal = sample1 * (1.0 - fraction) + sample2 * fraction
            feedback_signal = dry_signal + (wet_signal * feedback)
            write_indices_start = self._write_head
            if write_indices_start + block_size > self._buffer_size_samples:
                part1_len = self._buffer_size_samples - write_indices_start
                self._delay_buffer[:, write_indices_start:] = feedback_signal[:, :part1_len]
                self._delay_buffer[:, : block_size - part1_len] = feedback_signal[:, part1_len:]
            else:
                self._delay_buffer[:, write_indices_start : write_indices_start + block_size] = feedback_signal
            self._write_head = (self._write_head + block_size) % self._buffer_size_samples

        elif mode == ModulationEffectType.PHASER:
            min_freq, max_freq = 100.0, 4000.0
            sweep_width = (max_freq - min_freq) * depth
            center_freq = min_freq + sweep_width / 2.0 + ((max_freq - min_freq - sweep_width) / 2.0) * (lfo_out + 1)
            d = (1.0 - (torch.pi * center_freq / self._samplerate)) / (
                1.0 + (torch.pi * center_freq / self._samplerate) + EPSILON
            )

            wet_signal, new_phaser_z = _jit_phaser_pipeline(dry_signal, self._phaser_z, d, feedback)
            self._phaser_z.copy_(new_phaser_z)

        output_signal = (dry_signal * (1.0 - mix)) + (wet_signal * mix)
        return {"out": output_signal}

    def serialize_extra(self) -> dict:
        with self._lock:
            state = self._get_state_snapshot_locked()
            state["mode"] = state["mode"].name
            return state

    def deserialize_extra(self, data: dict):
        with self._lock:
            mode_name = data.get("mode", ModulationEffectType.CHORUS.name)
            self._mode = ModulationEffectType[mode_name]
            self._rate_hz = data.get("rate", 0.5)
            self._depth = data.get("depth", 0.5)
            self._feedback = data.get("feedback", 0.0)
            self._mix = data.get("mix", 0.5)
