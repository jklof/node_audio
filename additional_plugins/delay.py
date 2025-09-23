import torch
import numpy as np
import threading
import logging
from typing import Dict, Optional

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE

# --- UI and Qt Imports ---
from ui_elements import (
    ParameterNodeItem,
    NodeItem,
    NODE_CONTENT_PADDING,
)
from PySide6.QtWidgets import QWidget, QSlider, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, Signal, Slot, QObject, QSignalBlocker

# --- Configure logging ---
logger = logging.getLogger(__name__)

# --- Constants for Delay Node ---
MAX_DELAY_S = 2.0
MIN_DELAY_MS = 1.0
MAX_DELAY_MS = MAX_DELAY_S * 1000.0
MIN_FEEDBACK = 0.0
MAX_FEEDBACK = 1.0
MIN_MIX = 0.0
MAX_MIX = 1.0
EPSILON = 1e-9


# ==============================================================================
# 1. UI Class for the Delay Node
# ==============================================================================
class DelayNodeItem(ParameterNodeItem):
    """Custom UI for the DelayNode, with sliders for delay parameters."""

    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "DelayNode"):
        # Define the parameters for this node
        parameters = [
            {
                "key": "delay_time_ms",
                "name": "Delay Time",
                "min": MIN_DELAY_MS,
                "max": MAX_DELAY_MS,
                "format": "{:.0f} ms",
                "is_log": True,
            },
            {
                "key": "feedback",
                "name": "Feedback",
                "min": MIN_FEEDBACK,
                "max": MAX_FEEDBACK,
                "format": "{:.1%}",
                "is_log": False,
            },
            {
                "key": "mix",
                "name": "Mix",
                "min": MIN_MIX,
                "max": MAX_MIX,
                "format": "{:.1%}",
                "is_log": False,
            },
        ]

        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 2. Logic Class for the Delay Node
# ==============================================================================
class DelayNode(Node):
    NODE_TYPE = "Delay"
    UI_CLASS = DelayNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Creates echoes of the input signal."

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("delay_time_ms", data_type=float)
        self.add_input("feedback", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._lock = threading.Lock()
        self._delay_time_ms = 100.0
        self._feedback = 0.5
        self._mix = 0.5
        self._samplerate = DEFAULT_SAMPLERATE

        # DSP State
        self._buffer_size_samples = int(MAX_DELAY_S * self._samplerate)
        self._delay_buffer: Optional[torch.Tensor] = None
        self._write_head = 0
        self._expected_channels = None

    def _initialize_buffer(self, num_channels: int):
        """Initializes or re-initializes the delay buffer for a specific channel count."""
        self._delay_buffer = torch.zeros((num_channels, self._buffer_size_samples), dtype=DEFAULT_DTYPE)
        self._write_head = 0
        self._expected_channels = num_channels
        logger.info(
            f"[{self.name}] Delay buffer initialized for {num_channels} channels, size {self._buffer_size_samples} samples."
        )

    # --- Explicit, Thread-Safe Parameter Setters (for UI interaction) ---
    def set_delay_time_ms(self, value: float):
        state_to_emit = None
        with self._lock:
            clipped_value = np.clip(float(value), MIN_DELAY_MS, MAX_DELAY_MS).item()
            if self._delay_time_ms != clipped_value:
                self._delay_time_ms = clipped_value
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def set_feedback(self, value: float):
        state_to_emit = None
        with self._lock:
            clipped_value = np.clip(float(value), MIN_FEEDBACK, MAX_FEEDBACK).item()
            if self._feedback != clipped_value:
                self._feedback = clipped_value
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def set_mix(self, value: float):
        state_to_emit = None
        with self._lock:
            clipped_value = np.clip(float(value), MIN_MIX, MAX_MIX).item()
            if self._mix != clipped_value:
                self._mix = clipped_value
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def get_current_state_snapshot(self, locked: bool = False) -> Dict:
        """Returns a copy of the current parameters for UI or serialization."""
        state = {
            "delay_time_ms": self._delay_time_ms,
            "feedback": self._feedback,
            "mix": self._mix,
        }
        if locked:
            return state
        with self._lock:
            return state

    def start(self):
        """Called when processing starts. Resets the delay buffer."""
        with self._lock:
            self._write_head = 0
            if self._delay_buffer is not None:
                self._delay_buffer.zero_()
            self._expected_channels = None  # Force re-init on first process block

    def process(self, input_data: dict) -> dict:
        dry_signal = input_data.get("in")
        if not isinstance(dry_signal, torch.Tensor):
            return {"out": None}

        num_channels, block_size = dry_signal.shape
        if block_size != DEFAULT_BLOCKSIZE:
            logger.warning(f"[{self.name}] Input block size mismatch. Expected {DEFAULT_BLOCKSIZE}, got {block_size}.")
            return {"out": dry_signal}  # Pass through

        # --- CORRECTED: State update logic to prevent deadlock ---
        state_to_emit = None
        ui_update_needed = False
        with self._lock:
            if self._expected_channels != num_channels:
                self._initialize_buffer(num_channels)

            # Check for changes from input sockets and update values directly.
            # This avoids calling the locking setters from within this locked context.
            delay_socket_val = input_data.get("delay_time_ms")
            if delay_socket_val is not None:
                clipped_val = np.clip(float(delay_socket_val), MIN_DELAY_MS, MAX_DELAY_MS).item()
                if self._delay_time_ms != clipped_val:
                    self._delay_time_ms = clipped_val
                    ui_update_needed = True

            feedback_socket_val = input_data.get("feedback")
            if feedback_socket_val is not None:
                clipped_val = np.clip(float(feedback_socket_val), MIN_FEEDBACK, MAX_FEEDBACK).item()
                if self._feedback != clipped_val:
                    self._feedback = clipped_val
                    ui_update_needed = True

            mix_socket_val = input_data.get("mix")
            if mix_socket_val is not None:
                clipped_val = np.clip(float(mix_socket_val), MIN_MIX, MAX_MIX).item()
                if self._mix != clipped_val:
                    self._mix = clipped_val
                    ui_update_needed = True

            # Copy parameters to local variables for processing this tick
            delay_time = self._delay_time_ms
            feedback = self._feedback
            mix = self._mix

            # If a value changed, get a state snapshot to emit after releasing the lock
            if ui_update_needed:
                state_to_emit = self.get_current_state_snapshot(locked=True)

        # Emit signal to UI AFTER the lock is released
        if state_to_emit:
            self.ui_update_callback(state_to_emit)
        # --- END CORRECTION ---

        # --- DSP Processing ---
        delay_samples = delay_time / 1000.0 * self._samplerate

        # 1. Generate floating-point read indices for this block with interpolation
        read_head_float = self._write_head - delay_samples
        indices = torch.arange(block_size, dtype=torch.float32) + read_head_float
        indices_wrapped = torch.fmod(indices, self._buffer_size_samples)

        indices_floor = indices_wrapped.long()
        indices_ceil = torch.fmod(indices_floor + 1, self._buffer_size_samples).long()
        fraction = (indices_wrapped - indices_floor).unsqueeze(0)  # Shape: (1, block_size)

        # 2. Read from buffer using linear interpolation
        sample1 = self._delay_buffer[:, indices_floor]
        sample2 = self._delay_buffer[:, indices_ceil]
        delayed_signal = sample1 * (1.0 - fraction) + sample2 * fraction

        # 3. Mix dry and wet signals
        output_signal = (dry_signal * (1.0 - mix)) + (delayed_signal * mix)

        # 4. Create signal for feedback loop
        feedback_signal = dry_signal + (delayed_signal * feedback)

        # 5. Write back to the circular buffer, handling wrap-around
        write_indices_start = self._write_head
        if write_indices_start + block_size > self._buffer_size_samples:
            part1_len = self._buffer_size_samples - write_indices_start
            self._delay_buffer[:, write_indices_start:] = feedback_signal[:, :part1_len]
            self._delay_buffer[:, : block_size - part1_len] = feedback_signal[:, part1_len:]
        else:
            self._delay_buffer[:, write_indices_start : write_indices_start + block_size] = feedback_signal

        # 6. Advance write head
        self._write_head = (self._write_head + block_size) % self._buffer_size_samples

        return {"out": output_signal}

    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._delay_time_ms = data.get("delay_time_ms", 100.0)
            self._feedback = data.get("feedback", 0.5)
            self._mix = data.get("mix", 0.5)
