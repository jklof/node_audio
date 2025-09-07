import torch
import numpy as np
import threading
import logging
from typing import Dict, Optional

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE

# --- UI and Qt Imports ---
from ui_elements import NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING  # <-- IMPORTED NodeStateEmitter
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
class DelayNodeItem(NodeItem):
    """Custom UI for the DelayNode, with sliders for delay parameters."""

    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "DelayNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(5)

        # --- Create Slider Controls (Explicitly) ---
        self.controls = {}

        # Delay Time Slider
        self.delay_label = QLabel("Delay Time: ...")
        self.delay_slider = QSlider(Qt.Orientation.Horizontal)
        self.delay_slider.setRange(0, 1000)
        main_layout.addWidget(self.delay_label)
        main_layout.addWidget(self.delay_slider)
        self.controls["delay_time_ms"] = {
            "slider": self.delay_slider,
            "label": self.delay_label,
            "format": "{:.0f} ms",
            "min_val": MIN_DELAY_MS,
            "max_val": MAX_DELAY_MS,
            "is_log": True,
            "name": "Delay Time",
        }

        # Feedback Slider
        self.feedback_label = QLabel("Feedback: ...")
        self.feedback_slider = QSlider(Qt.Orientation.Horizontal)
        self.feedback_slider.setRange(0, 1000)
        main_layout.addWidget(self.feedback_label)
        main_layout.addWidget(self.feedback_slider)
        self.controls["feedback"] = {
            "slider": self.feedback_slider,
            "label": self.feedback_label,
            "format": "{:.1%}",
            "min_val": MIN_FEEDBACK,
            "max_val": MAX_FEEDBACK,
            "is_log": False,
            "name": "Feedback",
        }

        # Mix Slider
        self.mix_label = QLabel("Mix: ...")
        self.mix_slider = QSlider(Qt.Orientation.Horizontal)
        self.mix_slider.setRange(0, 1000)
        main_layout.addWidget(self.mix_label)
        main_layout.addWidget(self.mix_slider)
        self.controls["mix"] = {
            "slider": self.mix_slider,
            "label": self.mix_label,
            "format": "{:.1%}",
            "min_val": MIN_MIX,
            "max_val": MAX_MIX,
            "is_log": False,
            "name": "Mix",
        }

        # Connect slider signals to their specific handlers
        self.delay_slider.valueChanged.connect(self._handle_delay_slider_change)
        self.feedback_slider.valueChanged.connect(self._handle_feedback_slider_change)
        self.mix_slider.valueChanged.connect(self._handle_mix_slider_change)

        self.setContentWidget(self.container_widget)

        # Connect the logic node's state updates back to the UI
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        self.updateFromLogic()

    def _map_slider_to_logical(self, key: str, value: int) -> float:
        info = self.controls[key]
        norm = value / 1000.0
        if info["is_log"]:
            log_min = np.log10(info["min_val"])
            log_max = np.log10(info["max_val"])
            return 10 ** (log_min + norm * (log_max - log_min))
        else:
            return info["min_val"] + norm * (info["max_val"] - info["min_val"])

    def _map_logical_to_slider(self, key: str, value: float) -> int:
        info = self.controls[key]
        if info["is_log"]:
            log_min = np.log10(info["min_val"])
            log_max = np.log10(info["max_val"])
            range_val = log_max - log_min
            if abs(range_val) < EPSILON:
                return 0
            safe_val = np.clip(value, info["min_val"], info["max_val"])
            norm = (np.log10(safe_val) - log_min) / range_val
            return int(round(norm * 1000.0))
        else:
            range_val = info["max_val"] - info["min_val"]
            if abs(range_val) < EPSILON:
                return 0
            norm = (np.clip(value, info["min_val"], info["max_val"]) - info["min_val"]) / range_val
            return int(round(norm * 1000.0))

    @Slot(int)
    def _handle_delay_slider_change(self, value: int):
        logical_val = self._map_slider_to_logical("delay_time_ms", value)
        self.node_logic.set_delay_time_ms(logical_val)

    @Slot(int)
    def _handle_feedback_slider_change(self, value: int):
        logical_val = self._map_slider_to_logical("feedback", value)
        self.node_logic.set_feedback(logical_val)

    @Slot(int)
    def _handle_mix_slider_change(self, value: int):
        logical_val = self._map_slider_to_logical("mix", value)
        self.node_logic.set_mix(logical_val)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        for key, control_info in self.controls.items():
            value = state.get(key, control_info["min_val"])
            is_connected = key in self.node_logic.inputs and self.node_logic.inputs[key].connections
            control_info["slider"].setEnabled(not is_connected)

            with QSignalBlocker(control_info["slider"]):
                control_info["slider"].setValue(self._map_logical_to_slider(key, value))

            label_text = f"{control_info['name']}: {control_info['format'].format(value)}"
            if is_connected:
                label_text += " (ext)"
            control_info["label"].setText(label_text)

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


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
        self.emitter = NodeStateEmitter()
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
            self.emitter.stateUpdated.emit(state_to_emit)

    def set_feedback(self, value: float):
        state_to_emit = None
        with self._lock:
            clipped_value = np.clip(float(value), MIN_FEEDBACK, MAX_FEEDBACK).item()
            if self._feedback != clipped_value:
                self._feedback = clipped_value
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def set_mix(self, value: float):
        state_to_emit = None
        with self._lock:
            clipped_value = np.clip(float(value), MIN_MIX, MAX_MIX).item()
            if self._mix != clipped_value:
                self._mix = clipped_value
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

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
            self.emitter.stateUpdated.emit(state_to_emit)
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
