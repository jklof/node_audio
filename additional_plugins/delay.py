import ctypes
import torch
import numpy as np
import logging
from typing import Dict

from ffi_node import FFINodeBase
from ui_elements import ParameterNodeItem
from PySide6.QtCore import Slot
from constants import DEFAULT_SAMPLERATE, DEFAULT_DTYPE, DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS

logger = logging.getLogger(__name__)

# --- Constants for Delay Node ---
MAX_DELAY_S = 2.0
MIN_DELAY_MS = 1.0
MAX_DELAY_MS = MAX_DELAY_S * 1000.0

# Define a C-style pointer to a float pointer for our 2D audio data
float_pp = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))


# ==============================================================================
# 1. UI Class (Inherits directly from the old DelayNode's UI)
# ==============================================================================
class DelayNodeItem(ParameterNodeItem):
    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "DelayNode"):
        parameters = [
            {
                "key": "delay_time_ms",
                "name": "Delay Time",
                "min": MIN_DELAY_MS,
                "max": MAX_DELAY_MS,
                "format": "{:.0f} ms",
                "is_log": True,
            },
            {"key": "feedback", "name": "Feedback", "min": 0.0, "max": 1.0, "format": "{:.1%}"},
            {"key": "mix", "name": "Mix", "min": 0.0, "max": 1.0, "format": "{:.1%}"},
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 2. Logic Class for the Delay Node
# ==============================================================================
class DelayNode(FFINodeBase):
    NODE_TYPE = "Delay"
    UI_CLASS = DelayNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Creates echoes using a high-performance compiled C++ core."

    # --- Declarative API Definition for the C++ library ---
    LIB_NAME = "delay_processor"
    API = {
        "set_parameters": {
            "argtypes": [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        },
        "process_block": {"argtypes": [ctypes.c_void_p, float_pp, float_pp, ctypes.c_int, ctypes.c_int]},
    }
    MAX_CHANNELS = DEFAULT_CHANNELS
    BUFFER_SIZE_SAMPLES = int(MAX_DELAY_S * DEFAULT_SAMPLERATE)

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)  # This call loads and binds the C++ library

        self.add_input("in", data_type=torch.Tensor)
        self.add_input("delay_time_ms", data_type=float)
        self.add_input("feedback", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._delay_time_ms = 100.0
        self._feedback = 0.5
        self._mix = 0.5

        # Pre-allocate the output buffer for performance
        self._output_buffer = torch.zeros((self.MAX_CHANNELS, DEFAULT_BLOCKSIZE), dtype=torch.float32)

        # Synchronize parameters with the C++ object immediately on creation.
        self._update_cpp_params()

    def _get_state_snapshot_locked(self) -> Dict:
        return {"delay_time_ms": self._delay_time_ms, "feedback": self._feedback, "mix": self._mix}

    def _update_cpp_params(self):
        """Helper to send all current parameters to the C++ object."""
        if not self.dsp_handle:
            return
        delay_samples = self._delay_time_ms / 1000.0 * DEFAULT_SAMPLERATE
        self.lib.set_parameters(
            self.dsp_handle, self.BUFFER_SIZE_SAMPLES, self.MAX_CHANNELS, delay_samples, self._feedback, self._mix
        )

    # --- Public Setters for UI and Socket Control ---
    @Slot(float)
    def set_delay_time_ms(self, value: float):
        with self._lock:
            self._delay_time_ms = float(value)
            self._update_cpp_params()
        self.ui_update_callback(self._get_state_snapshot_locked())

    @Slot(float)
    def set_feedback(self, value: float):
        with self._lock:
            self._feedback = float(value)
            self._update_cpp_params()
        self.ui_update_callback(self._get_state_snapshot_locked())

    @Slot(float)
    def set_mix(self, value: float):
        with self._lock:
            self._mix = float(value)
            self._update_cpp_params()
        self.ui_update_callback(self._get_state_snapshot_locked())

    def process(self, input_data: dict) -> dict:
        if not self.dsp_handle or self.error_state:
            return {"out": None}

        # --- Handle parameter updates from sockets ---
        with self._lock:
            param_changed = False
            delay_socket = input_data.get("delay_time_ms")
            if delay_socket is not None and self._delay_time_ms != float(delay_socket):
                self._delay_time_ms = float(delay_socket)
                param_changed = True

            feedback_socket = input_data.get("feedback")
            if feedback_socket is not None and self._feedback != float(feedback_socket):
                self._feedback = float(feedback_socket)
                param_changed = True

            mix_socket = input_data.get("mix")
            if mix_socket is not None and self._mix != float(mix_socket):
                self._mix = float(mix_socket)
                param_changed = True

            if param_changed:
                self._update_cpp_params()
                self.ui_update_callback(self._get_state_snapshot_locked())

        # --- Marshall data and call C++ ---
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}

        # Ensure tensors are contiguous and float32
        in_tensor = signal.contiguous().to(torch.float32)
        out_tensor = self._output_buffer  # Use pre-allocated buffer

        num_channels, num_samples = in_tensor.shape

        # Create C-style arrays of pointers for our planar data
        in_channel_pointers = (ctypes.POINTER(ctypes.c_float) * num_channels)()
        out_channel_pointers = (ctypes.POINTER(ctypes.c_float) * num_channels)()

        for i in range(num_channels):
            in_channel_pointers[i] = ctypes.cast(in_tensor[i].data_ptr(), ctypes.POINTER(ctypes.c_float))
            out_channel_pointers[i] = ctypes.cast(out_tensor[i].data_ptr(), ctypes.POINTER(ctypes.c_float))

        # Call the C++ process function
        self.lib.process_block(self.dsp_handle, in_channel_pointers, out_channel_pointers, num_channels, num_samples)

        # Return a clone of the output to avoid overwriting in next process call
        return {"out": out_tensor[:num_channels, :].clone()}

    def serialize_extra(self) -> dict:
        with self._lock:
            return self._get_state_snapshot_locked()

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._delay_time_ms = data.get("delay_time_ms", 100.0)
            self._feedback = data.get("feedback", 0.5)
            self._mix = data.get("mix", 0.5)
            # Update the C++ side with all the new parameters at once.
            self._update_cpp_params()
        # No UI update callback is needed here. The engine's post-load sync
        # will handle updating the UI with the final state.
