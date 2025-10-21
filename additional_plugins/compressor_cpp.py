import torch
import ctypes
import logging
from typing import Dict, Optional

# --- Node System Imports ---
from ffi_node import FFINodeBase
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS

# --- UI and Qt Imports ---
from ui_elements import ParameterNodeItem
from PySide6.QtCore import Slot

# --- Helper Imports ---
from node_helpers import with_parameters, Parameter

# --- Configure logging ---
logger = logging.getLogger(__name__)

# --- Node-Specific Constants (Copied from original for UI) ---
MIN_THRESHOLD_DB = -60.0
MAX_THRESHOLD_DB = 0.0
MIN_RATIO = 1.0
MAX_RATIO = 20.0
MIN_ATTACK_MS = 0.1
MAX_ATTACK_MS = 100.0
MIN_RELEASE_MS = 1.0
MAX_RELEASE_MS = 2000.0
MIN_KNEE_DB = 0.0
MAX_KNEE_DB = 24.0

# Define a C-style pointer to a float pointer for our 2D audio data
float_pp = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))


# ==============================================================================
# 1. UI Class for the Compressor Node (Unchanged)
# ==============================================================================
class CompressorCppNodeItem(ParameterNodeItem):
    """Custom UI for the CompressorNode with slider controls."""

    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "CompressorCppNode"):
        parameters = [
            {
                "key": "threshold_db",
                "name": "Threshold",
                "min": MIN_THRESHOLD_DB,
                "max": MAX_THRESHOLD_DB,
                "format": "{:.1f} dB",
                "is_log": False,
            },
            {
                "key": "ratio",
                "name": "Ratio",
                "min": MIN_RATIO,
                "max": MAX_RATIO,
                "format": "{:.1f}:1",
                "is_log": False,
            },
            {
                "key": "attack_ms",
                "name": "Attack",
                "min": MIN_ATTACK_MS,
                "max": MAX_ATTACK_MS,
                "format": "{:.1f} ms",
                "is_log": True,
            },
            {
                "key": "release_ms",
                "name": "Release",
                "min": MIN_RELEASE_MS,
                "max": MAX_RELEASE_MS,
                "format": "{:.0f} ms",
                "is_log": True,
            },
            {
                "key": "knee_db",
                "name": "Knee",
                "min": MIN_KNEE_DB,
                "max": MAX_KNEE_DB,
                "format": "{:.1f} dB",
                "is_log": False,
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 2. Logic Class for the new C++ backed Compressor Node (FIXED)
# ==============================================================================
@with_parameters
class CompressorCppNode(FFINodeBase):
    NODE_TYPE = "Compressor++(faster)"
    UI_CLASS = CompressorCppNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Reduces the dynamic range of a signal using a high-performance C++ core."

    # --- Declarative API Definition for the C++ library ---
    LIB_NAME = "compressor_processor"
    API = {
        "set_parameters": {
            "argtypes": [
                ctypes.c_void_p,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
            ]
        },
        "process_block": {"argtypes": [ctypes.c_void_p, float_pp, float_pp, float_pp, ctypes.c_int, ctypes.c_int]},
    }

    # --- Declarative Parameters (FIX: on_change points to the _locked method) ---
    threshold_db = Parameter(
        default=-20.0, clip=(MIN_THRESHOLD_DB, MAX_THRESHOLD_DB), on_change="_update_cpp_params_locked"
    )
    ratio = Parameter(default=4.0, clip=(MIN_RATIO, MAX_RATIO), on_change="_update_cpp_params_locked")
    attack_ms = Parameter(default=5.0, clip=(MIN_ATTACK_MS, MAX_ATTACK_MS), on_change="_update_cpp_params_locked")
    release_ms = Parameter(default=100.0, clip=(MIN_RELEASE_MS, MAX_RELEASE_MS), on_change="_update_cpp_params_locked")
    knee_db = Parameter(default=6.0, clip=(MIN_KNEE_DB, MAX_KNEE_DB), on_change="_update_cpp_params_locked")

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)  # This call loads the C++ library

        self._init_parameters()

        self.add_input("in", data_type=torch.Tensor)
        self.add_input("threshold_db", data_type=float)
        self.add_input("ratio", data_type=float)
        self.add_input("attack_ms", data_type=float)
        self.add_input("release_ms", data_type=float)
        self.add_input("knee_db", data_type=float)
        self.add_input("sidechain_in", data_type=torch.Tensor)
        self.add_output("out", data_type=torch.Tensor)

        # Pre-allocate the output buffer for performance
        self._output_buffer = torch.zeros((DEFAULT_CHANNELS, DEFAULT_BLOCKSIZE), dtype=DEFAULT_DTYPE)
        self._last_shape = None

        # Synchronize parameters with the C++ object immediately on creation.
        self._update_cpp_params()

    def _get_state_snapshot_locked(self) -> dict:
        return self._get_parameters_state()

    def serialize_extra(self) -> dict:
        return self._serialize_parameters()

    def deserialize_extra(self, data: dict):
        # The helper acquires the lock, sets python-side state, and calls the on_change callbacks
        self._deserialize_parameters(data)
        # After loading state, ensure the C++ side is updated one final time.
        self._update_cpp_params()

    def _update_cpp_params_locked(self):
        """
        [LOCKED] Helper to send all current parameters to the C++ object.
        This method MUST be called when self._lock is already held.
        """
        if not self.dsp_handle:
            return

        self.lib.set_parameters(
            self.dsp_handle,
            DEFAULT_SAMPLERATE,
            self._threshold_db,
            self._ratio,
            self._attack_ms,
            self._release_ms,
            self._knee_db,
        )

    def _update_cpp_params(self):
        """[PUBLIC] Thread-safe method to update C++ parameters."""
        with self._lock:
            self._update_cpp_params_locked()

    def process(self, input_data: dict) -> dict:
        if not self.dsp_handle or self.error_state:
            return {"out": None}

        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}

        # Handle parameter updates from sockets. This will call the _locked on_change callback.
        self._update_parameters_from_sockets(input_data)

        # --- Marshall data and call C++ ---
        sidechain_signal = input_data.get("sidechain_in")

        # Resize output buffer if shape changes
        if signal.shape != self._last_shape:
            self._output_buffer = torch.zeros(signal.shape, dtype=DEFAULT_DTYPE)
            self._last_shape = signal.shape

        # Ensure tensors are contiguous and float32 for C++
        in_tensor = signal.contiguous().to(torch.float32)
        out_tensor = self._output_buffer

        num_channels, num_samples = in_tensor.shape

        # Create C-style arrays of pointers for planar data
        in_channel_pointers = (ctypes.POINTER(ctypes.c_float) * num_channels)()
        out_channel_pointers = (ctypes.POINTER(ctypes.c_float) * num_channels)()
        for i in range(num_channels):
            in_channel_pointers[i] = ctypes.cast(in_tensor[i].data_ptr(), ctypes.POINTER(ctypes.c_float))
            out_channel_pointers[i] = ctypes.cast(out_tensor[i].data_ptr(), ctypes.POINTER(ctypes.c_float))

        sidechain_channel_pointers = None
        if isinstance(sidechain_signal, torch.Tensor) and sidechain_signal.shape == in_tensor.shape:
            sidechain_tensor = sidechain_signal.contiguous().to(torch.float32)
            sidechain_channel_pointers = (ctypes.POINTER(ctypes.c_float) * num_channels)()
            for i in range(num_channels):
                sidechain_channel_pointers[i] = ctypes.cast(
                    sidechain_tensor[i].data_ptr(), ctypes.POINTER(ctypes.c_float)
                )

        # Call the C++ process function
        self.lib.process_block(
            self.dsp_handle,
            in_channel_pointers,
            sidechain_channel_pointers,
            out_channel_pointers,
            num_channels,
            num_samples,
        )

        return {"out": out_tensor.clone()}
