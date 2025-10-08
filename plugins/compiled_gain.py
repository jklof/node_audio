# === File: plugins/compiled_gain.py ===

import ctypes
import torch
import logging
from ffi_node import FFINodeBase
from ui_elements import ParameterNodeItem
from PySide6.QtCore import Slot

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. UI Class for the CompiledGainNode
# ==============================================================================


class CompiledGainNodeItem(ParameterNodeItem):
    """
    UI for the GainNode using the declarative ParameterNodeItem.
    """

    NODE_SPECIFIC_WIDTH = 160

    def __init__(self, node_logic: "CompiledGainNode"):
        parameters = [
            {
                "key": "gain_db",
                "name": "Gain",
                "type": "dial",
                "min": -60.0,
                "max": 12.0,
                "format": "{:.1f} dB",
                "is_log": False,
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 2. Logic Class for the CompiledGainNode
# ==============================================================================


class CompiledGainNode(FFINodeBase):
    NODE_TYPE = "Compiled Gain"
    UI_CLASS = CompiledGainNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Applies gain using a high-performance compiled C++ core."

    # --- Declarative API Definition ---
    # The FFINodeBase class will use this to find the library and bind the functions.
    LIB_NAME = "gain_processor"
    API = {
        "set_gain_db": {"argtypes": [ctypes.c_void_p, ctypes.c_float]},
        "process_block": {"argtypes": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]},
    }

    def __init__(self, name, node_id=None):
        # This calls the FFINodeBase constructor, which loads and binds the C++ library.
        super().__init__(name, node_id)

        # Define the node's sockets
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("gain_db", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        # Internal state for the parameter
        self._gain_db: float = 0.0

    def _get_state_snapshot_locked(self) -> dict:
        """Returns the current state for UI synchronization."""
        return {"gain_db": self._gain_db}

    @Slot(float)
    def set_gain_db(self, db_value: float):
        """
        Thread-safe slot called by the UI dial or from the process method.
        It updates the Python state and calls the specific C++ setter function.
        """
        state_to_emit = None
        new_db_value = float(db_value)

        with self._lock:
            # Only proceed if the value has actually changed
            if self._gain_db != new_db_value:
                self._gain_db = new_db_value
                if self.dsp_handle:
                    # Call the specific C function bound by the base class
                    self.lib.set_gain_db(self.dsp_handle, self._gain_db)
                # Get a snapshot of the new state to send back to the UI
                state_to_emit = self._get_state_snapshot_locked()

        # Emit the update signal outside the lock
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def process(self, input_data: dict) -> dict:
        """
        The main processing method, implemented by this subclass.
        """
        if not self.dsp_handle or self.error_state:
            return {"out": None}

        # --- Step 1: Handle parameter updates from input sockets ---
        # If the 'gain_db' input socket is connected, its value overrides the UI dial.
        gain_db_socket = input_data.get("gain_db")
        if gain_db_socket is not None:
            self.set_gain_db(float(gain_db_socket))

        # --- Step 2: Marshall the audio data for the C++ function ---
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}

        # Ensure the tensor's memory is contiguous and of type float32
        processed_signal = signal.contiguous().to(torch.float32)
        data_ptr = processed_signal.data_ptr()
        float_ptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_float))
        num_channels, num_samples = processed_signal.shape

        # --- Step 3: Call the C++ process function ---
        self.lib.process_block(self.dsp_handle, float_ptr, num_channels, num_samples)

        return {"out": processed_signal}

    def serialize_extra(self) -> dict:
        """Saves the node's state to a dictionary."""
        with self._lock:
            return {"gain_db": self._gain_db}

    def deserialize_extra(self, data: dict):
        """Loads the node's state from a dictionary."""
        # Use the public setter to ensure C++ state is also updated on load
        self.set_gain_db(data.get("gain_db", 0.0))
