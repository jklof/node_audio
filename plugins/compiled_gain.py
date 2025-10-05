import ctypes
import torch
import logging
from ffi_node import FFINodeBase
from ui_elements import ParameterNodeItem

logger = logging.getLogger(__name__)

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

class CompiledGainNode(FFINodeBase):
    NODE_TYPE = "Compiled Gain"
    UI_CLASS = CompiledGainNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Applies gain using a high-performance compiled C++ core."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("gain_db", data_type=float)
        self.add_output("out", data_type=torch.Tensor)
        self._gain_db: float = 0.0

        try:
            # We are using the corrected _load_library from the previous answer
            self._load_library("gain_processor", __file__)

            self.lib.set_gain_db.argtypes = [ctypes.c_void_p, ctypes.c_float]
            self.lib.process_block.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int
            ]
        except (FileNotFoundError, RuntimeError) as e:
            self.error_state = str(e)
            logger.error(f"[{self.name}] Failed to initialize: {e}")

    # --- FIX #1: Implement the state snapshot method ---
    def _get_state_snapshot_locked(self) -> dict:
        """Returns the current state for the UI."""
        return {"gain_db": self._gain_db}

    # --- FIX #2: Correctly implement the setter ---
    def set_gain_db(self, db_value: float):
        """Called by the UI dial. Updates C++ state and Python state, then notifies UI."""
        state_to_emit = None
        new_db_value = float(db_value)

        with self._lock:
            # Only proceed if the value has actually changed
            if self._gain_db != new_db_value:
                self._gain_db = new_db_value
                if self.dsp_handle:
                    # Update the state in the C++ object
                    self.lib.set_gain_db(self.dsp_handle, self._gain_db)
                # Get a snapshot of the new state to send back to the UI
                state_to_emit = self._get_state_snapshot_locked()
        
        # Emit the update signal outside the lock
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    # --- FIX #3: Add persistence methods ---
    def serialize_extra(self) -> dict:
        with self._lock:
            return {"gain_db": self._gain_db}

    def deserialize_extra(self, data: dict):
        # Use the setter to ensure C++ state is also updated
        self.set_gain_db(data.get("gain_db", 0.0))


    def process(self, input_data: dict) -> dict:
        if not self.dsp_handle or self.error_state:
            return {"out": None}

        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}
            
        # --- Handle external control from input socket ---
        gain_db_socket = input_data.get("gain_db")
        if gain_db_socket is not None:
            # If the socket is connected, its value overrides the dial.
            # Call our own setter to keep C++ and UI state in sync.
            self.set_gain_db(float(gain_db_socket))

        # --- The Magic Bridge ---
        processed_signal = signal.contiguous().to(torch.float32)
        data_ptr = processed_signal.data_ptr()
        float_ptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_float))
        num_channels, num_samples = processed_signal.shape

        # Call the C++ function to process the data in-place
        self.lib.process_block(self.dsp_handle, float_ptr, num_channels, num_samples)

        return {"out": processed_signal}