import numpy as np
from node_system import Node
from constants import DEFAULT_DTYPE

# Configure logging
import logging
logger = logging.getLogger(__name__)

class MultiplyAndAdd(Node):
    NODE_TYPE = "Multiply Add"
    CATEGORY = "Math"
    DESCRIPTION = "Outputs (value * multiply) + add. Inputs are floats."  # Updated description

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("value", data_type=float)
        self.add_input("multiply", data_type=float)
        self.add_input("add", data_type=float)
        self.add_output("out", data_type=float)
        logger.debug(f"MultiplyAndAdd Node [{self.name}] initialized.")

    def process(self, input_data):
        v = input_data.get("value")
        m = input_data.get("multiply")
        a = input_data.get("add")  # Renamed from 'c' for clarity

        # If main value is missing, output is undefined (None)
        if v is None:
            logger.debug(f"MultiplyAndAdd [{self.name}]: 'value' input is None.")
            return {"out": None}

        # Use defaults if multiply or add are missing
        if m is None:
            m = 1.0
        if a is None:
            a = 0.0

        # Perform calculation safely
        try:
            result = float((v * m) + a)
            return {"out": result}
        except (TypeError, ValueError) as e:
            logger.warning(f"MultiplyAndAdd [{self.name}]: Calculation error (Inputs: v={v}, m={m}, a={a}). Error: {e}")
            return {"out": None}

class ComparatorNode(Node):
    NODE_TYPE = "Comparator"
    CATEGORY = "Math"
    DESCRIPTION = "Outputs True if 'in' > 'threshold', else False."

    def __init__(self, name: str, node_id: str = None):
        super().__init__(name, node_id)
        
        # Sockets
        self.add_input("in", data_type=float)
        self.add_input("threshold", data_type=float)
        self.add_output("out", data_type=bool)
        
        self.default_threshold = 0.0

    def process(self, input_data: dict) -> dict:
        input_value = input_data.get("in")
        
        if input_value is None:
            return {"out": False}

        threshold = input_data.get("threshold", self.default_threshold)
        
        try:
            is_greater = float(input_value) > float(threshold)
            return {"out": is_greater}
        except (TypeError, ValueError):
            return {"out": False}

class AddSignalsNode(Node):
    NODE_TYPE = "Add Signals"
    CATEGORY = "Math"
    DESCRIPTION = "Adds two input signals element-wise."  # Updated description

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in1", data_type=np.ndarray)
        self.add_input("in2", data_type=np.ndarray)
        self.add_output("out", data_type=np.ndarray)
        logger.debug(f"AddSignalsNode [{self.name}] initialized.")

    def process(self, input_data):
        signal1 = input_data.get("in1")
        signal2 = input_data.get("in2")

        # Handle cases where one or both inputs are None
        if signal1 is None and signal2 is None:
            logger.debug(f"AddSignalsNode [{self.name}]: Both inputs are None.")
            return {"out": None}
        elif signal1 is None:
            # If only signal2 exists, pass it through (ensure correct type)
            if isinstance(signal2, np.ndarray):
                return {"out": signal2.astype(DEFAULT_DTYPE)}
            else:
                logger.warning(f"AddSignalsNode [{self.name}]: Input 'in1' is None, 'in2' is not a numpy array (type: {type(signal2)}).")
                return {"out": None}
        elif signal2 is None:
            # If only signal1 exists, pass it through (ensure correct type)
            if isinstance(signal1, np.ndarray):
                return {"out": signal1.astype(DEFAULT_DTYPE)}
            else:
                logger.warning(f"AddSignalsNode [{self.name}]: Input 'in2' is None, 'in1' is not a numpy array (type: {type(signal1)}).")
                return {"out": None}

        # Both signals are present, check types are numpy arrays
        if not isinstance(signal1, np.ndarray) or not isinstance(signal2, np.ndarray):
            logger.warning(f"AddSignalsNode [{self.name}]: Invalid input types. Expected numpy arrays, got ({type(signal1)}, {type(signal2)}).")
            return {"out": None}

        # Attempt addition (handles shape mismatches etc.)
        try:
            # Broadcasting might occur if shapes are compatible but not identical
            result = signal1 + signal2
            # Ensure output is the default float type
            return {"out": result.astype(DEFAULT_DTYPE)}
        except ValueError as e:
            # Specifically catch ValueError which often indicates shape mismatch
            logger.warning(f"AddNode [{self.name}]: Error adding signals (likely shape mismatch: {signal1.shape} vs {signal2.shape}). Error: {e}")
            return {"out": None}
        except Exception as e:
            logger.error(f"AddNode [{self.name}]: Unexpected error adding signals: {e}", exc_info=True)
            return {"out": None}

class GainBiasNode(Node):
    NODE_TYPE = "Gain & Bias"
    CATEGORY = "Math"
    DESCRIPTION = "Applies gain (multiply) and bias (add) to a signal."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=np.ndarray)
        self.add_input("gain", data_type=float)
        self.add_input("bias", data_type=float)
        self.add_output("out", data_type=np.ndarray)

    def process(self, input_data):
        signal = input_data.get("in")
        if signal is None:
            return {"out": None}

        # Handle missing or invalid signal input
        if signal is None:
            return {"out": None}
        if not isinstance(signal, np.ndarray):
            logger.warning(f"GainBiasNode [{self.name}]: 'in' signal is not a numpy array (type: {type(signal)}).")
            return {"out": None}

        gain_input = input_data.get("gain")
        bias_input = input_data.get("bias")

        gain = float(gain_input) if gain_input is not None else 1.0
        bias = float(bias_input) if bias_input is not None else 0.0
        
        result = (signal * gain) + bias
        return {"out": result.astype(DEFAULT_DTYPE)}