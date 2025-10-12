import torch
import numpy as np
import logging
from collections import deque
from node_system import Node

# ParameterNodeItem is now the standard UI for all parameter-driven nodes
# managed_parameters and Parameter are the core of the boilerplate reduction
from ui_elements import ParameterNodeItem
from node_helpers import managed_parameters, Parameter

from constants import DEFAULT_DTYPE, TICK_DURATION_S

logger = logging.getLogger(__name__)

EPSILON = 1e-9  # For safe division


# ============================================================
# 1. ValueNode
# ============================================================


class ValueNodeItem(ParameterNodeItem):
    """
    Generic UI for a node that provides a single float value via a spinbox.
    """

    NODE_SPECIFIC_WIDTH = 150

    def __init__(self, node_logic: "ValueNode"):
        parameters = [
            {
                "key": "value",
                "name": "Value",
                "type": "spinbox",  # Using a new spinbox type
                "min": -1000000.0,
                "max": 1000000.0,
                "decimals": 3,
                "step": 0.1,
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


@managed_parameters
class ValueNode(Node):
    NODE_TYPE = "Value"
    UI_CLASS = ValueNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Provides a constant floating-point value."

    # Declare the managed parameter
    value = Parameter(default=0.0)

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.add_output("out", data_type=float)

    def process(self, input_data: dict) -> dict:
        # The decorator handles everything. We just need to output the internal value.
        with self._lock:
            return {"out": self._value}


# ============================================================
# 2. DialNode (Refactored)
# ============================================================


class DialNodeItem(ParameterNodeItem):
    """UI for a 0-1 dial, using the declarative ParameterNodeItem."""

    NODE_SPECIFIC_WIDTH = 150

    def __init__(self, node_logic: "DialNode"):
        parameters = [
            {
                "key": "value",
                "name": "Value",
                "type": "dial",
                "min": 0.0,
                "max": 1.0,
                "format": "{:.2f}",
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


@managed_parameters
class DialNode(Node):
    NODE_TYPE = "Dial (0-1)"
    UI_CLASS = DialNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Provides a float value between 0.0 and 1.0."

    # Declare the parameter with clipping to enforce the 0-1 range
    value = Parameter(default=0.5, clip=(0.0, 1.0))

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.add_output("out", data_type=float)

    def process(self, input_data: dict) -> dict:
        with self._lock:
            return {"out": self._value}


# ============================================================
# 3. RunningAverageNode
# ============================================================


class RunningAverageNodeItem(ParameterNodeItem):
    """UI for RunningAverageNode, now using ParameterNodeItem."""

    NODE_SPECIFIC_WIDTH = 180

    def __init__(self, node_logic: "RunningAverageNode"):
        parameters = [
            {
                "key": "time",
                "name": "Avg. Time (s)",
                "type": "spinbox",
                "min": 0.01,
                "max": 60.0,
                "decimals": 2,
                "step": 0.1,
            }
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


@managed_parameters
class RunningAverageNode(Node):
    NODE_TYPE = "Running Average"
    UI_CLASS = RunningAverageNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Calculates the running average of a float value over a specified time."

    time = Parameter(default=1.0, clip=(0.01, 60.0))

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=float)
        self.add_input("time", data_type=float)  # Socket name matches parameter
        self.add_output("out", data_type=float)
        self._current_average = 0.0

    def process(self, input_data: dict) -> dict:
        input_value = input_data.get("in")
        if input_value is None:
            return {"out": self._current_average}

        self._update_params_from_sockets(input_data)

        with self._lock:
            current_time_s = self._time

        alpha = 1.0 - np.exp(-TICK_DURATION_S / (current_time_s + EPSILON))
        try:
            self._current_average += alpha * (float(input_value) - self._current_average)
        except (ValueError, TypeError):
            pass  # Hold last value if input is invalid

        return {"out": self._current_average}

    def start(self):
        with self._lock:  # may not need lock here?
            self._current_average = 0.0
        super().start()


# ============================================================
# 4. RouteNode
# ============================================================
class RouteNode(Node):
    NODE_TYPE = "Route"
    CATEGORY = "Utility"
    DESCRIPTION = "Simple pass-through node for organizing connections."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=None)
        self.add_output("out", data_type=None)

    def process(self, input_data):
        return {"out": input_data.get("in")}


# ==============================================================================
# 5. SignalAnalyzer Node
# ==============================================================================
class SignalAnalyzer(Node):
    NODE_TYPE = "Signal Analyzer"
    CATEGORY = "Utility"
    DESCRIPTION = "Analyzes a signal block and outputs its RMS, Peak, Crest Factor, DC Offset, and Zero-Crossing Rate."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_output("rms", data_type=float)
        self.add_output("peak", data_type=float)
        self.add_output("crest_factor", data_type=float)
        self.add_output("dc_offset", data_type=float)
        self.add_output("zero_crossing_rate", data_type=float)

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        stats = {"rms": None, "peak": None, "crest_factor": None, "dc_offset": None, "zero_crossing_rate": None}
        if not isinstance(signal, torch.Tensor) or signal.numel() == 0:
            return stats
        try:
            mono_signal = torch.mean(signal, dim=0) if signal.ndim > 1 else signal
            rms = torch.sqrt(torch.mean(torch.square(mono_signal)))
            peak = torch.max(torch.abs(mono_signal))
            stats = {
                "rms": rms.item(),
                "peak": peak.item(),
                "crest_factor": (peak / (rms + EPSILON)).item(),
                "dc_offset": torch.mean(mono_signal).item(),
                "zero_crossing_rate": (torch.mean(torch.abs(torch.diff(torch.sign(mono_signal)))) / 2.0).item(),
            }
        except Exception as e:
            logger.error(f"[{self.name}] Error during calculation: {e}", exc_info=True)
        return stats


# ==============================================================================
# 6. Dial (Hz) Node (Refactored)
# ==============================================================================


class DialHzNodeItem(ParameterNodeItem):
    """UI for the DialHzNode."""

    NODE_SPECIFIC_WIDTH = 160

    def __init__(self, node_logic: "DialHzNode"):
        parameters = [
            {
                "key": "frequency",
                "name": "Frequency",
                "type": "dial",
                "min": 20.0,
                "max": 20000.0,
                "format": "{:.1f} Hz",
                "is_log": True,
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


@managed_parameters
class DialHzNode(Node):
    NODE_TYPE = "Dial (Hz)"
    UI_CLASS = DialHzNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Provides a frequency value (20-20k Hz) with a logarithmic dial."

    frequency = Parameter(default=440.0, clip=(20.0, 20000.0))

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.add_input("frequency", data_type=float)
        self.add_output("freq_out", data_type=float)

    def process(self, input_data: dict) -> dict:
        self._update_params_from_sockets(input_data)
        with self._lock:
            return {"freq_out": self._frequency}


# ==============================================================================
# 7. Gain Node
# ==============================================================================


class GainNodeItem(ParameterNodeItem):
    """UI for the GainNode."""

    NODE_SPECIFIC_WIDTH = 160

    def __init__(self, node_logic: "GainNode"):
        parameters = [
            {
                "key": "gain_db",
                "name": "Gain",
                "type": "dial",
                "min": -60.0,
                "max": 12.0,
                "format": "{:.1f} dB",
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


@managed_parameters
class GainNode(Node):
    NODE_TYPE = "Gain"
    UI_CLASS = GainNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Applies gain (volume) to an audio signal, controlled in decibels."

    gain_db = Parameter(default=0.0, clip=(-60.0, 12.0))

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("gain_db", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}

        self._update_params_from_sockets(input_data)

        with self._lock:
            gain_db = self._gain_db

        amplitude_factor = 10.0 ** (gain_db / 20.0)
        return {"out": signal * amplitude_factor}
