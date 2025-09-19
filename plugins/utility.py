import torch
import numpy as np
import threading
import logging
from collections import deque
from node_system import Node

from ui_elements import NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING, ParameterNodeItem
from constants import DEFAULT_DTYPE, TICK_DURATION_S

from PySide6.QtWidgets import QDoubleSpinBox, QVBoxLayout, QWidget, QDial, QSizePolicy, QLabel
from PySide6.QtCore import Qt, Slot, QSignalBlocker, Signal, QObject
from PySide6.QtGui import QFontMetrics


logger = logging.getLogger(__name__)

EPSILON = 1e-9  # For safe division


# ============================================================
# UI for the ValueNode
# ============================================================
class ValueNodeItem(NodeItem):
    """Custom UI for ValueNode, featuring a QDoubleSpinBox."""

    def __init__(self, node_logic: "ValueNode"):
        super().__init__(node_logic)

        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spin_box = QDoubleSpinBox()
        self.spin_box.setMinimum(-1000000.0)
        self.spin_box.setMaximum(1000000.0)
        self.spin_box.setDecimals(3)
        self.spin_box.setSingleStep(0.1)
        self.spin_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.spin_box)
        self.container_widget.setLayout(layout)
        self.setContentWidget(self.container_widget)

        self.spin_box.valueChanged.connect(self.node_logic.set_value)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        """Updates the UI from a state dictionary."""
        value = state.get("value", 0.0)
        with QSignalBlocker(self.spin_box):
            self.spin_box.setValue(value)

    @Slot()
    def updateFromLogic(self):
        """Pulls the initial state from the logic node to initialize the UI."""
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ============================================================
# UI for the DialNode
# ============================================================
class DialNodeItem(NodeItem):
    """Custom UI for a 0-1 dial, demonstrating UI variation."""

    def __init__(self, node_logic: "ValueNode"):
        super().__init__(node_logic)
        self.DIAL_MAX = 100

        self.dial = QDial()
        self.dial.setRange(0, self.DIAL_MAX)
        self.dial.setNotchesVisible(True)
        self.dial.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.setContentWidget(self.dial)

        self.dial.valueChanged.connect(self._on_dial_change)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

    def _on_dial_change(self, dial_int_value: int):
        logical_value = max(0.0, min(1.0, dial_int_value / self.DIAL_MAX))
        self.node_logic.set_value(logical_value)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        """Updates the UI from a state dictionary."""
        value = state.get("value", 0.0)
        dial_int_value = int(round(value * self.DIAL_MAX))
        with QSignalBlocker(self.dial):
            self.dial.setValue(dial_int_value)

    @Slot()
    def updateFromLogic(self):
        """Pulls the initial state from the logic node to initialize the UI."""
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ============================================================
# Logic for the ValueNode (used by both UIs)
# ============================================================
class ValueNode(Node):
    NODE_TYPE = "Value"
    UI_CLASS = ValueNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Provides a constant floating-point value."

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self.add_output("out", data_type=float)
        self._lock = threading.Lock()
        self._value = 0.0

    def get_current_state_snapshot(self) -> dict:
        """Returns the current state for UI updates."""
        with self._lock:
            return {"value": self._value}

    @Slot(float)
    def set_value(self, value: float):
        """Thread-safe slot for the UI to set the value."""
        state_to_emit = None
        try:
            new_value = float(value)
            with self._lock:
                if self._value != new_value:
                    self._value = new_value
                    state_to_emit = {"value": self._value}
        except (ValueError, TypeError):
            logger.warning(f"ValueNode [{self.name}]: Invalid value received: {value}")
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def process(self, input_data: dict) -> dict:
        with self._lock:
            return {"out": self._value}

    def serialize_extra(self) -> dict:
        with self._lock:
            return {"value": self._value}

    def deserialize_extra(self, data: dict):
        self.set_value(data.get("value", 0.0))


# ============================================================
# Logic for the DialNode (inherits ValueNode, overrides UI and setter)
# ============================================================
class DialNode(ValueNode):
    NODE_TYPE = "Dial (0-1)"
    UI_CLASS = DialNodeItem
    DESCRIPTION = "Provides a float value between 0.0 and 1.0."

    @Slot(float)
    def set_value(self, value: float):
        """Overrides base setter to clamp value between 0 and 1."""
        clamped_value = max(0.0, min(1.0, float(value)))
        super().set_value(clamped_value)


# ============================================================
# UI for the RunningAverageNode
# ============================================================
class RunningAverageNodeItem(NodeItem):
    """Custom UI for RunningAverageNode."""

    def __init__(self, node_logic: "RunningAverageNode"):
        super().__init__(node_logic)

        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(4)

        self.time_label = QLabel("Avg. Time (s)")
        layout.addWidget(self.time_label)

        self.spin_box = QDoubleSpinBox()
        self.spin_box.setMinimum(0.01)  # A small minimum to avoid zero
        self.spin_box.setMaximum(60.0)  # A reasonable maximum
        self.spin_box.setDecimals(2)
        self.spin_box.setSingleStep(0.1)
        self.spin_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.spin_box)
        self.container_widget.setLayout(layout)
        self.setContentWidget(self.container_widget)

        self.spin_box.valueChanged.connect(self.node_logic.set_time)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        """Updates the UI from a state dictionary."""
        time_s = state.get("time_s", 1.0)
        is_ext_controlled = "time" in self.node_logic.inputs and self.node_logic.inputs["time"].connections

        with QSignalBlocker(self.spin_box):
            self.spin_box.setValue(time_s)

        self.spin_box.setEnabled(not is_ext_controlled)
        self.time_label.setText(f"Avg. Time (s){' (ext)' if is_ext_controlled else ''}")

    @Slot()
    def updateFromLogic(self):
        """Pulls the initial state from the logic node to initialize the UI."""
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ============================================================
# Logic for the RunningAverageNode
# ============================================================
class RunningAverageNode(Node):
    NODE_TYPE = "Running Average"
    UI_CLASS = RunningAverageNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Calculates the running average of a float value over a specified time."

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self.add_input("in", data_type=float)
        self.add_input("time", data_type=float)  # Time in seconds
        self.add_output("out", data_type=float)

        self._lock = threading.Lock()
        self._time_s = 1.0  # Default averaging time
        self._history = deque(maxlen=int(self._time_s / (TICK_DURATION_S + EPSILON)))

    def get_current_state_snapshot(self) -> dict:
        """Returns the current state for UI updates."""
        with self._lock:
            return {"time_s": self._time_s}

    @Slot(float)
    def set_time(self, time_s: float):
        """Thread-safe slot for the UI to set the averaging time."""
        state_to_emit = None
        try:
            new_time = float(time_s)
            if new_time < 0.01:  # Enforce a minimum
                new_time = 0.01
            with self._lock:
                if self._time_s != new_time:
                    self._time_s = new_time
                    state_to_emit = {"time_s": self._time_s}
        except (ValueError, TypeError):
            logger.warning(f"RunningAverageNode [{self.name}]: Invalid time received: {time_s}")
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def process(self, input_data: dict) -> dict:
        state_snapshot_to_emit = None
        output_value = None

        with self._lock:
            # Prioritize the external socket input for time
            time_socket = input_data.get("time")
            if time_socket is not None:
                current_time_s = max(0.01, float(time_socket))
                # If external control changes our state, we need to update the UI
                if abs(self._time_s - current_time_s) > 1e-6:
                    self._time_s = current_time_s
                    state_snapshot_to_emit = self.get_current_state_snapshot()
            else:
                current_time_s = self._time_s

            # Recalculate buffer size if needed
            new_maxlen = int(current_time_s / (TICK_DURATION_S + EPSILON))
            if new_maxlen < 1:
                new_maxlen = 1

            if self._history.maxlen != new_maxlen:
                # Create a new deque with the new maxlen, preserving existing data
                self._history = deque(self._history, maxlen=new_maxlen)

            # Add new value to history
            input_value = input_data.get("in")
            if input_value is not None:
                try:
                    self._history.append(float(input_value))
                except (ValueError, TypeError):
                    pass  # Ignore invalid inputs

            # Calculate average
            if self._history:
                output_value = sum(self._history) / len(self._history)

        # Emit UI update after releasing lock
        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

        return {"out": output_value}

    def start(self):
        """Clear history when processing starts to avoid stale data."""
        with self._lock:
            self._history.clear()
        super().start()

    def serialize_extra(self) -> dict:
        with self._lock:
            return {"time_s": self._time_s}

    def deserialize_extra(self, data: dict):
        self.set_time(data.get("time_s", 1.0))


# ============================================================
# Logic for the RouteNode
# ============================================================
class RouteNode(Node):
    NODE_TYPE = "Route"
    CATEGORY = "Utility"
    DESCRIPTION = "Simple pass-through node for organizing connections."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=None)
        self.add_output("out", data_type=None)
        logger.debug(f"RouteNode [{self.name}] initialized.")

    def process(self, input_data):
        signal = input_data.get("in")
        return {"out": signal}


# ==============================================================================
# Logic Class for the SignalAnalyzer Node
# ==============================================================================
class SignalAnalyzer(Node):
    """
    Analyzes an audio signal block and outputs various single-value metrics.
    """

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
        logger.debug(f"Node [{self.name}] initialized.")

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        stats = {
            "rms": None,
            "peak": None,
            "crest_factor": None,
            "dc_offset": None,
            "zero_crossing_rate": None,
        }
        if signal is None or not isinstance(signal, torch.Tensor) or signal.numel() == 0:
            return stats
        try:
            signal = signal.to(DEFAULT_DTYPE)
            mono_signal = torch.mean(signal, dim=0) if signal.ndim > 1 else signal
            rms = torch.sqrt(torch.mean(torch.square(mono_signal)))
            peak = torch.max(torch.abs(mono_signal))
            crest = peak / (rms + EPSILON)
            dc_offset = torch.mean(mono_signal)
            zcr = torch.mean(torch.abs(torch.diff(torch.sign(mono_signal)))) / 2.0
            stats = {
                "rms": rms.item(),
                "peak": peak.item(),
                "crest_factor": crest.item(),
                "dc_offset": dc_offset.item(),
                "zero_crossing_rate": zcr.item(),
            }
        except Exception as e:
            logger.error(f"[{self.name}] Error during calculation: {e}", exc_info=True)
        return stats

    def start(self):
        super().start()

    def stop(self):
        super().stop()

    def serialize_extra(self) -> dict:
        return {}

    def deserialize_extra(self, data: dict):
        pass


# ==============================================================================
# UI for the Dial (Hz) Node
# ==============================================================================
class DialHzNodeItem(ParameterNodeItem):
    """
    Refactored UI for the DialHzNode. The logarithmic mapping is now
    handled declaratively by the base class.
    """

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
                "is_log": True,  # This handles the logarithmic mapping
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# Logic for the Dial (Hz) Node
# ==============================================================================
class DialHzNode(Node):
    NODE_TYPE = "Dial (Hz)"
    UI_CLASS = DialHzNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Provides a frequency value (20-20k Hz) with a logarithmic dial."

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self.add_input("frequency", data_type=float)
        self.add_output("freq_out", data_type=float)
        self._lock = threading.Lock()
        self._frequency = 440.0

    def get_current_state_snapshot(self, locked: bool = False):
        if locked:
            return {"frequency": self._frequency}
        with self._lock:
            return {"frequency": self._frequency}

    @Slot(float)
    def set_frequency(self, frequency: float):
        state_to_emit = None
        with self._lock:
            new_freq = np.clip(float(frequency), 20.0, 20000.0)
            if self._frequency != new_freq:
                self._frequency = new_freq
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def process(self, input_data: dict) -> dict:
        state_snapshot_to_emit = None
        with self._lock:
            freq_socket = input_data.get("frequency")
            if freq_socket is not None:
                new_freq = np.clip(float(freq_socket), 20.0, 20000.0)
                if abs(self._frequency - new_freq) > 1e-6:
                    self._frequency = new_freq
                    state_snapshot_to_emit = self.get_current_state_snapshot(locked=True)
            output_freq = self._frequency
        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)
        return {"freq_out": output_freq}

    def serialize_extra(self) -> dict:
        with self._lock:
            return {"frequency": self._frequency}

    def deserialize_extra(self, data: dict):
        self.set_frequency(data.get("frequency", 440.0))


# ==============================================================================
# 1. UI Class for the Gain Node
# ==============================================================================
class GainNodeItem(ParameterNodeItem):
    """
    Refactored UI for the GainNode using the declarative ParameterNodeItem.
    """

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
                "is_log": False,
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 2. Logic Class for the Gain Node
# ==============================================================================
class GainNode(Node):
    NODE_TYPE = "Gain"
    UI_CLASS = GainNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Applies gain (volume) to an audio signal, controlled in decibels."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self._lock = threading.Lock()
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("gain_db", data_type=float)
        self.add_output("out", data_type=torch.Tensor)
        self._gain_db: float = 0.0

    def get_current_state_snapshot(self, locked: bool = False) -> dict:
        if locked:
            return {"gain_db": self._gain_db}
        with self._lock:
            return {"gain_db": self._gain_db}

    @Slot(float)
    def set_gain_db(self, db_value: float):
        state_to_emit = None
        with self._lock:
            new_db_value = float(db_value)
            if self._gain_db != new_db_value:
                self._gain_db = new_db_value
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}
        state_snapshot_to_emit = None
        with self._lock:
            gain_db_socket = input_data.get("gain_db")
            if gain_db_socket is not None:
                current_gain_db = float(gain_db_socket)
                if abs(self._gain_db - current_gain_db) > 1e-6:
                    self._gain_db = current_gain_db
                    state_snapshot_to_emit = self.get_current_state_snapshot(locked=True)
            else:
                current_gain_db = self._gain_db
        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

        amplitude_factor = 10.0 ** (np.clip(current_gain_db, -120.0, 60.0) / 20.0)
        output_signal = signal * amplitude_factor
        return {"out": output_signal}

    def serialize_extra(self) -> dict:
        with self._lock:
            return {"gain_db": self._gain_db}

    def deserialize_extra(self, data: dict):
        self.set_gain_db(data.get("gain_db", 0.0))
