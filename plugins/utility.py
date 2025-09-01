# === File: plugins/utility.py ===

import numpy as np
import threading
import logging
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING  # <-- Added NODE_CONTENT_PADDING
from constants import DEFAULT_DTYPE

from PySide6.QtWidgets import QDoubleSpinBox, QVBoxLayout, QWidget, QDial, QSizePolicy, QLabel  # <-- Added QLabel
from PySide6.QtCore import Qt, Slot, QSignalBlocker, Signal, QObject  # <-- Added Signal, QObject
from PySide6.QtGui import QFontMetrics  # <-- Added QFontMetrics


logger = logging.getLogger(__name__)

EPSILON = 1e-9  # For safe division


# ============================================================
# UI for the ValueNode
# ============================================================
class ValueNodeItem(NodeItem):
    """Custom UI for ValueNode, featuring a QDoubleSpinBox."""

    def __init__(self, node_logic: "ValueNode"):
        # 1. Initialize the base NodeItem.
        super().__init__(node_logic)

        # 2. Create and set the custom content widget
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

        # 3. Connect signals and perform initial sync
        self.spin_box.valueChanged.connect(self.node_logic.set_value)
        self.updateFromLogic()

    @Slot()
    def updateFromLogic(self):
        """Syncs the UI widget's state from the logic node."""
        current_logic_value = self.node_logic.get_value()
        if abs(self.spin_box.value() - current_logic_value) > 1e-9:
            self.spin_box.blockSignals(True)
            self.spin_box.setValue(current_logic_value)
            self.spin_box.blockSignals(False)
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
        self.updateFromLogic()

    def _on_dial_change(self, dial_int_value: int):
        logical_value = max(0.0, min(1.0, dial_int_value / self.DIAL_MAX))
        self.node_logic.set_value(logical_value)

    @Slot()
    def updateFromLogic(self):
        logical_value = self.node_logic.get_value()
        dial_int_value = int(round(logical_value * self.DIAL_MAX))
        if self.dial.value() != dial_int_value:
            self.dial.blockSignals(True)
            self.dial.setValue(dial_int_value)
            self.dial.blockSignals(False)
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
        self.add_output("out", data_type=float)
        self._lock = threading.Lock()
        self._value = 0.0

    def get_value(self) -> float:
        """Thread-safe getter for the current value."""
        with self._lock:
            return self._value

    @Slot(float)
    def set_value(self, value: float):
        """Thread-safe slot for the UI to set the value."""
        try:
            new_value = float(value)
            with self._lock:
                if self._value != new_value:
                    self._value = new_value
        except (ValueError, TypeError):
            logger.warning(f"ValueNode [{self.name}]: Invalid value received: {value}")

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
# Logic for the RouteNode
# ============================================================


class RouteNode(Node):
    NODE_TYPE = "Route"
    CATEGORY = "Utility"
    DESCRIPTION = "Simple pass-through node for organizing connections."  # Added description

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        # Use None for data_type to allow any type (more flexible routing)
        # Or keep np.ndarray if only audio routing is intended
        self.add_input("in", data_type=None)
        self.add_output("out", data_type=None)
        logger.debug(f"RouteNode [{self.name}] initialized.")

    def process(self, input_data):
        # Simply pass the input data directly to the output
        signal = input_data.get("in")
        # No processing needed, just return the input value
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

        # --- Sockets ---
        self.add_input("in", data_type=np.ndarray)
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

        # --- Handle Invalid Input ---
        if signal is None or not isinstance(signal, np.ndarray) or signal.size == 0:
            return stats

        # --- Signal Analysis ---
        try:
            # Ensure float dtype for calculations and get a mono signal
            signal = signal.astype(DEFAULT_DTYPE)
            if signal.ndim > 1:
                # Average across channels to get a mono representation
                mono_signal = np.mean(signal, axis=1)
            else:
                mono_signal = signal

            # RMS (Root Mean Square) - A measure of loudness
            rms = np.sqrt(np.mean(np.square(mono_signal)))

            # Peak - The maximum absolute amplitude
            peak = np.max(np.abs(mono_signal))

            # Crest Factor - Ratio of peak to RMS. Indicates dynamic range.
            crest = peak / (rms + EPSILON)  # Add epsilon to prevent division by zero

            # DC Offset - The average of all samples (should be near zero)
            dc_offset = np.mean(mono_signal)

            # Zero-Crossing Rate (ZCR) - A simple measure of spectral brightness
            # Normalized to be between 0 and 1
            zcr = np.mean(np.abs(np.diff(np.sign(mono_signal)))) / 2.0

            stats = {
                "rms": float(rms),
                "peak": float(peak),
                "crest_factor": float(crest),
                "dc_offset": float(dc_offset),
                "zero_crossing_rate": float(zcr),
            }
        except Exception as e:
            logger.error(f"[{self.name}] Error during calculation: {e}", exc_info=True)

        # --- Return values to output sockets ---
        return stats

    def start(self):
        super().start()

    def stop(self):
        super().stop()

    def serialize_extra(self) -> dict:
        # No state to save.
        return {}

    def deserialize_extra(self, data: dict):
        # No state to load.
        pass


# ==============================================================================
# --- NEW: UI for the Dial (Hz) Node ---
# ==============================================================================
class DialHzNodeItem(NodeItem):
    """Custom UI for the DialHzNode, featuring a logarithmic frequency dial."""

    NODE_SPECIFIC_WIDTH = 160

    def __init__(self, node_logic: "DialHzNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(6)

        # --- Create Dial and Labels ---
        self.freq_dial = QDial()
        self.freq_dial.setRange(0, 1000)
        self.freq_dial.setNotchesVisible(True)

        self.title_label = QLabel("Frequency")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label = QLabel("...")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Ensure minimum width to prevent UI jitter
        fm = QFontMetrics(self.value_label.font())
        min_width = fm.boundingRect("20000.0 Hz").width()
        self.title_label.setMinimumWidth(min_width)

        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.value_label)
        main_layout.addWidget(self.freq_dial)

        self.setContentWidget(self.container_widget)

        # --- Connect Signals ---
        self.freq_dial.valueChanged.connect(self._handle_freq_change)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

        self.updateFromLogic()

    @Slot(int)
    def _handle_freq_change(self, dial_value: int):
        # Logarithmic mapping for more intuitive frequency control
        min_f, max_f = 20.0, 20000.0
        log_min, log_max = np.log10(min_f), np.log10(max_f)
        freq = 10 ** (((dial_value / 1000.0) * (log_max - log_min)) + log_min)
        self.node_logic.set_frequency(freq)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        """Updates the UI from a state dictionary."""
        freq = state.get("frequency", 440.0)

        # Update Dial Position
        with QSignalBlocker(self.freq_dial):
            min_f, max_f = 20.0, 20000.0
            log_min, log_max = np.log10(min_f), np.log10(max_f)
            dial_val = int(1000.0 * (np.log10(freq) - log_min) / (log_max - log_min))
            self.freq_dial.setValue(dial_val)

        # Update Label Text
        is_freq_ext = "freq_in" in self.node_logic.inputs and self.node_logic.inputs["freq_in"].connections
        self.value_label.setText(f"{freq:.1f} Hz{' (ext)' if is_freq_ext else ''}")
        self.freq_dial.setEnabled(not is_freq_ext)

    @Slot()
    def updateFromLogic(self):
        """Requests a full state snapshot from the logic and updates the UI."""
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ==============================================================================
# --- NEW: Logic for the Dial (Hz) Node ---
# ==============================================================================
class DialHzNodeEmitter(QObject):
    """A dedicated QObject to safely emit signals from the logic to the UI thread."""

    stateUpdated = Signal(dict)


class DialHzNode(Node):
    NODE_TYPE = "Dial (Hz)"
    UI_CLASS = DialHzNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Provides a frequency value (20-20k Hz) with a logarithmic dial."

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.emitter = DialHzNodeEmitter()
        self.add_input("freq_in", data_type=float)
        self.add_output("freq_out", data_type=float)
        self._lock = threading.Lock()
        self._frequency = 440.0

    def get_current_state_snapshot(self, locked: bool = False):
        """Returns a copy of the current parameters for UI synchronization."""
        if locked:
            return {"frequency": self._frequency}
        with self._lock:
            return {"frequency": self._frequency}

    @Slot(float)
    def set_frequency(self, frequency: float):
        """Thread-safe slot for the UI to set the value."""
        with self._lock:
            new_freq = np.clip(float(frequency), 20.0, 20000.0)
            if self._frequency != new_freq:
                self._frequency = new_freq
                state_to_emit = self.get_current_state_snapshot(locked=True)
                self.emitter.stateUpdated.emit(state_to_emit)

    def process(self, input_data: dict) -> dict:
        state_snapshot_to_emit = None
        with self._lock:
            # Prioritize socket input for frequency
            freq_socket = input_data.get("freq_in")
            if freq_socket is not None:
                new_freq = np.clip(float(freq_socket), 20.0, 20000.0)
                if abs(self._frequency - new_freq) > 1e-6:
                    self._frequency = new_freq
                    state_snapshot_to_emit = self.get_current_state_snapshot(locked=True)

            output_freq = self._frequency

        # Emit signal after releasing the lock
        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

        return {"freq_out": output_freq}

    def serialize_extra(self) -> dict:
        with self._lock:
            return {"frequency": self._frequency}

    def deserialize_extra(self, data: dict):
        self.set_frequency(data.get("frequency", 440.0))
