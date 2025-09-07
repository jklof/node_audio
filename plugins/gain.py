import torch
import numpy as np
import threading
import logging
from typing import Dict

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE

# --- UI and Qt Imports ---
from ui_elements import NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING
from PySide6.QtWidgets import QWidget, QLabel, QDial, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt, Slot, QSignalBlocker
from PySide6.QtGui import QFontMetrics

# Configure logging
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. UI Class for the Gain Node
# ==============================================================================


class GainNodeItem(NodeItem):
    """Custom UI for the GainNode, featuring a decibel-scaled dial."""

    NODE_SPECIFIC_WIDTH = 160

    def __init__(self, node_logic: "GainNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        # --- Define the recommended dB range for the UI dial ---
        self.min_db = -60.0
        self.max_db = 12.0

        # --- Create Widgets ---
        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(6)

        self.gain_dial = QDial()
        self.gain_dial.setRange(0, 1000)
        self.gain_dial.setNotchesVisible(True)

        self.title_label = QLabel("Gain")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label = QLabel("...")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- Ensure minimum width to prevent UI jitter with new range ---
        fm = QFontMetrics(self.value_label.font())
        min_width = fm.boundingRect("-60.0 dB (ext)").width()
        self.title_label.setMinimumWidth(min_width)

        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.value_label)
        main_layout.addWidget(self.gain_dial)

        self.setContentWidget(self.container_widget)

        # --- Connect Signals ---
        self.gain_dial.valueChanged.connect(self._handle_dial_change)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

    def _map_dial_to_db(self, dial_value: int) -> float:
        """Linearly maps the integer dial value to the logarithmic dB scale."""
        norm = dial_value / 1000.0
        return self.min_db + norm * (self.max_db - self.min_db)

    def _map_db_to_dial(self, db_value: float) -> int:
        """Maps a dB value back to the integer dial position."""
        # Add a small epsilon to the range to avoid division by zero if min==max
        range_db = (self.max_db - self.min_db) or 1e-9
        safe_db = np.clip(db_value, self.min_db, self.max_db)
        norm = (safe_db - self.min_db) / range_db
        return int(round(norm * 1000.0))

    @Slot(int)
    def _handle_dial_change(self, dial_value: int):
        """Called when the user turns the dial."""
        db_val = self._map_dial_to_db(dial_value)
        self.node_logic.set_gain_db(db_val)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        """Updates the UI when the logic node's state changes."""
        gain_db = state.get("gain_db", 0.0)

        # Update the dial's position, blocking signals to prevent a loop
        with QSignalBlocker(self.gain_dial):
            self.gain_dial.setValue(self._map_db_to_dial(gain_db))

        # Check if the gain is being controlled by an external node
        is_ext_controlled = "gain_db" in self.node_logic.inputs and self.node_logic.inputs["gain_db"].connections
        self.gain_dial.setEnabled(not is_ext_controlled)

        # Update the text label
        label_text = f"{gain_db:.1f} dB"
        if is_ext_controlled:
            label_text += " (ext)"
        self.value_label.setText(label_text)

    @Slot()
    def updateFromLogic(self):
        """
        Pulls the current state from the logic node and updates the UI.
        This is essential for initializing the UI when the node is first created.
        """
        # The logic node holds the single source of truth for the state.
        state = self.node_logic.get_current_state_snapshot()
        # Call the existing update slot with this initial state.
        self._on_state_updated(state)
        # Call the base class implementation.
        super().updateFromLogic()


# ==============================================================================
# 2. Logic Class for the Gain Node (No changes needed here)
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

        # --- Define Sockets ---
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("gain_db", data_type=float)  # For external control
        self.add_output("out", data_type=torch.Tensor)

        # --- Internal State ---
        self._gain_db: float = 0.0  # Default to 0 dB (no change)

    def get_current_state_snapshot(self, locked: bool = False) -> Dict:
        """Returns the current state for UI updates."""
        if locked:
            return {"gain_db": self._gain_db}
        with self._lock:
            return {"gain_db": self._gain_db}

    @Slot(float)
    def set_gain_db(self, db_value: float):
        """Thread-safe method for the UI to set the gain."""
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
            # Prioritize the external socket input over the internal dial value
            gain_db_socket = input_data.get("gain_db")
            if gain_db_socket is not None:
                current_gain_db = float(gain_db_socket)
                # If external control changes our state, we need to update the UI
                if abs(self._gain_db - current_gain_db) > 1e-6:
                    self._gain_db = current_gain_db
                    state_snapshot_to_emit = self.get_current_state_snapshot(locked=True)
            else:
                current_gain_db = self._gain_db

        # Emit UI update after releasing lock
        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

        # --- Core Processing ---
        # Convert decibels to a linear amplitude multiplier
        # Formula: amplitude = 10^(dB / 20)
        amplitude_factor = 10.0 ** (current_gain_db / 20.0)

        # Apply gain
        output_signal = signal * amplitude_factor
        return {"out": output_signal}

    def serialize_extra(self) -> dict:
        """Save the gain setting."""
        with self._lock:
            return {"gain_db": self._gain_db}

    def deserialize_extra(self, data: dict):
        """Load the gain setting."""
        # Use the public setter to ensure the UI is also updated
        self.set_gain_db(data.get("gain_db", 0.0))
