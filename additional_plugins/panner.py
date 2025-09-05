import torch
import numpy as np
import threading
import logging
from typing import Dict

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE, DEFAULT_BLOCKSIZE

# --- UI and Qt Imports ---
from ui_elements import NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING # MODIFIED
from PySide6.QtWidgets import QWidget, QSlider, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, Slot, QSignalBlocker

# --- Configure logging ---
logger = logging.getLogger(__name__)

# --- Constants for Panner Node ---
MIN_PAN = -1.0  # Hard Left
MAX_PAN = 1.0   # Hard Right

# ==============================================================================
# 1. Emitter for UI Communication (REMOVED)
# The common NodeStateEmitter is now used.
# ==============================================================================

# ==============================================================================
# 2. UI Class for the Panner Node
# ==============================================================================
class PannerNodeItem(NodeItem):
    """Custom UI for the PannerNode, featuring a single slider for pan control."""
    NODE_SPECIFIC_WIDTH = 180

    def __init__(self, node_logic: "PannerNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(5)

        self.pan_label = QLabel("Pan: Center")
        self.pan_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pan_slider = QSlider(Qt.Orientation.Horizontal)
        self.pan_slider.setRange(int(MIN_PAN * 100), int(MAX_PAN * 100)) # Range -100 to 100

        main_layout.addWidget(self.pan_label)
        main_layout.addWidget(self.pan_slider)

        self.setContentWidget(self.container_widget)

        self.pan_slider.valueChanged.connect(self._handle_slider_change)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        # The initial state is set by the graph scene's sync logic.

    def _format_pan_label(self, value: float, is_external: bool) -> str:
        """Formats the pan value into a user-friendly string (e.g., L 50%, Center, R 100%)."""
        if abs(value) < 0.01:
            label = "Center"
        elif value < 0:
            label = f"L {abs(value):.0%}"
        else:
            label = f"R {value:.0%}"
        
        if is_external:
            label += " (ext)"
        return f"Pan: {label}"

    @Slot(int)
    def _handle_slider_change(self, value: int):
        logical_val = value / 100.0
        self.node_logic.set_pan(logical_val)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        pan_value = state.get("pan", 0.0)
        
        # Check if the pan input socket is connected
        is_connected = "pan" in self.node_logic.inputs and self.node_logic.inputs["pan"].connections
        self.pan_slider.setEnabled(not is_connected)

        # Update slider position
        with QSignalBlocker(self.pan_slider):
            self.pan_slider.setValue(int(np.clip(pan_value, MIN_PAN, MAX_PAN) * 100))
        
        # Update text label
        self.pan_label.setText(self._format_pan_label(pan_value, is_connected))

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()

# ==============================================================================
# 3. Logic Class for the Panner Node
# ==============================================================================
class PannerNode(Node):
    NODE_TYPE = "Panner"
    UI_CLASS = PannerNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Positions a mono signal in the stereo field using constant power panning."

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter() # MODIFIED
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("pan", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._lock = threading.Lock()
        self._pan = 0.0 # -1.0 (L) to 1.0 (R)

    @Slot(float)
    def set_pan(self, value: float):
        """Thread-safe setter for the pan parameter."""
        state_to_emit = None
        with self._lock:
            clipped_value = np.clip(float(value), MIN_PAN, MAX_PAN).item()
            if self._pan != clipped_value:
                self._pan = clipped_value
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def get_current_state_snapshot(self, locked: bool = False) -> Dict:
        """Returns a copy of the current parameters for UI or serialization."""
        state = {"pan": self._pan}
        if locked:
            return state
        with self._lock:
            return state

    def process(self, input_data: dict) -> dict:
        in_signal = input_data.get("in")
        if not isinstance(in_signal, torch.Tensor):
            # Create a silent stereo block if there's no input
            return {"out": torch.zeros((2, DEFAULT_BLOCKSIZE), dtype=DEFAULT_DTYPE)}

        state_snapshot_to_emit = None
        with self._lock:
            # Check for external pan control from the input socket
            pan_socket_val = input_data.get("pan")
            if pan_socket_val is not None:
                new_pan = np.clip(float(pan_socket_val), MIN_PAN, MAX_PAN).item()
                if self._pan != new_pan:
                    self._pan = new_pan
                    state_snapshot_to_emit = self.get_current_state_snapshot(locked=True)
            
            pan_value = self._pan
        
        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

        # --- DSP Processing ---
        
        # 1. Ensure input signal is mono for panning
        if in_signal.shape[0] > 1:
            # If stereo or multi-channel, mix down to mono
            mono_signal = torch.mean(in_signal, dim=0, keepdim=True)
        else:
            mono_signal = in_signal
            
        # 2. Apply constant power panning law
        # Map pan value from [-1, 1] to angle in radians [0, pi/2]
        pan_rad = (pan_value + 1.0) * 0.5 * (torch.pi / 2.0)
        
        gain_left = torch.cos(torch.tensor(pan_rad, dtype=DEFAULT_DTYPE))
        gain_right = torch.sin(torch.tensor(pan_rad, dtype=DEFAULT_DTYPE))
        
        # 3. Apply gains to create L and R channels
        # Broadcasting handles applying the scalar gain to the entire mono tensor
        left_channel = mono_signal * gain_left
        right_channel = mono_signal * gain_right
        
        # 4. Stack the channels to create a stereo output tensor
        stereo_out = torch.cat((left_channel, right_channel), dim=0)
        
        return {"out": stereo_out}

    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()

    def deserialize_extra(self, data: dict):
        # MODIFIED: Use the public setter to ensure the UI is updated on load
        self.set_pan(data.get("pan", 0.0))