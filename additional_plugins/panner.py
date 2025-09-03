import torch
import numpy as np
import threading
import logging
from typing import Dict

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE, DEFAULT_BLOCKSIZE

# --- UI and Qt Imports ---
from ui_elements import ParameterNodeItem, NodeStateEmitter, NODE_CONTENT_PADDING
from PySide6.QtCore import Qt, Slot

# --- Configure logging ---
logger = logging.getLogger(__name__)

# --- Constants for Panner Node ---
MIN_PAN = -1.0  # Hard Left
MAX_PAN = 1.0  # Hard Right


# ==============================================================================
# 1. UI Class for the Panner Node (REFACTORED)
# ==============================================================================
class PannerNodeItem(ParameterNodeItem):
    """
    Refactored UI for the PannerNode.
    Inherits from ParameterNodeItem to auto-generate controls and overrides
    the state update method to provide custom label formatting.
    """

    NODE_SPECIFIC_WIDTH = 180

    def __init__(self, node_logic: "PannerNode"):
        # Define the parameters declaratively. ParameterNodeItem will create the slider.
        parameters = [
            {
                "key": "pan",
                "name": "Pan",
                "min": MIN_PAN,
                "max": MAX_PAN,
                "format": "{:.2f}",  # A default format, will be overridden by custom label logic
            }
        ]
        # Initialize the parent class, which creates all UI elements
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)

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

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        """
        Overrides the base class method to apply custom text formatting to the label
        after the base functionality (like updating the slider) has been executed.
        """
        # First, let the parent class handle standard updates (slider position, enabled state).
        super()._on_state_updated(state)

        # Now, apply our custom formatting to the label that the parent class created.
        pan_value = state.get("pan", 0.0)
        is_connected = "pan" in self.node_logic.inputs and self.node_logic.inputs["pan"].connections

        # Access the label widget created by the parent class
        if "pan" in self._controls and "label" in self._controls["pan"]:
            label_widget = self._controls["pan"]["label"]
            label_widget.setText(self._format_pan_label(pan_value, is_connected))
            # Center the custom label text
            label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)


# ==============================================================================
# 2. Logic Class for the Panner Node (Unchanged)
# ==============================================================================
class PannerNode(Node):
    NODE_TYPE = "Panner"
    UI_CLASS = PannerNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Positions a mono signal in the stereo field using constant power panning."

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("pan", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._lock = threading.Lock()
        self._pan = 0.0  # -1.0 (L) to 1.0 (R)

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
        # Use the public setter to ensure the UI is updated on load
        self.set_pan(data.get("pan", 0.0))
