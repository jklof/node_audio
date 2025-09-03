import torch
import numpy as np
import threading
import logging
from enum import Enum
from typing import Dict

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE

# --- UI and Qt Imports ---
from ui_elements import ParameterNodeItem, NodeStateEmitter
from PySide6.QtCore import Slot

# Configure logging
logger = logging.getLogger(__name__)


# ==============================================================================
# Enum for Shaper Types
# ==============================================================================
class ShaperType(Enum):
    SOFT_CLIP = "Soft Clip (Tanh)"
    HARD_CLIP = "Hard Clip"
    FOLD = "Foldback"
    SINE = "Sine Distortion"


# ==============================================================================
# UI Class for the WaveShaper Node (REFACTORED)
# ==============================================================================
class WaveShaperNodeItem(ParameterNodeItem):
    """
    Refactored UI for the WaveShaperNode.
    This class now fully leverages the enhanced ParameterNodeItem by defining its
    entire UI declaratively, including the new combobox and dial.
    """

    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "WaveShaperNode"):
        # Define the parameters and their control types for this node
        parameters = [
            {
                "key": "shaper_type",
                "name": "Shaper Type",
                "type": "combobox",
                "items": [(st.value, st) for st in ShaperType],
            },
            {
                "key": "drive",
                "name": "Drive",
                "type": "dial",  # Use the new dial control
                "min": 1.0,
                "max": 100.0,
                "format": "{:.1f}",
                "is_log": True,  # Logarithmic scale feels better for drive
            },
            {
                "key": "mix",
                "name": "Mix",
                "type": "slider",  # Keep mix as a slider
                "min": 0.0,
                "max": 1.0,
                "format": "{:.0%}",
                "is_log": False,
            },
        ]

        # The parent class now handles the creation of all specified controls.
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# WaveShaper Logic Node (Unchanged)
# ==============================================================================
class WaveShaperNode(Node):
    NODE_TYPE = "WaveShaper"
    UI_CLASS = WaveShaperNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Applies non-linear distortion to a signal to add harmonics."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("drive", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._lock = threading.Lock()
        self._shaper_type: ShaperType = ShaperType.SOFT_CLIP
        self._drive: float = 1.0
        self._mix: float = 1.0

    def get_current_state_snapshot(self, locked: bool = False) -> Dict:
        """Returns a copy of the current parameters for UI or serialization."""
        state = {"shaper_type": self._shaper_type, "drive": self._drive, "mix": self._mix}
        if locked:
            return state
        with self._lock:
            return state

    @Slot(ShaperType)
    def set_shaper_type(self, shaper_type: ShaperType):
        state_to_emit = None
        with self._lock:
            if self._shaper_type != shaper_type:
                self._shaper_type = shaper_type
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_drive(self, drive: float):
        state_to_emit = None
        with self._lock:
            new_drive = np.clip(float(drive), 1.0, 100.0).item()
            if self._drive != new_drive:
                self._drive = new_drive
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_mix(self, mix: float):
        state_to_emit = None
        with self._lock:
            new_mix = np.clip(float(mix), 0.0, 1.0).item()
            if self._mix != new_mix:
                self._mix = new_mix
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}

        # --- CORRECTED: State update logic to prevent calling setters from audio thread ---
        state_to_emit = None
        ui_update_needed = False
        with self._lock:
            # Check for changes from input sockets and update values directly
            drive_socket_val = input_data.get("drive")
            if drive_socket_val is not None:
                clipped_val = np.clip(float(drive_socket_val), 1.0, 100.0).item()
                if self._drive != clipped_val:
                    self._drive = clipped_val
                    ui_update_needed = True

            mix_socket_val = input_data.get("mix")
            if mix_socket_val is not None:
                clipped_val = np.clip(float(mix_socket_val), 0.0, 1.0).item()
                if self._mix != clipped_val:
                    self._mix = clipped_val
                    ui_update_needed = True

            # Copy current state to local variables for processing
            drive = self._drive
            mix = self._mix
            shaper_type = self._shaper_type

            # If a value changed, get a state snapshot to emit after releasing the lock
            if ui_update_needed:
                state_to_emit = self.get_current_state_snapshot(locked=True)

        # Emit signal to UI AFTER the lock is released
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)
        # --- END CORRECTION ---

        # All processing is now done with PyTorch
        driven_signal = signal * drive
        output_signal = None

        if shaper_type == ShaperType.SOFT_CLIP:
            output_signal = torch.tanh(driven_signal)
        elif shaper_type == ShaperType.HARD_CLIP:
            output_signal = torch.clamp(driven_signal, -1.0, 1.0)
        elif shaper_type == ShaperType.FOLD:
            # This foldback logic is a simple, effective approximation
            output_signal = torch.abs(torch.fmod(driven_signal + 1, 4) - 2) - 1
        elif shaper_type == ShaperType.SINE:
            # Clip the input to +/- 1 to keep the sine function within a single cycle
            clipped_driven_signal = torch.clamp(driven_signal, -1.0, 1.0)
            output_signal = torch.sin(0.5 * torch.pi * clipped_driven_signal)

        # Apply dry-wet mix
        final_signal = signal * (1 - mix) + output_signal * mix

        return {"out": final_signal.to(DEFAULT_DTYPE)}

    def serialize_extra(self) -> dict:
        with self._lock:
            return {
                "shaper_type": self._shaper_type.name,
                "drive": self._drive,
                "mix": self._mix,
            }

    def deserialize_extra(self, data: dict):
        shaper_type_name = data.get("shaper_type", ShaperType.SOFT_CLIP.name)
        try:
            shaper_type_enum = ShaperType[shaper_type_name]
        except KeyError:
            shaper_type_enum = ShaperType.SOFT_CLIP

        # Use the public setters during deserialization. This happens on the main
        # thread when the graph is loading, so it is perfectly safe and ensures
        # the UI is correctly initialized with the loaded state.
        self.set_shaper_type(shaper_type_enum)
        self.set_drive(data.get("drive", 1.0))
        self.set_mix(data.get("mix", 1.0))
