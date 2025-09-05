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
from ui_elements import NodeItem, NodeStateEmitter
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QComboBox,
    QDial,
    QVBoxLayout,
    QHBoxLayout,
)
from PySide6.QtCore import Qt, Slot, QSignalBlocker
from PySide6.QtGui import QFontMetrics

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
# UI Class for the WaveShaper Node
# ==============================================================================
class WaveShaperNodeItem(NodeItem):
    """Custom NodeItem for the WaveShaperNode, providing user controls."""

    NODE_SPECIFIC_WIDTH = 180

    def __init__(self, node_logic: "WaveShaperNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(6)

        # --- Shaper Type Selection ---
        main_layout.addWidget(QLabel("Shaper Type:"))
        self.type_combo = QComboBox()
        for st in ShaperType:
            self.type_combo.addItem(st.value, st)
        main_layout.addWidget(self.type_combo)

        # --- Drive Control ---
        drive_layout = QHBoxLayout()
        drive_layout.setSpacing(5)

        self.drive_dial = QDial()
        self.drive_dial.setRange(0, 1000)
        self.drive_dial.setNotchesVisible(True)

        drive_labels_vbox = QVBoxLayout()
        drive_labels_vbox.setSpacing(1)
        self.drive_title_label = QLabel("Drive")
        self.drive_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drive_value_label = QLabel("...")
        self.drive_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fm = QFontMetrics(self.drive_value_label.font())
        min_width = fm.boundingRect("100.0 (ext)").width()
        self.drive_title_label.setMinimumWidth(min_width)

        drive_labels_vbox.addWidget(self.drive_title_label)
        drive_labels_vbox.addWidget(self.drive_value_label)

        drive_layout.addWidget(self.drive_dial)
        drive_layout.addLayout(drive_labels_vbox)
        main_layout.addLayout(drive_layout)

        # --- Mix Control ---
        mix_layout = QHBoxLayout()
        mix_layout.setSpacing(5)

        self.mix_dial = QDial()
        self.mix_dial.setRange(0, 1000)
        self.mix_dial.setNotchesVisible(True)

        mix_labels_vbox = QVBoxLayout()
        mix_labels_vbox.setSpacing(1)
        self.mix_title_label = QLabel("Mix")
        self.mix_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mix_title_label.setMinimumWidth(min_width)
        self.mix_value_label = QLabel("...")
        self.mix_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        mix_labels_vbox.addWidget(self.mix_title_label)
        mix_labels_vbox.addWidget(self.mix_value_label)

        mix_layout.addWidget(self.mix_dial)
        mix_layout.addLayout(mix_labels_vbox)
        main_layout.addLayout(mix_layout)

        self.setContentWidget(self.container_widget)

        # --- Connect Signals ---
        self.type_combo.currentTextChanged.connect(self._handle_type_change)
        self.drive_dial.valueChanged.connect(self._handle_drive_change)
        self.mix_dial.valueChanged.connect(self._handle_mix_change)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

        self.updateFromLogic()

    @Slot(str)
    def _handle_type_change(self, type_text: str):
        selected_enum = self.type_combo.currentData()
        if isinstance(selected_enum, ShaperType):
            self.node_logic.set_shaper_type(selected_enum)

    @Slot(int)
    def _handle_drive_change(self, dial_value: int):
        logical_drive = 1.0 + (dial_value / 1000.0) * 99.0
        self.node_logic.set_drive(logical_drive)

    @Slot(int)
    def _handle_mix_change(self, dial_value: int):
        logical_mix = dial_value / 1000.0
        self.node_logic.set_mix(logical_mix)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        shaper_type = state.get("shaper_type")
        drive = state.get("drive", 1.0)
        mix = state.get("mix", 1.0)

        with QSignalBlocker(self.type_combo):
            index = self.type_combo.findData(shaper_type)
            if index != -1:
                self.type_combo.setCurrentIndex(index)

        with QSignalBlocker(self.drive_dial):
            dial_value = int(((drive - 1.0) / 99.0) * 1000.0)
            self.drive_dial.setValue(dial_value)

        is_drive_socket_connected = "drive" in self.node_logic.inputs and self.node_logic.inputs["drive"].connections
        self.drive_dial.setEnabled(not is_drive_socket_connected)
        label_text = f"{drive:.1f}{' (ext)' if is_drive_socket_connected else ''}"
        self.drive_value_label.setText(label_text)

        with QSignalBlocker(self.mix_dial):
            dial_value = int(mix * 1000.0)
            self.mix_dial.setValue(dial_value)

        is_mix_socket_connected = "mix" in self.node_logic.inputs and self.node_logic.inputs["mix"].connections
        self.mix_dial.setEnabled(not is_mix_socket_connected)
        label_text = f"{mix:.2f}{' (ext)' if is_mix_socket_connected else ''}"
        self.mix_value_label.setText(label_text)

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ==============================================================================
# WaveShaper Logic Node
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