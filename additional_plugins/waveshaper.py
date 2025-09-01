# File: additional_plugins/effects_waveshaper.py

import numpy as np
import threading
import logging
from enum import Enum
from typing import Dict

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE

# --- UI and Qt Imports ---
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QComboBox,
    QDial,
    QVBoxLayout,
    QHBoxLayout,
)
from PySide6.QtCore import Qt, Slot, QSignalBlocker, Signal, QObject
from PySide6.QtGui import QFontMetrics

# Configure logging
logger = logging.getLogger(__name__)


# ==============================================================================
# Enum for Shaper Types and Emitter for UI Communication
# ==============================================================================
class ShaperType(Enum):
    SOFT_CLIP = "Soft Clip (Tanh)"
    HARD_CLIP = "Hard Clip"
    FOLD = "Foldback"
    SINE = "Sine Distortion"

class WaveShaperEmitter(QObject):
    """A dedicated QObject to safely emit signals from the logic to the UI thread."""
    stateUpdated = Signal(dict)


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
        main_layout.setContentsMargins(NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING)
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
        # Map dial's 0-1000 range to a logical 1.0-100.0 range
        logical_drive = 1.0 + (dial_value / 1000.0) * 99.0
        self.node_logic.set_drive(logical_drive)

    @Slot(int)
    def _handle_mix_change(self, dial_value: int):
        # Map dial's 0-1000 range to a logical 0.0-1.0 range
        logical_mix = dial_value / 1000.0
        self.node_logic.set_mix(logical_mix)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        """Central slot to update all UI controls from a state dictionary."""
        shaper_type = state.get("shaper_type")
        drive = state.get("drive", 1.0)
        mix = state.get("mix", 1.0)

        # Update Shaper Type
        with QSignalBlocker(self.type_combo):
            index = self.type_combo.findData(shaper_type)
            if index != -1: self.type_combo.setCurrentIndex(index)

        # Update Drive
        with QSignalBlocker(self.drive_dial):
            dial_value = int(((drive - 1.0) / 99.0) * 1000.0)
            self.drive_dial.setValue(dial_value)

        is_drive_socket_connected = "drive" in self.node_logic.inputs and self.node_logic.inputs["drive"].connections
        self.drive_dial.setEnabled(not is_drive_socket_connected)

        label_text = f"{drive:.1f}"
        if is_drive_socket_connected: label_text += " (ext)"
        self.drive_value_label.setText(label_text)

        # Update Mix
        with QSignalBlocker(self.mix_dial):
            dial_value = int(mix * 1000.0)
            self.mix_dial.setValue(dial_value)

        is_mix_socket_connected = "mix" in self.node_logic.inputs and self.node_logic.inputs["mix"].connections
        self.mix_dial.setEnabled(not is_mix_socket_connected)

        label_text = f"{mix:.2f}"
        if is_mix_socket_connected: label_text += " (ext)"
        self.mix_value_label.setText(label_text)

    @Slot()
    def updateFromLogic(self):
        """Requests a full state snapshot from the logic and updates the UI."""
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
        self.emitter = WaveShaperEmitter()
        self.add_input("in", data_type=np.ndarray)
        self.add_input("drive", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("out", data_type=np.ndarray)

        self._lock = threading.Lock()
        self._shaper_type: ShaperType = ShaperType.SOFT_CLIP
        self._drive: float = 1.0
        self._mix: float = 1.0

    def _get_current_state_snapshot_locked(self) -> Dict:
        """Returns a copy of the current parameters for UI synchronization. Assumes caller holds the lock."""
        return {"shaper_type": self._shaper_type, "drive": self._drive, "mix": self._mix}

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    @Slot(ShaperType)
    def set_shaper_type(self, shaper_type: ShaperType):
        state_to_emit = None
        with self._lock:
            if self._shaper_type != shaper_type:
                self._shaper_type = shaper_type
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_drive(self, drive: float):
        state_to_emit = None
        with self._lock:
            new_drive = np.clip(float(drive), 1.0, 100.0)
            if self._drive != new_drive:
                self._drive = new_drive
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_mix(self, mix: float):
        state_to_emit = None
        with self._lock:
            new_mix = np.clip(float(mix), 0.0, 1.0)
            if self._mix != new_mix:
                self._mix = new_mix
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        if not isinstance(signal, np.ndarray):
            return {"out": None}

        state_snapshot_to_emit = None
        with self._lock:
            # Prioritize socket input for drive
            drive_socket = input_data.get("drive")
            if drive_socket is not None:
                new_drive = np.clip(float(drive_socket), 1.0, 100.0)
                if abs(self._drive - new_drive) > 1e-6:
                    self._drive = new_drive
                    state_snapshot_to_emit = self._get_current_state_snapshot_locked()

            # Prioritize socket input for mix
            mix_socket = input_data.get("mix")
            if mix_socket is not None:
                new_mix = np.clip(float(mix_socket), 0.0, 1.0)
                if abs(self._mix - new_mix) > 1e-6:
                    self._mix = new_mix
                    state_snapshot_to_emit = self._get_current_state_snapshot_locked()

            drive = self._drive
            shaper_type = self._shaper_type
        
        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

        # 1. Apply drive
        driven_signal = signal * drive
        
        # 2. Apply shaping function
        output_signal = None
        if shaper_type == ShaperType.SOFT_CLIP:
            output_signal = np.tanh(driven_signal)
        elif shaper_type == ShaperType.HARD_CLIP:
            output_signal = np.clip(driven_signal, -1.0, 1.0)
        elif shaper_type == ShaperType.FOLD:
            output_signal = np.abs(np.mod(driven_signal + 1, 4) - 2) - 1
        elif shaper_type == ShaperType.SINE:
            # Clip the input to [-1, 1] first to ensure the sine function maps correctly
            clipped_driven_signal = np.clip(driven_signal, -1.0, 1.0)
            output_signal = np.sin(0.5 * np.pi * clipped_driven_signal)

        # Apply dry-wet mix
        final_mix = self._mix
        output_signal = signal * (1 - final_mix) + output_signal * final_mix

        return {"out": output_signal.astype(DEFAULT_DTYPE)}

    def serialize_extra(self) -> dict:
        with self._lock:
            return {
                "shaper_type": self._shaper_type.name,
                "drive": self._drive,
                "mix": self._mix,
            }

    def deserialize_extra(self, data: dict):
        with self._lock:
            shaper_type_name = data.get("shaper_type", ShaperType.SOFT_CLIP.name)
            try:
                self._shaper_type = ShaperType[shaper_type_name]
            except KeyError:
                self._shaper_type = ShaperType.SOFT_CLIP
            self._drive = np.clip(float(data.get("drive", 1.0)), 1.0, 100.0)
            self._mix = np.clip(float(data.get("mix", 1.0)), 0.0, 1.0)