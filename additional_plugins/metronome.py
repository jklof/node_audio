import threading
import logging
import time
from collections import deque
from typing import Dict, Optional

import numpy as np
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE

from PySide6.QtCore import Qt, Signal, Slot, QObject, QTimer, QSignalBlocker
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QComboBox, QCheckBox
)

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_BPM = 120.0
MIN_BPM = 20.0
MAX_BPM = 300.0

# ==============================================================================
# 1. State Emitter for UI Communication
# ==============================================================================
class MetronomeEmitter(QObject):
    stateUpdated = Signal(dict)
    beatTicked = Signal()

# ==============================================================================
# 2. Custom UI Class (BPMMetronomeNodeItem)
# ==============================================================================
class BPMMetronomeNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "BPMMetronomeNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING)
        main_layout.setSpacing(5)

        # --- Top controls layout (Indicator and Gate Checkbox) ---
        top_controls_layout = QHBoxLayout()
        self.beat_indicator = QLabel("●")
        font = self.beat_indicator.font()
        font.setPointSize(16)
        self.beat_indicator.setFont(font)
        self.beat_indicator.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.beat_indicator.setStyleSheet("color: #444;")
        
        self.gate_checkbox = QCheckBox("Running")

        top_controls_layout.addWidget(self.beat_indicator)
        top_controls_layout.addStretch()
        top_controls_layout.addWidget(self.gate_checkbox)
        main_layout.addLayout(top_controls_layout)
        
        # --- BPM Controls ---
        bpm_layout = QHBoxLayout()
        bpm_label = QLabel("BPM:")
        self.bpm_spinbox = QDoubleSpinBox()
        self.bpm_spinbox.setRange(MIN_BPM, MAX_BPM)
        self.bpm_spinbox.setDecimals(1)
        self.bpm_spinbox.setSingleStep(1.0)
        bpm_layout.addWidget(bpm_label)
        bpm_layout.addWidget(self.bpm_spinbox)
        main_layout.addLayout(bpm_layout)

        # --- Subdivision Controls ---
        subdiv_layout = QHBoxLayout()
        subdiv_label = QLabel("Subdivision:")
        self.subdiv_combo = QComboBox()
        self.subdiv_combo.addItems(["1/4", "1/8", "1/16", "1/32"])
        subdiv_layout.addWidget(subdiv_label)
        subdiv_layout.addWidget(self.subdiv_combo)
        main_layout.addLayout(subdiv_layout)

        # --- REMOVED: Tap Tempo Button ---

        self.setContentWidget(self.container_widget)

        # --- Connect Signals ---
        self.bpm_spinbox.valueChanged.connect(self.node_logic.set_bpm)
        self.subdiv_combo.currentIndexChanged.connect(self._on_subdivision_change)
        # --- REMOVED: Tap button signal connection ---
        self.gate_checkbox.toggled.connect(self.node_logic.set_internal_gate)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        self.node_logic.emitter.beatTicked.connect(self._on_beat_ticked)

        self.updateFromLogic()

    @Slot(int)
    def _on_subdivision_change(self, index: int):
        subdivision_text = self.subdiv_combo.itemText(index)
        self.node_logic.set_subdivision(subdivision_text)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        bpm = state.get("bpm", DEFAULT_BPM)
        subdivision = state.get("subdivision", "1/4")
        internal_gate_state = state.get("internal_gate", True)

        with QSignalBlocker(self.bpm_spinbox):
            self.bpm_spinbox.setValue(bpm)

        with QSignalBlocker(self.subdiv_combo):
            index = self.subdiv_combo.findText(subdivision)
            if index != -1:
                self.subdiv_combo.setCurrentIndex(index)
        
        with QSignalBlocker(self.gate_checkbox):
            self.gate_checkbox.setChecked(internal_gate_state)

        # Disable UI controls if they are being driven by an input socket
        is_bpm_socket_connected = "bpm" in self.node_logic.inputs and self.node_logic.inputs["bpm"].connections
        self.bpm_spinbox.setEnabled(not is_bpm_socket_connected)
        
        is_gate_socket_connected = "gate" in self.node_logic.inputs and self.node_logic.inputs["gate"].connections
        self.gate_checkbox.setEnabled(not is_gate_socket_connected)

    @Slot()
    def _on_beat_ticked(self):
        self.beat_indicator.setStyleSheet("color: orange;")
        QTimer.singleShot(100, lambda: self.beat_indicator.setStyleSheet("color: #444;"))
        
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()

# ==============================================================================
# 3. Node Logic Class (BPMMetronomeNode)
# ==============================================================================
class BPMMetronomeNode(Node):
    NODE_TYPE = "BPM Metronome"
    UI_CLASS = BPMMetronomeNodeItem
    CATEGORY = "Modulation"
    DESCRIPTION = "Generates triggers at a specified BPM and subdivision."

    SUBDIVISION_MAP = {"1/4": 1.0, "1/8": 0.5, "1/16": 0.25, "1/32": 0.125}

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = MetronomeEmitter()

        self.add_input("gate", data_type=bool)
        self.add_input("reset", data_type=bool)
        self.add_input("bpm", data_type=float)
        self.add_output("trigger_out", data_type=bool)
        self.add_output("beat_count_out", data_type=int)

        self._lock = threading.Lock()
        self._bpm = DEFAULT_BPM
        self._subdivision = "1/4"
        self._subdivision_multiplier = 1.0
        self._is_running = False
        self._samples_until_next_tick = 0
        self._beat_counter = 0
        self._internal_gate_state = True
        

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return {
                "bpm": self._bpm, 
                "subdivision": self._subdivision,
                "internal_gate": self._internal_gate_state
            }

    @Slot(float)
    def set_bpm(self, bpm: float):
        state_to_emit = None
        with self._lock:
            new_bpm = np.clip(bpm, MIN_BPM, MAX_BPM)
            if self._bpm != new_bpm:
                self._bpm = new_bpm
                # Construct state dictionary directly to avoid re-acquiring lock.
                state_to_emit = {
                    "bpm": self._bpm, 
                    "subdivision": self._subdivision,
                    "internal_gate": self._internal_gate_state
                }
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)
    
    @Slot(bool)
    def set_internal_gate(self, state: bool):
        state_to_emit = None
        with self._lock:
            if self._internal_gate_state != state:
                self._internal_gate_state = state
                # Construct state dictionary directly to avoid re-acquiring lock.
                state_to_emit = {
                    "bpm": self._bpm, 
                    "subdivision": self._subdivision,
                    "internal_gate": self._internal_gate_state
                }
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)
    
    @Slot(str)
    def set_subdivision(self, subdivision_text: str):
        state_to_emit = None
        with self._lock:
            if subdivision_text in self.SUBDIVISION_MAP and self._subdivision != subdivision_text:
                self._subdivision = subdivision_text
                self._subdivision_multiplier = self.SUBDIVISION_MAP[subdivision_text]
                # Construct state dictionary directly to avoid re-acquiring lock.
                state_to_emit = {
                    "bpm": self._bpm, 
                    "subdivision": self._subdivision,
                    "internal_gate": self._internal_gate_state
                }
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)
    

    def process(self, input_data: dict) -> dict:
        trigger_out = False
        state_update_needed = False
        tick_to_emit = False
        
        with self._lock:
            reset = bool(input_data.get("reset", False))
            
            gate_socket_val = input_data.get("gate")
            if gate_socket_val is not None:
                self._is_running = bool(gate_socket_val)
            else:
                self._is_running = self._internal_gate_state
            
            bpm_socket_val = input_data.get("bpm")
            if bpm_socket_val is not None:
                current_bpm = np.clip(float(bpm_socket_val), MIN_BPM, MAX_BPM)
                if abs(self._bpm - current_bpm) > 0.01:
                    self._bpm = current_bpm
                    state_update_needed = True # Set flag instead of emitting
            else:
                current_bpm = self._bpm
            
            if reset:
                self._beat_counter = 0
                self._samples_until_next_tick = 0

            if self._is_running:
                samples_per_beat = (60.0 / current_bpm) * DEFAULT_SAMPLERATE
                samples_per_tick = samples_per_beat * self._subdivision_multiplier
                
                self._samples_until_next_tick -= DEFAULT_BLOCKSIZE
                
                if self._samples_until_next_tick <= 0:
                    trigger_out = True
                    self._beat_counter += 1
                    self._samples_until_next_tick += samples_per_tick
                    tick_to_emit = True # Set flag instead of emitting

            current_beat_count = self._beat_counter if self._is_running else 0

        # --- Emit signals AFTER releasing the lock ---
        if state_update_needed:
            self.emitter.stateUpdated.emit(self.get_current_state_snapshot())
        
        if tick_to_emit:
            self.emitter.beatTicked.emit()
        
        return {"trigger_out": trigger_out, "beat_count_out": current_beat_count}

    def start(self):
        with self._lock:
            self._samples_until_next_tick = 0
            self._beat_counter = 0

    def serialize_extra(self) -> dict:
        with self._lock:
            return {
                "bpm": self._bpm, 
                "subdivision": self._subdivision,
                "internal_gate": self._internal_gate_state
            }

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._bpm = data.get("bpm", DEFAULT_BPM)
            self._internal_gate_state = data.get("internal_gate", True)
            subdiv = data.get("subdivision", "1/4")
            if subdiv in self.SUBDIVISION_MAP:
                self._subdivision = subdiv
                self._subdivision_multiplier = self.SUBDIVISION_MAP[subdiv]