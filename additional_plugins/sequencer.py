import threading
import logging
from typing import Dict, Optional, List

import numpy as np
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING

from PySide6.QtCore import Qt, Signal, Slot, QObject, QSignalBlocker
from PySide6.QtWidgets import QWidget, QCheckBox, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_STEPS = 16
MAX_STEPS = 32


# ==============================================================================
# 1. State Emitter for UI Communication
# ==============================================================================
class SequencerEmitter(QObject):
    """A dedicated QObject to safely emit signals from the logic to the UI thread."""

    stateUpdated = Signal(dict)
    playheadUpdated = Signal(int)


# ==============================================================================
# 2. Custom UI Class (SequencerNodeItem)
# ==============================================================================


class SequencerNodeItem(NodeItem):

    def __init__(self, node_logic: "SequencerNode"):
        super().__init__(node_logic)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(5)

        controls_layout = QHBoxLayout()
        steps_label = QLabel("Steps:")
        self.steps_spinbox = QSpinBox()
        self.steps_spinbox.setRange(1, MAX_STEPS)
        controls_layout.addWidget(steps_label)
        controls_layout.addWidget(self.steps_spinbox)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        self.steps_layout = QHBoxLayout()
        self.steps_layout.setSpacing(2)
        main_layout.addLayout(self.steps_layout)
        self.step_checkboxes: List[QCheckBox] = []
        self._current_playhead = -1

        self.setContentWidget(self.container_widget)

        # --- Connect Signals ---
        self.steps_spinbox.valueChanged.connect(self.node_logic.set_num_steps)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        self.node_logic.emitter.playheadUpdated.connect(self._on_playhead_updated)

        self.updateFromLogic()

    def _rebuild_grid(self, num_steps: int):
        """Clears and rebuilds the checkbox grid in a single horizontal row."""
        while self.steps_layout.count():
            item = self.steps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.step_checkboxes.clear()

        for i in range(num_steps):
            checkbox = QCheckBox()
            checkbox.setFixedSize(20, 20)
            checkbox.setProperty("step_index", i)
            checkbox.toggled.connect(self._on_step_toggled)
            self.step_checkboxes.append(checkbox)
            self.steps_layout.addWidget(checkbox)

        self.steps_layout.addStretch()
        self.update_geometry()

    @Slot(bool)
    def _on_step_toggled(self):
        checkbox = self.sender()
        index = checkbox.property("step_index")
        state = checkbox.isChecked()
        self.node_logic.set_step_state(index, state)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        num_steps = state.get("num_steps", DEFAULT_STEPS)
        sequence = state.get("sequence", [])

        if num_steps != len(self.step_checkboxes):
            self._rebuild_grid(num_steps)

        with QSignalBlocker(self.steps_spinbox):
            self.steps_spinbox.setValue(num_steps)

        for i, checkbox in enumerate(self.step_checkboxes):
            with QSignalBlocker(checkbox):
                checkbox.setChecked(sequence[i] if i < len(sequence) else False)

    @Slot(int)
    def _on_playhead_updated(self, step_index: int):
        if self._current_playhead != step_index:
            if 0 <= self._current_playhead < len(self.step_checkboxes):
                self.step_checkboxes[self._current_playhead].setStyleSheet("")

            if 0 <= step_index < len(self.step_checkboxes):
                self.step_checkboxes[step_index].setStyleSheet("QCheckBox::indicator { background-color: orange; }")

            self._current_playhead = step_index

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ==============================================================================
# 3. Node Logic Class (SequencerNode)
# ==============================================================================
class SequencerNode(Node):
    NODE_TYPE = "Sequencer"
    UI_CLASS = SequencerNodeItem
    CATEGORY = "Modulation"
    DESCRIPTION = "A step sequencer that outputs triggers based on a clock."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = SequencerEmitter()

        self.add_input("clock", data_type=bool)
        self.add_input("reset", data_type=bool)
        self.add_output("trigger_out", data_type=bool)

        self._lock = threading.Lock()
        self._num_steps = DEFAULT_STEPS
        self._sequence = [False] * DEFAULT_STEPS
        self._current_step = -1
        self._prev_clock_state = False

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return {
                "num_steps": self._num_steps,
                "sequence": list(self._sequence),
            }

    @Slot(int)
    def set_num_steps(self, num_steps: int):
        state_to_emit = None
        with self._lock:
            num_steps = max(1, min(MAX_STEPS, num_steps))
            if num_steps == self._num_steps:
                return

            new_sequence = [False] * num_steps
            common_length = min(num_steps, self._num_steps)
            new_sequence[:common_length] = self._sequence[:common_length]

            self._sequence = new_sequence
            self._num_steps = num_steps
            if self._current_step >= num_steps:
                self._current_step = -1

            state_to_emit = {"num_steps": self._num_steps, "sequence": list(self._sequence)}

        # Now, emit the signal safely after the lock has been released.
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(int, bool)
    def set_step_state(self, index: int, state: bool):
        with self._lock:
            if 0 <= index < self._num_steps:
                self._sequence[index] = state

    def process(self, input_data: dict) -> dict:
        clock_signal = bool(input_data.get("clock", False))
        reset_signal = bool(input_data.get("reset", False))

        trigger_out = False
        playhead_to_emit = None

        with self._lock:
            # Detect rising edge for reset
            if reset_signal:
                if self._current_step != -1:
                    self._current_step = -1
                    playhead_to_emit = self._current_step

            # Detect rising edge for clock
            is_rising_edge = clock_signal and not self._prev_clock_state
            self._prev_clock_state = clock_signal

            if is_rising_edge:
                self._current_step = (self._current_step + 1) % self._num_steps
                trigger_out = self._sequence[self._current_step]
                playhead_to_emit = self._current_step

        # Emit signal AFTER the lock is released
        if playhead_to_emit is not None:
            self.emitter.playheadUpdated.emit(playhead_to_emit)

        return {"trigger_out": trigger_out}

    def serialize_extra(self) -> dict:
        with self._lock:
            return {
                "num_steps": self._num_steps,
                "sequence": self._sequence,
            }

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._num_steps = data.get("num_steps", DEFAULT_STEPS)
            self._sequence = data.get("sequence", [False] * self._num_steps)
            # Ensure sequence length matches num_steps
            if len(self._sequence) != self._num_steps:
                self._sequence = [False] * self._num_steps
