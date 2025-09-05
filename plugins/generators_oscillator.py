import torch
import numpy as np  # Kept for UI-specific logarithmic mapping
import threading
import logging
from enum import Enum
from typing import Dict

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS

# --- UI and Qt Imports ---
from ui_elements import NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING
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
# Enum for Waveform Types
# ==============================================================================
class Waveform(Enum):
    SINE = "Sine"
    SQUARE = "Square"
    SAWTOOTH = "Sawtooth"
    TRIANGLE = "Triangle"





# ==============================================================================
# UI Class for the Oscillator Node
# ==============================================================================
class OscillatorNodeItem(NodeItem):
    """Custom NodeItem for the OscillatorNode, providing user controls."""

    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "OscillatorNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(6)

        # --- Waveform Selection ---
        main_layout.addWidget(QLabel("Waveform:"))
        self.waveform_combo = QComboBox()
        for wf in Waveform:
            self.waveform_combo.addItem(wf.value, wf)
        main_layout.addWidget(self.waveform_combo)

        # --- Dials Layout ---
        dials_layout = QHBoxLayout()
        dials_layout.setSpacing(10)

        # --- Frequency Dial ---
        self.freq_dial, self.freq_label_vbox = self._create_dial_with_labels("Freq (Hz)", "440.0")
        dials_layout.addLayout(self.freq_label_vbox, stretch=1)
        dials_layout.addWidget(self.freq_dial, stretch=2)

        # --- Pulse Width Dial ---
        self.pw_dial, self.pw_label_vbox = self._create_dial_with_labels("Pulse Width", "0.50")
        self.pw_widget = QWidget()  # Container to easily show/hide
        pw_layout = QHBoxLayout(self.pw_widget)
        pw_layout.setContentsMargins(0, 0, 0, 0)
        pw_layout.setSpacing(10)
        pw_layout.addLayout(self.pw_label_vbox, stretch=1)
        pw_layout.addWidget(self.pw_dial, stretch=2)

        main_layout.addLayout(dials_layout)
        main_layout.addWidget(self.pw_widget)

        self.setContentWidget(self.container_widget)

        # --- Connect Signals ---
        self.waveform_combo.currentTextChanged.connect(self._handle_waveform_change)
        self.freq_dial.valueChanged.connect(self._handle_freq_change)
        self.pw_dial.valueChanged.connect(self._handle_pw_change)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)


    def _create_dial_with_labels(self, title: str, initial_value: str) -> tuple[QDial, QVBoxLayout]:
        """Helper factory to create a dial and its associated labels."""
        dial = QDial()
        dial.setRange(0, 1000)  # Use a large integer range for precision
        dial.setNotchesVisible(True)

        label_vbox = QVBoxLayout()
        label_vbox.setSpacing(1)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label = QLabel(initial_value)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Ensure minimum width to prevent UI jitter when text changes
        fm = QFontMetrics(value_label.font())
        min_width = fm.boundingRect("9999.9 Hz (ext)").width()
        title_label.setMinimumWidth(min_width)

        label_vbox.addWidget(title_label)
        label_vbox.addWidget(value_label)

        return (dial, label_vbox)

    @Slot(str)
    def _handle_waveform_change(self, text: str):
        selected_enum = self.waveform_combo.currentData()
        if isinstance(selected_enum, Waveform):
            self.node_logic.set_waveform(selected_enum)

    @Slot(int)
    def _handle_freq_change(self, dial_value: int):
        # Logarithmic mapping for more intuitive frequency control
        min_f, max_f = 20.0, 20000.0
        log_min, log_max = np.log10(min_f), np.log10(max_f)
        freq = 10 ** (((dial_value / 1000.0) * (log_max - log_min)) + log_min)
        self.node_logic.set_frequency(freq)

    @Slot(int)
    def _handle_pw_change(self, dial_value: int):
        pw = dial_value / 1000.0
        self.node_logic.set_pulse_width(pw)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        """Central slot to update all UI controls from a state dictionary."""
        waveform = state.get("waveform")
        freq = state.get("frequency", 440.0)
        pw = state.get("pulse_width", 0.5)

        # --- Update Waveform Selector ---
        with QSignalBlocker(self.waveform_combo):
            index = self.waveform_combo.findData(waveform)
            if index != -1:
                self.waveform_combo.setCurrentIndex(index)

        # --- Update Frequency Dial and Label ---
        with QSignalBlocker(self.freq_dial):
            min_f, max_f = 20.0, 20000.0
            log_min, log_max = np.log10(min_f), np.log10(max_f)
            dial_val = int(1000.0 * (np.log10(freq) - log_min) / (log_max - log_min))
            self.freq_dial.setValue(dial_val)

        freq_label_widget = self.freq_label_vbox.itemAt(1).widget()
        is_freq_ext = "freq" in self.node_logic.inputs and self.node_logic.inputs["freq"].connections
        freq_label_widget.setText(f"{freq:.1f} Hz{' (ext)' if is_freq_ext else ''}")
        self.freq_dial.setEnabled(not is_freq_ext)

        # --- Update Pulse Width Dial and Label ---
        with QSignalBlocker(self.pw_dial):
            self.pw_dial.setValue(int(pw * 1000.0))

        pw_label_widget = self.pw_label_vbox.itemAt(1).widget()
        is_pw_ext = "pulse_width" in self.node_logic.inputs and self.node_logic.inputs["pulse_width"].connections
        pw_label_widget.setText(f"{pw:.2f}{' (ext)' if is_pw_ext else ''}")
        self.pw_dial.setEnabled(not is_pw_ext)

        # --- Show/Hide Pulse Width Control ---
        self.pw_widget.setVisible(waveform == Waveform.SQUARE)
        self.update_geometry()  # Request geometry update when visibility changes




# ==============================================================================
# Oscillator Logic Node
# ==============================================================================
class OscillatorNode(Node):
    NODE_TYPE = "Oscillator"
    UI_CLASS = OscillatorNodeItem
    CATEGORY = "Generators"
    DESCRIPTION = "Generates classic synthesizer waveforms (Sine, Square, Saw, Triangle)."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self.add_input("freq", data_type=float)
        self.add_input("pulse_width", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._lock = threading.Lock()
        self.samplerate = DEFAULT_SAMPLERATE
        self.blocksize = DEFAULT_BLOCKSIZE
        self.channels = DEFAULT_CHANNELS

        # --- Internal State ---
        self._phase = 0.0
        self._waveform: Waveform = Waveform.SINE
        self._frequency: float = 440.0
        self._pulse_width: float = 0.5

    def _get_current_state_snapshot_locked(self) -> Dict:
        """Returns a copy of the current parameters for UI synchronization. Assumes lock is held."""
        return {"waveform": self._waveform, "frequency": self._frequency, "pulse_width": self._pulse_width}

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    @Slot(Waveform)
    def set_waveform(self, waveform: Waveform):
        state_to_emit = None
        with self._lock:
            if self._waveform != waveform:
                self._waveform = waveform
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_frequency(self, frequency: float):
        state_to_emit = None
        with self._lock:
            new_freq = np.clip(float(frequency), 20.0, 20000.0)
            if self._frequency != new_freq:
                self._frequency = new_freq
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_pulse_width(self, pulse_width: float):
        state_to_emit = None
        with self._lock:
            new_pw = np.clip(float(pulse_width), 0.01, 0.99)  # Avoid extremes
            if self._pulse_width != new_pw:
                self._pulse_width = new_pw
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def process(self, input_data: dict) -> dict:
        state_snapshot_to_emit = None
        with self._lock:
            # Check for external modulation via input sockets
            freq_socket = input_data.get("freq")
            if freq_socket is not None:
                new_freq = np.clip(float(freq_socket), 20.0, 20000.0)
                if abs(self._frequency - new_freq) > 1e-6:
                    self._frequency = new_freq
                    state_snapshot_to_emit = self._get_current_state_snapshot_locked()

            pw_socket = input_data.get("pulse_width")
            if pw_socket is not None:
                new_pw = np.clip(float(pw_socket), 0.01, 0.99)
                if abs(self._pulse_width - new_pw) > 1e-6:
                    self._pulse_width = new_pw
                    state_snapshot_to_emit = self._get_current_state_snapshot_locked()

            # Copy state to local variables for processing
            frequency = self._frequency
            pulse_width = self._pulse_width
            waveform = self._waveform

        # Emit signal after releasing the lock to avoid deadlocks
        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

        # --- Generate Waveform using PyTorch ---
        phase_increment = (2 * torch.pi * frequency) / self.samplerate
        phases = self._phase + torch.arange(self.blocksize, dtype=DEFAULT_DTYPE) * phase_increment

        output_1d = None
        # Normalize phase to [0, 2*pi) for periodic functions
        norm_phases = torch.fmod(phases, 2 * torch.pi)

        if waveform == Waveform.SINE:
            output_1d = 0.5 * torch.sin(norm_phases)
        elif waveform == Waveform.SQUARE:
            # Create square wave from -0.5 to 0.5
            output_1d = torch.where(norm_phases < (2 * torch.pi * pulse_width), 0.5, -0.5)
        elif waveform == Waveform.SAWTOOTH:
            # Create sawtooth from -0.5 to 0.5
            output_1d = (norm_phases / (2 * torch.pi)) - 0.5
        elif waveform == Waveform.TRIANGLE:
            # Create triangle from -0.5 to 0.5
            output_1d = 2 * torch.abs((norm_phases / (2 * torch.pi)) - 0.5) - 0.5

        # Update phase for the next block
        self._phase = torch.fmod(phases[-1] + phase_increment, 2 * torch.pi).item()

        # Tile to match channel count, creating (channels, samples) shape
        output_2d = output_1d.unsqueeze(0).expand(self.channels, -1)

        return {"out": output_2d}

    def serialize_extra(self) -> dict:
        with self._lock:
            return {
                "waveform": self._waveform.name,
                "frequency": self._frequency,
                "pulse_width": self._pulse_width,
            }

    def deserialize_extra(self, data: dict):
        with self._lock:
            waveform_name = data.get("waveform", Waveform.SINE.name)
            try:
                self._waveform = Waveform[waveform_name]
            except KeyError:
                self._waveform = Waveform.SINE

            self._frequency = float(data.get("frequency", 440.0))
            self._pulse_width = float(data.get("pulse_width", 0.5))
