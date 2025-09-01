import numpy as np
import scipy.signal
import threading
import logging
from enum import Enum
from typing import Dict

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS

# --- UI and Qt Imports ---
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QComboBox,
    QDial,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Slot, QSignalBlocker, Signal, QObject
from PySide6.QtGui import QFontMetrics

# Configure logging
logger = logging.getLogger(__name__)


# ==============================================================================
# Enum for Noise Types
# ==============================================================================
class NoiseType(Enum):
    WHITE = "White"
    PINK = "Pink"
    BROWN = "Brown"  # Also known as Red Noise
    BLUE = "Blue"
    VIOLET = "Violet"  # Also known as Purple Noise


# ==============================================================================
# Emitter for UI Communication
# ==============================================================================
class NoiseGeneratorEmitter(QObject):
    """A dedicated QObject to safely emit signals from the logic to the UI thread."""

    stateUpdated = Signal(dict)


# ==============================================================================
# UI Class for Noise Generator
# ==============================================================================
class NoiseGeneratorNodeItem(NodeItem):
    """Custom NodeItem for NoiseGeneratorNode, architected for signal/slot communication."""

    NODE_SPECIFIC_WIDTH = 160

    def __init__(self, node_logic: "NoiseGeneratorNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(6)

        # --- Noise Type Selection ---
        self.type_label = QLabel("Noise Type:")
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.type_combo = QComboBox()
        for nt in NoiseType:
            self.type_combo.addItem(nt.value, nt)
        self.type_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(self.type_label)
        main_layout.addWidget(self.type_combo)

        # --- Level Control ---
        level_controls_layout = QHBoxLayout()
        level_controls_layout.setSpacing(5)

        self.level_dial = QDial()
        self.level_dial.setRange(0, 100)
        self.level_dial.setNotchesVisible(True)

        level_label_vbox = QVBoxLayout()
        level_label_vbox.setSpacing(1)
        self.level_title_label = QLabel("Level")
        self.level_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.level_value_label = QLabel("...")
        self.level_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fm = QFontMetrics(self.level_value_label.font())
        min_width = fm.boundingRect("1.00 (ext)").width()
        self.level_title_label.setMinimumWidth(min_width)

        level_label_vbox.addWidget(self.level_title_label)
        level_label_vbox.addWidget(self.level_value_label)

        level_controls_layout.addWidget(self.level_dial)
        level_controls_layout.addLayout(level_label_vbox)
        main_layout.addLayout(level_controls_layout)

        self.setContentWidget(self.container_widget)

        # --- Connect Signals ---
        self.type_combo.currentTextChanged.connect(self._handle_type_change)
        self.level_dial.valueChanged.connect(self._handle_level_change)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

        self.updateFromLogic()

    @Slot(str)
    def _handle_type_change(self, type_text: str):
        selected_enum_member = self.type_combo.currentData()
        if isinstance(selected_enum_member, NoiseType):
            self.node_logic.set_noise_type(selected_enum_member)

    @Slot(int)
    def _handle_level_change(self, dial_value: int):
        logical_level = dial_value / 100.0
        self.node_logic.set_level(logical_level)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        """Central slot to update all UI controls from a state dictionary."""
        # Update Noise Type
        noise_type = state.get("noise_type")
        with QSignalBlocker(self.type_combo):
            index = self.type_combo.findData(noise_type)
            if index != -1:
                self.type_combo.setCurrentIndex(index)

        # Update Level
        level = state.get("level", 0.0)
        with QSignalBlocker(self.level_dial):
            self.level_dial.setValue(int(round(level * 100.0)))

        is_level_socket_connected = "level" in self.node_logic.inputs and self.node_logic.inputs["level"].connections
        self.level_dial.setEnabled(not is_level_socket_connected)

        label_text = f"{level:.2f}"
        if is_level_socket_connected:
            label_text += " (ext)"
        self.level_value_label.setText(label_text)

    @Slot()
    def updateFromLogic(self):
        """Requests a full state snapshot from the logic and updates the UI."""
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ==============================================================================
# Noise Generator Logic Node
# ==============================================================================
class NoiseGeneratorNode(Node):
    NODE_TYPE = "Noise Generator"
    UI_CLASS = NoiseGeneratorNodeItem
    CATEGORY = "Generators"
    DESCRIPTION = "Generates various types of noise (White, Pink, Brown, Blue, Violet)."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.emitter = NoiseGeneratorEmitter()
        self.add_input("level", data_type=float)
        self.add_output("out", data_type=np.ndarray)

        self._lock = threading.Lock()
        self.samplerate = DEFAULT_SAMPLERATE
        self.blocksize = DEFAULT_BLOCKSIZE
        self.channels = DEFAULT_CHANNELS
        self._noise_type: NoiseType = NoiseType.WHITE
        self._level: float = 0.5

        # Filter states
        self._b_pink = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        self._a_pink = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        self._b_brown, self._a_brown = None, None
        self._b_blue, self._a_blue = None, None
        self._b_violet, self._a_violet = None, None
        self._zi_pink, self._zi_brown, self._zi_blue, self._zi_violet = [None] * 4

        self._init_filter_coeffs_and_states()
        logger.info(f"NoiseGeneratorNode [{self.name}] initialized.")

    def _init_filter_coeffs_and_states(self):
        nyquist = self.samplerate / 2.0
        # Brown Noise: 1st order LPF @ 40Hz
        self._b_brown, self._a_brown = scipy.signal.butter(1, 40.0 / nyquist, btype="low")
        # Blue Noise: 1st order HPF @ 1000Hz
        self._b_blue, self._a_blue = scipy.signal.butter(1, 1000.0 / nyquist, btype="high")
        # Violet Noise: 2nd order HPF @ 4000Hz
        self._b_violet, self._a_violet = scipy.signal.butter(2, 4000.0 / nyquist, btype="high")

        # Reset initial filter conditions (zi)
        self._zi_pink = scipy.signal.lfilter_zi(self._b_pink, self._a_pink) * 0.0
        self._zi_brown = scipy.signal.lfilter_zi(self._b_brown, self._a_brown) * 0.0
        self._zi_blue = scipy.signal.lfilter_zi(self._b_blue, self._a_blue) * 0.0
        self._zi_violet = scipy.signal.lfilter_zi(self._b_violet, self._a_violet) * 0.0
        logger.debug(f"[{self.name}] Filters initialized for samplerate {self.samplerate}Hz.")

    def get_current_state_snapshot(self, locked: bool = False) -> Dict:
        """Returns a copy of the current parameters for UI synchronization."""
        if locked:
            return {"noise_type": self._noise_type, "level": self._level}
        with self._lock:
            return {"noise_type": self._noise_type, "level": self._level}

    @Slot(NoiseType)
    def set_noise_type(self, noise_type: NoiseType):
        state_to_emit = None
        with self._lock:
            if self._noise_type != noise_type:
                logger.info(f"[{self.name}] Changing noise type to: {noise_type.value}")
                self._noise_type = noise_type
                self._init_filter_coeffs_and_states()
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_level(self, level: float):
        state_to_emit = None
        with self._lock:
            new_level = np.clip(float(level), 0.0, 1.0)
            if self._level != new_level:
                self._level = new_level
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def process(self, input_data: dict) -> dict:
        state_snapshot_to_emit = None
        with self._lock:
            # Prioritize socket input for level
            level_socket = input_data.get("level")
            if level_socket is not None:
                new_level = np.clip(float(level_socket), 0.0, 1.0)
                if abs(self._level - new_level) > 1e-6:
                    self._level = new_level
                    state_snapshot_to_emit = self.get_current_state_snapshot(locked=True)

            # Get parameters for this processing tick
            noise_type = self._noise_type
            level = self._level
            zi_pink = self._zi_pink.copy()
            zi_brown = self._zi_brown.copy()
            zi_blue = self._zi_blue.copy()
            zi_violet = self._zi_violet.copy()

        # Emit signal after releasing the lock
        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

        # Generate mono white noise with headroom
        mono_white_noise = np.random.uniform(-0.8, 0.8, self.blocksize).astype(DEFAULT_DTYPE)
        processed_mono_signal = mono_white_noise

        # Apply filter based on noise type
        if noise_type == NoiseType.PINK:
            processed_mono_signal, zf = scipy.signal.lfilter(self._b_pink, self._a_pink, mono_white_noise, zi=zi_pink)
            with self._lock:
                self._zi_pink = zf
        elif noise_type == NoiseType.BROWN:
            processed_mono_signal, zf = scipy.signal.lfilter(
                self._b_brown, self._a_brown, mono_white_noise, zi=zi_brown
            )
            with self._lock:
                self._zi_brown = zf
        elif noise_type == NoiseType.BLUE:
            processed_mono_signal, zf = scipy.signal.lfilter(self._b_blue, self._a_blue, mono_white_noise, zi=zi_blue)
            with self._lock:
                self._zi_blue = zf
        elif noise_type == NoiseType.VIOLET:
            processed_mono_signal, zf = scipy.signal.lfilter(
                self._b_violet, self._a_violet, mono_white_noise, zi=zi_violet
            )
            with self._lock:
                self._zi_violet = zf

        processed_mono_signal *= level
        np.clip(processed_mono_signal, -1.0, 1.0, out=processed_mono_signal)
        output_block = np.tile(processed_mono_signal[:, np.newaxis], (1, self.channels))

        return {"out": output_block.astype(DEFAULT_DTYPE)}

    def start(self):
        with self._lock:
            self._init_filter_coeffs_and_states()
        logger.debug(f"[{self.name}] started, filter states reset.")

    def serialize_extra(self) -> dict:
        with self._lock:
            return {
                "noise_type": self._noise_type.name,
                "level": self._level,
                "channels": self.channels,
            }

    def deserialize_extra(self, data: dict):
        with self._lock:
            noise_type_name = data.get("noise_type", NoiseType.WHITE.name)
            try:
                self._noise_type = NoiseType[noise_type_name]
            except KeyError:
                self._noise_type = NoiseType.WHITE
            self._level = np.clip(float(data.get("level", 0.5)), 0.0, 1.0)
            self.channels = int(data.get("channels", DEFAULT_CHANNELS))
        self._init_filter_coeffs_and_states()
