import torch
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
from ui_elements import ParameterNodeItem
from PySide6.QtCore import Slot  # <-- FIXED: Added the missing import for Slot

# Configure logging
logger = logging.getLogger(__name__)


# ==============================================================================
# Enum for Noise Types
# ==============================================================================
class NoiseType(Enum):
    WHITE = "White"
    PINK = "Pink"
    BROWN = "Brown"
    BLUE = "Blue"
    VIOLET = "Violet"


# ==============================================================================
# REFACTORED: UI Class for Noise Generator
# ==============================================================================
class NoiseGeneratorNodeItem(ParameterNodeItem):
    """
    Refactored UI for the NoiseGeneratorNode.
    This class now fully leverages ParameterNodeItem by defining its entire UI
    declaratively, which handles widget creation, signal/slot connections,
    and state updates automatically.
    """

    NODE_SPECIFIC_WIDTH = 160

    def __init__(self, node_logic: "NoiseGeneratorNode"):
        # Define the parameters and their control types for this node.
        parameters = [
            {
                "key": "noise_type",
                "name": "Noise Type",
                "type": "combobox",
                # The items list provides the display text and the data (enum member)
                # to be sent to the logic node's setter.
                "items": [(nt.value, nt) for nt in NoiseType],
            },
            {
                "key": "level",
                "name": "Level",
                "type": "dial",  # Use a dial for the level control
                "min": 0.0,
                "max": 1.0,
                "format": "{:.2f}",  # Display format for the label
                "is_log": False,
            },
        ]

        # The parent class now handles the creation of all specified controls.
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# Noise Generator Logic Node (Unchanged logic, only import was missing)
# ==============================================================================
class NoiseGeneratorNode(Node):
    NODE_TYPE = "Noise Generator"
    UI_CLASS = NoiseGeneratorNodeItem
    CATEGORY = "Generators"
    DESCRIPTION = "Generates various types of noise (White, Pink, Brown, Blue, Violet)."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("level", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._lock = threading.Lock()
        self.samplerate = DEFAULT_SAMPLERATE
        self.blocksize = DEFAULT_BLOCKSIZE
        self.channels = DEFAULT_CHANNELS
        self._noise_type: NoiseType = NoiseType.WHITE
        self._level: float = 0.5

        # Filter states and coefficients will be NumPy arrays for SciPy
        self._filter_coeffs = {}
        self._filter_states = {}

        self._init_filter_coeffs_and_states()
        logger.info(f"NoiseGeneratorNode [{self.name}] initialized.")

    def _init_filter_coeffs_and_states(self):
        """Initialize SciPy filter coefficients and state arrays."""
        nyquist = self.samplerate / 2.0
        # Pink Noise approximation
        b_pink = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        a_pink = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        # Brown Noise: 1st order LPF @ 40Hz
        b_brown, a_brown = scipy.signal.butter(1, 40.0 / nyquist, btype="low")
        # Blue Noise: 1st order HPF @ 1000Hz
        b_blue, a_blue = scipy.signal.butter(1, 1000.0 / nyquist, btype="high")
        # Violet Noise: 2nd order HPF @ 4000Hz
        b_violet, a_violet = scipy.signal.butter(2, 4000.0 / nyquist, btype="high")

        self._filter_coeffs = {
            NoiseType.PINK: (b_pink, a_pink),
            NoiseType.BROWN: (b_brown, a_brown),
            NoiseType.BLUE: (b_blue, a_blue),
            NoiseType.VIOLET: (b_violet, a_violet),
        }

        # Reset initial filter conditions (zi) for each channel
        for nt, (b, a) in self._filter_coeffs.items():
            zi = scipy.signal.lfilter_zi(b, a)
            # The shape must be (num_channels, filter_order)
            self._filter_states[nt] = np.tile(zi, (self.channels, 1))

        logger.debug(f"[{self.name}] Filters initialized for {self.channels} channels at {self.samplerate}Hz.")

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {"noise_type": self._noise_type, "level": self._level}

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    @Slot(NoiseType)
    def set_noise_type(self, noise_type: NoiseType):
        state_to_emit = None
        with self._lock:
            if self._noise_type != noise_type:
                logger.info(f"[{self.name}] Changing noise type to: {noise_type.value}")
                self._noise_type = noise_type
                # No need to re-init filters, just use the correct one
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_level(self, level: float):
        state_to_emit = None
        with self._lock:
            new_level = np.clip(float(level), 0.0, 1.0)
            if self._level != new_level:
                self._level = new_level
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def process(self, input_data: dict) -> dict:
        state_snapshot_to_emit = None
        with self._lock:
            level_socket = input_data.get("level")
            if level_socket is not None:
                new_level = np.clip(float(level_socket), 0.0, 1.0)
                if abs(self._level - new_level) > 1e-6:
                    self._level = new_level
                    state_snapshot_to_emit = self._get_current_state_snapshot_locked()
            noise_type = self._noise_type
            level = self._level

        if state_snapshot_to_emit:
            self.ui_update_callback(state_snapshot_to_emit)

        # Generate white noise directly as a torch tensor
        white_noise = torch.rand(self.channels, self.blocksize, dtype=DEFAULT_DTYPE) * 2.0 - 1.0

        if noise_type == NoiseType.WHITE:
            processed_signal = white_noise
        else:
            # Hybrid approach: Convert to NumPy for fast filtering
            white_noise_np = white_noise.numpy()

            with self._lock:  # Lock only when accessing shared filter state
                b, a = self._filter_coeffs[noise_type]
                zi = self._filter_states[noise_type]
                processed_signal_np, zf = scipy.signal.lfilter(b, a, white_noise_np, axis=1, zi=zi)
                self._filter_states[noise_type] = zf

            # Convert back to a torch tensor
            processed_signal = torch.from_numpy(processed_signal_np.astype(np.float32))

        # Apply level and clipping
        output_block = torch.clamp(processed_signal * level, -1.0, 1.0)

        return {"out": output_block}

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
