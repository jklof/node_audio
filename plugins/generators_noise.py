import torch
import numpy as np
import scipy.signal
import logging
from enum import Enum
from typing import Dict

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS

# --- UI and Helper Imports ---
from ui_elements import ParameterNodeItem
from node_helpers import with_parameters, Parameter

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
# Noise Generator Logic Node
# ==============================================================================
@with_parameters
class NoiseGeneratorNode(Node):
    NODE_TYPE = "Noise Generator"
    UI_CLASS = NoiseGeneratorNodeItem
    CATEGORY = "Generators"
    DESCRIPTION = "Generates various types of noise (White, Pink, Brown, Blue, Violet)."

    noise_type = Parameter(default=NoiseType.WHITE)
    level = Parameter(default=0.5, clip=(0.0, 1.0))

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)

        self._init_parameters()

        # Custom state
        self.samplerate = DEFAULT_SAMPLERATE
        self.blocksize = DEFAULT_BLOCKSIZE
        self.channels = DEFAULT_CHANNELS
        self._filter_coeffs = {}
        self._filter_states = {}

        self.add_input("level", data_type=float)
        self.add_output("out", data_type=torch.Tensor)
        self._init_filter_coeffs_and_states()

    def _get_state_snapshot_locked(self) -> dict:
        # Get the state from the helper
        state = self._get_parameters_state()
        # Add our custom state
        state["channels"] = self.channels
        return state

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

        # Each channel needs its own independent state.
        for nt, (b, a) in self._filter_coeffs.items():
            zi_single = scipy.signal.lfilter_zi(b, a)
            self._filter_states[nt] = np.array([zi_single] * self.channels)

        logger.debug(f"[{self.name}] Filters initialized for {self.channels} channels at {self.samplerate}Hz.")

    def process(self, input_data: dict) -> dict:
        # The name of this helper is now _update_parameters_from_sockets
        self._update_parameters_from_sockets(input_data)

        with self._lock:
            noise_type = self._noise_type
            level = self._level

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

        # Apply level
        output_block = processed_signal * level

        return {"out": output_block}

    def start(self):
        with self._lock:
            self._init_filter_coeffs_and_states()

    def serialize_extra(self) -> dict:
        # Get serialized state for managed params from the helper
        state = self._serialize_parameters()
        # Add our custom serialization logic
        with self._lock:
            state["channels"] = self.channels
        return state

    def deserialize_extra(self, data: dict):
        # Let the helper handle its parameters first
        self._deserialize_parameters(data)
        # Now, handle our custom deserialization logic
        with self._lock:
            self.channels = int(data.get("channels", DEFAULT_CHANNELS))
            # Re-initialize filters since channel count might have changed
            self._init_filter_coeffs_and_states()
