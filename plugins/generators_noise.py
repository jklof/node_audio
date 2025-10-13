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
from node_helpers import managed_parameters, Parameter

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
@managed_parameters
class NoiseGeneratorNode(Node):
    NODE_TYPE = "Noise Generator"
    UI_CLASS = NoiseGeneratorNodeItem
    CATEGORY = "Generators"
    DESCRIPTION = "Generates various types of noise (White, Pink, Brown, Blue, Violet)."

    # --- Declarative managed parameters ---
    # The decorator automatically creates thread-safe setters (e.g., set_noise_type, set_level),
    # serialization, deserialization, and the UI update callback mechanism.
    noise_type = Parameter(default=NoiseType.WHITE)
    level = Parameter(default=0.5, clip=(0.0, 1.0))

    def __init__(self, name, node_id=None):
        # The decorator handles initializing self._noise_type and self._level.
        # The original __init__ is called after the parameters are set up.
        super().__init__(name, node_id)
        self.add_input("level", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self.samplerate = DEFAULT_SAMPLERATE
        self.blocksize = DEFAULT_BLOCKSIZE
        self.channels = DEFAULT_CHANNELS

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

        # Each channel needs its own independent state.
        for nt, (b, a) in self._filter_coeffs.items():
            zi_single = scipy.signal.lfilter_zi(b, a)
            self._filter_states[nt] = np.array([zi_single] * self.channels)

        logger.debug(f"[{self.name}] Filters initialized for {self.channels} channels at {self.samplerate}Hz.")

    # Boilerplate methods like set_noise_type, set_level, _get_state_snapshot_locked,
    # serialize_extra, and deserialize_extra are now automatically handled by the
    # @managed_parameters decorator and can be removed.

    def process(self, input_data: dict) -> dict:
        # The injected helper method handles updating parameters from sockets,
        # including clipping, side-effects, and emitting a single UI update signal if needed.
        self._update_params_from_sockets(input_data)

        # Create a consistent snapshot of parameters for this processing block.
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
            # It's good practice to reset filter states when processing starts
            self._init_filter_coeffs_and_states()
        logger.debug(f"[{self.name}] started, filter states reset.")

    # The decorator will generate a serialize_extra that saves 'noise_type' and 'level'.
    # We only need to add the 'channels' key, which is not a managed parameter.
    def serialize_extra(self) -> dict:
        with self._lock:
            # Get the state from the decorator's injected method
            state = self._get_state_snapshot_locked()
            # Add any non-managed parameters
            state["channels"] = self.channels
            # The decorator's serialize method will handle converting the Enum to a string
            return state

    # The decorator will handle loading 'noise_type' and 'level'.
    # We just need to load our custom 'channels' data.
    def deserialize_extra(self, data: dict):
        # Let the decorator handle its own parameters first
        super().deserialize_extra(data)
        # Now, handle any custom data
        with self._lock:
            self.channels = int(data.get("channels", DEFAULT_CHANNELS))
        # Re-initialize filters since channel count might have changed
        self._init_filter_coeffs_and_states()
