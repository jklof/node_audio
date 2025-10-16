import torch
import logging
from node_system import Node
from ui_elements import ParameterNodeItem
from node_helpers import with_parameters, Parameter
from PySide6.QtCore import Qt, Slot

from constants import DEFAULT_DTYPE

logger = logging.getLogger(__name__)


class MonoMixdownNode(Node):
    NODE_TYPE = "Mono Mixdown"
    CATEGORY = "Utility"
    DESCRIPTION = "Mixes a multi-channel signal to mono by averaging input channels. Preserves dtype."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_output("out", data_type=torch.Tensor)
        logger.debug(f"[{self.name}] MonoMixdownNode initialized.")

    def process(self, input_data: dict) -> dict:
        in_signal = input_data.get("in")

        if not isinstance(in_signal, torch.Tensor) or in_signal.ndim != 2:
            if in_signal is not None:
                logger.warning(
                    f"[{self.name}] Input is not a valid 2D torch Tensor. "
                    f"Received type: {type(in_signal)}, ndim: {getattr(in_signal, 'ndim', 'N/A')}."
                )
            return {"out": None}

        num_channels, _blocksize = in_signal.shape

        if num_channels == 0:
            logger.warning(f"[{self.name}] Input has 0 channels.")
            return {"out": None}

        if num_channels == 1:
            # Pass through if already mono, preserving dtype
            return {"out": in_signal}

        # Mix down by averaging channels, preserving dtype as much as possible
        try:
            # torch.mean preserves float and complex dtypes
            mono_signal = torch.mean(in_signal, dim=0, keepdim=True)
            return {"out": mono_signal}
        except Exception as e:
            logger.error(f"[{self.name}] Error during mono mixdown: {e}", exc_info=True)
            return {"out": None}


class StereoJoinNode(Node):
    NODE_TYPE = "Stereo Join Channels"
    CATEGORY = "Utility"
    DESCRIPTION = "Joins two mono signals into a stereo (L, R) signal. Preserves dtype. Missing inputs become silent channels of compatible dtype."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in_left", data_type=torch.Tensor)
        self.add_input("in_right", data_type=torch.Tensor)
        self.add_output("out", data_type=torch.Tensor)
        logger.debug(f"[{self.name}] StereoJoinNode initialized.")

    def _validate_mono_input(
        self, signal: torch.Tensor, expected_blocksize: int | None, channel_name: str
    ) -> torch.Tensor | None:
        if not isinstance(signal, torch.Tensor) or signal.ndim != 2:
            logger.warning(f"[{self.name}] Input '{channel_name}' is not a valid 2D torch Tensor.")
            return None

        num_channels, blocksize = signal.shape

        if num_channels != 1:
            # logger.warning(f"[{self.name}] Input '{channel_name}' is not mono (channels: {num_channels}).")
            return None

        if expected_blocksize is not None and blocksize != expected_blocksize:
            logger.warning(
                f"[{self.name}] Input '{channel_name}' blocksize ({blocksize}) "
                f"does not match expected ({expected_blocksize})."
            )
            return None
        # Return original signal, preserving dtype
        return signal

    def process(self, input_data: dict) -> dict:
        in_left = input_data.get("in_left")
        in_right = input_data.get("in_right")

        valid_left = None
        valid_right = None
        determined_blocksize = None

        if in_left is not None:
            valid_left = self._validate_mono_input(in_left, None, "in_left")
            if valid_left is not None:
                determined_blocksize = valid_left.shape[1]

        if in_right is not None:
            valid_right = self._validate_mono_input(in_right, determined_blocksize, "in_right")
            if valid_right is not None:
                if determined_blocksize is None:
                    determined_blocksize = valid_right.shape[1]
                elif valid_left is not None and valid_left.shape[1] != determined_blocksize:
                    logger.warning(
                        f"[{self.name}] Input 'in_left' blocksize ({valid_left.shape[1]}) "
                        f"mismatches 'in_right' blocksize ({determined_blocksize}). Discarding 'in_left'."
                    )
                    valid_left = None

        if valid_left is None and valid_right is None:
            # logger.debug(f"[{self.name}] Both inputs are invalid or missing. Outputting None.")
            return {"out": None}

        if determined_blocksize is None:
            logger.warning(f"[{self.name}] Could not determine blocksize from inputs. Outputting None.")
            return {"out": None}

        # Determine output dtype based on valid inputs
        output_dtype = DEFAULT_DTYPE  # Fallback
        if valid_left is not None:
            output_dtype = valid_left.dtype
        elif valid_right is not None:
            output_dtype = valid_right.dtype

        left_channel_data = (
            valid_left if valid_left is not None else torch.zeros((1, determined_blocksize), dtype=output_dtype)
        )
        right_channel_data = (
            valid_right if valid_right is not None else torch.zeros((1, determined_blocksize), dtype=output_dtype)
        )

        try:
            # torch.vstack will promote dtype if they are different
            stereo_out = torch.vstack((left_channel_data, right_channel_data))
            return {"out": stereo_out}
        except ValueError as e:
            logger.error(
                f"[{self.name}] Error during vstack, likely due to final blocksize mismatch: {e}", exc_info=True
            )
            return {"out": None}
        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error joining stereo channels: {e}", exc_info=True)
            return {"out": None}


class StereoChannelSplitterNode(Node):
    NODE_TYPE = "Stereo Channel Splitter"
    CATEGORY = "Utility"
    DESCRIPTION = "Splits a stereo signal into L/R mono. Mono input duplicates. Preserves dtype."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_output("out_left", data_type=torch.Tensor)
        self.add_output("out_right", data_type=torch.Tensor)
        logger.debug(f"[{self.name}] StereoChannelSplitterNode initialized.")

    def process(self, input_data: dict) -> dict:
        in_signal = input_data.get("in")

        if not isinstance(in_signal, torch.Tensor) or in_signal.ndim != 2:
            if in_signal is not None:
                logger.warning(
                    f"[{self.name}] Input is not a valid 2D torch Tensor. "
                    f"Received type: {type(in_signal)}, ndim: {getattr(in_signal, 'ndim', 'N/A')}."
                )
            return {"out_left": None, "out_right": None}

        num_channels, _blocksize = in_signal.shape

        if num_channels == 0:
            logger.warning(f"[{self.name}] Input has 0 channels.")
            return {"out_left": None, "out_right": None}

        out_left_data = None
        out_right_data = None

        try:
            if num_channels == 1:
                # Mono input: send same signal to L and R (views or copies, dtype preserved)
                out_left_data = in_signal
                out_right_data = in_signal.clone()  # Make a copy for right if it might be modified
            elif num_channels >= 2:
                # Stereo or multi-channel input: take first as L, second as R
                out_left_data = in_signal[0:1, :]  # Slice, preserves dtype
                out_right_data = in_signal[1:2, :]  # Slice, preserves dtype

            return {"out_left": out_left_data, "out_right": out_right_data}

        except Exception as e:
            logger.error(f"[{self.name}] Error splitting channels: {e}", exc_info=True)
            return {"out_left": None, "out_right": None}


# ==============================================================================
# 1. UI Class for the Panner Node (UNCHANGED)
# ==============================================================================

# --- Constants for Panner Node ---
MIN_PAN = -1.0  # Hard Left
MAX_PAN = 1.0  # Hard Right


class PannerNodeItem(ParameterNodeItem):
    """
    Refactored UI for the PannerNode.
    Inherits from ParameterNodeItem to auto-generate controls and overrides
    the state update method to provide custom label formatting.
    """

    NODE_SPECIFIC_WIDTH = 180

    def __init__(self, node_logic: "PannerNode"):
        # Define the parameters declaratively. ParameterNodeItem will create the slider.
        parameters = [
            {
                "key": "pan",
                "name": "Pan",
                "min": MIN_PAN,
                "max": MAX_PAN,
                "format": "{:.2f}",  # A default format, will be overridden by custom label logic
            }
        ]
        # Initialize the parent class, which creates all UI elements
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)

    def _format_pan_label(self, value: float, is_external: bool) -> str:
        """Formats the pan value into a user-friendly string (e.g., L 50%, Center, R 100%)."""
        if abs(value) < 0.01:
            label = "Center"
        elif value < 0:
            label = f"L {abs(value):.0%}"
        else:
            label = f"R {value:.0%}"

        if is_external:
            label += " (ext)"
        return f"Pan: {label}"

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        """
        Overrides the base class method to apply custom text formatting to the label
        after the base functionality (like updating the slider) has been executed.
        """
        # First, let the parent class handle standard updates (slider position, enabled state).
        super()._on_state_updated_from_logic(state)

        # Now, apply our custom formatting to the label that the parent class created.
        pan_value = state.get("pan", 0.0)
        is_connected = "pan" in self.node_logic.inputs and self.node_logic.inputs["pan"].connections

        # Access the label widget created by the parent class
        if "pan" in self._controls and "label" in self._controls["pan"]:
            label_widget = self._controls["pan"]["label"]
            label_widget.setText(self._format_pan_label(pan_value, is_connected))
            # Center the custom label text
            label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)


# ==============================================================================
# 2. Logic Class for the Panner Node
# ==============================================================================
@with_parameters
class PannerNode(Node):
    NODE_TYPE = "Panner"
    UI_CLASS = PannerNodeItem
    CATEGORY = "Utility"
    DESCRIPTION = "Positions a mono signal in the stereo field using constant power panning."

    # --- Declarative managed parameter ---
    pan = Parameter(default=0.0, clip=(MIN_PAN, MAX_PAN))

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)

        self._init_parameters()

        self.add_input("in", data_type=torch.Tensor)
        self.add_input("pan", data_type=float)  # Socket name must match parameter key
        self.add_output("out", data_type=torch.Tensor)

    def _get_state_snapshot_locked(self) -> dict:
        return self._get_parameters_state()

    def serialize_extra(self) -> dict:
        return self._serialize_parameters()

    def deserialize_extra(self, data: dict):
        self._deserialize_parameters(data)

    def process(self, input_data: dict) -> dict:
        in_signal = input_data.get("in")
        if not isinstance(in_signal, torch.Tensor):
            # no input
            return {"out": None}

        # --- Update parameters from sockets using the injected helper method ---
        self._update_parameters_from_sockets(input_data)

        # Create a consistent snapshot of the parameter for this processing block
        with self._lock:
            pan_value = self._pan  # Read the managed parameter

        # --- DSP Processing (unchanged) ---

        # 1. Ensure input signal is mono for panning
        if in_signal.shape[0] > 1:
            # If stereo or multi-channel, mix down to mono
            mono_signal = torch.mean(in_signal, dim=0, keepdim=True)
        else:
            mono_signal = in_signal

        # 2. Apply constant power panning law
        # Map pan value from [-1, 1] to angle in radians [0, pi/2]
        pan_rad = (pan_value + 1.0) * 0.5 * (torch.pi / 2.0)

        gain_left = torch.cos(torch.tensor(pan_rad, dtype=DEFAULT_DTYPE))
        gain_right = torch.sin(torch.tensor(pan_rad, dtype=DEFAULT_DTYPE))

        # 3. Apply gains to create L and R channels
        # Broadcasting handles applying the scalar gain to the entire mono tensor
        left_channel = mono_signal * gain_left
        right_channel = mono_signal * gain_right

        # 4. Stack the channels to create a stereo output tensor
        stereo_out = torch.cat((left_channel, right_channel), dim=0)

        return {"out": stereo_out}
