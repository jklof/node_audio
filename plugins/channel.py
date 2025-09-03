import torch
import logging
from node_system import Node

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
