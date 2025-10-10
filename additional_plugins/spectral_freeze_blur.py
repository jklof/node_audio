import torch
import threading
import logging
from typing import Dict, Optional

# --- Node System Imports ---
from node_system import Node
from constants import SpectralFrame, DEFAULT_COMPLEX_DTYPE
from ui_elements import ParameterNodeItem

# --- Qt Imports ---
from PySide6.QtCore import Slot

# Configure logging
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. UI Class for the Spectral Freeze/Blur Node
# ==============================================================================
class SpectralFreezeBlurNodeItem(ParameterNodeItem):
    """
    UI for the SpectralFreezeBlurNode, created declaratively using the
    ParameterNodeItem base class for consistency and simplicity.
    """

    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "SpectralFreezeBlurNode"):
        # Define the UI controls and their properties.
        parameters = [
            {
                "key": "blur_amount",
                "name": "Blur Amount",
                "min": 0.0,
                "max": 0.99,  # Max is just under 1.0 to prevent a complete lock-up
                "format": "{:.0%}",
                "is_log": False,
            },
            {
                "key": "mix",
                "name": "Mix",
                "min": 0.0,
                "max": 1.0,
                "format": "{:.0%}",
                "is_log": False,
            },
        ]

        # The parent constructor handles all widget creation and signal connections.
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)


# ==============================================================================
# 2. Logic Class for the Spectral Freeze/Blur Node
# ==============================================================================
class SpectralFreezeBlurNode(Node):
    NODE_TYPE = "Spectral Freeze/Blur"
    UI_CLASS = SpectralFreezeBlurNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "Freezes the current spectrum on a gate signal or blurs it over time."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)

        # --- Define Sockets ---
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("freeze_gate", data_type=bool)
        self.add_input("blur_amount", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        # --- Internal State Parameters ---
        self._blur_amount: float = 0.0
        self._mix: float = 1.0

        # --- DSP State ---
        self._frozen_frame_data: Optional[torch.Tensor] = None
        self._blurred_frame_data: Optional[torch.Tensor] = None
        self._is_frozen: bool = False
        self._previous_gate: bool = False

    def _get_state_snapshot_locked(self) -> Dict:
        """Returns a copy of the current parameters for UI or serialization."""
        return {"blur_amount": self._blur_amount, "mix": self._mix}

    # --- Thread-safe setters for UI interaction ---
    @Slot(float)
    def set_blur_amount(self, value: float):
        state_to_emit = None
        with self._lock:
            # Clip value to a safe range to prevent the blur filter from locking up
            clipped_value = max(0.0, min(0.99, float(value)))
            if self._blur_amount != clipped_value:
                self._blur_amount = clipped_value
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_mix(self, value: float):
        state_to_emit = None
        with self._lock:
            clipped_value = max(0.0, min(1.0, float(value)))
            if self._mix != clipped_value:
                self._mix = clipped_value
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        state_snapshot_to_emit = None
        with self._lock:
            ui_update_needed = False

            # --- Read socket inputs, falling back to internal state ---
            freeze_gate = bool(input_data.get("freeze_gate", False))

            blur_socket = input_data.get("blur_amount")
            effective_blur = float(blur_socket) if blur_socket is not None else self._blur_amount
            if self._blur_amount != effective_blur:
                self._blur_amount = max(0.0, min(0.99, effective_blur))
                ui_update_needed = True

            mix_socket = input_data.get("mix")
            effective_mix = float(mix_socket) if mix_socket is not None else self._mix
            if self._mix != effective_mix:
                self._mix = max(0.0, min(1.0, effective_mix))
                ui_update_needed = True

            if ui_update_needed:
                state_snapshot_to_emit = self._get_state_snapshot_locked()

            # --- Freeze Logic (State Machine) ---
            # On a rising edge of the gate, capture the frame and set the freeze state.
            if freeze_gate and not self._previous_gate:
                self._is_frozen = True
                self._frozen_frame_data = frame.data.clone()
                logger.debug(f"[{self.name}] Spectral frame frozen.")
            # If the gate is released, unfreeze.
            elif not freeze_gate:
                self._is_frozen = False
            self._previous_gate = freeze_gate

            # --- Core DSP Logic ---
            wet_signal: torch.Tensor
            if self._is_frozen and self._frozen_frame_data is not None:
                # If frozen, the wet signal is simply the stored frame.
                wet_signal = self._frozen_frame_data
            else:
                # --- Blur Logic (Exponential Moving Average) ---
                if self._blurred_frame_data is None or self._blurred_frame_data.shape != frame.data.shape:
                    # Initialize or resize the blur buffer on the first run or format change.
                    self._blurred_frame_data = frame.data.clone()

                # Apply the one-pole smoothing filter.
                # A blur_amount of 0 means the output is 100% the new frame (no blur).
                # A high blur_amount means the output is mostly the old frame (heavy blur).
                alpha = self._blur_amount
                self._blurred_frame_data = alpha * self._blurred_frame_data + (1.0 - alpha) * frame.data
                wet_signal = self._blurred_frame_data

            # --- Final Dry/Wet Mix ---
            dry_signal = frame.data
            output_fft = (dry_signal * (1.0 - self._mix)) + (wet_signal * self._mix)

        # Emit signal to UI AFTER the lock is released
        if state_snapshot_to_emit:
            self.ui_update_callback(state_snapshot_to_emit)

        # Create the output frame, preserving all metadata from the input.
        output_frame = SpectralFrame(
            data=output_fft.to(DEFAULT_COMPLEX_DTYPE),
            fft_size=frame.fft_size,
            hop_size=frame.hop_size,
            window_size=frame.window_size,
            sample_rate=frame.sample_rate,
            analysis_window=frame.analysis_window,
        )
        return {"spectral_frame_out": output_frame}

    def start(self):
        """Called when graph processing starts. Reset all DSP state."""
        with self._lock:
            self._frozen_frame_data = None
            self._blurred_frame_data = None
            self._is_frozen = False
            self._previous_gate = False
        logger.debug(f"[{self.name}] Node started and state reset.")

    def serialize_extra(self) -> dict:
        """Save the node's user-configured parameters."""
        with self._lock:
            return self._get_state_snapshot_locked()

    def deserialize_extra(self, data: dict):
        """Load the node's parameters from a file, using the public setters."""
        self.set_blur_amount(data.get("blur_amount", 0.0))
        self.set_mix(data.get("mix", 1.0))
