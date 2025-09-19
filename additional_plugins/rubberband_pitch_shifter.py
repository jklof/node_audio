# additional_plugins/rubberband_pitch_shifter.py

import torch
import numpy as np
import threading
import logging
from pylibrb import RubberBandStretcher, Option
from typing import Dict, Optional

from node_system import Node
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE
from ui_elements import ParameterNodeItem, NodeStateEmitter
from PySide6.QtCore import Slot

logger = logging.getLogger(__name__)


# --- UI Class ---
class RubberBandPitchShiftNodeItem(ParameterNodeItem):
    """Modern UI for the RubberBand Pitch Shifter using ParameterNodeItem."""

    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "RubberBandPitchShiftNode"):
        parameters = [
            {
                "key": "pitch_shift_st",
                "name": "Pitch Shift",
                "type": "slider",
                "min": -24.0,
                "max": 24.0,
                "format": "{:+.1f} st",
            },
            {
                "key": "formant_mode",
                "name": "Formants",
                "type": "combobox",
                "items": [("Preserve", "Preserve"), ("Shifted", "Shifted"), ("Off (Classic)", "Off")],
            },
            {
                "key": "formant_shift_st",
                "name": "Formant Shift",
                "type": "slider",
                "min": -12.0,
                "max": 12.0,
                "format": "{:+.1f} st",
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        """Overrides base method to add custom UI logic."""
        # First, call the parent method to handle standard updates.
        super()._on_state_updated(state)

        # Custom logic: only enable the formant shift dial when mode is "Shifted"
        formant_mode = state.get("formant_mode")
        is_shift_mode = formant_mode == "Shifted"

        formant_shift_control = self._controls.get("formant_shift_st")
        if formant_shift_control:
            formant_shift_control["widget"].setEnabled(is_shift_mode)
            # You could also dim the label if you prefer
            # label_palette = formant_shift_control["label"].palette()
            # label_palette.setColor(label_palette.ColorRole.WindowText, QColor("gray" if not is_shift_mode else "white"))
            # formant_shift_control["label"].setPalette(label_palette)


# --- Logic Class ---
class RubberBandPitchShiftNode(Node):
    NODE_TYPE = "RubberBand Pitch Shifter"
    UI_CLASS = RubberBandPitchShiftNodeItem
    CATEGORY = "Audio Effects"
    DESCRIPTION = "High-quality pitch/formant shifter using the RubberBand time-stretching library."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self.add_input("audio_in", data_type=torch.Tensor)
        self.add_input("pitch_shift_st", data_type=float)
        self.add_input("formant_shift_st", data_type=float)
        self.add_output("audio_out", data_type=torch.Tensor)

        self._lock = threading.Lock()
        self._pitch_shift_st: float = 0.0
        self._formant_shift_st: float = 0.0
        self._formant_mode: str = "Preserve"

        self._stretcher: Optional[RubberBandStretcher] = None
        self._current_channels: int = -1
        self._output_buffer: Optional[torch.Tensor] = None

    def start(self):
        with self._lock:
            self._cleanup_stretcher_locked()
        logger.debug(f"[{self.name}] Started and reset.")

    def stop(self):
        with self._lock:
            self._cleanup_stretcher_locked()
        logger.debug(f"[{self.name}] Stopped and cleaned up.")

    def remove(self):
        self.stop()
        super().remove()

    def _cleanup_stretcher_locked(self):
        """Cleans up the stretcher and buffers. Must be called with lock held."""
        self._stretcher = None
        self._current_channels = -1
        self._output_buffer = None

    def _recreate_stretcher_locked(self, num_channels: int):
        """Creates a new RubberBandStretcher instance. Must be called with lock held."""
        logger.info(f"[{self.name}] Recreating stretcher for {num_channels} channels (Mode: {self._formant_mode}).")
        self._cleanup_stretcher_locked()

        options = Option.PROCESS_REALTIME | Option.ENGINE_FINER | Option.WINDOW_SHORT | Option.SMOOTHING_ON
        if self._formant_mode == "Preserve":
            options |= Option.FORMANT_PRESERVED

        try:
            self._stretcher = RubberBandStretcher(
                sample_rate=DEFAULT_SAMPLERATE,
                channels=num_channels,
                options=options,
                initial_time_ratio=1.0,
                initial_pitch_scale=2.0 ** (self._pitch_shift_st / 12.0),
            )
            self._current_channels = num_channels
            self._output_buffer = torch.empty((num_channels, 0), dtype=DEFAULT_DTYPE)
        except Exception as e:
            logger.error(f"[{self.name}] Failed to create RubberBandStretcher: {e}", exc_info=True)
            self._stretcher = None

    @Slot(float)
    def set_pitch_shift_st(self, value: float):
        self._update_parameter("_pitch_shift_st", value)

    @Slot(float)
    def set_formant_shift_st(self, value: float):
        self._update_parameter("_formant_shift_st", value)

    @Slot(str)
    def set_formant_mode(self, value: str):
        state_to_emit = None  # Initialize the variable to None
        with self._lock:
            if self._formant_mode != value:
                self._formant_mode = value
                # Formant mode change requires full recreation of the stretcher
                self._cleanup_stretcher_locked()
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def _update_parameter(self, attr_name: str, value):
        state_to_emit = None
        with self._lock:
            if getattr(self, attr_name) != value:
                setattr(self, attr_name, value)
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {
            "pitch_shift_st": self._pitch_shift_st,
            "formant_shift_st": self._formant_shift_st,
            "formant_mode": self._formant_mode,
        }

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    def process(self, input_data: dict) -> dict:
        audio_in = input_data.get("audio_in")

        with self._lock:
            state_to_emit = None
            ui_update_needed = False

            # --- Update parameters from sockets ---
            pitch_socket = input_data.get("pitch_shift_st")
            effective_pitch = float(pitch_socket) if pitch_socket is not None else self._pitch_shift_st
            if self._pitch_shift_st != effective_pitch:
                self._pitch_shift_st = effective_pitch
                ui_update_needed = True

            formant_socket = input_data.get("formant_shift_st")
            effective_formant = float(formant_socket) if formant_socket is not None else self._formant_shift_st
            if self._formant_shift_st != effective_formant:
                self._formant_shift_st = effective_formant
                ui_update_needed = True

            if ui_update_needed:
                state_to_emit = self._get_current_state_snapshot_locked()

            # --- Handle input and stretcher state ---
            if audio_in is not None and isinstance(audio_in, torch.Tensor):
                num_channels, num_samples = audio_in.shape
                if self._stretcher is None or self._current_channels != num_channels:
                    self._recreate_stretcher_locked(num_channels)

                if self._stretcher:
                    # Convert torch tensor to numpy for pylibrb
                    numpy_in = audio_in.numpy().astype(np.float32)

                    # Set stretcher parameters
                    self._stretcher.pitch_scale = 2.0 ** (self._pitch_shift_st / 12.0)
                    if self._formant_mode == "Shifted":
                        self._stretcher.formant_scale = 2.0 ** (self._formant_shift_st / 12.0)

                    # Process and retrieve
                    self._stretcher.process(numpy_in, final=False)
                    available_samples = self._stretcher.available()
                    if available_samples > 0:
                        numpy_out = self._stretcher.retrieve(available_samples)
                        # Convert numpy back to torch and append to buffer
                        torch_out = torch.from_numpy(numpy_out.astype(np.float32))
                        self._output_buffer = torch.cat((self._output_buffer, torch_out), dim=1)

            # --- Manage output buffer ---
            if self._output_buffer is not None and self._output_buffer.shape[1] >= DEFAULT_BLOCKSIZE:
                output_block = self._output_buffer[:, :DEFAULT_BLOCKSIZE]
                self._output_buffer = self._output_buffer[:, DEFAULT_BLOCKSIZE:]
            else:
                # Output silence if not enough data is ready
                channels = self._current_channels if self._current_channels > 0 else 1
                output_block = torch.zeros((channels, DEFAULT_BLOCKSIZE), dtype=DEFAULT_DTYPE)

        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

        return {"audio_out": output_block}

    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()

    def deserialize_extra(self, data: dict):
        self.set_pitch_shift_st(data.get("pitch_shift_st", 0.0))
        self.set_formant_shift_st(data.get("formant_shift_st", 0.0))
        # Important: set_formant_mode also handles stretcher recreation
        self.set_formant_mode(data.get("formant_mode", "Preserve"))
