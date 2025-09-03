import os
import threading
import logging
import weakref
from typing import Dict, Optional

import torch
import numpy as np  # Kept for UI slider mapping
import soundfile as sf
import torchaudio.transforms as T

from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE, DEFAULT_CHANNELS
from ui_elements import NodeItem, NODE_CONTENT_PADDING

from PySide6.QtCore import Qt, Signal, Slot, QObject, QRunnable, QThreadPool
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog

# --- Logging ---
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Background Sample Loader (QRunnable with Signal Emitter)
# ==============================================================================
class SampleLoadSignaller(QObject):
    load_finished = Signal(tuple)  # ('status', data, filepath)


class SampleLoadRunnable(QRunnable):
    def __init__(self, file_path: str, target_sr: int, signaller: SampleLoadSignaller):
        super().__init__()
        self.file_path = file_path
        self.target_sr = target_sr
        self.signaller = signaller

    def run(self):
        try:
            # Load as numpy array (samples, channels)
            audio_data_np, source_sr = sf.read(self.file_path, dtype="float32", always_2d=True)
            # Convert to torch tensor and transpose to (channels, samples)
            audio_data = torch.from_numpy(audio_data_np.T)

            # Use torchaudio for resampling
            if source_sr != self.target_sr:
                resampler = T.Resample(orig_freq=source_sr, new_freq=self.target_sr, dtype=audio_data.dtype)
                audio_data = resampler(audio_data)  # Input is (channels, time)

            # Normalize
            max_val = torch.max(torch.abs(audio_data))
            if max_val > 0:
                audio_data /= max_val

            self.signaller.load_finished.emit(("success", audio_data, self.file_path))
        except Exception as e:
            err_msg = f"Failed to load sample: {e}"
            logger.error(f"Error loading '{self.file_path}': {err_msg}", exc_info=True)
            self.signaller.load_finished.emit(("failure", err_msg, self.file_path))


# ==============================================================================
# 2. State Emitter for UI Communication
# ==============================================================================
class SampleEmitter(QObject):
    stateUpdated = Signal(dict)


# ==============================================================================
# 3. Custom UI Class (SampleNodeItem)
# ==============================================================================
class SampleNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "SampleNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )

        self.load_button = QPushButton("Load Sample")
        self.status_label = QLabel("Status: No Sample")
        self.filename_label = QLabel("File: None")
        self.filename_label.setWordWrap(True)

        layout.addWidget(self.load_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.filename_label)

        self.setContentWidget(self.container_widget)

        self.load_button.clicked.connect(self._on_load_clicked)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        self.updateFromLogic()

    @Slot()
    def _on_load_clicked(self):
        parent = self.scene().views()[0] if self.scene() and self.scene().views() else None
        file_path, _ = QFileDialog.getOpenFileName(parent, "Load Sample", "", "Audio Files (*.wav *.flac *.aiff)")
        if file_path:
            self.node_logic.load_file(file_path)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        status = state.get("status", "Unknown")
        filepath = state.get("filepath")

        self.status_label.setText(f"Status: {status}")
        self.filename_label.setText(f"File: {os.path.basename(filepath) if filepath else 'None'}")
        self.filename_label.setToolTip(filepath or "")

        if "Error" in status:
            self.status_label.setStyleSheet("color: red;")
        elif "Loading" in status:
            self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setStyleSheet("color: lightgreen;")

    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ==============================================================================
# 4. Node Logic Class (SampleNode)
# ==============================================================================
class SampleNode(Node):
    NODE_TYPE = "Sample Player"
    UI_CLASS = SampleNodeItem
    CATEGORY = "Generators"
    DESCRIPTION = "Loads an audio sample into memory and plays it on trigger."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = SampleEmitter()

        self.add_input("trigger", data_type=bool)
        self.add_input("pitch", data_type=float)
        self.add_output("out", data_type=torch.Tensor)
        self.add_output("on_end", data_type=bool)

        self._lock = threading.Lock()
        self._filepath: Optional[str] = None
        self._status: str = "No Sample"
        self._is_loading: bool = False

        self._audio_data: Optional[torch.Tensor] = None
        self._play_pos: float = 0.0
        self._is_playing: bool = False
        self._prev_trigger: bool = False

        self.loader_signaller = SampleLoadSignaller()
        self.loader_signaller.load_finished.connect(self._on_load_finished)

    def load_file(self, file_path: str):
        state_to_emit = None
        with self._lock:
            if self._is_loading:
                return
            self._is_loading = True
            self._filepath = file_path
            self._status = "Loading..."
            state_to_emit = self._get_current_state_snapshot_locked()

        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

        runnable = SampleLoadRunnable(file_path, DEFAULT_SAMPLERATE, self.loader_signaller)
        QThreadPool.globalInstance().start(runnable)

    @Slot(tuple)
    def _on_load_finished(self, result: tuple):
        status, data, filepath = result
        state_to_emit = None
        with self._lock:
            if filepath != self._filepath:
                return
            self._is_loading = False
            if status == "success":
                self._audio_data = data
                self._status = "Ready"
                self._is_playing = False
                self._play_pos = 0.0
            else:
                self._audio_data = None
                self._status = f"Error: {data}"
            state_to_emit = self._get_current_state_snapshot_locked()

        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {"status": self._status, "filepath": self._filepath}

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    def process(self, input_data: dict) -> dict:
        trigger_in = input_data.get("trigger")
        trigger = bool(trigger_in) if trigger_in is not None else False

        pitch_in = input_data.get("pitch")
        pitch = float(pitch_in) if pitch_in is not None else 1.0

        on_end_signal = False
        # Output block is (channels, samples)
        output_block = torch.zeros((DEFAULT_CHANNELS, DEFAULT_BLOCKSIZE), dtype=DEFAULT_DTYPE)

        with self._lock:
            if self._audio_data is None or self._audio_data.shape[1] < 2:  # Check for samples
                return {"out": output_block, "on_end": False}

            # Rising edge trigger
            if trigger and not self._prev_trigger:
                self._is_playing = True
                self._play_pos = 0.0
            self._prev_trigger = trigger

            if self._is_playing:
                num_channels_sample, num_frames_total = self._audio_data.shape

                # 1. Generate floating-point sample indices for this block
                indices = self._play_pos + torch.arange(DEFAULT_BLOCKSIZE, dtype=torch.float32) * pitch

                # 2. Find which indices are valid for interpolation
                valid_mask = indices < (num_frames_total - 1)
                valid_float_indices = indices[valid_mask]

                num_valid = len(valid_float_indices)
                if num_valid > 0:
                    # 3. Calculate floor indices and fractional parts
                    indices_floor = valid_float_indices.long()
                    indices_ceil = indices_floor + 1
                    fraction = (valid_float_indices - indices_floor).unsqueeze(0)  # Shape (1, num_valid)

                    # 4. Get sample data for floor and ceil indices
                    sample_floor = self._audio_data[:, indices_floor]
                    sample_ceil = self._audio_data[:, indices_ceil]

                    # 5. Perform linear interpolation
                    interpolated_samples = sample_floor * (1.0 - fraction) + sample_ceil * fraction

                    # 6. Channel matching for the output block
                    num_output_channels = output_block.shape[0]
                    if num_channels_sample == 1:
                        output_block[:, :num_valid] = interpolated_samples.expand(num_output_channels, -1)
                    else:
                        ch_to_copy = min(num_output_channels, num_channels_sample)
                        output_block[:ch_to_copy, :num_valid] = interpolated_samples[:ch_to_copy, :]

                # Update play position
                self._play_pos += DEFAULT_BLOCKSIZE * pitch

                if self._play_pos >= num_frames_total:
                    self._is_playing = False
                    on_end_signal = True

        return {"out": output_block, "on_end": on_end_signal}

    def serialize_extra(self) -> dict:
        with self._lock:
            return {"filepath": self._filepath}

    def deserialize_extra(self, data: dict):
        filepath = data.get("filepath")
        if filepath and os.path.exists(filepath):
            self.load_file(filepath)
