import os
import threading
import logging
import weakref
from typing import Dict, Optional

import torch
import numpy as np
import soundfile as sf
import torchaudio.transforms as T

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE, DEFAULT_COMPLEX_DTYPE
from ui_elements import ParameterNodeItem, NODE_CONTENT_PADDING

# --- Qt Imports ---
from PySide6.QtCore import Qt, Signal, Slot, QObject, QRunnable, QThreadPool
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog

# --- Logging ---
logger = logging.getLogger(__name__)


# --- Node-Specific Constants ---
PARTITION_SIZE = DEFAULT_BLOCKSIZE
FFT_SIZE = PARTITION_SIZE * 2


# ==============================================================================
# 1. Background IR Loader (Unchanged)
# ==============================================================================
class IRLoadSignaller(QObject):
    load_finished = Signal(tuple)


class IRLoadRunnable(QRunnable):
    def __init__(self, node_ref: weakref.ReferenceType, file_path: str, target_sr: int, signaller: IRLoadSignaller):
        super().__init__()
        self._node_ref = node_ref
        self.file_path = file_path
        self.target_sr = target_sr
        self.signaller = signaller

    def run(self):
        node = self._node_ref()
        if not node:
            return
        try:
            ir_data_np, source_sr = sf.read(self.file_path, dtype="float32", always_2d=True)
            ir_data = torch.from_numpy(ir_data_np.T).to(DEFAULT_DTYPE)
            if ir_data.shape[0] > 1:
                ir_data = torch.mean(ir_data, dim=0, keepdim=True)
            if source_sr != self.target_sr:
                resampler = T.Resample(orig_freq=source_sr, new_freq=self.target_sr, dtype=DEFAULT_DTYPE)
                ir_data = resampler(ir_data)
            max_val = torch.max(torch.abs(ir_data))
            if max_val > 0:
                ir_data /= max_val
            ir_samples = ir_data.squeeze(0)
            num_partitions = int(np.ceil(len(ir_samples) / PARTITION_SIZE))
            padded_len = num_partitions * PARTITION_SIZE
            pad_amount = padded_len - len(ir_samples)
            if pad_amount > 0:
                ir_samples = torch.nn.functional.pad(ir_samples, (0, pad_amount), "constant", 0)
            partition_ffts = []
            for i in range(num_partitions):
                partition = ir_samples[i * PARTITION_SIZE : (i + 1) * PARTITION_SIZE]
                padded_partition = torch.nn.functional.pad(partition, (0, PARTITION_SIZE), "constant", 0)
                partition_fft = torch.fft.rfft(padded_partition, n=FFT_SIZE)
                partition_ffts.append(partition_fft)
            stacked_ffts = torch.stack(partition_ffts).unsqueeze(1).to(DEFAULT_COMPLEX_DTYPE)
            logger.info(
                f"[{node.name}] IR prepared: {len(ir_samples)} samples -> {num_partitions} partitions, shape {stacked_ffts.shape}"
            )
            self.signaller.load_finished.emit(("success", stacked_ffts, self.file_path))
        except Exception as e:
            err_msg = f"Failed to load or process IR '{os.path.basename(self.file_path)}': {e}"
            logger.error(err_msg, exc_info=True)
            self.signaller.load_finished.emit(("failure", err_msg, self.file_path))


# ==============================================================================
# 2. Custom UI Class (CORRECTED)
# ==============================================================================
class ConvolutionReverbNodeItem(ParameterNodeItem):
    """
    Refactored UI for the ConvolutionReverbNode with corrected constructor order.
    """

    NODE_SPECIFIC_WIDTH = 240

    def __init__(self, node_logic: "ConvolutionReverbNode"):
        # 1. Define standard parameters
        parameters = [
            {"key": "input_gain_db", "name": "Input", "min": -24.0, "max": 24.0, "format": "{:+.1f} dB"},
            {"key": "mix", "name": "Mix", "min": 0.0, "max": 1.0, "format": "{:.0%}"},
            {"key": "output_gain_db", "name": "Output", "min": -24.0, "max": 24.0, "format": "{:+.1f} dB"},
        ]

        # --- FIX: Create custom widgets and assign to self BEFORE calling super().__init__ ---
        self.load_button = QPushButton("Load Impulse Response")
        self.status_label = QLabel("Status: No IR Loaded")
        self.filename_label = QLabel("File: None")
        # --- END FIX ---

        # 2. Call the parent constructor. This will create the layout and also call
        #    _on_state_updated, which now works because self.status_label exists.
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)

        # 3. Configure and add the custom widgets to the layout created by the parent.
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filename_label.setWordWrap(True)
        self.status_label.setWordWrap(True)

        main_layout = self.container_widget.layout()
        main_layout.insertWidget(0, self.filename_label)
        main_layout.insertWidget(0, self.status_label)
        main_layout.insertWidget(0, self.load_button)

        # 4. Connect signals for the custom widgets.
        self.load_button.clicked.connect(self._handle_load_button)

    @Slot()
    def _handle_load_button(self):
        """Handles the file dialog for loading an impulse response."""
        parent = self.scene().views()[0] if self.scene() and self.scene().views() else None
        file_path, _ = QFileDialog.getOpenFileName(
            parent, "Open Impulse Response", "", "Audio Files (*.wav *.flac *.aiff)"
        )
        if file_path:
            self.node_logic.load_file(file_path)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        """
        Handles state updates from the logic node for both custom and parent widgets.
        """
        # --- Handle custom widget logic ---
        status = state.get("status", "Unknown")
        self.status_label.setText(f"Status: {status}")

        ir_filepath = state.get("ir_filepath")
        if ir_filepath:
            self.filename_label.setText(f"File: {os.path.basename(ir_filepath)}")
            self.filename_label.setToolTip(ir_filepath)
        else:
            self.filename_label.setText("File: None")
            self.filename_label.setToolTip("")

        if "Error" in status:
            self.status_label.setStyleSheet("color: red;")
        elif "Loading" in status:
            self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setStyleSheet("color: lightgreen;")

        # --- Call parent to handle the sliders, labels, and external control state ---
        super()._on_state_updated(state)


# ==============================================================================
# 3. Node Logic Class (ConvolutionReverbNode) (Unchanged)
# ==============================================================================
class ConvolutionReverbNode(Node):
    NODE_TYPE = "Convolution Reverb"
    UI_CLASS = ConvolutionReverbNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Applies reverb by convolving audio with an impulse response."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("input_gain_db", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_input("output_gain_db", data_type=float)
        self.add_output("out", data_type=torch.Tensor)
        self._lock = threading.Lock()
        self._ir_filepath: Optional[str] = None
        self._input_gain_db: float = 0.0
        self._output_gain_db: float = 0.0
        self._mix: float = 0.5
        self._current_input_gain = 1.0
        self._current_output_gain = 1.0
        self._current_mix = 0.5
        self._smoothing_coeff = 1.0 - np.exp(-1 / (0.01 * (DEFAULT_SAMPLERATE / DEFAULT_BLOCKSIZE)))
        self._status: str = "No IR Loaded"
        self._is_loading: bool = False
        self._ir_partitions_fft: Optional[torch.Tensor] = None
        self._input_fft_history: Optional[torch.Tensor] = None
        self._overlap_add_buffer: Optional[torch.Tensor] = None
        self._expected_channels: Optional[int] = None
        self._is_initialized: bool = False
        self.ir_loader_signaller = IRLoadSignaller()
        self.ir_loader_signaller.load_finished.connect(self._on_ir_load_finished)

    def load_file(self, file_path: str):
        if not os.path.exists(file_path):
            logger.error(f"IR file does not exist: {file_path}")
            return
        with self._lock:
            if self._is_loading:
                logger.warning("IR load already in progress, ignoring new request")
                return
            self._is_loading = True
            self._ir_filepath = file_path
            self._status = "Loading..."
        self.ui_update_callback(self._get_current_state_snapshot_locked())
        runnable = IRLoadRunnable(weakref.ref(self), file_path, DEFAULT_SAMPLERATE, self.ir_loader_signaller)
        QThreadPool.globalInstance().start(runnable)

    @Slot(tuple)
    def _on_ir_load_finished(self, result: tuple):
        status, data, filepath = result
        with self._lock:
            if filepath != self._ir_filepath:
                logger.debug(f"Ignoring stale IR load result for {filepath}")
                return
            self._is_loading = False
            if status == "success":
                self._ir_partitions_fft = data
                self._status = "Ready"
                if self._expected_channels is not None:
                    self._reset_dsp_state_locked(self._expected_channels)
                logger.info(f"[{self.name}] IR loaded successfully: {self._ir_partitions_fft.shape}")
            else:
                self._ir_partitions_fft = None
                self._status = f"Error: {data}"
                logger.error(f"[{self.name}] IR load failed: {data}")
        self.ui_update_callback(self._get_current_state_snapshot_locked())

    @Slot(float)
    def set_input_gain_db(self, value: float):
        with self._lock:
            self._input_gain_db = np.clip(value, -60.0, 60.0)
        self.ui_update_callback(self.get_current_state_snapshot())

    @Slot(float)
    def set_output_gain_db(self, value: float):
        with self._lock:
            self._output_gain_db = np.clip(value, -60.0, 60.0)
        self.ui_update_callback(self.get_current_state_snapshot())

    @Slot(float)
    def set_mix(self, value: float):
        with self._lock:
            self._mix = np.clip(value, 0.0, 1.0)
        self.ui_update_callback(self.get_current_state_snapshot())

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {
            "status": self._status,
            "ir_filepath": self._ir_filepath,
            "input_gain_db": self._input_gain_db,
            "output_gain_db": self._output_gain_db,
            "mix": self._mix,
        }

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    def _reset_dsp_state_locked(self, num_channels: int):
        self._expected_channels = num_channels
        if self._ir_partitions_fft is not None:
            num_ir_partitions = self._ir_partitions_fft.shape[0]
            fft_bins = self._ir_partitions_fft.shape[2]
            self._input_fft_history = torch.zeros(
                (num_ir_partitions, num_channels, fft_bins), dtype=DEFAULT_COMPLEX_DTYPE
            )
        else:
            self._input_fft_history = None
        self._overlap_add_buffer = torch.zeros((num_channels, PARTITION_SIZE), dtype=DEFAULT_DTYPE)
        self._is_initialized = True
        logger.debug(f"[{self.name}] DSP state reset for {num_channels} channels.")

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}
        if signal.dim() != 2:
            logger.error(f"Expected 2D tensor [channels, samples], got shape {signal.shape}")
            return {"out": signal}
        num_channels, num_samples = signal.shape
        if num_samples != PARTITION_SIZE:
            if num_samples < PARTITION_SIZE:
                padded_signal = torch.zeros((num_channels, PARTITION_SIZE), dtype=DEFAULT_DTYPE)
                padded_signal[:, :num_samples] = signal
                signal = padded_signal
            else:
                signal = signal[:, :PARTITION_SIZE]
                logger.warning(f"Input signal truncated from {num_samples} to {PARTITION_SIZE} samples")
        with torch.no_grad():
            with self._lock:
                input_gain_socket = input_data.get("input_gain_db")
                if input_gain_socket is not None:
                    self._input_gain_db = np.clip(float(input_gain_socket), -60.0, 60.0)
                output_gain_socket = input_data.get("output_gain_db")
                if output_gain_socket is not None:
                    self._output_gain_db = np.clip(float(output_gain_socket), -60.0, 60.0)
                mix_socket = input_data.get("mix")
                if mix_socket is not None:
                    self._mix = np.clip(float(mix_socket), 0.0, 1.0)
                target_input_gain = 10 ** (self._input_gain_db / 20.0)
                target_output_gain = 10 ** (self._output_gain_db / 20.0)
                target_mix = self._mix
                self._current_input_gain += self._smoothing_coeff * (target_input_gain - self._current_input_gain)
                self._current_output_gain += self._smoothing_coeff * (target_output_gain - self._current_output_gain)
                self._current_mix += self._smoothing_coeff * (target_mix - self._current_mix)
                if self._ir_partitions_fft is None:
                    return {"out": signal}
                if not self._is_initialized or self._expected_channels != num_channels:
                    self._reset_dsp_state_locked(num_channels)
                if self._input_fft_history is None:
                    logger.error("DSP state not properly initialized")
                    return {"out": signal}
                gained_signal = signal * self._current_input_gain
                padded_input = torch.nn.functional.pad(gained_signal, (0, PARTITION_SIZE), "constant", 0)
                input_fft = torch.fft.rfft(padded_input, n=FFT_SIZE, dim=1).to(DEFAULT_COMPLEX_DTYPE)
                self._input_fft_history = torch.roll(self._input_fft_history, shifts=1, dims=0)
                self._input_fft_history[0] = input_fft
                convolved_partitions = self._input_fft_history * self._ir_partitions_fft
                output_fft_sum = torch.sum(convolved_partitions, dim=0)
                full_ifft = torch.fft.irfft(output_fft_sum, n=FFT_SIZE, dim=1)
                wet_signal = full_ifft[:, :PARTITION_SIZE] + self._overlap_add_buffer
                self._overlap_add_buffer = full_ifft[:, PARTITION_SIZE:].clone()
                wet_signal_gained = wet_signal * self._current_output_gain
                dry_mix = 1.0 - self._current_mix
                wet_mix = self._current_mix
                output_signal = (signal * dry_mix) + (wet_signal_gained * wet_mix)
        return {"out": output_signal}

    def start(self):
        with self._lock:
            self._current_input_gain = 10 ** (self._input_gain_db / 20.0)
            self._current_output_gain = 10 ** (self._output_gain_db / 20.0)
            self._current_mix = self._mix
            if self._expected_channels is not None:
                self._reset_dsp_state_locked(self._expected_channels)
            else:
                self._is_initialized = False
        logger.info(f"[{self.name}] Node started")

    def stop(self):
        with self._lock:
            self._is_initialized = False
        logger.info(f"[{self.name}] Node stopped")

    def remove(self):
        self.ir_loader_signaller.load_finished.disconnect(self._on_ir_load_finished)
        super().remove()

    def serialize_extra(self) -> dict:
        with self._lock:
            return {
                "ir_filepath": self._ir_filepath,
                "input_gain_db": self._input_gain_db,
                "output_gain_db": self._output_gain_db,
                "mix": self._mix,
            }

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._input_gain_db = data.get("input_gain_db", 0.0)
            self._output_gain_db = data.get("output_gain_db", 0.0)
            self._mix = data.get("mix", 0.5)
        filepath = data.get("ir_filepath")
        if filepath and os.path.exists(filepath):
            self.load_file(filepath)
        elif filepath:
            with self._lock:
                self._status = "Error: File not found"
                self._ir_filepath = filepath
            self.ui_update_callback(self._get_current_state_snapshot_locked())
