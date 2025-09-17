import os
import threading
import logging
import weakref
from typing import Dict, Optional, List

import torch
import numpy as np
import soundfile as sf
import torchaudio.transforms as T

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE, DEFAULT_COMPLEX_DTYPE
from ui_elements import NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING

# --- Qt Imports ---
from PySide6.QtCore import Qt, Signal, Slot, QObject, QRunnable, QThreadPool, QSignalBlocker
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QSlider, QCheckBox

# --- Logging ---
logger = logging.getLogger(__name__)


# --- Node-Specific Constants ---
PARTITION_SIZE = DEFAULT_BLOCKSIZE
FFT_SIZE = PARTITION_SIZE * 2


# ==============================================================================
# 1. Background IR Loader
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

            # Convert to mono if multi-channel
            if ir_data.shape[0] > 1:
                ir_data = torch.mean(ir_data, dim=0, keepdim=True)

            # Resample if necessary
            if source_sr != self.target_sr:
                resampler = T.Resample(orig_freq=source_sr, new_freq=self.target_sr, dtype=DEFAULT_DTYPE)
                ir_data = resampler(ir_data)

            # Normalize to prevent clipping
            max_val = torch.max(torch.abs(ir_data))
            if max_val > 0:
                ir_data /= max_val

            # Partition the IR into blocks and compute FFTs
            ir_samples = ir_data.squeeze(0)  # Remove channel dimension for processing
            num_partitions = int(np.ceil(len(ir_samples) / PARTITION_SIZE))

            # Pad IR to exact partition boundary
            padded_len = num_partitions * PARTITION_SIZE
            pad_amount = padded_len - len(ir_samples)
            if pad_amount > 0:
                ir_samples = torch.nn.functional.pad(ir_samples, (0, pad_amount), "constant", 0)

            # Compute FFT for each partition
            partition_ffts = []
            for i in range(num_partitions):
                partition = ir_samples[i * PARTITION_SIZE : (i + 1) * PARTITION_SIZE]
                # Zero-pad to FFT_SIZE for linear convolution
                padded_partition = torch.nn.functional.pad(partition, (0, PARTITION_SIZE), "constant", 0)
                partition_fft = torch.fft.rfft(padded_partition, n=FFT_SIZE)
                partition_ffts.append(partition_fft)

            # Stack into tensor [num_partitions, fft_bins] -> [num_partitions, 1, fft_bins]
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
# 2. Custom UI Class
# ==============================================================================
class ConvolutionReverbNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 240

    def __init__(self, node_logic: "ConvolutionReverbNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)
        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(5)

        # IR Loading controls
        self.load_button = QPushButton("Load Impulse Response")
        self.status_label = QLabel("Status: No IR Loaded")
        self.filename_label = QLabel("File: None")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filename_label.setWordWrap(True)
        self.status_label.setWordWrap(True)

        main_layout.addWidget(self.load_button)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.filename_label)

        # Parameter controls
        self.input_gain_slider, self.input_gain_label = self._create_slider_control("Input", -24.0, 24.0, "{:+.1f} dB")
        self.mix_slider, self.mix_label = self._create_slider_control("Mix", 0.0, 1.0, "{:.0%}")
        self.output_gain_slider, self.output_gain_label = self._create_slider_control(
            "Output", -24.0, 24.0, "{:+.1f} dB"
        )

        main_layout.addWidget(self.input_gain_label)
        main_layout.addWidget(self.input_gain_slider)
        main_layout.addWidget(self.mix_label)
        main_layout.addWidget(self.mix_slider)
        main_layout.addWidget(self.output_gain_label)
        main_layout.addWidget(self.output_gain_slider)

        self.setContentWidget(self.container_widget)

        # Connect signals
        self.load_button.clicked.connect(self._handle_load_button)
        self.input_gain_slider.valueChanged.connect(self._on_input_gain_changed)
        self.mix_slider.valueChanged.connect(self._on_mix_changed)
        self.output_gain_slider.valueChanged.connect(self._on_output_gain_changed)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

    def _create_slider_control(self, name: str, min_val: float, max_val: float, fmt: str) -> tuple[QSlider, QLabel]:
        label = QLabel(f"{name}: ...")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)
        slider.setProperty("min_val", min_val)
        slider.setProperty("max_val", max_val)
        slider.setProperty("name", name)
        slider.setProperty("format", fmt)
        return slider, label

    def _map_slider_value_to_logical(self, slider: QSlider, value: int) -> float:
        min_val = slider.property("min_val")
        max_val = slider.property("max_val")
        norm = value / 1000.0
        return min_val + norm * (max_val - min_val)

    def _map_logical_to_slider_value(self, slider: QSlider, value: float) -> int:
        min_val = slider.property("min_val")
        max_val = slider.property("max_val")
        range_val = max_val - min_val
        if range_val == 0:
            return 0
        norm = (value - min_val) / range_val
        return int(np.clip(norm, 0.0, 1.0) * 1000.0)

    @Slot()
    def _handle_load_button(self):
        parent = self.scene().views()[0] if self.scene() and self.scene().views() else None
        file_path, _ = QFileDialog.getOpenFileName(
            parent, "Open Impulse Response", "", "Audio Files (*.wav *.flac *.aiff)"
        )
        if file_path:
            self.node_logic.load_file(file_path)

    @Slot(int)
    def _on_input_gain_changed(self, value: int):
        self.node_logic.set_input_gain_db(self._map_slider_value_to_logical(self.input_gain_slider, value))

    @Slot(int)
    def _on_mix_changed(self, value: int):
        self.node_logic.set_mix(self._map_slider_value_to_logical(self.mix_slider, value))

    @Slot(int)
    def _on_output_gain_changed(self, value: int):
        self.node_logic.set_output_gain_db(self._map_slider_value_to_logical(self.output_gain_slider, value))

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        status = state.get("status", "Unknown")
        self.status_label.setText(f"Status: {status}")

        ir_filepath = state.get("ir_filepath")
        if ir_filepath:
            self.filename_label.setText(f"File: {os.path.basename(ir_filepath)}")
            self.filename_label.setToolTip(ir_filepath)
        else:
            self.filename_label.setText("File: None")
            self.filename_label.setToolTip("")

        # Color coding for status
        if "Error" in status:
            self.status_label.setStyleSheet("color: red;")
        elif "Loading" in status:
            self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setStyleSheet("color: lightgreen;")

        # Update sliders
        sliders_map = {
            "input_gain_db": (self.input_gain_slider, self.input_gain_label),
            "mix": (self.mix_slider, self.mix_label),
            "output_gain_db": (self.output_gain_slider, self.output_gain_label),
        }

        for key, (slider, label) in sliders_map.items():
            value = state.get(key, slider.property("min_val"))
            is_connected = key in self.node_logic.inputs and self.node_logic.inputs[key].connections
            slider.setEnabled(not is_connected)

            with QSignalBlocker(slider):
                slider.setValue(self._map_logical_to_slider_value(slider, value))

            label_text = f"{slider.property('name')}: {slider.property('format').format(value)}"
            if is_connected:
                label_text += " (ext)"
            label.setText(label_text)

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ==============================================================================
# 3. Node Logic Class (ConvolutionReverbNode)
# ==============================================================================
class ConvolutionReverbNode(Node):
    NODE_TYPE = "Convolution Reverb"
    UI_CLASS = ConvolutionReverbNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Applies reverb by convolving audio with an impulse response."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()

        # Define inputs and outputs
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("input_gain_db", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_input("output_gain_db", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._lock = threading.Lock()

        # --- Target parameters set by UI/Sockets ---
        self._ir_filepath: Optional[str] = None
        self._input_gain_db: float = 0.0
        self._output_gain_db: float = 0.0
        self._mix: float = 0.5

        # --- Smoothed parameters for processing ---
        self._current_input_gain = 1.0
        self._current_output_gain = 1.0
        self._current_mix = 0.5
        self._smoothing_coeff = 1.0 - np.exp(-1 / (0.01 * (DEFAULT_SAMPLERATE / DEFAULT_BLOCKSIZE)))

        # --- DSP State ---
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
        """Load an impulse response file in the background."""
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

        self.emitter.stateUpdated.emit(self._get_current_state_snapshot_locked())
        runnable = IRLoadRunnable(weakref.ref(self), file_path, DEFAULT_SAMPLERATE, self.ir_loader_signaller)
        QThreadPool.globalInstance().start(runnable)

    @Slot(tuple)
    def _on_ir_load_finished(self, result: tuple):
        """Handle completion of IR loading."""
        status, data, filepath = result

        with self._lock:
            # Check if this result is for the current file request
            if filepath != self._ir_filepath:
                logger.debug(f"Ignoring stale IR load result for {filepath}")
                return

            self._is_loading = False

            if status == "success":
                self._ir_partitions_fft = data
                self._status = "Ready"
                # Reset DSP state if we already know the expected channels
                if self._expected_channels is not None:
                    self._reset_dsp_state_locked(self._expected_channels)
                logger.info(f"[{self.name}] IR loaded successfully: {self._ir_partitions_fft.shape}")
            else:
                self._ir_partitions_fft = None
                self._status = f"Error: {data}"
                logger.error(f"[{self.name}] IR load failed: {data}")

        self.emitter.stateUpdated.emit(self._get_current_state_snapshot_locked())

    @Slot(float)
    def set_input_gain_db(self, value: float):
        with self._lock:
            self._input_gain_db = np.clip(value, -60.0, 60.0)
        self.emitter.stateUpdated.emit(self.get_current_state_snapshot())

    @Slot(float)
    def set_output_gain_db(self, value: float):
        with self._lock:
            self._output_gain_db = np.clip(value, -60.0, 60.0)
        self.emitter.stateUpdated.emit(self.get_current_state_snapshot())

    @Slot(float)
    def set_mix(self, value: float):
        with self._lock:
            self._mix = np.clip(value, 0.0, 1.0)
        self.emitter.stateUpdated.emit(self.get_current_state_snapshot())

    def _get_current_state_snapshot_locked(self) -> Dict:
        """Get current state snapshot while holding lock."""
        return {
            "status": self._status,
            "ir_filepath": self._ir_filepath,
            "input_gain_db": self._input_gain_db,
            "output_gain_db": self._output_gain_db,
            "mix": self._mix,
        }

    def get_current_state_snapshot(self) -> Dict:
        """Get current state snapshot (thread-safe)."""
        with self._lock:
            return self._get_current_state_snapshot_locked()

    def _reset_dsp_state_locked(self, num_channels: int):
        """Reset all DSP processing state for the given number of channels."""
        self._expected_channels = num_channels

        if self._ir_partitions_fft is not None:
            num_ir_partitions = self._ir_partitions_fft.shape[0]
            fft_bins = self._ir_partitions_fft.shape[2]  # Shape: [partitions, 1, fft_bins]

            # Initialize history buffer: [num_partitions, channels, fft_bins]
            self._input_fft_history = torch.zeros(
                (num_ir_partitions, num_channels, fft_bins), dtype=DEFAULT_COMPLEX_DTYPE
            )
        else:
            self._input_fft_history = None

        # Initialize overlap-add buffer
        self._overlap_add_buffer = torch.zeros((num_channels, PARTITION_SIZE), dtype=DEFAULT_DTYPE)
        self._is_initialized = True

        logger.debug(f"[{self.name}] DSP state reset for {num_channels} channels.")

    def process(self, input_data: dict) -> dict:
        """Process one block of audio."""
        signal = input_data.get("in")

        # Input validation
        if not isinstance(signal, torch.Tensor):
            return {"out": None}

        if signal.dim() != 2:
            logger.error(f"Expected 2D tensor [channels, samples], got shape {signal.shape}")
            return {"out": signal}  # Pass through

        # Ensure we have the correct block size
        num_channels, num_samples = signal.shape
        if num_samples != PARTITION_SIZE:
            # Pad or truncate to partition size
            if num_samples < PARTITION_SIZE:
                padded_signal = torch.zeros((num_channels, PARTITION_SIZE), dtype=DEFAULT_DTYPE)
                padded_signal[:, :num_samples] = signal
                signal = padded_signal
            else:
                signal = signal[:, :PARTITION_SIZE]
                logger.warning(f"Input signal truncated from {num_samples} to {PARTITION_SIZE} samples")

        with torch.no_grad():
            with self._lock:
                # Update parameters from socket inputs
                input_gain_socket = input_data.get("input_gain_db")
                if input_gain_socket is not None:
                    self._input_gain_db = np.clip(float(input_gain_socket), -60.0, 60.0)

                output_gain_socket = input_data.get("output_gain_db")
                if output_gain_socket is not None:
                    self._output_gain_db = np.clip(float(output_gain_socket), -60.0, 60.0)

                mix_socket = input_data.get("mix")
                if mix_socket is not None:
                    self._mix = np.clip(float(mix_socket), 0.0, 1.0)

                # Smooth parameter changes to avoid clicks
                target_input_gain = 10 ** (self._input_gain_db / 20.0)
                target_output_gain = 10 ** (self._output_gain_db / 20.0)
                target_mix = self._mix

                self._current_input_gain += self._smoothing_coeff * (target_input_gain - self._current_input_gain)
                self._current_output_gain += self._smoothing_coeff * (target_output_gain - self._current_output_gain)
                self._current_mix += self._smoothing_coeff * (target_mix - self._current_mix)

                # Early return if no IR loaded
                if self._ir_partitions_fft is None:
                    return {"out": signal}

                # Initialize or reset DSP state if channel count changed
                if not self._is_initialized or self._expected_channels != num_channels:
                    self._reset_dsp_state_locked(num_channels)

                if self._input_fft_history is None:
                    logger.error("DSP state not properly initialized")
                    return {"out": signal}

                # Apply input gain
                gained_signal = signal * self._current_input_gain

                # Compute FFT of current input block (zero-padded for linear convolution)
                padded_input = torch.nn.functional.pad(gained_signal, (0, PARTITION_SIZE), "constant", 0)
                input_fft = torch.fft.rfft(padded_input, n=FFT_SIZE, dim=1).to(DEFAULT_COMPLEX_DTYPE)

                # Update circular history buffer
                # Shift all partitions back by 1, newest goes to position 0
                self._input_fft_history = torch.roll(self._input_fft_history, shifts=1, dims=0)
                self._input_fft_history[0] = input_fft  # Most recent input

                # Perform convolution with each IR partition
                # ir_partitions_fft shape: [num_partitions, 1, fft_bins]
                # input_fft_history shape: [num_partitions, channels, fft_bins]
                convolved_partitions = self._input_fft_history * self._ir_partitions_fft

                # Sum contributions from all partitions
                output_fft_sum = torch.sum(convolved_partitions, dim=0)  # [channels, fft_bins]

                # Convert back to time domain
                full_ifft = torch.fft.irfft(output_fft_sum, n=FFT_SIZE, dim=1)  # [channels, FFT_SIZE]

                # Standard overlap-add for convolution tails (no windowing)
                wet_signal = full_ifft[:, :PARTITION_SIZE] + self._overlap_add_buffer

                # Store second half (convolution tail) for next block
                self._overlap_add_buffer = full_ifft[:, PARTITION_SIZE:].clone()

                # Apply output gain to wet signal
                wet_signal_gained = wet_signal * self._current_output_gain

                # Mix dry and wet signals
                dry_mix = 1.0 - self._current_mix
                wet_mix = self._current_mix
                output_signal = (signal * dry_mix) + (wet_signal_gained * wet_mix)

        return {"out": output_signal}

    def start(self):
        """Initialize the node for processing."""
        with self._lock:
            # Initialize smoothed parameters to their targets
            self._current_input_gain = 10 ** (self._input_gain_db / 20.0)
            self._current_output_gain = 10 ** (self._output_gain_db / 20.0)
            self._current_mix = self._mix

            # Reset DSP state
            if self._expected_channels is not None:
                self._reset_dsp_state_locked(self._expected_channels)
            else:
                self._is_initialized = False

        logger.info(f"[{self.name}] Node started")

    def stop(self):
        """Clean up when stopping."""
        with self._lock:
            self._is_initialized = False
        logger.info(f"[{self.name}] Node stopped")

    def remove(self):
        """Clean up when removing node."""
        # Disconnect signal to prevent callbacks after deletion
        self.ir_loader_signaller.load_finished.disconnect(self._on_ir_load_finished)
        super().remove()

    def serialize_extra(self) -> dict:
        """Serialize node-specific data."""
        with self._lock:
            return {
                "ir_filepath": self._ir_filepath,
                "input_gain_db": self._input_gain_db,
                "output_gain_db": self._output_gain_db,
                "mix": self._mix,
            }

    def deserialize_extra(self, data: dict):
        """Deserialize node-specific data."""
        with self._lock:
            self._input_gain_db = data.get("input_gain_db", 0.0)
            self._output_gain_db = data.get("output_gain_db", 0.0)
            self._mix = data.get("mix", 0.5)

        # Load IR file if it exists
        filepath = data.get("ir_filepath")
        if filepath and os.path.exists(filepath):
            self.load_file(filepath)
        elif filepath:
            with self._lock:
                self._status = "Error: File not found"
                self._ir_filepath = filepath  # Keep the path for UI display
            self.emitter.stateUpdated.emit(self._get_current_state_snapshot_locked())
