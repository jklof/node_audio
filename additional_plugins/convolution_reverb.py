import os
import threading
import logging
import weakref
from collections import deque
from typing import Dict, Optional, List

import numpy as np
import soundfile as sf
import resampy

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE
from ui_elements import NodeItem, NODE_CONTENT_PADDING

# --- Qt Imports ---
from PySide6.QtCore import Qt, Signal, Slot, QObject, QRunnable, QThreadPool, QSignalBlocker
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QSlider, QCheckBox
)

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Numba Setup with Fallback ---
try:
    from numba import njit
    logging.info("Numba detected in convolution_reverb. JIT compilation enabled.")
except ImportError:
    logging.warning("Numba not found in convolution_reverb. Using pure Python fallback.")
    def njit(func=None, **kwargs):
        if func:
            return func
        else:
            return lambda f: f



# --- Node-Specific Constants ---
PARTITION_SIZE = DEFAULT_BLOCKSIZE
FFT_SIZE = PARTITION_SIZE * 2

# ==============================================================================
# 1. Background IR Loader (QRunnable with Signal Emitter)
# ==============================================================================

class IRLoadSignaller(QObject):
    """Signal emitter to communicate from background thread to node."""
    load_finished = Signal(tuple) # ('status', data, filepath)

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
            logger.info(f"[{node.name}] Starting IR load for: {self.file_path}")
            ir_data, source_sr = sf.read(self.file_path, dtype='float32')
            if ir_data.ndim > 1: ir_data = np.mean(ir_data, axis=1)
            if source_sr != self.target_sr:
                ir_data = resampy.resample(ir_data, source_sr, self.target_sr, filter='kaiser_fast')
            max_val = np.max(np.abs(ir_data))
            if max_val > 0: ir_data /= max_val
            logger.info(f"[{node.name}] IR loaded. Now calculating FFT partitions...")

            num_partitions = int(np.ceil(len(ir_data) / PARTITION_SIZE))
            padded_len = num_partitions * PARTITION_SIZE
            padded_ir = np.pad(ir_data, (0, padded_len - len(ir_data)), 'constant')
            
            fft_partitions = [
                np.ascontiguousarray(
                    np.fft.rfft(padded_ir[i*PARTITION_SIZE:(i+1)*PARTITION_SIZE], n=FFT_SIZE).astype(np.complex64)
                )
                for i in range(num_partitions)
            ]

            logger.info(f"[{node.name}] IR prepared into {num_partitions} FFT partitions.")
            self.signaller.load_finished.emit(('success', fft_partitions, self.file_path))

        except Exception as e:
            err_msg = f"Failed to load or process IR '{os.path.basename(self.file_path)}': {e}"
            logger.error(err_msg, exc_info=True)
            self.signaller.load_finished.emit(('failure', err_msg, self.file_path))

# ==============================================================================
# 2. Signal Emitter for UI Communication
# ==============================================================================
class ConvolutionReverbEmitter(QObject):
    stateUpdated = Signal(dict)

# ==============================================================================
# 3. Custom UI Class (ConvolutionReverbNodeItem)
# ==============================================================================
class ConvolutionReverbNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 240

    def __init__(self, node_logic: "ConvolutionReverbNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING)
        main_layout.setSpacing(5)

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
        
        self.input_gain_slider, self.input_gain_label = self._create_slider_control("Input", -24.0, 24.0, "{:+.1f} dB")
        self.mix_slider, self.mix_label = self._create_slider_control("Mix", 0.0, 1.0, "{:.0%}")
        self.output_gain_slider, self.output_gain_label = self._create_slider_control("Output", -24.0, 24.0, "{:+.1f} dB")

        main_layout.addWidget(self.input_gain_label)
        main_layout.addWidget(self.input_gain_slider)
        main_layout.addWidget(self.mix_label)
        main_layout.addWidget(self.mix_slider)
        main_layout.addWidget(self.output_gain_label)
        main_layout.addWidget(self.output_gain_slider)

        self.bypass_checkbox = QCheckBox("Bypass")
        main_layout.addWidget(self.bypass_checkbox)

        self.setContentWidget(self.container_widget)

        self.load_button.clicked.connect(self._handle_load_button)
        self.input_gain_slider.valueChanged.connect(self._on_input_gain_changed)
        self.mix_slider.valueChanged.connect(self._on_mix_changed)
        self.output_gain_slider.valueChanged.connect(self._on_output_gain_changed)
        self.bypass_checkbox.toggled.connect(self.node_logic.set_bypass)

        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        
        self.updateFromLogic()

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
        if range_val == 0: return 0
        norm = (value - min_val) / range_val
        return int(np.clip(norm, 0.0, 1.0) * 1000.0)

    @Slot()
    def _handle_load_button(self):
        parent = self.scene().views()[0] if self.scene() and self.scene().views() else None
        file_path, _ = QFileDialog.getOpenFileName(parent, "Open Impulse Response", "", "Audio Files (*.wav *.flac *.aiff)")
        if file_path:
            self.node_logic.load_file(file_path)

    @Slot(int)
    def _on_input_gain_changed(self, value: int):
        logical_val = self._map_slider_value_to_logical(self.input_gain_slider, value)
        self.node_logic.set_input_gain_db(logical_val)
        self.input_gain_label.setText(f"Input: {self.input_gain_slider.property('format').format(logical_val)}")

    @Slot(int)
    def _on_mix_changed(self, value: int):
        logical_val = self._map_slider_value_to_logical(self.mix_slider, value)
        self.node_logic.set_mix(logical_val)
        self.mix_label.setText(f"Mix: {self.mix_slider.property('format').format(logical_val)}")

    @Slot(int)
    def _on_output_gain_changed(self, value: int):
        logical_val = self._map_slider_value_to_logical(self.output_gain_slider, value)
        self.node_logic.set_output_gain_db(logical_val)
        self.output_gain_label.setText(f"Output: {self.output_gain_slider.property('format').format(logical_val)}")

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        status = state.get('status', 'Unknown')
        self.status_label.setText(f"Status: {status}")
        
        ir_filepath = state.get('ir_filepath')
        if ir_filepath:
            self.filename_label.setText(f"File: {os.path.basename(ir_filepath)}")
            self.filename_label.setToolTip(ir_filepath)
        else:
            self.filename_label.setText("File: None")
            self.filename_label.setToolTip("")

        if "Error" in status: self.status_label.setStyleSheet("color: red;")
        elif "Loading" in status: self.status_label.setStyleSheet("color: orange;")
        else: self.status_label.setStyleSheet("color: lightgreen;")
        
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
        
        with QSignalBlocker(self.bypass_checkbox):
            self.bypass_checkbox.setChecked(state.get('bypass', False))

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ==============================================================================
# 4. Node Logic Class (ConvolutionReverbNode)
# ==============================================================================
@njit(fastmath=True, cache=True)
def perform_convolution_sum(ir_partitions_fft, input_fft_history, current_history_idx):
    num_parts = len(ir_partitions_fft)
    history_len = len(input_fft_history)
    
    # Create the output buffer for the sum
    output_fft_sum = np.zeros_like(input_fft_history[0])

    for i in range(num_parts):
        # This is the tricky part: access the circular buffer correctly.
        # The most recent item is at current_history_idx.
        # The one before that is at (current_history_idx - 1), etc.
        # We use modulo arithmetic to wrap around the buffer.
        history_index = (current_history_idx - i + history_len) % history_len
        
        input_block_fft = input_fft_history[history_index]
        ir_part_fft = ir_partitions_fft[i]
        
        output_fft_sum += input_block_fft * ir_part_fft[:, np.newaxis]
        
    return output_fft_sum

class ConvolutionReverbNode(Node):
    NODE_TYPE = "Convolution Reverb"
    UI_CLASS = ConvolutionReverbNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Applies reverb by convolving audio with an impulse response."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = ConvolutionReverbEmitter()
        self.add_input("in", data_type=np.ndarray)
        self.add_input("input_gain_db", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_input("output_gain_db", data_type=float)
        self.add_output("out", data_type=np.ndarray)
        self._lock = threading.Lock()
        self._ir_filepath: Optional[str] = None
        self._input_gain_db: float = 0.0
        self._output_gain_db: float = 0.0
        self._mix: float = 0.5
        self._bypass: bool = False
        self._status: str = "No IR Loaded"
        self._is_loading: bool = False
        
        # Use a List for IR partitions, as they are read-only
        self._ir_partitions_fft: List[np.ndarray] = []
        
        # circular buffer for input FFT history
        self._input_fft_history: Optional[np.ndarray] = None
        self._history_idx: int = 0 # Pointer for our circular buffer

        self._overlap_add_buffer: Optional[np.ndarray] = None
        self._expected_channels: Optional[int] = None
        self.ir_loader_signaller = IRLoadSignaller()
        self.ir_loader_signaller.load_finished.connect(self._on_ir_load_finished)



    def load_file(self, file_path: str):
        state_snapshot = None
        with self._lock:
            if self._is_loading:
                logger.warning(f"[{self.name}] Load request ignored, already loading.")
                return
            self._is_loading = True
            self._ir_filepath = file_path
            self._status = "Loading..."
            state_snapshot = self.get_current_state_snapshot(locked=True)
        
        self.emitter.stateUpdated.emit(state_snapshot)
        
        runnable = IRLoadRunnable(weakref.ref(self), file_path, DEFAULT_SAMPLERATE, self.ir_loader_signaller)
        QThreadPool.globalInstance().start(runnable)
        
    @Slot(tuple)
    def _on_ir_load_finished(self, result: tuple):
        """
        This slot is connected to the IRLoadSignaller and is executed on the main thread
        when the background IR loading task is complete.
        """
        result_type, data, file_path = result
        
        state_snapshot = None
        with self._lock:
            if file_path != self._ir_filepath:
                logger.warning(f"[{self.name}] Ignoring stale load result for {file_path}")
                return

            self._is_loading = False
            if result_type == 'success':
                # Store IR partitions as a list of arrays
                self._ir_partitions_fft = data
                if self._expected_channels is not None:
                    self._reset_dsp_state_locked(self._expected_channels)
                self._status = "Ready"
            else: # 'failure'
                self._ir_partitions_fft = []
                self._status = f"Error: {data}"
            
            state_snapshot = self.get_current_state_snapshot(locked=True)
        
        self.emitter.stateUpdated.emit(state_snapshot)

    @Slot(float)
    def set_input_gain_db(self, value: float):
        with self._lock: self._input_gain_db = value
    @Slot(float)
    def set_output_gain_db(self, value: float):
        with self._lock: self._output_gain_db = value
    @Slot(float)
    def set_mix(self, value: float):
        with self._lock: self._mix = np.clip(value, 0.0, 1.0)
    @Slot(bool)
    def set_bypass(self, value: bool):
        with self._lock: self._bypass = value

    def get_current_state_snapshot(self, locked: bool = False) -> Dict:
        if locked:
            return {
                "status": self._status, "ir_filepath": self._ir_filepath,
                "input_gain_db": self._input_gain_db, "output_gain_db": self._output_gain_db,
                "mix": self._mix, "bypass": self._bypass,
            }
        with self._lock:
            return {
                "status": self._status, "ir_filepath": self._ir_filepath,
                "input_gain_db": self._input_gain_db, "output_gain_db": self._output_gain_db,
                "mix": self._mix, "bypass": self._bypass,
            }

    def _reset_dsp_state_locked(self, num_channels: int):
        self._expected_channels = num_channels
        num_ir_partitions = len(self._ir_partitions_fft)

        # Pre-allocate the entire history buffer as one big NumPy array
        if num_ir_partitions > 0:
            self._input_fft_history = np.zeros(
                (num_ir_partitions, FFT_SIZE // 2 + 1, num_channels), 
                dtype=np.complex64
            )
        else:
            self._input_fft_history = None
        
        self._history_idx = 0
        self._overlap_add_buffer = np.zeros((PARTITION_SIZE, num_channels), dtype=DEFAULT_DTYPE)
        logger.debug(f"[{self.name}] DSP state reset for {num_channels} channels.")


    def process(self, input_data: dict) -> dict:
        dry_signal = input_data.get("in")
        if dry_signal is None: return {"out": None}

        # --- State Update from Sockets ---
        state_snapshot_to_emit = None
        with self._lock:
            ui_update_needed = False
            
            # Check for socket inputs and update the node's internal state.
            input_gain_db_socket = input_data.get("input_gain_db")
            if input_gain_db_socket is not None:
                try:
                    val = float(input_gain_db_socket)
                    if self._input_gain_db != val:
                        self._input_gain_db = val
                        ui_update_needed = True
                except (ValueError, TypeError): pass

            output_gain_db_socket = input_data.get("output_gain_db")
            if output_gain_db_socket is not None:
                try:
                    val = float(output_gain_db_socket)
                    if self._output_gain_db != val:
                        self._output_gain_db = val
                        ui_update_needed = True
                except (ValueError, TypeError): pass

            mix_socket = input_data.get("mix")
            if mix_socket is not None:
                try:
                    val = np.clip(float(mix_socket), 0.0, 1.0)
                    if self._mix != val:
                        self._mix = val
                        ui_update_needed = True
                except (ValueError, TypeError): pass
            
            # If the state changed, prepare a snapshot to be emitted after releasing the lock.
            if ui_update_needed:
                state_snapshot_to_emit = self.get_current_state_snapshot(locked=True)

            # Get parameters for this processing tick using the most up-to-date state.
            input_gain_db = self._input_gain_db
            output_gain_db = self._output_gain_db
            mix = self._mix
            bypass = self._bypass
            has_ir = bool(self._ir_partitions_fft)

        # Emit the signal after the lock is released to avoid holding it during signal emission.
        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)
            
        # --- Audio Processing ---
        input_gain = 10**(input_gain_db / 20.0)
        output_gain = 10**(output_gain_db / 20.0)

        if bypass or not has_ir:
            return {"out": dry_signal}
            
        dry_signal_gained = dry_signal * input_gain

        if dry_signal_gained.ndim == 1:
            dry_signal_gained = dry_signal_gained[:, np.newaxis]
        
        num_samples, num_channels = dry_signal_gained.shape
        if num_samples != PARTITION_SIZE:
            logger.warning(f"[{self.name}] Input block size ({num_samples}) differs from partition size ({PARTITION_SIZE}). Skipping.")
            return {"out": dry_signal}

        with self._lock:
            if self._expected_channels != num_channels:
                self._reset_dsp_state_locked(num_channels)

            if self._input_fft_history is None or not self._ir_partitions_fft:
                return {"out": dry_signal}

            fft_result = np.fft.rfft(dry_signal_gained, n=FFT_SIZE, axis=0).astype(np.complex64)
            input_fft = np.ascontiguousarray(fft_result)

            # Update the circular buffer
            # Increment index, wrapping around if necessary
            self._history_idx = (self._history_idx + 1) % len(self._input_fft_history)
            # Place the new FFT into the correct slot in our pre-allocated array
            self._input_fft_history[self._history_idx] = input_fft

            # Call the JIT function
            output_fft_sum = perform_convolution_sum(
                self._ir_partitions_fft,
                self._input_fft_history, 
                self._history_idx
            )
            
            convolved_block = np.fft.irfft(output_fft_sum, n=FFT_SIZE, axis=0)
            
            wet_signal = (convolved_block[:PARTITION_SIZE] + self._overlap_add_buffer).astype(DEFAULT_DTYPE)
            self._overlap_add_buffer = convolved_block[PARTITION_SIZE:].astype(DEFAULT_DTYPE)

        wet_signal_gained = wet_signal * output_gain
        final_output = (dry_signal * (1.0 - mix)) + (wet_signal_gained * mix)
        
        return {"out": final_output}

    def start(self):
        super().start()
        with self._lock:
            if self._expected_channels:
                self._reset_dsp_state_locked(self._expected_channels)

    def remove(self):
        logger.info(f"[{self.name}] Remove called.")
        self.ir_loader_signaller.load_finished.disconnect(self._on_ir_load_finished)
        super().remove()

    def serialize_extra(self) -> dict:
        with self._lock:
            return {
                "ir_filepath": self._ir_filepath,
                "input_gain_db": self._input_gain_db,
                "output_gain_db": self._output_gain_db,
                "mix": self._mix,
                "bypass": self._bypass,
            }

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._input_gain_db = data.get("input_gain_db", 0.0)
            self._output_gain_db = data.get("output_gain_db", 0.0)
            self._mix = data.get("mix", 0.5)
            self._bypass = data.get("bypass", False)
        
        filepath = data.get("ir_filepath")
        if filepath and os.path.exists(filepath):
            self.load_file(filepath)
        elif filepath:
            with self._lock:
                self._status = "Error: File not found"