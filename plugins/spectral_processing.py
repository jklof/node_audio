import numpy as np
import threading
import logging
from collections import deque

# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from constants import (
    DEFAULT_SAMPLERATE,
    DEFAULT_BLOCKSIZE, 
    DEFAULT_DTYPE,
    DEFAULT_COMPLEX_DTYPE,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_FFT_SIZE,
    SpectralFrame  # <-- IMPORT THE CENTRAL DEFINITION
)

# --- Qt Imports ---
from PySide6.QtWidgets import (
    QWidget, QLabel, QSlider, QVBoxLayout, QHBoxLayout, QSizePolicy, QComboBox
)
from PySide6.QtCore import Qt, Slot, QSignalBlocker, Signal, QObject

# --- Logging ---
logger = logging.getLogger(__name__)

# NOTE: The SpectralFrame dataclass definition has been REMOVED from this file.

# ==============================================================================
# 2. STFT Node (Time Domain -> Spectral Domain) - UI and Logic REVISED
# ==============================================================================

class STFTNodeItem(NodeItem):
    """--- REVISED --- UI for the STFTNode with simplified controls."""

    def __init__(self, node_logic: "STFTNode"):
        super().__init__(node_logic)

        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING)
        layout.setSpacing(5)

        layout.addWidget(QLabel("Window Size (Overlap):"))
        self.window_size_combo = QComboBox()
        self.window_sizes = [512, 1024, 2048, 4096] # Must be >= block_size
        self.window_size_combo.addItems([f"{s} ({100*(1-DEFAULT_BLOCKSIZE/s):.0f}%)" for s in self.window_sizes])
        layout.addWidget(self.window_size_combo)
        
        self.setContentWidget(self.container_widget)
        self.window_size_combo.activated.connect(self._on_window_size_change)
        self.updateFromLogic()

    @Slot(int)
    def _on_window_size_change(self, index: int):
        new_size = self.window_sizes[index]
        self.node_logic.set_window_size(new_size)

    @Slot()
    def updateFromLogic(self):
        with QSignalBlocker(self.window_size_combo):
            win_size = self.node_logic.get_window_size()
            try:
                index = self.window_sizes.index(win_size)
                self.window_size_combo.setCurrentIndex(index)
            except ValueError:
                closest_size = min(self.window_sizes, key=lambda x:abs(x-win_size))
                index = self.window_sizes.index(closest_size)
                self.window_size_combo.setCurrentIndex(index)

        super().updateFromLogic()

class STFTNode(Node):
    NODE_TYPE = "STFT"
    UI_CLASS = STFTNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "Converts audio into a stream of spectral frames. Hop size is fixed to block size."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("audio_in", data_type=np.ndarray)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        self._lock = threading.Lock()
        
        self._hop_size = DEFAULT_BLOCKSIZE
        self._window_size = DEFAULT_WINDOW_SIZE
        self._fft_size = DEFAULT_FFT_SIZE
        self._sample_rate = DEFAULT_SAMPLERATE
        self._analysis_window = None
        self._buffer = np.array([], dtype=DEFAULT_DTYPE)
        self._expected_channels = None
        
        self._recalculate_params()

    def _recalculate_params(self):
        with self._lock:
            self._window_size = max(self._hop_size, int(self._window_size))
            self._fft_size = int(2**np.ceil(np.log2(self._window_size)))
            self._analysis_window = np.hanning(self._window_size).astype(DEFAULT_DTYPE)
            logger.info(f"[{self.name}] Recalculated STFT params: Win={self._window_size}, Hop={self._hop_size} (Fixed), FFT={self._fft_size}")

    @Slot(int)
    def set_window_size(self, value: int):
        with self._lock:
            self._window_size = value
        self._recalculate_params()
        self.start()

    def get_window_size(self):
        with self._lock: return self._window_size
        
    def start(self):
        with self._lock:
            self._buffer = np.array([], dtype=DEFAULT_DTYPE)
            self._expected_channels = None

    def process(self, input_data: dict) -> dict:
        audio_chunk = input_data.get("audio_in")
        
        if audio_chunk is None:
            return {"spectral_frame_out": None}

        proc_chunk = np.atleast_2d(audio_chunk.astype(DEFAULT_DTYPE))
        if proc_chunk.shape[0] < proc_chunk.shape[1]: proc_chunk = proc_chunk.T
        
        _, num_channels = proc_chunk.shape
        
        with self._lock:
            # If channel count is uninitialized or has changed, reset state here.
            if self._expected_channels is None or self._expected_channels != num_channels:
                logger.info(f"[{self.name}] Channel count changed from {self._expected_channels} to {num_channels}. Resetting buffer.")
                self._expected_channels = num_channels
                # Correctly initialize the buffer as a 2D array with the new channel count.
                self._buffer = np.zeros((0, num_channels), dtype=DEFAULT_DTYPE)

            self._buffer = np.vstack((self._buffer, proc_chunk))
            
            if len(self._buffer) >= self._window_size:
                frame_data = self._buffer[:self._window_size]
                windowed_frame = frame_data * self._analysis_window[:, np.newaxis]
                fft_data = np.fft.rfft(windowed_frame, n=self._fft_size, axis=0).astype(DEFAULT_COMPLEX_DTYPE)
                
                output_frame = SpectralFrame(
                    data=fft_data, fft_size=self._fft_size, hop_size=self._hop_size,
                    window_size=self._window_size, sample_rate=self._sample_rate,
                    analysis_window=self._analysis_window
                )
                
                self._buffer = self._buffer[self._hop_size:]
                return {"spectral_frame_out": output_frame}

        return {"spectral_frame_out": None}

# ==============================================================================
# 3. ISTFT Node (Spectral Domain -> Time Domain)
# ==============================================================================

class ISTFTNode(Node):
    NODE_TYPE = "ISTFT"
    CATEGORY = "Spectral"
    DESCRIPTION = "Reconstructs audio from spectral frames using overlap-add."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_output("audio_out", data_type=np.ndarray)

        self._lock = threading.Lock()
        self._ola_buffer = None
        self._expected_channels = None
        self._synthesis_window = None
        self._last_params = (0, 0, 0)
        
    def _recalculate_synthesis_params(self, frame: SpectralFrame):
        win, hop, win_size = frame.analysis_window, frame.hop_size, frame.window_size
        sum_of_squares = np.zeros(win_size, dtype=DEFAULT_DTYPE)
        for i in range(0, win_size, hop):
            sum_of_squares += np.roll(win**2, i)
        sum_of_squares[sum_of_squares < 1e-9] = 1.0
        self._synthesis_window = (win / sum_of_squares)[:, np.newaxis]
        logger.info(f"[{self.name}] Recalculated synthesis window for Win={win_size}, Hop={hop}.")

    def _initialize_buffers(self, frame: SpectralFrame, num_channels: int):
        self._recalculate_synthesis_params(frame)
        self._ola_buffer = np.zeros((frame.window_size, num_channels), dtype=DEFAULT_DTYPE)
        self._expected_channels = num_channels
        self._last_params = (frame.window_size, frame.hop_size, frame.fft_size)
        logger.info(f"[{self.name}] Initialized ISTFT for {num_channels} channels.")

    def start(self):
        with self._lock:
            self._ola_buffer = None
            self._expected_channels = None
            self._synthesis_window = None

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        
        if not isinstance(frame, SpectralFrame):
            channels = self._expected_channels if self._expected_channels is not None else 1
            return {"audio_out": np.zeros((DEFAULT_BLOCKSIZE, channels), dtype=DEFAULT_DTYPE)}

        num_channels = frame.data.shape[1]
        
        with self._lock:
            if self._ola_buffer is None or self._expected_channels != num_channels or self._last_params != (frame.window_size, frame.hop_size, frame.fft_size):
                self._initialize_buffers(frame, num_channels)
            
            if self._ola_buffer is None:
                return {"audio_out": np.zeros((DEFAULT_BLOCKSIZE, num_channels), dtype=DEFAULT_DTYPE)}

            ifft_frame_full = np.fft.irfft(frame.data, n=frame.fft_size, axis=0).astype(DEFAULT_DTYPE)
            ifft_frame = ifft_frame_full[:frame.window_size]
            
            windowed_ifft = ifft_frame * self._synthesis_window
            
            self._ola_buffer[:frame.window_size] += windowed_ifft
            
            output_block = self._ola_buffer[:frame.hop_size].copy()
            
            self._ola_buffer = np.roll(self._ola_buffer, -frame.hop_size, axis=0)
            self._ola_buffer[-frame.hop_size:, :] = 0.0

            return {"audio_out": output_block}

# ==============================================================================
# 4. Spectral Filter Node
# ==============================================================================
class SpectralFilterNodeItem(NodeItem):

    def __init__(self, node_logic: "SpectralFilterNode"):
        super().__init__(node_logic)

        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING)
        layout.setSpacing(5)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["Low Pass", "High Pass", "Band Pass"])
        layout.addWidget(self.type_combo)

        self.fc1_slider, self.fc1_label = self._create_slider("Cutoff Freq")
        layout.addWidget(self.fc1_label)
        layout.addWidget(self.fc1_slider)
        
        self.fc2_widget = QWidget()
        fc2_layout = QVBoxLayout(self.fc2_widget)
        fc2_layout.setContentsMargins(0, 0, 0, 0)
        self.fc2_slider, self.fc2_label = self._create_slider("Cutoff Freq 2")
        fc2_layout.addWidget(self.fc2_label)
        fc2_layout.addWidget(self.fc2_slider)
        layout.addWidget(self.fc2_widget)

        self.setContentWidget(self.container_widget)

        self.type_combo.currentTextChanged.connect(self.node_logic.set_filter_type)
        self.fc1_slider.valueChanged.connect(lambda v: self._on_slider_change(v, 'fc1'))
        self.fc2_slider.valueChanged.connect(lambda v: self._on_slider_change(v, 'fc2'))
        
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        self.updateFromLogic()

    def _create_slider(self, name):
        label = QLabel(f"{name}: ...")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)
        return slider, label

    def _map_slider_to_freq(self, value):
        min_freq_log = np.log10(20)
        max_freq_log = np.log10(20000)
        norm = value / 1000.0
        return 10**(min_freq_log + norm * (max_freq_log - min_freq_log))
        
    def _map_freq_to_slider(self, freq):
        min_freq_log = np.log10(20)
        max_freq_log = np.log10(20000)
        safe_freq = max(20.0, freq)
        norm = (np.log10(safe_freq) - min_freq_log) / (max_freq_log - min_freq_log)
        return int(np.clip(norm, 0, 1) * 1000)

    def _on_slider_change(self, value, key):
        freq = self._map_slider_to_freq(value)
        label_text_parts = (self.fc1_label if key == 'fc1' else self.fc2_label).text().split(':')
        name_part = label_text_parts[0]
        
        if key == 'fc1':
            self.node_logic.set_cutoff_freq_1(freq)
            self.fc1_label.setText(f"{name_part}: {freq:.0f} Hz")
        else:
            self.node_logic.set_cutoff_freq_2(freq)
            self.fc2_label.setText(f"Cutoff Freq 2: {freq:.0f} Hz")
    
    @Slot(dict)
    def _on_state_updated(self, state):
        filter_type = state.get('type')
        fc1 = state.get('fc1')
        fc2 = state.get('fc2')
        
        with QSignalBlocker(self.type_combo):
            self.type_combo.setCurrentText(filter_type)
        
        with QSignalBlocker(self.fc1_slider):
            self.fc1_slider.setValue(self._map_freq_to_slider(fc1))
            
        with QSignalBlocker(self.fc2_slider):
            self.fc2_slider.setValue(self._map_freq_to_slider(fc2))
        
        self.fc2_widget.setVisible(filter_type == "Band Pass")
        self.fc1_label.setText(f"{'Cutoff Freq 1' if filter_type == 'Band Pass' else 'Cutoff Freq'}: {fc1:.0f} Hz")
        self.fc2_label.setText(f"Cutoff Freq 2: {fc2:.0f} Hz")

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_state()
        self._on_state_updated(state)
        super().updateFromLogic()

class SpectralFilterNode(Node):
    NODE_TYPE = "Spectral Filter"
    CATEGORY = "Spectral"
    DESCRIPTION = "Applies a brick-wall filter to spectral frames."
    UI_CLASS = SpectralFilterNodeItem

    class Emitter(QObject):
        stateUpdated = Signal(dict)
        
    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)
        self.emitter = self.Emitter()
        self._lock = threading.Lock()
        
        self._filter_type = "Low Pass"
        self._cutoff_freq_1 = 1000.0
        self._cutoff_freq_2 = 4000.0
    
    def _get_state_locked(self):
        return {"type": self._filter_type, "fc1": self._cutoff_freq_1, "fc2": self._cutoff_freq_2}

    def get_state(self):
        with self._lock:
            return self._get_state_locked()

    @Slot(str)
    def set_filter_type(self, f_type: str):
        state_to_emit = None
        with self._lock:
            if self._filter_type != f_type:
                self._filter_type = f_type
                state_to_emit = self._get_state_locked()
        
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_cutoff_freq_1(self, freq: float):
        with self._lock: self._cutoff_freq_1 = freq
    @Slot(float)
    def set_cutoff_freq_2(self, freq: float):
        with self._lock: self._cutoff_freq_2 = freq

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        with self._lock:
            filter_type = self._filter_type
            fc1 = self._cutoff_freq_1
            fc2 = self._cutoff_freq_2

        modified_data = frame.data.copy()
        freq_per_bin = frame.sample_rate / frame.fft_size
        
        bin1 = int(round(fc1 / freq_per_bin))
        bin2 = int(round(fc2 / freq_per_bin))
        
        num_bins = modified_data.shape[0]
        bin1 = np.clip(bin1, 0, num_bins)
        bin2 = np.clip(bin2, 0, num_bins)

        if filter_type == "Low Pass":
            modified_data[bin1:, :] = 0.0
        elif filter_type == "High Pass":
            modified_data[:bin1, :] = 0.0
        elif filter_type == "Band Pass":
            low, high = min(bin1, bin2), max(bin1, bin2)
            modified_data[:low, :] = 0.0
            modified_data[high:, :] = 0.0
            
        output_frame = SpectralFrame(
            data=modified_data,
            fft_size=frame.fft_size, hop_size=frame.hop_size,
            window_size=frame.window_size, sample_rate=frame.sample_rate,
            analysis_window=frame.analysis_window
        )
        return {"spectral_frame_out": output_frame}