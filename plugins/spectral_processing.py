import torch
import numpy as np
import threading
import logging
from collections import deque

# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING
from constants import (
    DEFAULT_SAMPLERATE,
    DEFAULT_BLOCKSIZE,
    DEFAULT_DTYPE,
    DEFAULT_COMPLEX_DTYPE,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_FFT_SIZE,
    SpectralFrame,
)

# --- Qt Imports ---
from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout, QHBoxLayout, QSizePolicy, QComboBox
from PySide6.QtCore import Qt, Slot, QSignalBlocker, Signal, QObject

# --- Logging ---
logger = logging.getLogger(__name__)


# ==============================================================================
# 2. STFT Node (Time Domain -> Spectral Domain)
# ==============================================================================
class STFTNodeItem(NodeItem):

    def __init__(self, node_logic: "STFTNode"):
        super().__init__(node_logic)

        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(5)

        layout.addWidget(QLabel("Window Size (Overlap):"))
        self.window_size_combo = QComboBox()
        self.window_sizes = [512, 1024, 2048, 4096]  # Must be >= block_size
        self.window_size_combo.addItems([f"{s} ({100*(1-DEFAULT_BLOCKSIZE/s):.0f}%)" for s in self.window_sizes])
        layout.addWidget(self.window_size_combo)

        self.setContentWidget(self.container_widget)
        self.window_size_combo.activated.connect(self._on_window_size_change)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

    @Slot()
    def updateFromLogic(self):
        state = {"window_size": self.node_logic.get_window_size()}
        self._on_state_updated(state)
        super().updateFromLogic()

    @Slot(int)
    def _on_window_size_change(self, index: int):
        new_size = self.window_sizes[index]
        self.node_logic.set_window_size(new_size)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        window_size = state.get("window_size", DEFAULT_WINDOW_SIZE)
        try:
            index = self.window_sizes.index(window_size)
            with QSignalBlocker(self.window_size_combo):
                self.window_size_combo.setCurrentIndex(index)
        except ValueError:
            closest_size = min(self.window_sizes, key=lambda x: abs(x - window_size))
            index = self.window_sizes.index(closest_size)
            with QSignalBlocker(self.window_size_combo):
                self.window_size_combo.setCurrentIndex(index)


class STFTNode(Node):
    NODE_TYPE = "STFT"
    UI_CLASS = STFTNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "Converts audio into a stream of spectral frames. Hop size is fixed to block size."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self.add_input("audio_in", data_type=torch.Tensor)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)
        self._lock = threading.Lock()
        self._hop_size = DEFAULT_BLOCKSIZE
        self._window_size = DEFAULT_WINDOW_SIZE
        self._fft_size = DEFAULT_FFT_SIZE
        self._sample_rate = DEFAULT_SAMPLERATE
        self._analysis_window = None
        self._buffer = torch.tensor([], dtype=DEFAULT_DTYPE)
        self._expected_channels = None
        self._recalculate_params()

    def _recalculate_params(self):
        with self._lock:
            self._window_size = max(self._hop_size, int(self._window_size))
            self._fft_size = int(2 ** np.ceil(np.log2(self._window_size)))
            self._analysis_window = torch.hann_window(self._window_size, dtype=DEFAULT_DTYPE)
            logger.info(
                f"[{self.name}] Recalculated STFT params: Win={self._window_size}, Hop={self._hop_size} (Fixed), FFT={self._fft_size}"
            )

    @Slot(int)
    def set_window_size(self, value: int):
        state_to_emit = None
        with self._lock:
            if self._window_size != value:
                self._window_size = value
                state_to_emit = {"window_size": self._window_size}

        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

        self._recalculate_params()
        self.start()

    def get_window_size(self):
        with self._lock:
            return self._window_size

    def start(self):
        with self._lock:
            self._buffer = torch.tensor([], dtype=DEFAULT_DTYPE)
            self._expected_channels = None

    def process(self, input_data: dict) -> dict:
        audio_chunk = input_data.get("audio_in")
        if audio_chunk is None:
            return {"spectral_frame_out": None}
        proc_chunk = torch.atleast_2d(audio_chunk.to(DEFAULT_DTYPE))
        num_channels, _ = proc_chunk.shape
        with self._lock:
            if self._expected_channels is None or self._expected_channels != num_channels:
                logger.info(
                    f"[{self.name}] Channel count changed from {self._expected_channels} to {num_channels}. Resetting buffer."
                )
                self._expected_channels = num_channels
                self._buffer = torch.zeros((num_channels, 0), dtype=DEFAULT_DTYPE)
            self._buffer = torch.cat((self._buffer, proc_chunk), dim=1)
            if self._buffer.shape[1] >= self._window_size:
                frame_data = self._buffer[:, : self._window_size]
                windowed_frame = frame_data * self._analysis_window
                fft_data = torch.fft.rfft(windowed_frame, n=self._fft_size, dim=1).to(DEFAULT_COMPLEX_DTYPE)
                output_frame = SpectralFrame(
                    data=fft_data,
                    fft_size=self._fft_size,
                    hop_size=self._hop_size,
                    window_size=self._window_size,
                    sample_rate=self._sample_rate,
                    analysis_window=self._analysis_window,
                )
                self._buffer = self._buffer[:, self._hop_size :]
                return {"spectral_frame_out": output_frame}
        return {"spectral_frame_out": None}

    def serialize_extra(self) -> dict:
        """Saves the current window size."""
        with self._lock:
            return {"window_size": self._window_size}

    def deserialize_extra(self, data: dict):
        """Loads and applies the window size from saved data."""
        # Use the public setter to ensure all internal params are recalculated and the UI is updated.
        self.set_window_size(data.get("window_size", DEFAULT_WINDOW_SIZE))


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
        self.add_output("audio_out", data_type=torch.Tensor)
        self._lock = threading.Lock()
        self._ola_buffer = None
        self._expected_channels = None
        self._synthesis_window = None
        self._last_params = (0, 0, 0)

    def _recalculate_synthesis_params(self, frame: SpectralFrame):
        win, hop, win_size = frame.analysis_window, frame.hop_size, frame.window_size
        sum_of_squares = torch.zeros(win_size, dtype=DEFAULT_DTYPE)
        for i in range(0, win_size, hop):
            sum_of_squares += torch.roll(win**2, i)
        sum_of_squares[sum_of_squares < 1e-9] = 1.0
        self._synthesis_window = win / sum_of_squares
        logger.info(f"[{self.name}] Recalculated synthesis window for Win={win_size}, Hop={hop}.")

    def _initialize_buffers(self, frame: SpectralFrame, num_channels: int):
        self._recalculate_synthesis_params(frame)
        self._ola_buffer = torch.zeros((num_channels, frame.window_size), dtype=DEFAULT_DTYPE)
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
            return {"audio_out": torch.zeros((channels, DEFAULT_BLOCKSIZE), dtype=DEFAULT_DTYPE)}
        num_channels = frame.data.shape[0]
        with self._lock:
            if (
                self._ola_buffer is None
                or self._expected_channels != num_channels
                or self._last_params != (frame.window_size, frame.hop_size, frame.fft_size)
            ):
                self._initialize_buffers(frame, num_channels)
            if self._ola_buffer is None:
                return {"audio_out": torch.zeros((num_channels, DEFAULT_BLOCKSIZE), dtype=DEFAULT_DTYPE)}
            ifft_frame_full = torch.fft.irfft(frame.data, n=frame.fft_size, dim=1).to(DEFAULT_DTYPE)
            ifft_frame = ifft_frame_full[:, : frame.window_size]
            windowed_ifft = ifft_frame * self._synthesis_window
            self._ola_buffer[:, : frame.window_size] += windowed_ifft
            output_block = self._ola_buffer[:, : frame.hop_size].clone()
            self._ola_buffer = torch.roll(self._ola_buffer, -frame.hop_size, dims=1)
            self._ola_buffer[:, -frame.hop_size :] = 0.0
            return {"audio_out": output_block}


# ==============================================================================
# 4. Spectral Filter Node
# ==============================================================================
class SpectralFilterNodeItem(NodeItem):

    def __init__(self, node_logic: "SpectralFilterNode"):
        super().__init__(node_logic)

        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
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
        self.fc1_slider.valueChanged.connect(lambda v: self._on_slider_change(v, "fc1"))
        self.fc2_slider.valueChanged.connect(lambda v: self._on_slider_change(v, "fc2"))

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
        return 10 ** (min_freq_log + norm * (max_freq_log - min_freq_log))

    def _map_freq_to_slider(self, freq):
        min_freq_log = np.log10(20)
        max_freq_log = np.log10(20000)
        safe_freq = max(20.0, freq)
        norm = (np.log10(safe_freq) - min_freq_log) / (max_freq_log - min_freq_log)
        return int(np.clip(norm, 0, 1) * 1000)

    def _on_slider_change(self, value, key):
        freq = self._map_slider_to_freq(value)
        if key == "fc1":
            self.node_logic.set_cutoff_freq_1(freq)
        else:
            self.node_logic.set_cutoff_freq_2(freq)

    @Slot(dict)
    def _on_state_updated(self, state):
        filter_type = state.get("filter_type")
        fc1 = state.get("fc1")
        fc2 = state.get("fc2")

        with QSignalBlocker(self.type_combo):
            self.type_combo.setCurrentText(filter_type)

        is_fc1_ext = "cutoff_freq_1" in self.node_logic.inputs and self.node_logic.inputs["cutoff_freq_1"].connections
        is_fc2_ext = "cutoff_freq_2" in self.node_logic.inputs and self.node_logic.inputs["cutoff_freq_2"].connections

        self.fc1_slider.setEnabled(not is_fc1_ext)
        self.fc2_slider.setEnabled(not is_fc2_ext)

        with QSignalBlocker(self.fc1_slider):
            self.fc1_slider.setValue(self._map_freq_to_slider(fc1))

        with QSignalBlocker(self.fc2_slider):
            self.fc2_slider.setValue(self._map_freq_to_slider(fc2))

        self.fc2_widget.setVisible(filter_type == "Band Pass")

        fc1_label_base = "Cutoff Freq 1" if filter_type == "Band Pass" else "Cutoff Freq"
        fc1_label_text = f"{fc1_label_base}: {fc1:.0f} Hz"
        if is_fc1_ext:
            fc1_label_text += " (ext)"
        self.fc1_label.setText(fc1_label_text)

        fc2_label_text = f"Cutoff Freq 2: {fc2:.0f} Hz"
        if is_fc2_ext:
            fc2_label_text += " (ext)"
        self.fc2_label.setText(fc2_label_text)
        self.container_widget.adjustSize()
        self.update_geometry()

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

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("cutoff_freq_1", data_type=float)
        self.add_input("cutoff_freq_2", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)
        self.emitter = NodeStateEmitter()
        self._lock = threading.Lock()

        self._filter_type = "Low Pass"
        self._cutoff_freq_1 = 1000.0
        self._cutoff_freq_2 = 4000.0

    def _get_state_locked(self):
        return {"filter_type": self._filter_type, "fc1": self._cutoff_freq_1, "fc2": self._cutoff_freq_2}

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
        state_to_emit = None
        with self._lock:
            if self._cutoff_freq_1 != freq:
                self._cutoff_freq_1 = freq
                state_to_emit = self._get_state_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_cutoff_freq_2(self, freq: float):
        state_to_emit = None
        with self._lock:
            if self._cutoff_freq_2 != freq:
                self._cutoff_freq_2 = freq
                state_to_emit = self._get_state_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        state_snapshot_to_emit = None
        with self._lock:
            ui_update_needed = False

            fc1_socket = input_data.get("cutoff_freq_1")
            if fc1_socket is not None:
                new_val = float(fc1_socket)
                if self._cutoff_freq_1 != new_val:
                    self._cutoff_freq_1 = new_val
                    ui_update_needed = True

            fc2_socket = input_data.get("cutoff_freq_2")
            if fc2_socket is not None:
                new_val = float(fc2_socket)
                if self._cutoff_freq_2 != new_val:
                    self._cutoff_freq_2 = new_val
                    ui_update_needed = True

            if ui_update_needed:
                state_snapshot_to_emit = self._get_state_locked()

            filter_type = self._filter_type
            fc1 = self._cutoff_freq_1
            fc2 = self._cutoff_freq_2

        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

        modified_data = frame.data.clone()
        freq_per_bin = frame.sample_rate / frame.fft_size

        bin1 = int(round(fc1 / freq_per_bin))
        bin2 = int(round(fc2 / freq_per_bin))

        num_bins = modified_data.shape[1]
        bin1 = np.clip(bin1, 0, num_bins)
        bin2 = np.clip(bin2, 0, num_bins)

        if filter_type == "Low Pass":
            modified_data[:, bin1:] = 0.0
        elif filter_type == "High Pass":
            modified_data[:, :bin1] = 0.0
        elif filter_type == "Band Pass":
            low, high = min(bin1, bin2), max(bin1, bin2)
            modified_data[:, :low] = 0.0
            modified_data[:, high:] = 0.0

        output_frame = SpectralFrame(
            data=modified_data,
            fft_size=frame.fft_size,
            hop_size=frame.hop_size,
            window_size=frame.window_size,
            sample_rate=frame.sample_rate,
            analysis_window=frame.analysis_window,
        )
        return {"spectral_frame_out": output_frame}

    def serialize_extra(self) -> dict:
        return self.get_state()

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._filter_type = data.get("filter_type", "Low Pass")
            self._cutoff_freq_1 = data.get("fc1", 1000.0)
            self._cutoff_freq_2 = data.get("fc2", 4000.0)
