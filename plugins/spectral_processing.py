import torch
import numpy as np
import threading
import logging
from collections import deque

# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING, ParameterNodeItem
from node_helpers import managed_parameters, Parameter
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
class STFTNodeItem(ParameterNodeItem):
    def __init__(self, node_logic: "STFTNode"):
        self.window_sizes = [512, 1024, 2048, 4096]
        parameters = [
            {
                "key": "window_size",
                "name": "Window Size (Overlap)",
                "type": "combobox",
                "items": [(f"{s} ({100*(1-DEFAULT_BLOCKSIZE/s):.0f}%)", s) for s in self.window_sizes],
            }
        ]
        super().__init__(node_logic, parameters)


@managed_parameters
class STFTNode(Node):
    NODE_TYPE = "STFT"
    UI_CLASS = STFTNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "Converts audio into a stream of spectral frames. Hop size is fixed to block size."

    window_size = Parameter(default=DEFAULT_WINDOW_SIZE, on_change="_on_window_size_changed")

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("audio_in", data_type=torch.Tensor)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)
        self._hop_size = DEFAULT_BLOCKSIZE
        self._fft_size = DEFAULT_FFT_SIZE
        self._sample_rate = DEFAULT_SAMPLERATE
        self._analysis_window = None
        self._buffer = torch.tensor([], dtype=DEFAULT_DTYPE)
        self._expected_channels = None
        self._recalculate_params()

    def _recalculate_params(self):
        # This method is called from within a locked context, so it's safe.
        win_size = max(self._hop_size, int(self._window_size))
        self._fft_size = int(2 ** np.ceil(np.log2(win_size)))
        self._analysis_window = torch.hann_window(win_size, dtype=DEFAULT_DTYPE)
        logger.info(
            f"[{self.name}] Recalculated STFT params: Win={win_size}, Hop={self._hop_size} (Fixed), FFT={self._fft_size}"
        )

    def _reset_buffer_state_locked(self):
        """
        Resets the internal buffer state. ASSUMES THE CALLING THREAD HOLDS THE LOCK.
        This prevents a re-entrant lock deadlock.
        """
        self._buffer = torch.tensor([], dtype=DEFAULT_DTYPE)
        self._expected_channels = None

    def _on_window_size_changed(self):
        """
        Callback triggered by the decorator when window_size changes.
        This method is executed WHILE THE LOCK IS HELD.
        """
        self._recalculate_params()
        # Call the non-locking version of the reset logic.
        self._reset_buffer_state_locked()

    def start(self):
        """
        Called by the Engine when graph processing starts.
        It acquires the lock once and is safe.
        """
        with self._lock:
            self._reset_buffer_state_locked()

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
                # Since we're already in a lock, we can safely call this.
                self._reset_buffer_state_locked()
                self._expected_channels = num_channels

            self._buffer = torch.cat((self._buffer, proc_chunk), dim=1)
            current_window_size = self._window_size
            if self._buffer.shape[1] >= current_window_size:
                frame_data = self._buffer[:, :current_window_size]
                windowed_frame = frame_data * self._analysis_window
                fft_data = torch.fft.rfft(windowed_frame, n=self._fft_size, dim=1).to(DEFAULT_COMPLEX_DTYPE)
                output_frame = SpectralFrame(
                    data=fft_data,
                    fft_size=self._fft_size,
                    hop_size=self._hop_size,
                    window_size=current_window_size,
                    sample_rate=self._sample_rate,
                    analysis_window=self._analysis_window,
                )
                self._buffer = self._buffer[:, self._hop_size :]
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
        self.add_output("audio_out", data_type=torch.Tensor)
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
class SpectralFilterNodeItem(ParameterNodeItem):
    def __init__(self, node_logic: "SpectralFilterNode"):
        parameters = [
            {
                "key": "filter_type",
                "name": "Filter Type",
                "type": "combobox",
                "items": [("Low Pass", "Low Pass"), ("High Pass", "High Pass"), ("Band Pass", "Band Pass")],
            },
            {
                "key": "cutoff_freq_1",
                "name": "Cutoff Freq",
                "type": "dial",
                "min": 20.0,
                "max": 20000.0,
                "format": "{:.0f} Hz",
                "is_log": True,
            },
            {
                "key": "cutoff_freq_2",
                "name": "Cutoff Freq 2",
                "type": "dial",
                "min": 20.0,
                "max": 20000.0,
                "format": "{:.0f} Hz",
                "is_log": True,
            },
        ]
        super().__init__(node_logic, parameters)

        self._on_state_updated_from_logic(self.node_logic.get_current_state_snapshot())

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        # Call the parent first to handle all standard UI updates
        super()._on_state_updated_from_logic(state)

        # Custom logic for visibility based on the received state
        filter_type = state.get("filter_type", "Low Pass")
        is_bandpass = filter_type == "Band Pass"

        # Show/hide the second frequency control
        self._controls["cutoff_freq_2"]["widget"].setVisible(is_bandpass)
        self._controls["cutoff_freq_2"]["label"].setVisible(is_bandpass)

        # Update the label of the first frequency control
        label_widget = self._controls["cutoff_freq_1"]["label"]
        current_text = label_widget.text()
        base_name = "Cutoff Freq 1" if is_bandpass else "Cutoff Freq"

        # Update the base name in the control dictionary to be used by the superclass method
        self._controls["cutoff_freq_1"]["name"] = base_name

        # Manually trigger a label text update
        value = state.get("cutoff_freq_1", 0.0)
        label_text = f"{base_name}: {self._controls['cutoff_freq_1']['format'].format(value)}"
        is_connected = "cutoff_freq_1" in self.node_logic.inputs and self.node_logic.inputs["cutoff_freq_1"].connections
        if is_connected:
            label_text += " (ext)"
        label_widget.setText(label_text)

        # Request a geometry update for the node
        self.container_widget.adjustSize()
        self.update_geometry()


@managed_parameters
class SpectralFilterNode(Node):
    NODE_TYPE = "Spectral Filter"
    CATEGORY = "Spectral"
    DESCRIPTION = "Applies a brick-wall filter to a spectral frame."
    UI_CLASS = SpectralFilterNodeItem

    filter_type = Parameter(default="Low Pass")
    cutoff_freq_1 = Parameter(default=1000.0, clip=(20.0, 20000.0))
    cutoff_freq_2 = Parameter(default=4000.0, clip=(20.0, 20000.0))

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("cutoff_freq_1", data_type=float)
        self.add_input("cutoff_freq_2", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        self._update_params_from_sockets(input_data)

        with self._lock:
            filter_type = self._filter_type
            fc1 = self._cutoff_freq_1
            fc2 = self._cutoff_freq_2

        modified_data = frame.data.clone()
        freq_per_bin = frame.sample_rate / frame.fft_size

        bin1 = int(round(fc1 / freq_per_bin))
        bin2 = int(round(fc2 / freq_per_bin))

        num_bins = modified_data.shape[1]
        bin1 = np.clip(bin1, 0, num_bins - 1)
        bin2 = np.clip(bin2, 0, num_bins - 1)

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
