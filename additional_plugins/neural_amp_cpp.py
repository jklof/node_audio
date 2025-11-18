import torch
import ctypes
import logging
import os
import json
from typing import Dict, Optional

from ffi_node import FFINodeBase
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog
from PySide6.QtCore import Slot

# --- Import the new ResamplingStream wrapper ---
try:
    from resampler import ResamplingStream
except ImportError:
    logging.error("Could not import ResamplingStream from resampler.py")

logger = logging.getLogger(__name__)


# --- UI Class  ---
class NAMCppNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, n):
        super().__init__(n, width=self.NODE_SPECIFIC_WIDTH)
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(5, 5, 5, 5)
        l.setSpacing(5)
        self.b = QPushButton("Load .nam Model...")
        self.s = QLabel("No model loaded.")
        self.s.setWordWrap(True)
        self.s.setStyleSheet("color: lightgray;")
        l.addWidget(self.b)
        l.addWidget(self.s)
        self.setContentWidget(w)
        self.b.clicked.connect(self._on_load)

    @Slot()
    def _on_load(self):
        p = self.scene().views()[0] if self.scene() and self.scene().views() else None
        fp, _ = QFileDialog.getOpenFileName(p, "Open NAM Model", "", "NAM Files (*.nam)")
        if fp:
            self.node_logic.load_model(fp)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        super()._on_state_updated_from_logic(state)
        fp, err = state.get("file_path"), state.get("error_message")
        if err:
            self.s.setText(f"Error: {err}")
            self.s.setStyleSheet("color: red;")
            self.s.setToolTip(err)
        elif fp:
            self.s.setText(f"Loaded: {os.path.basename(fp)}")
            self.s.setStyleSheet("color: lightgreen;")
            self.s.setToolTip(fp)
        else:
            self.s.setText("No model loaded.")
            self.s.setStyleSheet("color: lightgray;")
            self.s.setToolTip("")


class NAMCppNode(FFINodeBase):
    NODE_TYPE = "Neural Amp Modeler++ (faster)"
    UI_CLASS = NAMCppNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Processes audio using a Neural Amp Model (C++). Multi-channel audio is mixed to mono."

    LIB_NAME = "neural_amp_processor"
    API = {
        "set_parameters": {"argtypes": [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_int]},
        "process_block": {
            "argtypes": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        },
    }

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_output("out", data_type=torch.Tensor)
        self._file_path: Optional[str] = None
        self._error_message: Optional[str] = None
        self._model_samplerate: int = DEFAULT_SAMPLERATE

        # --- Updated: Use ResamplingStream for clean continuity ---
        self._resampler_stream_in: Optional[ResamplingStream] = None
        self._resampler_stream_out: Optional[ResamplingStream] = None

        if self._file_path and self.dsp_handle:
            self.load_model(self._file_path)

    @Slot(str)
    def load_model(self, file_path: str):
        state_to_emit = None
        if not os.path.exists(file_path):
            error_msg = f"NAM file not found: {file_path}"
            self._error_message, self._file_path, self.error_state = error_msg, None, error_msg
            self.ui_update_callback(self._get_state_snapshot_locked())
            return
        with self._lock:
            # Reset streams
            self._resampler_stream_in = None
            self._resampler_stream_out = None

            try:
                with open(file_path, "r") as f:
                    config = json.load(f)
                model_config = config.get("config", {})
                self._model_samplerate = int(model_config.get("sample_rate", model_config.get("fs", 48000)))

                if self._model_samplerate != DEFAULT_SAMPLERATE:
                    logger.info(
                        f"[{self.name}] Resampling configured: App ({DEFAULT_SAMPLERATE} Hz) <-> Model ({self._model_samplerate} Hz)"
                    )
                    # Initialize streaming resamplers (NAM is mono -> 1 channel)
                    self._resampler_stream_in = ResamplingStream(
                        orig_sr=DEFAULT_SAMPLERATE,
                        target_sr=self._model_samplerate,
                        num_channels=1,
                        dtype=DEFAULT_DTYPE,
                    )
                    self._resampler_stream_out = ResamplingStream(
                        orig_sr=self._model_samplerate,
                        target_sr=DEFAULT_SAMPLERATE,
                        num_channels=1,
                        dtype=DEFAULT_DTYPE,
                    )

                # The C++ module's block size can be dynamic, so we just pass a representative value.
                self.lib.set_parameters(
                    self.dsp_handle, file_path.encode("utf-8"), self._model_samplerate, DEFAULT_BLOCKSIZE * 2
                )

                self._file_path, self._error_message = file_path, None
                self.clear_error_state()
            except Exception as e:
                self._error_message, self.error_state, self._file_path = f"Failed to load NAM model: {e}", str(e), None
            state_to_emit = self._get_state_snapshot_locked()
        self.ui_update_callback(state_to_emit)

    def start(self):
        with self._lock:
            if self._resampler_stream_in:
                self._resampler_stream_in.reset()
            if self._resampler_stream_out:
                self._resampler_stream_out.reset()

    def process(self, input_data: Dict) -> Dict:
        if not self.dsp_handle or self.error_state:
            return {"out": None}
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}

        # NAM processes mono. Mix down if necessary.
        mono_signal = torch.mean(signal, dim=0, keepdim=True) if signal.shape[0] > 1 else signal

        # --- Run processing chain ---
        audio_for_cpp = None

        # 1. Input Resampling
        if self._resampler_stream_in:
            # Push current block
            self._resampler_stream_in.push(mono_signal)
            # Pull ALL available samples at the model's rate
            audio_for_cpp = self._resampler_stream_in.pull()
        else:
            audio_for_cpp = mono_signal

        # 2. C++ Processing
        # If resampling buffer yielded no samples (unlikely but possible), skip C++
        if audio_for_cpp.shape[1] > 0:
            processed_by_cpp = self._process_cpp_block(audio_for_cpp)
        else:
            processed_by_cpp = torch.zeros_like(audio_for_cpp)

        # 3. Output Resampling & Buffering
        final_output = None

        if self._resampler_stream_out:
            # Push processed audio (model rate)
            self._resampler_stream_out.push(processed_by_cpp)

            # Pull exactly one block for the graph output
            if self._resampler_stream_out.can_pull(DEFAULT_BLOCKSIZE):
                final_output = self._resampler_stream_out.pull(DEFAULT_BLOCKSIZE)
            else:
                # Underflow handling: output silence
                final_output = torch.zeros((1, DEFAULT_BLOCKSIZE), dtype=DEFAULT_DTYPE)
        else:
            # If no resampling, we assume 1:1 input/output ratio, so logic matches input size
            final_output = processed_by_cpp

        return {"out": final_output}

    def _process_cpp_block(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Helper function to encapsulate the C++ call."""
        if not self.dsp_handle:
            return torch.zeros_like(input_tensor)

        input_tensor = input_tensor.contiguous().to(torch.float32)
        num_samples = input_tensor.shape[1]

        # Output buffer must match input size for the C++ processor
        output_buffer = torch.zeros((1, num_samples), dtype=torch.float32)

        in_ptr = ctypes.cast(input_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
        out_ptr = ctypes.cast(output_buffer.data_ptr(), ctypes.POINTER(ctypes.c_float))

        self.lib.process_block(self.dsp_handle, in_ptr, out_ptr, num_samples)

        return output_buffer

    def _get_state_snapshot_locked(self):
        return {"file_path": self._file_path, "error_message": self._error_message}

    def serialize_extra(self):
        with self._lock:
            return {"file_path": self._file_path}

    def deserialize_extra(self, data: Dict):
        fp = data.get("file_path")
        if fp:
            with self._lock:
                self._file_path = fp
