import torch
import ctypes
import logging
import os
import json
from typing import Dict, Optional

import torchaudio.transforms as T

from ffi_node import FFINodeBase
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog
from PySide6.QtCore import Slot

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
    DESCRIPTION = "Processes audio using a Neural Amp Model. Multi-channel audio is mixed to mono before processing."

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

        self._resampler_to_model: Optional[T.Resample] = None
        self._resampler_from_model: Optional[T.Resample] = None
        self._input_fifo = torch.tensor([], dtype=DEFAULT_DTYPE)
        self._output_fifo = torch.tensor([], dtype=DEFAULT_DTYPE)

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
            self._resampler_to_model = None
            self._resampler_from_model = None
            try:
                with open(file_path, "r") as f:
                    config = json.load(f)
                model_config = config.get("config", {})
                self._model_samplerate = int(model_config.get("sample_rate", model_config.get("fs", 48000)))

                if self._model_samplerate != DEFAULT_SAMPLERATE:
                    logger.info(
                        f"[{self.name}] Resampling required: App SR ({DEFAULT_SAMPLERATE} Hz) -> Model SR ({self._model_samplerate} Hz)"
                    )
                    self._resampler_to_model = T.Resample(
                        DEFAULT_SAMPLERATE, self._model_samplerate, dtype=DEFAULT_DTYPE
                    )
                    self._resampler_from_model = T.Resample(
                        self._model_samplerate, DEFAULT_SAMPLERATE, dtype=DEFAULT_DTYPE
                    )

                # The C++ module's block size can be dynamic, so we just pass a representative value during setup.
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
            self._input_fifo = torch.tensor([], dtype=DEFAULT_DTYPE)
            self._output_fifo = torch.tensor([], dtype=DEFAULT_DTYPE)

    def process(self, input_data: Dict) -> Dict:
        if not self.dsp_handle or self.error_state:
            return {"out": None}
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}

        mono_signal = torch.mean(signal, dim=0, keepdim=True) if signal.shape[0] > 1 else signal

        with self._lock:
            resampler_to = self._resampler_to_model
            resampler_from = self._resampler_from_model

        # --- Run processing chain ---
        if resampler_to and resampler_from:
            self._input_fifo = torch.cat((self._input_fifo, mono_signal), dim=1)
            signal_for_cpp = resampler_to(self._input_fifo)

            # This is the C++ processing step
            processed_by_cpp = self._process_cpp_block(signal_for_cpp)

            resampled_output = resampler_from(processed_by_cpp)
            self._output_fifo = torch.cat((self._output_fifo, resampled_output), dim=1)

            consumed_input_samples = int(signal_for_cpp.shape[1] * (DEFAULT_SAMPLERATE / self._model_samplerate))
            self._input_fifo = self._input_fifo[:, consumed_input_samples:]
        else:
            processed_output = self._process_cpp_block(mono_signal)
            self._output_fifo = torch.cat((self._output_fifo, processed_output), dim=1)

        # --- Return output block if available ---
        if self._output_fifo.shape[1] >= DEFAULT_BLOCKSIZE:
            output_block = self._output_fifo[:, :DEFAULT_BLOCKSIZE]
            self._output_fifo = self._output_fifo[:, DEFAULT_BLOCKSIZE:]
            return {"out": output_block}
        else:
            return {"out": torch.zeros((1, DEFAULT_BLOCKSIZE), dtype=DEFAULT_DTYPE)}

    def _process_cpp_block(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Helper function to encapsulate the C++ call."""
        if not self.dsp_handle:
            return torch.zeros_like(input_tensor)

        input_tensor = input_tensor.contiguous().to(torch.float32)
        num_samples = input_tensor.shape[1]

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
