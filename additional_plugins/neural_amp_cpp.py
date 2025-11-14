import torch
import ctypes
import logging
import os
from typing import Dict, Optional

# --- Node System Imports ---
from ffi_node import FFINodeBase
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE

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
    NODE_TYPE = "Neural Amp Modeler (C++)"
    UI_CLASS = NAMCppNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Processes audio using a .nam file. Multi-channel audio is mixed to mono before processing."

    LIB_NAME = "nam_processor"
    API = {
        "set_parameters": {"argtypes": [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_int]},
        "process_block": {
            "argtypes": [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
        },
    }

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_output("out", data_type=torch.Tensor)

        self._file_path: Optional[str] = None
        self._error_message: Optional[str] = None

        # Pre-allocate a mono output buffer for performance
        self._output_buffer = torch.zeros((1, DEFAULT_BLOCKSIZE), dtype=torch.float32)

        if self._file_path and self.dsp_handle:
            self.load_model(self._file_path)

    @Slot(str)
    def load_model(self, file_path: str):
        logger.info(f"[{self.name}] Loading model: {file_path}")
        state_to_emit = None
        if not os.path.exists(file_path):
            error_msg = f"NAM file not found: {file_path}"
            logger.error(error_msg)
            self._error_message, self._file_path, self.error_state = error_msg, None, error_msg
            self.ui_update_callback(self._get_state_snapshot_locked())
            return
        with self._lock:
            try:
                self.lib.set_parameters(
                    self.dsp_handle, file_path.encode("utf-8"), DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE
                )
                self._file_path, self._error_message = file_path, None
                self.clear_error_state()
            except Exception as e:
                error_msg = f"Failed to load NAM model in C++: {e}"
                self._error_message, self.error_state, self._file_path = error_msg, error_msg, None
                logger.error(error_msg, exc_info=True)
            state_to_emit = self._get_state_snapshot_locked()
        self.ui_update_callback(state_to_emit)

    def process(self, input_data: Dict) -> Dict:
        if not self.dsp_handle or self.error_state:
            return {"out": None}

        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor) or signal.numel() == 0:
            return {"out": None}

        # TODO: need to handle sample rate conversion if model was trained at different rate

        # --- Auto-Mixdown Logic ---
        num_channels = signal.shape[0]
        if num_channels > 1:
            # If multi-channel, average to create a mono signal. Use keepdim=True to maintain 2D shape.
            mono_input_tensor = torch.mean(signal, dim=0, keepdim=True)
        else:
            mono_input_tensor = signal

        # Ensure the resulting mono tensor is float32 and contiguous for the C++ layer.
        mono_input_tensor = mono_input_tensor.contiguous().to(torch.float32)

        num_samples = mono_input_tensor.shape[1]

        # Ensure output buffer has the correct shape
        if self._output_buffer.shape != (1, num_samples):
            self._output_buffer = torch.zeros((1, num_samples), dtype=torch.float32)

        # Get direct pointers to the data buffers (now guaranteed to be mono)
        in_ptr = ctypes.cast(mono_input_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
        out_ptr = ctypes.cast(self._output_buffer.data_ptr(), ctypes.POINTER(ctypes.c_float))

        # Call the simplified C++ process function
        self.lib.process_block(
            self.dsp_handle,
            in_ptr,
            out_ptr,
            num_samples,
        )

        return {"out": self._output_buffer.clone().to(DEFAULT_DTYPE)}

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
