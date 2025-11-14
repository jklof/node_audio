import torch
import ctypes
import logging
import os
from typing import Dict, Optional

# --- Node System Imports ---
from ffi_node import FFINodeBase
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog
from PySide6.QtCore import Slot

logger = logging.getLogger(__name__)

# Define a C-style pointer to a float pointer for our 2D audio data
float_pp = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))


# UI Class (Copied from existing neural_amp.py)
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
    DESCRIPTION = "Processes audio using a trained Neural Amp Model (.nam) file with the high-performance C++ core."

    # --- Declarative API Definition for the C++ library ---
    LIB_NAME = "nam_processor"
    API = {
        "set_parameters": {"argtypes": [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_int]},
        "process_block": {"argtypes": [ctypes.c_void_p, float_pp, float_pp, ctypes.c_int, ctypes.c_int]},
    }

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_output("out", data_type=torch.Tensor)

        self._file_path: Optional[str] = None
        self._error_message: Optional[str] = None

        # Pre-allocate the output buffer for performance (Stereo, max block size)
        self._output_buffer = torch.zeros((DEFAULT_CHANNELS, DEFAULT_BLOCKSIZE), dtype=torch.float32)

        # Attempt to load the file path if it was restored from a session (handled by deserialize_extra)
        if self._file_path and self.dsp_handle:
            self.load_model(self._file_path)

    @Slot(str)
    def load_model(self, file_path: str):
        logger.info(f"[{self.name}] Loading model: {file_path}")
        state_to_emit = None

        if not os.path.exists(file_path):
            error_msg = f"NAM file not found: {file_path}"
            logger.error(error_msg)
            self._error_message = error_msg
            self._file_path = None
            self.error_state = error_msg
            self.ui_update_callback(self._get_state_snapshot_locked())
            return

        with self._lock:
            try:
                # Call the C++ set_parameters function to load the model
                self.lib.set_parameters(
                    self.dsp_handle,
                    file_path.encode("utf-8"),  # Convert string to C-string
                    DEFAULT_SAMPLERATE,
                    DEFAULT_BLOCKSIZE,
                )

                self._file_path, self._error_message = file_path, None
                self.clear_error_state()
            except Exception as e:
                error_msg = f"Failed to load NAM model in C++: {e}"
                self._error_message, self.error_state = error_msg, error_msg
                self._file_path = None
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

        # The C++ core expects (channels, samples) and float32
        in_tensor = signal.contiguous().to(torch.float32)

        # Ensure we are operating on a mono signal if the NAM core expects it.
        # The C++ code will only use the first channel, so we ensure the input is at least 1 channel.
        if in_tensor.ndim == 1:
            in_tensor = in_tensor.unsqueeze(0)

        num_input_channels, num_samples = in_tensor.shape

        # Ensure output buffer matches expected max channels and current block size
        # We enforce at least stereo output for the graph environment
        num_output_channels = DEFAULT_CHANNELS

        out_tensor = self._output_buffer[:num_output_channels, :num_samples]

        # Create C-style arrays of pointers for planar data
        # Note: We only need to provide pointers for the number of channels actually used.
        in_channel_pointers = (ctypes.POINTER(ctypes.c_float) * num_output_channels)()
        out_channel_pointers = (ctypes.POINTER(ctypes.c_float) * num_output_channels)()

        for i in range(num_output_channels):
            # For input, we just repeat the first input channel's pointer for all C++ channels
            # or use the actual channel if present (only input[0] is ever read in the C++ side)
            input_ptr_index = min(i, num_input_channels - 1)
            in_channel_pointers[i] = ctypes.cast(in_tensor[input_ptr_index].data_ptr(), ctypes.POINTER(ctypes.c_float))
            out_channel_pointers[i] = ctypes.cast(out_tensor[i].data_ptr(), ctypes.POINTER(ctypes.c_float))

        # Call the C++ process function
        self.lib.process_block(
            self.dsp_handle,
            in_channel_pointers,
            out_channel_pointers,
            num_output_channels,
            num_samples,
        )

        # Return a clone of the output buffer to avoid corruption in next call
        return {"out": out_tensor.clone().to(DEFAULT_DTYPE)}

    def _get_state_snapshot_locked(self):
        return {"file_path": self._file_path, "error_message": self._error_message}

    def serialize_extra(self):
        with self._lock:
            return {"file_path": self._file_path}

    def deserialize_extra(self, data: Dict):
        fp = data.get("file_path")
        if fp:
            with self._lock:
                # Store the path for the UI and later C++ loading (in __init__)
                self._file_path = fp
