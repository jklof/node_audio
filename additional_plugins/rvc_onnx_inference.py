import os
import json
import torch
import numpy as np
import onnxruntime
import threading
import collections
import logging
import time
from typing import Any, Dict, Optional

import resampy  # --- FIX: Use resampy as requested ---
from constants import DEFAULT_BLOCKSIZE, DEFAULT_SAMPLERATE, DEFAULT_DTYPE

# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING

# --- Qt Imports ---
from PySide6.QtCore import Qt, Slot, QSignalBlocker, QTimer
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QComboBox, QFileDialog, QHBoxLayout, QSpinBox

logger = logging.getLogger(__name__)
MAX_BUFFER_CHUNKS = 5


# ==============================================================================
# 1. UI Class (No changes needed here)
# ==============================================================================
class RVCOnnxInferenceNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 280

    def __init__(self, node_logic: "RVCOnnxInferenceNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)
        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        self.load_model_button = QPushButton("Load RVC ONNX Model (.onnx)")
        layout.addWidget(self.load_model_button)
        self.model_path_label = QLabel("Model: Not loaded")
        self.model_path_label.setStyleSheet("font-size: 9px; color: gray;")
        self.model_path_label.setWordWrap(True)
        layout.addWidget(self.model_path_label)
        controls_layout = QHBoxLayout()
        device_v_layout = QVBoxLayout()
        device_v_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        device_v_layout.addWidget(self.device_combo)
        controls_layout.addLayout(device_v_layout)
        spk_v_layout = QVBoxLayout()
        spk_v_layout.addWidget(QLabel("Speaker ID:"))
        self.spk_id_spin = QSpinBox()
        self.spk_id_spin.setRange(0, 100)
        spk_v_layout.addWidget(self.spk_id_spin)
        controls_layout.addLayout(spk_v_layout)
        layout.addLayout(controls_layout)
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Idle")
        self.buffer_label = QLabel("Buffer: 0 | 0")
        self.buffer_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.buffer_label.setStyleSheet("color: gray;")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.buffer_label)
        layout.addLayout(status_layout)
        self.info_label = QLabel("Info: Awaiting model")
        self.info_label.setStyleSheet("font-size: 9px; color: gray;")
        layout.addWidget(self.info_label)
        self.setContentWidget(self.container_widget)
        self._populate_devices()
        self.load_model_button.clicked.connect(self._on_load_model_clicked)
        self.device_combo.currentIndexChanged.connect(self._on_device_change)
        self.spk_id_spin.valueChanged.connect(self.node_logic.set_speaker_id)
        self.ui_updater = QTimer(self)
        self.ui_updater.setInterval(200)
        self.ui_updater.timeout.connect(self.updateFromLogic)
        self.ui_updater.start()
        self.updateFromLogic()

    def _populate_devices(self):
        self.device_combo.clear()
        self.device_combo.addItem("CPU", -1)
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            try:
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        self.device_combo.addItem(f"GPU {i}", i)
            except Exception as e:
                logger.warning(f"Could not enumerate CUDA devices for ONNX: {e}")

    @Slot()
    def _on_load_model_clicked(self):
        view = self.scene().views()[0] if self.scene() and self.scene().views() else None
        path, _ = QFileDialog.getOpenFileName(view, "Select RVC ONNX Model", "", "ONNX Models (*.onnx)")
        if path:
            self.node_logic.set_model_path(path)

    @Slot(int)
    def _on_device_change(self, index: int):
        if index != -1:
            self.node_logic.set_device(self.device_combo.itemData(index))

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        model_path = state.get("model_path")
        self.model_path_label.setText(f"Model: ...{model_path[-40:]}" if model_path else "Model: Not loaded")
        with QSignalBlocker(self.device_combo):
            self.device_combo.setCurrentIndex(self.device_combo.findData(state.get("device", -1)))
        with QSignalBlocker(self.spk_id_spin):
            self.spk_id_spin.setValue(state.get("speaker_id", 0))
        status_text = "Status: OK" if state.get("is_processing") else state.get("status", "Idle")
        is_strained = state.get("is_strained", False)
        if is_strained:
            status_text = "Status: Overloaded!"
            self.status_label.setStyleSheet("color: red;")
            self.buffer_label.setStyleSheet("color: red;")
        else:
            self.status_label.setStyleSheet("color: white;")
            self.buffer_label.setStyleSheet("color: gray;")
        self.status_label.setText(status_text)
        task_queue = state.get("task_queue_size", 0)
        output_buffer = state.get("output_buffer_size", 0)
        self.buffer_label.setText(f"Buffer: In {task_queue} | Out {output_buffer}")
        info = state.get("info", {})
        info_text = (
            f"SR: {info.get('sr')} | F0: {info.get('f0')} | Half: {info.get('is_half')}"
            if info
            else "Info: Awaiting model"
        )
        self.info_label.setText(info_text)
        super().updateFromLogic()


# ==============================================================================
# 2. Node Logic Class (Corrected)
# ==============================================================================
class RVCOnnxInferenceNode(Node):
    NODE_TYPE = "RVC ONNX Inference"
    UI_CLASS = RVCOnnxInferenceNodeItem
    CATEGORY = "RVC"
    DESCRIPTION = "Performs voice conversion using a loaded RVC ONNX model."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("features", data_type=np.ndarray)
        self.add_input("f0_coarse", data_type=np.ndarray)
        self.add_input("pitchf", data_type=np.ndarray)
        self.add_output("audio_out", data_type=np.ndarray)
        self._lock = threading.Lock()
        self._model_path: Optional[str] = None
        self._device: int = -1
        self._speaker_id: int = 0
        self._session: Optional[onnxruntime.InferenceSession] = None
        self._metadata: Dict[str, Any] = {}
        self._status: str = "Idle"
        self._is_half: bool = False
        self._is_processing = False
        self._is_strained = False
        self._tasks_deque = collections.deque()
        self._results_deque = collections.deque()  # --- FIX: This is now the final audio buffer ---
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def _load_model(self):
        with self._lock:
            model_path, gpu = self._model_path, self._device
            self._status = "Idle"
            self._is_half = False
        if not model_path or not os.path.exists(model_path):
            self._session, self._metadata = None, {}
            return

        provider = "CPUExecutionProvider"
        provider_options = {}
        if gpu >= 0 and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            provider = "CUDAExecutionProvider"
            provider_options = {"device_id": str(gpu)}

        try:
            with self._lock:
                self._status = "Loading..."
            so = onnxruntime.SessionOptions()
            so.log_severity_level = 3
            session = onnxruntime.InferenceSession(
                model_path, sess_options=so, providers=[provider], provider_options=[provider_options]
            )
            meta = session.get_modelmeta()
            if "metadata" not in meta.custom_metadata_map:
                raise ValueError("ONNX model is missing metadata.")
            metadata = json.loads(meta.custom_metadata_map["metadata"])
            first_input_type = session.get_inputs()[0].type
            is_half = first_input_type == "tensor(float16)"
            self._session, self._metadata, self._is_half = session, metadata, is_half
            with self._lock:
                self._status = "Loaded"
            logger.info(f"[{self.name}] ONNX model loaded. Info: {metadata}, is_half: {is_half}")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to load ONNX model: {e}", exc_info=True)
            self._session, self._metadata, self._is_half = None, {}, False
            with self._lock:
                self._status = "Error"

    def _inference_loop(self):
        while not self._stop_event.is_set():
            task = None
            with self._lock:
                if self._tasks_deque:
                    task = self._tasks_deque.popleft()
                self._is_strained = len(self._tasks_deque) >= MAX_BUFFER_CHUNKS

            if task and self._session:
                try:
                    # --- FIX: Worker thread now handles resampling ---
                    audio_out = self._session.run(["audio"], task)[0].squeeze()

                    model_sr = self._metadata.get("samplingRate")
                    if model_sr != DEFAULT_SAMPLERATE:
                        # Use resampy for high-quality and fast resampling
                        audio_out = resampy.resample(
                            audio_out, sr_orig=model_sr, sr_new=DEFAULT_SAMPLERATE, filter="kaiser_fast"
                        )

                    with self._lock:
                        # Add final, ready-to-play audio to the results deque
                        self._results_deque.extend(audio_out.tolist())

                except Exception as e:
                    logger.error(f"[{self.name}] Error in ONNX worker: {e}", exc_info=True)
            else:
                time.sleep(0.005)

    @Slot(str)
    def set_model_path(self, path: str):
        self._model_path = path
        self._load_model()

    @Slot(int)
    def set_device(self, device_id: int):
        self._device = device_id
        self._load_model()

    @Slot(int)
    def set_speaker_id(self, sid: int):
        self._speaker_id = sid

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            info = {"sr": self._metadata.get("samplingRate"), "f0": self._metadata.get("f0"), "is_half": self._is_half}
            return {
                "model_path": self._model_path,
                "device": self._device,
                "speaker_id": self._speaker_id,
                "is_processing": self._is_processing,
                "is_strained": self._is_strained,
                "status": self._status,
                "info": info if self._session else None,
                "task_queue_size": len(self._tasks_deque),
                "output_buffer_size": len(self._results_deque),  # --- FIX: Report the correct buffer ---
            }

    def process(self, input_data: dict) -> dict:
        features, f0_coarse, pitchf = (input_data.get(k) for k in ["features", "f0_coarse", "pitchf"])

        with self._lock:
            session_active = self._session is not None
            is_f0_model = self._metadata.get("f0", 1) == 1
            feats_dtype = np.float16 if self._is_half else np.float32

        if session_active and features is not None:
            if not is_f0_model or (f0_coarse is not None and pitchf is not None):
                min_len = len(features)
                if is_f0_model:
                    min_len = min(min_len, len(f0_coarse), len(pitchf))

                if min_len > 0:
                    input_dict = {
                        "feats": np.expand_dims(features[:min_len], 0).astype(feats_dtype),
                        "p_len": np.array([min_len], dtype=np.int64),
                        "sid": np.array([self._speaker_id], dtype=np.int64),
                    }
                    if is_f0_model:
                        input_dict["pitch"] = np.expand_dims(f0_coarse[:min_len], 0).astype(np.int64)
                        input_dict["pitchf"] = np.expand_dims(pitchf[:min_len], 0).astype(np.float32)

                    with self._lock:
                        self._tasks_deque.append(input_dict)

        with self._lock:
            # --- FIX: More robust output buffering ---
            if len(self._results_deque) >= DEFAULT_BLOCKSIZE:
                # Pop exactly one block's worth of samples
                output_samples = [self._results_deque.popleft() for _ in range(DEFAULT_BLOCKSIZE)]
                # Create a 2D mono array with the correct dtype
                output_block = np.array(output_samples, dtype=DEFAULT_DTYPE).reshape(-1, 1)
                return {"audio_out": output_block}
            else:
                # If not enough samples, return silence of the correct shape and type
                silent_block = np.zeros((DEFAULT_BLOCKSIZE, 1), dtype=DEFAULT_DTYPE)
                return {"audio_out": silent_block}

    def start(self):
        with self._lock:
            self._tasks_deque.clear()
            self._results_deque.clear()
            self._is_processing = True
            self._is_strained = False
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._worker_thread.start()
        self._load_model()

    def stop(self):
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=0.5)
        with self._lock:
            self._is_processing = False

    def remove(self):
        self.stop()
        super().remove()

    def serialize_extra(self) -> dict:
        return {"model_path": self._model_path, "device": self._device, "speaker_id": self._speaker_id}

    def deserialize_extra(self, data: dict):
        self._model_path = data.get("model_path")
        self._device = data.get("device", -1)
        self._speaker_id = data.get("speaker_id", 0)
