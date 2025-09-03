import numpy as np
import threading
import logging
import time
import collections
import os
from typing import Dict, Optional

# --- Core Dependencies for this Plugin ---
try:
    import torch
    import torchaudio
    from torchaudio.pipelines import HUBERT_BASE
    import resampy

    RVC_HUBERT_DEPS_AVAILABLE = True
except ImportError:
    RVC_HUBERT_DEPS_AVAILABLE = False

    class RVCContentEncoder:
        pass


# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING

# --- MODIFIED: Import torch.Tensor ---
from constants import DEFAULT_SAMPLERATE, torch

# --- Qt Imports ---
from PySide6.QtCore import Qt, QTimer, Slot, QSignalBlocker
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QComboBox, QFileDialog, QHBoxLayout

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Node-Specific Constants ---
REQUIRED_SR = 16000
WORKER_CHUNK_SAMPLES = 16384
UI_UPDATE_INTERVAL_MS = 100
MAX_BUFFER_CHUNKS = 5


if RVC_HUBERT_DEPS_AVAILABLE:

    class RVCContentEncoder:
        def __init__(self, gpu: int = 0):
            if gpu < 0 or not torch.cuda.is_available():
                self.device = torch.device("cpu")
                self.is_half = False
            else:
                self.device = torch.device(f"cuda:{gpu}")
                try:
                    gpu_name = torch.cuda.get_device_name(gpu).upper()
                    self.is_half = not (
                        "16" in gpu_name or "P40" in gpu_name or "1070" in gpu_name or "1080" in gpu_name
                    )
                except Exception:
                    self.is_half = False
            logger.info(f"Loading HuBERT model from torchaudio bundle on device: {self.device}, Half: {self.is_half}")
            bundle = HUBERT_BASE
            model = bundle.get_model().to(self.device)
            model.eval()
            if self.is_half:
                model = model.half()
            self.model = model

        @torch.no_grad()
        def encode(self, audio_16khz: np.ndarray, emb_output_layer=9) -> np.ndarray:
            feats = torch.from_numpy(audio_16khz).float().view(1, -1).to(self.device)
            if self.is_half:
                feats = feats.half()
            hidden_states, _ = self.model.extract_features(feats)
            layer_idx = min(emb_output_layer, len(hidden_states) - 1)
            if emb_output_layer >= len(hidden_states):
                logger.warning(f"Requested layer {emb_output_layer} out of bounds. Using last layer {layer_idx}.")
            return hidden_states[layer_idx].squeeze(0).cpu().numpy()


class RVCContentEncoderNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 250

    # ... (UI code remains unchanged) ...
    def __init__(self, node_logic: "RVCContentEncoderNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)
        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(5)

        model_source_label = QLabel("Model Source: torchaudio HuBERT Base")
        model_source_label.setStyleSheet("font-size: 10px; color: lightgray;")
        model_source_label.setWordWrap(True)
        layout.addWidget(model_source_label)

        controls_layout = QHBoxLayout()
        version_v_layout = QVBoxLayout()
        version_v_layout.addWidget(QLabel("Version:"))
        self.version_combo = QComboBox()
        self.version_combo.addItems(["v1", "v2"])
        version_v_layout.addWidget(self.version_combo)
        controls_layout.addLayout(version_v_layout)

        device_v_layout = QVBoxLayout()
        device_v_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        device_v_layout.addWidget(self.device_combo)
        controls_layout.addLayout(device_v_layout)
        layout.addLayout(controls_layout)

        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label)

        if not RVC_HUBERT_DEPS_AVAILABLE:
            error_label = QLabel("Missing dependencies:\n(torch, torchaudio, resampy)")
            error_label.setStyleSheet("color: orange;")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.device_combo.setEnabled(False)
            self.version_combo.setEnabled(False)
            layout.addWidget(error_label)

        self.setContentWidget(self.container_widget)

        self._populate_devices()

        self.device_combo.currentIndexChanged.connect(self._on_device_change)
        self.version_combo.currentIndexChanged.connect(self._on_version_change)

        self.ui_updater = QTimer(self)
        self.ui_updater.setInterval(UI_UPDATE_INTERVAL_MS)
        self.ui_updater.timeout.connect(self.updateFromLogic)
        self.ui_updater.start()

    def _populate_devices(self):
        logger.info("Populating device list for RVC Content Encoder...")
        self.device_combo.clear()
        self.device_combo.addItem("CPU", -1)
        if not RVC_HUBERT_DEPS_AVAILABLE:
            return

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA is available. Found {device_count} GPU(s).")
            for i in range(device_count):
                self.device_combo.addItem(f"GPU {i}: {torch.cuda.get_device_name(i)}", i)
        else:
            logger.warning("CUDA is not available. Check PyTorch installation and NVIDIA drivers.")

    @Slot(int)
    def _on_device_change(self, index: int):
        if index != -1:
            self.node_logic.set_device(self.device_combo.itemData(index))

    @Slot(int)
    def _on_version_change(self, index: int):
        if index != -1:
            self.node_logic.set_rvc_version(self.version_combo.itemText(index))

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()

        with QSignalBlocker(self.device_combo):
            index = self.device_combo.findData(state.get("device", -1))
            if index != -1:
                self.device_combo.setCurrentIndex(index)

        with QSignalBlocker(self.version_combo):
            index = self.version_combo.findText(state.get("version", "v1"))
            if index != -1:
                self.version_combo.setCurrentIndex(index)

        if state.get("is_processing"):
            proc_time = state.get("last_proc_time_ms", 0)
            buffer_size_ms = state.get("buffer_size_ms", 0)
            self.status_label.setText(f"Buffer: {buffer_size_ms:.0f} ms | Proc: {proc_time:.1f} ms")

            is_strained = state.get("is_strained", False)
            if is_strained:
                self.status_label.setStyleSheet("color: red;")
            else:
                self.status_label.setStyleSheet("color: white;")
        else:
            self.status_label.setText("Status: Idle")
            self.status_label.setStyleSheet("color: white;")
        super.updateFromLogic()


class RVCContentEncoderNode(Node):
    NODE_TYPE = "RVC Content Encoder"
    UI_CLASS = RVCContentEncoderNodeItem
    CATEGORY = "RVC"
    DESCRIPTION = "Extracts content features from audio using a HuBERT model."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        # --- MODIFIED: Use torch.Tensor ---
        self.add_input("in", data_type=torch.Tensor)
        self.add_output("features", data_type=torch.Tensor)
        self._lock = threading.Lock()
        self._device: int = -1
        self._rvc_version: str = "v1"
        self._encoder_instance: Optional[RVCContentEncoder] = None
        self._is_processing = False
        self._last_proc_time_ms: float = 0.0
        self._is_strained: bool = False

        # --- MODIFIED: More efficient tensor-based buffer ---
        self._audio_buffer = torch.tensor([], dtype=torch.float32)
        self._results_deque = collections.deque(maxlen=10)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def _process_audio_chunk(
        self, audio_chunk: np.ndarray, encoder: RVCContentEncoder, version: str = "v1"
    ) -> np.ndarray:
        resampled = resampy.resample(audio_chunk, sr_orig=DEFAULT_SAMPLERATE, sr_new=REQUIRED_SR, filter="kaiser_fast")
        emb_layer = 9 if version == "v1" else 12
        return encoder.encode(resampled.astype(np.float32), emb_layer)

    def _encoder_loop(self):
        logger.info(f"[{self.name}] Encoder worker thread started.")
        while not self._stop_event.is_set():
            with self._lock:
                encoder = self._encoder_instance
                current_buffer_size = len(self._audio_buffer)
            has_enough_data = current_buffer_size >= WORKER_CHUNK_SAMPLES

            if has_enough_data and encoder:
                with self._lock:
                    self._is_strained = current_buffer_size > WORKER_CHUNK_SAMPLES * MAX_BUFFER_CHUNKS
                    if self._is_strained:
                        samples_to_discard = current_buffer_size - (WORKER_CHUNK_SAMPLES * (MAX_BUFFER_CHUNKS - 1))
                        self._audio_buffer = self._audio_buffer[samples_to_discard:]
                        logger.warning(f"[{self.name}] Buffer overflow. Discarding old audio.")
                    audio_chunk_tensor = self._audio_buffer[:WORKER_CHUNK_SAMPLES]
                    self._audio_buffer = self._audio_buffer[WORKER_CHUNK_SAMPLES:]
                    version = self._rvc_version

                try:
                    # Convert to numpy only when needed
                    audio_chunk_np = audio_chunk_tensor.numpy()
                    start_time = time.monotonic()
                    features = self._process_audio_chunk(audio_chunk_np, encoder, version)
                    proc_duration_ms = (time.monotonic() - start_time) * 1000
                    # Convert back to tensor to store result
                    with self._lock:
                        self._results_deque.append(torch.from_numpy(features))
                        self._last_proc_time_ms = proc_duration_ms
                except Exception as e:
                    logger.error(f"[{self.name}] Error in worker thread: {e}", exc_info=True)
                    time.sleep(0.1)
            else:
                time.sleep(0.01)
        logger.info(f"[{self.name}] Encoder worker thread stopped.")

    def _load_model(self):
        with self._lock:
            if self._is_processing:
                return
            device_id, version = self._device, self._rvc_version
        encoder = None
        try:
            if RVC_HUBERT_DEPS_AVAILABLE:
                encoder = RVCContentEncoder(device_id)
                dummy_input = np.random.randn(WORKER_CHUNK_SAMPLES).astype(np.float32)
                for _ in range(3):
                    _ = self._process_audio_chunk(dummy_input, encoder, version)
                if "cuda" in str(encoder.device):
                    torch.cuda.synchronize()
                logger.info(f"[{self.name}] Model is warmed up and ready.")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load or warm up HuBERT model: {e}", exc_info=True)
        with self._lock:
            self._encoder_instance = encoder

    @Slot(int)
    def set_device(self, device_id: int):
        with self._lock:
            if self._is_processing or self._device == device_id:
                return
            self._device = device_id
        self._load_model()

    @Slot(str)
    def set_rvc_version(self, version: str):
        with self._lock:
            if self._rvc_version != version:
                self._rvc_version = version

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return {
                "device": self._device,
                "version": self._rvc_version,
                "is_processing": self._is_processing,
                "buffer_size_ms": (len(self._audio_buffer) / DEFAULT_SAMPLERATE) * 1000,
                "last_proc_time_ms": self._last_proc_time_ms,
                "is_strained": self._is_strained,
            }

    def process(self, input_data: dict) -> dict:
        audio_in = input_data.get("in")
        if isinstance(audio_in, torch.Tensor):
            mono_signal = torch.mean(audio_in, dim=0) if audio_in.ndim > 1 else audio_in
            with self._lock:
                if self._encoder_instance:
                    self._audio_buffer = torch.cat((self._audio_buffer, mono_signal.to(torch.float32)))
        latest_features = None
        with self._lock:
            if self._results_deque:
                latest_features = self._results_deque.popleft()
        return {"features": latest_features}

    def start(self):
        if self._encoder_instance is None:
            self._load_model()
        super().start()
        with self._lock:
            self._audio_buffer = torch.tensor([], dtype=torch.float32)
            self._results_deque.clear()
            self._is_processing, self._last_proc_time_ms, self._is_strained = True, 0.0, False
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._encoder_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        super().stop()
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=0.5)
            self._worker_thread = None
        with self._lock:
            self._is_processing, self._audio_buffer = False, torch.tensor([], dtype=torch.float32)
            self._results_deque.clear()

    def remove(self):
        self.stop()
        super().remove()

    def serialize_extra(self) -> dict:
        with self._lock:
            return {"device": self._device, "version": self._rvc_version}

    def deserialize_extra(self, data: dict):
        self._device, self._rvc_version = data.get("device", -1), data.get("version", "v1")
        self._load_model()
