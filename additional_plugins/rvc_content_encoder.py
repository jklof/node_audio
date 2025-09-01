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
    import resampy
    from fairseq import checkpoint_utils

    RVC_HUBERT_DEPS_AVAILABLE = True
except ImportError:
    RVC_HUBERT_DEPS_AVAILABLE = False

    class RVCContentEncoder:
        pass


# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from constants import DEFAULT_SAMPLERATE

# --- Qt Imports ---
from PySide6.QtCore import Qt, QTimer, Slot, QSignalBlocker
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QComboBox, QFileDialog, QHBoxLayout

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Node-Specific Constants ---
REQUIRED_SR = 16000
# --- REVISED: Increased chunk size for better efficiency on slower hardware ---
WORKER_CHUNK_SAMPLES = 16384
# WORKER_CHUNK_SAMPLES = 8192
UI_UPDATE_INTERVAL_MS = 100
# If the buffer grows beyond this many chunks, the worker will discard old audio.
MAX_BUFFER_CHUNKS = 5


if RVC_HUBERT_DEPS_AVAILABLE:

    class RVCContentEncoder:
        def __init__(self, model_path: str, gpu: int = 0):
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"HuBERT model not found at {model_path}")

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

            logger.info(f"Loading HuBERT model from: {model_path}")
            logger.info(f"Using device: {self.device}, Half precision: {self.is_half}")

            models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                [model_path],
                suffix="",
            )
            model = models[0].eval().to(self.device)

            if self.is_half:
                model = model.half()

            self.model = model

        @torch.no_grad()
        def encode(self, audio_16khz: np.ndarray, emb_output_layer=9, use_final_proj=True) -> np.ndarray:
            feats = torch.from_numpy(audio_16khz).float().view(1, -1).to(self.device)
            if self.is_half:
                feats = feats.half()

            padding_mask = torch.BoolTensor(feats.shape).fill_(False).to(self.device)

            inputs = {"source": feats, "padding_mask": padding_mask, "output_layer": emb_output_layer}

            logits = self.model.extract_features(**inputs)

            if use_final_proj:
                feats_out = self.model.final_proj(logits[0])
            else:
                feats_out = logits[0]

            return feats_out.squeeze(0).cpu().numpy()


class RVCContentEncoderNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 250

    def __init__(self, node_logic: "RVCContentEncoderNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)
        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(5)

        self.load_button = QPushButton("Load HuBERT Model")
        layout.addWidget(self.load_button)

        self.model_path_label = QLabel("Model: Not loaded")
        self.model_path_label.setStyleSheet("font-size: 9px; color: gray;")
        self.model_path_label.setWordWrap(True)
        layout.addWidget(self.model_path_label)

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
            error_label = QLabel("Missing dependencies:\n(torch, fairseq, resampy)")
            error_label.setStyleSheet("color: orange;")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.load_button.setEnabled(False)
            self.device_combo.setEnabled(False)
            self.version_combo.setEnabled(False)
            layout.addWidget(error_label)

        self.setContentWidget(self.container_widget)

        self._populate_devices()

        self.load_button.clicked.connect(self._on_load_model_clicked)
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

    @Slot()
    def _on_load_model_clicked(self):
        parent = self.scene().views()[0] if self.scene() and self.scene().views() else None
        path, _ = QFileDialog.getOpenFileName(parent, "Select HuBERT Model", "", "PyTorch Models (*.pt)")
        if path:
            self.node_logic.set_model_path(path)

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
        path = state.get("model_path")
        self.model_path_label.setText(f"Model: ...{path[-35:]}" if path else "Model: Not loaded")

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
            deque_size = state.get("deque_size", 0)
            self.status_label.setText(f"Buffer: {deque_size} | Proc: {proc_time:.1f} ms")

            # --- NEW: Visual feedback for system strain ---
            is_strained = state.get("is_strained", False)
            if is_strained:
                self.status_label.setStyleSheet("color: red;")
            else:
                self.status_label.setStyleSheet("color: white;")  # Or your theme's default
        else:
            self.status_label.setText("Status: Idle")
            self.status_label.setStyleSheet("color: white;")

        super().updateFromLogic()


class RVCContentEncoderNode(Node):
    NODE_TYPE = "RVC Content Encoder"
    UI_CLASS = RVCContentEncoderNodeItem
    CATEGORY = "RVC"
    DESCRIPTION = "Extracts content features from audio using a HuBERT model."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=np.ndarray)
        self.add_output("features", data_type=np.ndarray)
        self._lock = threading.Lock()

        self._model_path: Optional[str] = None
        self._device: int = -1
        self._rvc_version: str = "v1"
        self._encoder_instance: Optional[RVCContentEncoder] = None
        self._is_processing = False
        self._last_proc_time_ms: float = 0.0
        self._is_strained: bool = False  # For UI feedback

        self._audio_deque = collections.deque()
        self._results_deque = collections.deque(maxlen=10)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def _process_audio_chunk(
        self, audio_chunk: np.ndarray, encoder: RVCContentEncoder, version: str = "v1"
    ) -> np.ndarray:
        """Process a single audio chunk through resampling and encoding."""
        resampled = resampy.resample(audio_chunk, sr_orig=DEFAULT_SAMPLERATE, sr_new=REQUIRED_SR, filter="kaiser_fast")
        emb_layer, final_proj = (9, True) if version == "v1" else (12, False)
        features = encoder.encode(resampled.astype(np.float32), emb_layer, final_proj)
        return features

    def _encoder_loop(self):
        logger.info(f"[{self.name}] Encoder worker thread started.")

        while not self._stop_event.is_set():
            with self._lock:
                encoder = self._encoder_instance
                current_deque_size = len(self._audio_deque)

            has_enough_data = current_deque_size >= WORKER_CHUNK_SAMPLES

            if has_enough_data and encoder:
                is_strained_now = False  # Reset strain flag for this iteration
                if current_deque_size > WORKER_CHUNK_SAMPLES * MAX_BUFFER_CHUNKS:
                    is_strained_now = True  # Set strain flag
                    with self._lock:
                        samples_to_discard = current_deque_size - (WORKER_CHUNK_SAMPLES * (MAX_BUFFER_CHUNKS - 1))
                        for _ in range(samples_to_discard):
                            self._audio_deque.popleft()
                        logger.warning(f"[{self.name}] Buffer overflow. Discarding {samples_to_discard} old samples.")

                with self._lock:
                    self._is_strained = is_strained_now  # Update shared state for UI

                try:
                    with self._lock:
                        # Create the potentially non-contiguous array from the deque
                        audio_chunk_non_contiguous = np.array(
                            [self._audio_deque.popleft() for _ in range(WORKER_CHUNK_SAMPLES)]
                        )
                    # Explicitly create a C-contiguous copy.
                    audio_chunk = np.ascontiguousarray(audio_chunk_non_contiguous, dtype=np.float32)

                    start_time = time.monotonic()
                    with self._lock:
                        version = self._rvc_version
                    features = self._process_audio_chunk(audio_chunk, encoder, version)
                    proc_duration_ms = (time.monotonic() - start_time) * 1000
                    logger.info(f"[{self.name}] Chunk processed in {proc_duration_ms:.2f} ms.")

                    with self._lock:
                        self._results_deque.append(features)
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
                logger.warning(f"[{self.name}] Denied model load while processing is active.")
                return
            model_path, device_id, version = self._model_path, self._device, self._rvc_version

        encoder = None
        try:
            if RVC_HUBERT_DEPS_AVAILABLE and model_path and os.path.exists(model_path):
                encoder = RVCContentEncoder(model_path, device_id)

                logger.info(f"[{self.name}] Warming up the model on device {encoder.device}...")
                # Warm up by processing the dummy chunk with version-specific parameters.
                # This ensures both resampling and encoding run with the same parameters used at runtime.
                for _ in range(3):  # Increased warm-up iterations for better stability
                    dummy_input_chunk = np.random.randn(WORKER_CHUNK_SAMPLES).astype(np.float32)
                    _ = self._process_audio_chunk(dummy_input_chunk, encoder, version)

                # If on GPU, explicitly wait for all setup operations to complete.
                if "cuda" in str(encoder.device):
                    logger.info(f"[{self.name}] Synchronizing CUDA stream to finalize model setup...")
                    torch.cuda.synchronize()
                    logger.info(f"[{self.name}] CUDA stream synchronized.")

                logger.info(f"[{self.name}] Model is warmed up and ready.")

            elif model_path:
                logger.error(f"[{self.name}] Failed to load: File not found at '{model_path}'")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load or warm up HuBERT model: {e}", exc_info=True)

        with self._lock:
            self._encoder_instance = encoder

    @Slot(str)
    def set_model_path(self, path: str):
        with self._lock:
            if self._is_processing:
                return
            self._model_path = path
        self._load_model()

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
            self._rvc_version = version

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return {
                "model_path": self._model_path,
                "device": self._device,
                "version": self._rvc_version,
                "is_processing": self._is_processing,
                "deque_size": len(self._audio_deque),
                "last_proc_time_ms": self._last_proc_time_ms,
                "is_strained": self._is_strained,
            }

    def process(self, input_data: dict) -> dict:
        audio_in = input_data.get("in")
        if audio_in is not None:
            mono_signal = np.mean(audio_in, axis=1) if audio_in.ndim > 1 else audio_in
            with self._lock:
                if self._encoder_instance:
                    self._audio_deque.extend(mono_signal.astype(np.float32).tolist())

        latest_features = None
        with self._lock:
            if self._results_deque:
                latest_features = self._results_deque.popleft()
        return {"features": latest_features}

    def start(self):
        super().start()
        with self._lock:
            self._audio_deque.clear()
            self._results_deque.clear()
            self._is_processing = True
            self._last_proc_time_ms = 0.0
            self._is_strained = False

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._encoder_loop, daemon=True)
        self._worker_thread.start()
        logger.info(f"[{self.name}] Started processing.")

    def stop(self):
        super().stop()
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=0.5)
            self._worker_thread = None
        with self._lock:
            self._is_processing = False
            self._audio_deque.clear()
            self._results_deque.clear()
        logger.info(f"[{self.name}] Stopped processing.")

    def remove(self):
        self.stop()
        super().remove()

    def serialize_extra(self) -> dict:
        with self._lock:
            return {"model_path": self._model_path, "device": self._device, "version": self._rvc_version}

    def deserialize_extra(self, data: dict):
        self._device = data.get("device", -1)
        self._rvc_version = data.get("version", "v1")
        self._model_path = data.get("model_path")
        if self._model_path:
            self._load_model()
