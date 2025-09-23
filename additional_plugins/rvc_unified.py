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

# --- MODIFIED: Added torchaudio import ---
import torchaudio
from torchaudio.pipelines import HUBERT_BASE
import resampy
from constants import DEFAULT_BLOCKSIZE, DEFAULT_SAMPLERATE, DEFAULT_DTYPE

# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING

# --- Qt Imports ---
from PySide6.QtCore import Qt, Slot, QSignalBlocker, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
)

logger = logging.getLogger(__name__)

# --- Node-Specific Constants ---
RVC_REQUIRED_SR = 16000
UI_UPDATE_INTERVAL_MS = 100
MAX_BUFFER_CHUNKS_INPUT = 20
EPSILON = 1e-9


# ==============================================================================
# MODIFIED: Helper class for RVC HuBERT now uses torchaudio
# ==============================================================================
class RVC_Hubert:
    """
    MODIFIED: This class now loads the HuBERT model from the torchaudio
    bundle and does not require a local model file.
    """

    def __init__(self, gpu: int = 0):
        if gpu < 0 or not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.is_half = False
        else:
            self.device = torch.device(f"cuda:{gpu}")
            try:
                gpu_name = torch.cuda.get_device_name(gpu).upper()
                self.is_half = not ("16" in gpu_name or "P40" in gpu_name or "1070" in gpu_name or "1080" in gpu_name)
            except Exception:
                self.is_half = False

        logger.info("Loading HuBERT model from torchaudio bundle...")
        bundle = HUBERT_BASE
        model = bundle.get_model().to(self.device)
        model.eval()

        if self.is_half:
            model = model.half()
        self.model = model

    @torch.no_grad()
    def encode(self, audio_16khz: np.ndarray, emb_output_layer=9, use_final_proj=True) -> np.ndarray:
        feats = torch.from_numpy(audio_16khz).float().view(1, -1).to(self.device)
        if self.is_half:
            feats = feats.half()
        hidden_states, _ = self.model.extract_features(feats)

        if emb_output_layer < len(hidden_states):
            feats_out = hidden_states[emb_output_layer]
        else:
            feats_out = hidden_states[-1]
        return feats_out.squeeze(0).cpu().numpy()


# ==============================================================================
# 1. UI Class for the Unified Node (With Pre-Buffer Control)
# ==============================================================================
class RVCUnifiedNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 300

    def __init__(self, node_logic: "RVCUnifiedNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container = QWidget()
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(5)

        self.load_rvc_button = QPushButton("Load RVC Model (.onnx)")
        self.load_rmvpe_button = QPushButton("Load RMVPE Model (.onnx)")

        self.rvc_label = QLabel("RVC: Not loaded")
        self.rvc_label.setStyleSheet("font-size: 9px; color: gray;")
        self.hubert_label = QLabel("HuBERT: torchaudio (built-in)")
        self.hubert_label.setStyleSheet("font-size: 9px; color: lightgray;")
        self.rmvpe_label = QLabel("RMVPE: Not loaded")
        self.rmvpe_label.setStyleSheet("font-size: 9px; color: gray;")

        layout.addWidget(self.load_rvc_button)
        layout.addWidget(self.rvc_label)
        layout.addWidget(self.hubert_label)
        layout.addWidget(self.load_rmvpe_button)
        layout.addWidget(self.rmvpe_label)

        controls_layout = QHBoxLayout()
        device_layout = QVBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        device_layout.addWidget(self.device_combo)
        controls_layout.addLayout(device_layout)

        spk_layout = QVBoxLayout()
        spk_layout.addWidget(QLabel("Speaker ID:"))
        self.spk_id_spin = QSpinBox()
        self.spk_id_spin.setRange(0, 100)
        spk_layout.addWidget(self.spk_id_spin)
        controls_layout.addLayout(spk_layout)

        pitch_layout = QVBoxLayout()
        pitch_layout.addWidget(QLabel("Pitch Shift:"))
        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(-24.0, 24.0)
        self.pitch_spin.setSingleStep(0.1)
        pitch_layout.addWidget(self.pitch_spin)
        controls_layout.addLayout(pitch_layout)
        layout.addLayout(controls_layout)

        perf_layout = QHBoxLayout()
        vad_controls_layout = QVBoxLayout()
        self.vad_checkbox = QCheckBox("VAD Enabled")
        self.vad_checkbox.setChecked(True)
        vad_controls_layout.addWidget(self.vad_checkbox)
        self.vad_spin = QDoubleSpinBox()
        self.vad_spin.setRange(0.0, 1.0)
        self.vad_spin.setSingleStep(0.001)
        self.vad_spin.setDecimals(4)
        self.vad_spin.setToolTip("Voice Activity Detection threshold.\nProcessing is skipped on silent chunks.")
        vad_controls_layout.addWidget(self.vad_spin)
        perf_layout.addLayout(vad_controls_layout)

        chunk_layout = QVBoxLayout()
        chunk_layout.addWidget(QLabel("Chunk (ms):"))
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(40, 2000)
        self.chunk_spin.setSingleStep(20)
        self.chunk_spin.setValue(320)
        self.chunk_spin.setToolTip("Larger chunks reduce CPU overhead but increase latency.")
        chunk_layout.addWidget(self.chunk_spin)
        perf_layout.addLayout(chunk_layout)
        layout.addLayout(perf_layout)

        sola_layout = QHBoxLayout()
        crossfade_layout = QVBoxLayout()
        crossfade_layout.addWidget(QLabel("Crossfade (ms):"))
        self.crossfade_spin = QSpinBox()
        self.crossfade_spin.setRange(0, 500)
        self.crossfade_spin.setSingleStep(10)
        self.crossfade_spin.setValue(100)
        self.crossfade_spin.setToolTip("Duration of the audio crossfade to hide seams between chunks.")
        crossfade_layout.addWidget(self.crossfade_spin)
        sola_layout.addLayout(crossfade_layout)

        sola_search_layout = QVBoxLayout()
        sola_search_layout.addWidget(QLabel("SOLA Search (ms):"))
        self.sola_search_spin = QSpinBox()
        self.sola_search_spin.setRange(0, 100)
        self.sola_search_spin.setSingleStep(2)
        self.sola_search_spin.setValue(10)
        self.sola_search_spin.setToolTip(
            "Additional time to search for the best audio overlap point.\nHelps reduce artifacts."
        )
        sola_search_layout.addWidget(self.sola_search_spin)
        sola_layout.addLayout(sola_search_layout)
        layout.addLayout(sola_layout)

        prebuffer_layout = QVBoxLayout()
        self.prebuffer_spin = QSpinBox()
        self.prebuffer_spin.setRange(0, 1000)
        self.prebuffer_spin.setSingleStep(10)
        self.prebuffer_spin.setValue(100)
        self.prebuffer_spin.setToolTip(
            "Context audio given to the model before the main chunk (ms).\nCRITICAL for reducing robotic artifacts."
        )
        prebuffer_layout.addWidget(QLabel("Pre-Buffer (ms):"))
        prebuffer_layout.addWidget(self.prebuffer_spin)
        layout.addLayout(prebuffer_layout)

        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Idle")
        self.buffer_label = QLabel("Buffer: In 0 | Out 0")
        self.buffer_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.buffer_label.setStyleSheet("color: gray;")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.buffer_label)
        layout.addLayout(status_layout)

        self.info_label = QLabel("Info: Awaiting models")
        self.info_label.setStyleSheet("font-size: 9px; color: gray;")
        layout.addWidget(self.info_label)

        self.setContentWidget(self.container)
        self._populate_devices()

        self.load_rvc_button.clicked.connect(lambda: self._on_load_model_clicked("rvc"))
        self.load_rmvpe_button.clicked.connect(lambda: self._on_load_model_clicked("rmvpe"))
        self.device_combo.currentIndexChanged.connect(self._on_device_change)
        self.spk_id_spin.valueChanged.connect(self.node_logic.set_speaker_id)
        self.pitch_spin.valueChanged.connect(self.node_logic.set_pitch_shift)
        self.vad_checkbox.toggled.connect(self.node_logic.set_vad_enabled)
        self.vad_spin.valueChanged.connect(self.node_logic.set_silent_threshold)
        self.chunk_spin.valueChanged.connect(self.node_logic.set_chunk_size_ms)
        self.crossfade_spin.valueChanged.connect(self.node_logic.set_crossfade_ms)
        self.sola_search_spin.valueChanged.connect(self.node_logic.set_sola_search_ms)
        self.prebuffer_spin.valueChanged.connect(self.node_logic.set_extra_conversion_ms)

        self.ui_updater = QTimer(self)
        self.ui_updater.setInterval(UI_UPDATE_INTERVAL_MS)
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
                logger.warning(f"Could not enumerate CUDA devices: {e}")

    @Slot(str)
    def _on_load_model_clicked(self, model_type: str):
        view = self.scene().views()[0] if self.scene() and self.scene().views() else None
        title_map = {"rvc": "RVC ONNX", "rmvpe": "RMVPE ONNX"}
        ext_map = {"rvc": "ONNX (*.onnx)", "rmvpe": "ONNX (*.onnx)"}
        path, _ = QFileDialog.getOpenFileName(view, f"Select {title_map[model_type]} Model", "", ext_map[model_type])
        if path:
            self.node_logic.set_model_path(model_type, path)

    @Slot(int)
    def _on_device_change(self, index: int):
        if index != -1:
            self.node_logic.set_device(self.device_combo.itemData(index))

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()

        def set_label(label, key):
            path = state.get(f"{key}_path")
            label.setText(f"{key.upper()}: ...{path[-40:]}" if path else f"{key.upper()}: Not loaded")

        set_label(self.rvc_label, "rvc")
        set_label(self.rmvpe_label, "rmvpe")

        with QSignalBlocker(self.device_combo):
            self.device_combo.setCurrentIndex(self.device_combo.findData(state.get("device", -1)))
        with QSignalBlocker(self.spk_id_spin):
            self.spk_id_spin.setValue(state.get("speaker_id", 0))
        with QSignalBlocker(self.pitch_spin):
            self.pitch_spin.setValue(state.get("pitch_shift", 0.0))
        with QSignalBlocker(self.vad_checkbox):
            self.vad_checkbox.setChecked(state.get("vad_enabled", True))
        with QSignalBlocker(self.vad_spin):
            self.vad_spin.setValue(state.get("silent_threshold", 0.001))
        with QSignalBlocker(self.chunk_spin):
            self.chunk_spin.setValue(state.get("chunk_size_ms", 320))
        with QSignalBlocker(self.crossfade_spin):
            self.crossfade_spin.setValue(state.get("crossfade_ms", 100))
        with QSignalBlocker(self.sola_search_spin):
            self.sola_search_spin.setValue(state.get("sola_search_ms", 10))
        with QSignalBlocker(self.prebuffer_spin):
            self.prebuffer_spin.setValue(state.get("extra_conversion_ms", 100))

        is_strained = state.get("is_strained", False)
        status_text = (
            "Status: Overloaded!"
            if is_strained
            else ("Status: Processing" if state.get("is_processing") else state.get("status", "Idle"))
        )
        self.status_label.setStyleSheet("color: red;" if is_strained else "color: white;")
        self.status_label.setText(status_text)
        self.buffer_label.setStyleSheet("color: red;" if is_strained else "color: gray;")

        input_q_len = state.get("input_buffer_len_ms", 0)
        output_q_len = state.get("output_buffer_len_ms", 0)
        self.buffer_label.setText(f"In: {input_q_len:.0f}ms | Out: {output_q_len:.0f}ms")

        info = state.get("info", {})
        info_text = (
            f"SR: {info.get('sr')} | F0: {info.get('f0')} | Half: {info.get('is_half')}"
            if info
            else "Info: Awaiting models"
        )
        self.info_label.setText(info_text)
        super().updateFromLogic()


# ==============================================================================
# 2. Corrected Node Logic Class
# ==============================================================================
class RVCUnifiedNode(Node):
    NODE_TYPE = "RVC Unified Pipeline"
    UI_CLASS = RVCUnifiedNodeItem
    CATEGORY = "RVC"
    DESCRIPTION = "A self-contained RVC pipeline for robust, real-time voice conversion. Combines pitch, content, and inference into one node to prevent synchronization issues."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        # --- MODIFIED: Sockets now use torch.Tensor ---
        self.add_input("audio_in", data_type=torch.Tensor)
        self.add_output("audio_out", data_type=torch.Tensor)

        self._model_paths = {"rvc": None, "rmvpe": None}
        self._rvc_session: Optional[onnxruntime.InferenceSession] = None
        self._hubert_model: Optional[RVC_Hubert] = None
        self._rmvpe_session: Optional[onnxruntime.InferenceSession] = None

        self._device: int = -1
        self._speaker_id: int = 0
        self._pitch_shift: float = 0.0
        self._vad_enabled: bool = True
        self._silent_threshold: float = 0.001
        self._chunk_size_ms: int = 320
        self._crossfade_ms: int = 100
        self._sola_search_ms: int = 10
        self._extra_conversion_ms: int = 100
        self._metadata: Dict[str, Any] = {}
        self._is_half: bool = False
        self._model_sr: int = DEFAULT_SAMPLERATE
        self.f0_min = 50.0
        self.f0_max = 1100.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        self._status: str = "Idle"
        self._is_processing = False
        self._is_strained = False
        # --- MODIFIED: Buffers are now torch.Tensors ---
        self._input_buffer = torch.tensor([], dtype=DEFAULT_DTYPE)
        self._output_buffer = torch.tensor([], dtype=DEFAULT_DTYPE)
        self._last_valid_output = torch.zeros(DEFAULT_BLOCKSIZE, dtype=DEFAULT_DTYPE)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # SOLA buffers remain NumPy as the algorithm is NumPy-based
        self._sola_buffer: Optional[np.ndarray] = None
        self._np_prev_strength: Optional[np.ndarray] = None
        self._np_cur_strength: Optional[np.ndarray] = None
        self._last_crossfade_ms = -1
        self._generate_strength_curves()

    def _are_all_models_loaded(self):
        return all([self._rvc_session, self._hubert_model, self._rmvpe_session])

    def _load_all_models(self):
        with self._lock:
            rvc_path, rmvpe_path = (self._model_paths["rvc"], self._model_paths["rmvpe"])
            gpu = self._device

        def load_onnx(path, model_name):
            if not path or not os.path.exists(path):
                return None
            logger.info(f"[{self.name}] Loading {model_name} ONNX model...")
            providers, opts = self._get_onnx_provider(gpu)
            so = onnxruntime.SessionOptions()
            so.log_severity_level = 3
            return onnxruntime.InferenceSession(path, sess_options=so, providers=providers, provider_options=opts)

        try:
            rvc_session = load_onnx(rvc_path, "RVC")
            rmvpe_session = load_onnx(rmvpe_path, "RMVPE")
            hubert_model = RVC_Hubert(gpu)

            metadata, is_half, model_sr = {}, False, DEFAULT_SAMPLERATE
            if rvc_session:
                meta = rvc_session.get_modelmeta()
                if "metadata" in meta.custom_metadata_map:
                    metadata = json.loads(meta.custom_metadata_map["metadata"])
                    is_half = rvc_session.get_inputs()[0].type == "tensor(float16)"
                    model_sr = metadata.get("samplingRate")
                    if not model_sr:
                        raise ValueError("ONNX model metadata is missing 'samplingRate'.")
                else:
                    raise ValueError("ONNX model is missing required metadata.")

            with self._lock:
                self._rvc_session, self._hubert_model, self._rmvpe_session = rvc_session, hubert_model, rmvpe_session
                self._metadata, self._is_half, self._model_sr = metadata, is_half, model_sr
                self._status = "Ready" if self._are_all_models_loaded() else "Awaiting Models"

        except Exception as e:
            logger.error(f"[{self.name}] Failed to load one or more models: {e}", exc_info=True)
            with self._lock:
                self._status = "Error loading models"

    def _get_onnx_provider(self, gpu: int):
        providers, opts = ["CPUExecutionProvider"], [{"intra_op_num_threads": 4}]
        if gpu >= 0 and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            try:
                if torch.cuda.is_available() and gpu < torch.cuda.device_count():
                    return ["CUDAExecutionProvider"], [{"device_id": gpu}]
            except Exception:
                pass
        return providers, opts

    def _generate_strength_curves(self):
        with self._lock:
            crossfade_samples = int(self._crossfade_ms / 1000 * DEFAULT_SAMPLERATE)
            if self._last_crossfade_ms == self._crossfade_ms:
                return
            if crossfade_samples <= 0:
                self._np_prev_strength = np.array([], dtype=np.float32)
                self._np_cur_strength = np.array([], dtype=np.float32)
            else:
                fade_range = np.arange(crossfade_samples) / crossfade_samples
                self._np_prev_strength = np.cos(fade_range * 0.5 * np.pi) ** 2
                self._np_cur_strength = np.cos((1 - fade_range) * 0.5 * np.pi) ** 2
            self._last_crossfade_ms = self._crossfade_ms
            self._sola_buffer = None

    def _pipeline_worker_loop(self):
        logger.info(f"[{self.name}] Unified pipeline worker started.")
        while not self._stop_event.is_set():
            with self._lock:
                chunk_samples = int(self._chunk_size_ms / 1000 * DEFAULT_SAMPLERATE)
                crossfade_samples = int(self._crossfade_ms / 1000 * DEFAULT_SAMPLERATE)
                sola_search_samples = int(self._sola_search_ms / 1000 * DEFAULT_SAMPLERATE)
                extra_samples = int(self._extra_conversion_ms / 1000 * DEFAULT_SAMPLERATE)

                required_input_len = chunk_samples + crossfade_samples + sola_search_samples + extra_samples

                # --- MODIFIED: Use tensor shape for length check ---
                has_enough_data = self._input_buffer.shape[0] >= required_input_len
                self._is_strained = self._input_buffer.shape[0] > required_input_len * MAX_BUFFER_CHUNKS_INPUT

            if has_enough_data and self._are_all_models_loaded():
                try:
                    with self._lock:
                        # --- MODIFIED: Slice the tensor buffer ---
                        audio_to_process_tensor = self._input_buffer[-required_input_len:]
                        self._input_buffer = self._input_buffer[-(required_input_len - chunk_samples) :]

                        hubert_model, rmvpe_session, rvc_session = (
                            self._hubert_model,
                            self._rmvpe_session,
                            self._rvc_session,
                        )
                        pitch_shift, speaker_id, metadata, is_half = (
                            self._pitch_shift,
                            self._speaker_id,
                            self._metadata,
                            self._is_half,
                        )
                        silent_threshold, vad_enabled, model_sr = (
                            self._silent_threshold,
                            self._vad_enabled,
                            self._model_sr,
                        )

                    # --- MODIFIED: VAD check on the tensor ---
                    vad_check_region = audio_to_process_tensor[extra_samples : extra_samples + chunk_samples]
                    rms = torch.sqrt(torch.mean(torch.square(vad_check_region)) + EPSILON)

                    if vad_enabled and rms < silent_threshold:
                        stable_audio_out = np.zeros(chunk_samples, dtype=np.float32)
                    else:
                        # --- MODIFIED: Convert tensor to NumPy array for processing ---
                        audio_to_process_np = audio_to_process_tensor.numpy()

                        audio_16k = resampy.resample(
                            np.ascontiguousarray(audio_to_process_np),
                            sr_orig=DEFAULT_SAMPLERATE,
                            sr_new=RVC_REQUIRED_SR,
                            filter="kaiser_fast",
                        )
                        pitchf = rmvpe_session.run(
                            ["pitchf"],
                            {"waveform": audio_16k[np.newaxis, :], "threshold": np.array([0.3]).astype(np.float32)},
                        )[0].squeeze()
                        pitchf *= pow(2, pitch_shift / 12)
                        f0_mel = 1127.0 * np.log(1.0 + pitchf / 700.0)
                        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
                            self.f0_mel_max - self.f0_mel_min
                        ) + 1
                        f0_mel[f0_mel <= 1] = 1
                        f0_mel[f0_mel > 255] = 255
                        f0_coarse = np.rint(f0_mel).astype(np.int64)
                        emb_layer = 9 if metadata.get("version", "v1") == "v1" else 12
                        features = hubert_model.encode(audio_16k, emb_layer)
                        min_len = min(len(features), len(f0_coarse))
                        feats_dtype = np.float16 if is_half else np.float32
                        input_dict = {
                            "feats": np.expand_dims(features[:min_len], 0).astype(feats_dtype),
                            "p_len": np.array([min_len], dtype=np.int64),
                            "sid": np.array([speaker_id], dtype=np.int64),
                        }
                        if metadata.get("f0", 1) == 1:
                            input_dict["pitch"] = np.expand_dims(f0_coarse[:min_len], 0)
                            input_dict["pitchf"] = np.expand_dims(pitchf[:min_len], 0).astype(np.float32)

                        audio_out_model_sr = rvc_session.run(["audio"], input_dict)[0].squeeze()

                        extra_samples_at_model_sr = int(self._extra_conversion_ms / 1000 * model_sr)
                        stable_audio_out_model_sr = audio_out_model_sr[extra_samples_at_model_sr:]

                        if len(stable_audio_out_model_sr) == 0:
                            stable_audio_out = np.zeros(chunk_samples, dtype=np.float32)
                        elif model_sr != DEFAULT_SAMPLERATE:
                            stable_audio_out = resampy.resample(
                                np.ascontiguousarray(stable_audio_out_model_sr),
                                sr_orig=model_sr,
                                sr_new=DEFAULT_SAMPLERATE,
                                filter="kaiser_fast",
                            )
                        else:
                            stable_audio_out = stable_audio_out_model_sr

                    self._perform_sola_stitching(
                        stable_audio_out, chunk_samples, crossfade_samples, sola_search_samples
                    )

                except Exception as e:
                    logger.error(f"[{self.name}] Error in pipeline worker: {e}", exc_info=True)
            else:
                time.sleep(0.005)
        logger.info(f"[{self.name}] Unified pipeline worker stopped.")

    def _perform_sola_stitching(
        self, stable_audio_out: np.ndarray, chunk_samples: int, crossfade_samples: int, sola_search_samples: int
    ):
        with self._lock:
            if self._sola_buffer is not None:
                search_region = stable_audio_out[: crossfade_samples + sola_search_samples]

                if len(search_region) == 0 or len(self._sola_buffer) == 0:
                    sola_offset = 0
                else:
                    cor_nom = np.convolve(search_region, np.flip(self._sola_buffer), "valid")
                    cor_den = np.sqrt(np.convolve(search_region**2, np.ones(crossfade_samples), "valid") + EPSILON)
                    if len(cor_nom) > 0 and len(cor_den) > 0:
                        sola_offset = np.argmax(cor_nom / cor_den)
                    else:
                        sola_offset = 0

                output_block_np = stable_audio_out[sola_offset : sola_offset + chunk_samples]

                if crossfade_samples > 0 and len(output_block_np) >= crossfade_samples:
                    min_len = min(len(output_block_np), len(self._sola_buffer), len(self._np_cur_strength))
                    output_block_np[:min_len] *= self._np_cur_strength[:min_len]
                    output_block_np[:min_len] += self._sola_buffer[:min_len]

                buffer_start_idx = sola_offset + chunk_samples
                sola_slice = stable_audio_out[buffer_start_idx : buffer_start_idx + crossfade_samples]
                padded_slice = (
                    np.pad(sola_slice, (0, crossfade_samples - len(sola_slice)), "constant")
                    if len(sola_slice) < crossfade_samples
                    else sola_slice[:crossfade_samples]
                )
                self._sola_buffer = padded_slice * self._np_prev_strength
            else:
                logger.info(f"[{self.name}] Priming SOLA buffer.")
                output_block_np = stable_audio_out[:chunk_samples]
                sola_slice = stable_audio_out[chunk_samples : chunk_samples + crossfade_samples]
                padded_slice = (
                    np.pad(sola_slice, (0, crossfade_samples - len(sola_slice)), "constant")
                    if len(sola_slice) < crossfade_samples
                    else sola_slice[:crossfade_samples]
                )
                self._sola_buffer = padded_slice * self._np_prev_strength

            # --- MODIFIED: Convert output NumPy block to a tensor and append to buffer ---
            output_block_tensor = torch.from_numpy(output_block_np.astype(np.float32))
            self._output_buffer = torch.cat((self._output_buffer, output_block_tensor))

    @Slot(str, str)
    def set_model_path(self, model_type: str, path: str):
        if model_type in self._model_paths:
            with self._lock:
                self._model_paths[model_type] = path
            self._load_all_models()
        else:
            logger.warning(f"[{self.name}] Unknown model type specified: {model_type}")

    @Slot(int)
    def set_device(self, device_id: int):
        with self._lock:
            self._device = device_id
        self._load_all_models()

    @Slot(int)
    def set_speaker_id(self, sid: int):
        with self._lock:
            self._speaker_id = sid

    @Slot(float)
    def set_pitch_shift(self, pitch: float):
        with self._lock:
            self._pitch_shift = pitch

    @Slot(bool)
    def set_vad_enabled(self, enabled: bool):
        with self._lock:
            self._vad_enabled = enabled

    @Slot(float)
    def set_silent_threshold(self, threshold: float):
        with self._lock:
            self._silent_threshold = threshold

    @Slot(int)
    def set_chunk_size_ms(self, ms: int):
        with self._lock:
            self._chunk_size_ms = ms

    @Slot(int)
    def set_crossfade_ms(self, ms: int):
        with self._lock:
            self._crossfade_ms = ms
        self._generate_strength_curves()

    @Slot(int)
    def set_sola_search_ms(self, ms: int):
        with self._lock:
            self._sola_search_ms = ms

    @Slot(int)
    def set_extra_conversion_ms(self, ms: int):
        with self._lock:
            self._extra_conversion_ms = ms

    def _get_current_state_snapshot(self) -> Dict:
        info = (
            {"sr": self._model_sr, "f0": self._metadata.get("f0", 1) == 1, "is_half": self._is_half}
            if self._are_all_models_loaded()
            else None
        )
        return {
            "rvc_path": self._model_paths["rvc"],
            "rmvpe_path": self._model_paths["rmvpe"],
            "device": self._device,
            "speaker_id": self._speaker_id,
            "pitch_shift": self._pitch_shift,
            "vad_enabled": self._vad_enabled,
            "silent_threshold": self._silent_threshold,
            "chunk_size_ms": self._chunk_size_ms,
            "crossfade_ms": self._crossfade_ms,
            "sola_search_ms": self._sola_search_ms,
            "extra_conversion_ms": self._extra_conversion_ms,
            "is_processing": self._is_processing,
            "is_strained": self._is_strained,
            "status": self._status,
            "info": info,
            "input_buffer_len_ms": self._input_buffer.shape[0] / DEFAULT_SAMPLERATE * 1000,
            "output_buffer_len_ms": self._output_buffer.shape[0] / DEFAULT_SAMPLERATE * 1000,
        }

    def process(self, input_data: dict) -> dict:
        audio_in = input_data.get("audio_in")
        with self._lock:
            # --- MODIFIED: Process input tensor ---
            if isinstance(audio_in, torch.Tensor):
                mono_signal = torch.mean(audio_in, dim=0) if audio_in.ndim > 1 else audio_in
                self._input_buffer = torch.cat((self._input_buffer, mono_signal.to(DEFAULT_DTYPE)))

            if self._output_buffer.shape[0] >= DEFAULT_BLOCKSIZE:
                final_output_block = self._output_buffer[:DEFAULT_BLOCKSIZE]
                self._output_buffer = self._output_buffer[DEFAULT_BLOCKSIZE:]
                self._last_valid_output = final_output_block
            else:
                final_output_block = self._last_valid_output

        # --- MODIFIED: Reshape tensor for output (1, samples) ---
        return {"audio_out": final_output_block.unsqueeze(0)}

    def start(self):
        with self._lock:
            crossfade_samples = int(self._crossfade_ms / 1000 * DEFAULT_SAMPLERATE)
            sola_search_samples = int(self._sola_search_ms / 1000 * DEFAULT_SAMPLERATE)
            extra_samples = int(self._extra_conversion_ms / 1000 * DEFAULT_SAMPLERATE)
            initial_pad_len = extra_samples + crossfade_samples + sola_search_samples

            # --- MODIFIED: Use torch to initialize buffers ---
            self._input_buffer = torch.zeros(initial_pad_len, dtype=DEFAULT_DTYPE)
            self._output_buffer = torch.tensor([], dtype=DEFAULT_DTYPE)
            self._sola_buffer = None
            self._last_valid_output = torch.zeros(DEFAULT_BLOCKSIZE, dtype=DEFAULT_DTYPE)
            self._is_processing = True
            self._is_strained = False

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._pipeline_worker_loop, daemon=True)
        self._worker_thread.start()
        self._load_all_models()

    def stop(self):
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
        with self._lock:
            self._is_processing = False

    def remove(self):
        self.stop()
        super().remove()

    def serialize_extra(self) -> dict:
        with self._lock:
            return {
                "rvc_model_path": self._model_paths["rvc"],
                "rmvpe_model_path": self._model_paths["rmvpe"],
                "device": self._device,
                "speaker_id": self._speaker_id,
                "pitch_shift": self._pitch_shift,
                "vad_enabled": self._vad_enabled,
                "silent_threshold": self._silent_threshold,
                "chunk_size_ms": self._chunk_size_ms,
                "crossfade_ms": self._crossfade_ms,
                "sola_search_ms": self._sola_search_ms,
                "extra_conversion_ms": self._extra_conversion_ms,
            }

    def deserialize_extra(self, data: dict):
        self._model_paths["rvc"] = data.get("rvc_model_path")
        self._model_paths["rmvpe"] = data.get("rmvpe_model_path")
        self._device = data.get("device", -1)
        self._speaker_id = data.get("speaker_id", 0)
        self._pitch_shift = data.get("pitch_shift", 0.0)
        self._vad_enabled = data.get("vad_enabled", True)
        self._silent_threshold = data.get("silent_threshold", 0.001)
        self._chunk_size_ms = data.get("chunk_size_ms", 320)
        self._crossfade_ms = data.get("crossfade_ms", 100)
        self._sola_search_ms = data.get("sola_search_ms", 10)
        self._extra_conversion_ms = data.get("extra_conversion_ms", 100)
        self._generate_strength_curves()
