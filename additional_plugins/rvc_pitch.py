import numpy as np
import threading
import logging
import time
import collections
from typing import Dict, Optional

# --- Core Dependencies ---
try:
    import onnxruntime
    import torch
    import resampy
    RVC_DEPS_AVAILABLE = True
except ImportError:
    RVC_DEPS_AVAILABLE = False

# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from constants import DEFAULT_SAMPLERATE

# --- Qt Imports ---
from PySide6.QtCore import Qt, Signal, Slot, QObject, QSignalBlocker, QTimer
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QComboBox, QFileDialog, QHBoxLayout

logger = logging.getLogger(__name__)

# --- Node-Specific Constants ---
REQUIRED_SR = 16000
HOP_SIZE = 160
# This is the amount of audio (at the original sample rate) needed to produce one RVC chunk
REQUIRED_INPUT_SAMPLES = int(960 * (DEFAULT_SAMPLERATE / REQUIRED_SR))
# The number of samples the model expects after resampling
MODEL_INPUT_LENGTH = 960 
HISTORY_FRAMES = (3840 // HOP_SIZE)
UI_UPDATE_INTERVAL_MS = 100 # For the UI timer

# ==============================================================================
# 1. State Emitter for UI Communication
# ==============================================================================
class RmvpeEmitter(QObject):
    stateUpdated = Signal(dict)

# ==============================================================================
# 2. Custom UI Class (RMVPEF0EstimatorNodeItem)
# ==============================================================================
class RMVPEF0EstimatorNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "RMVPEF0EstimatorNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)
        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(5)
        self.load_button = QPushButton("Load RMVPE Model")
        layout.addWidget(self.load_button)
        self.model_path_label = QLabel("Model: Not loaded")
        self.model_path_label.setStyleSheet("font-size: 9px; color: gray;"); self.model_path_label.setWordWrap(True)
        layout.addWidget(self.model_path_label)
        layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        layout.addWidget(self.device_combo)
        
        status_layout = QHBoxLayout()
        self.f0_label = QLabel("F0: ... Hz")
        self.f0_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.deque_label = QLabel("Deque: 0")
        self.deque_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        status_layout.addWidget(self.f0_label)
        status_layout.addWidget(self.deque_label)
        layout.addLayout(status_layout)

        if not RVC_DEPS_AVAILABLE:
            error_label = QLabel("Missing dependencies:\n(onnxruntime, torch, resampy)")
            error_label.setStyleSheet("color: orange;"); error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.load_button.setEnabled(False); self.device_combo.setEnabled(False)
            layout.addWidget(error_label)
        self.setContentWidget(self.container_widget)

        self.load_button.clicked.connect(self._on_load_model_clicked)
        self.device_combo.currentIndexChanged.connect(self._on_device_change)
        
        # This timer will safely poll the logic for UI updates
        self.ui_updater = QTimer(self)
        self.ui_updater.setInterval(UI_UPDATE_INTERVAL_MS)
        self.ui_updater.timeout.connect(self.updateFromLogic)
        self.ui_updater.start()

    def _populate_devices(self):
        current_data = self.device_combo.currentData()
        self.device_combo.clear()
        self.device_combo.addItem("CPU", -1)
        if RVC_DEPS_AVAILABLE and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            try:
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()): self.device_combo.addItem(f"GPU {i}: {torch.cuda.get_device_name(i)}", i)
            except Exception as e: logger.warning(f"Could not enumerate CUDA devices: {e}")
        
        index = self.device_combo.findData(current_data)
        if index != -1: self.device_combo.setCurrentIndex(index)


    @Slot()
    def _on_load_model_clicked(self):
        parent_widget = self.scene().views()[0] if self.scene() and self.scene().views() else None
        file_path, _ = QFileDialog.getOpenFileName(parent_widget, "Select RMVPE ONNX Model", "", "ONNX Models (*.onnx)")
        if file_path: self.node_logic.set_model_path(file_path)

    @Slot(int)
    def _on_device_change(self, index: int):
        if index != -1: self.node_logic.set_device(self.device_combo.itemData(index))

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        model_path = state.get("model_path")
        if model_path: self.model_path_label.setText(f"Model: ...{model_path[-30:]}")
        else: self.model_path_label.setText("Model: Not loaded")
        
        with QSignalBlocker(self.device_combo):
            device = state.get("device", -1)
            # Populate the device list if it hasn't been populated yet.
            if self.device_combo.count() <= 1:
                self._populate_devices()
            
            index = self.device_combo.findData(device)
            if index != -1: self.device_combo.setCurrentIndex(index)

        f0 = state.get("f0_hz", 0.0)
        if f0 > 0: self.f0_label.setText(f"F0: {f0:.1f} Hz")
        else: self.f0_label.setText("F0: Unvoiced")

        deque_size = state.get("deque_size", 0)
        self.deque_label.setText(f"Deque: {deque_size}")
        
        warn_threshold = REQUIRED_INPUT_SAMPLES * 2
        crit_threshold = REQUIRED_INPUT_SAMPLES * 5
        if deque_size > crit_threshold:
            self.deque_label.setStyleSheet("color: red;")
        elif deque_size > warn_threshold:
            self.deque_label.setStyleSheet("color: orange;")
        else:
            self.deque_label.setStyleSheet("color: gray;")

        is_processing = state.get("is_processing", False)
        self.load_button.setEnabled(not is_processing)
        self.device_combo.setEnabled(not is_processing)
        super().updateFromLogic()


# ==============================================================================
# 3. Node Logic Class (RMVPEF0EstimatorNode)
# ==============================================================================
class RMVPEF0EstimatorNode(Node):
    NODE_TYPE = "RVC RMVPE Pitch"
    UI_CLASS = RMVPEF0EstimatorNodeItem
    CATEGORY = "RVC"
    DESCRIPTION = "Estimates F0 using the RMVPE ONNX model for RVC."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=np.ndarray)
        self.add_input("f0_up_key", data_type=float)
        self.add_output("f0_hz", data_type=float)
        self.add_output("f0_coarse", data_type=np.ndarray)
        self.add_output("pitchf", data_type=np.ndarray)  # <--- FIX 1: ADDED THE MISSING OUTPUT
        
        self._lock = threading.Lock()
        self._model_path: Optional[str] = None
        self._device: int = -1
        self._onnx_session: Optional[onnxruntime.InferenceSession] = None
        self._is_processing = False
        self._last_f0_hz = 0.0
        self._audio_deque = collections.deque()
        self._results_deque = collections.deque(maxlen=10)
        self._pitchf_buffer = np.zeros(HISTORY_FRAMES, dtype=np.float32)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self.f0_min = 50.0
        self.f0_max = 1100.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def _pitch_estimation_loop(self):
        logger.info(f"[{self.name}] Pitch estimation worker thread started.")
        while not self._stop_event.is_set():
            with self._lock:
                session = self._onnx_session
                has_enough_data = len(self._audio_deque) >= REQUIRED_INPUT_SAMPLES

            if has_enough_data and session:
                try:

                    with self._lock:
                        # Create the potentially non-contiguous array from the deque
                        audio_chunk_non_contiguous = np.array([self._audio_deque.popleft() for _ in range(REQUIRED_INPUT_SAMPLES)])
                    # Explicitly create a C-contiguous copy.
                    audio_chunk = np.ascontiguousarray(audio_chunk_non_contiguous, dtype=np.float32)

                    resampled = resampy.resample(audio_chunk, sr_orig=DEFAULT_SAMPLERATE, sr_new=REQUIRED_SR)
                    
                    # Enforce fixed input size for consistent performance ---
                    if len(resampled) < MODEL_INPUT_LENGTH:
                        resampled = np.pad(resampled, (0, MODEL_INPUT_LENGTH - len(resampled)))
                    else:
                        resampled = resampled[:MODEL_INPUT_LENGTH]

                    audio_batch = np.expand_dims(resampled, axis=0).astype(np.float32)

                    onnx_out = session.run(["pitchf"], {"waveform": audio_batch, "threshold": np.array([0.3]).astype(np.float32)})
                    pitchf = onnx_out[0].squeeze()
                    
                    num_new_frames = len(resampled) // HOP_SIZE
                    
                    with self._lock:
                        self._pitchf_buffer = np.roll(self._pitchf_buffer, -num_new_frames)
                        self._pitchf_buffer[-num_new_frames:] = pitchf[-num_new_frames:]
                        self._results_deque.append((self._pitchf_buffer.copy(), num_new_frames))

                except Exception as e:
                    logger.error(f"[{self.name}] Error in worker thread: {e}", exc_info=True)
                    time.sleep(0.1)
            else:
                time.sleep(0.01)
        logger.info(f"[{self.name}] Pitch estimation worker thread stopped.")


    def _load_model(self):
        with self._lock:
            if self._is_processing:
                logger.warning(f"[{self.name}] Denied request to load model while processing is active.")
                return
            model_path, device = self._model_path, self._device
        
        session = None
        try:
            if RVC_DEPS_AVAILABLE and model_path:
                providers, opts = self._get_onnx_provider(device)
                so = onnxruntime.SessionOptions(); so.log_severity_level = 3
                logger.info(f"[{self.name}] Loading RMVPE model '{model_path}' for device {device} ({providers[0]})")
                session = onnxruntime.InferenceSession(model_path, sess_options=so, providers=providers, provider_options=opts)
                
                # --- REVISED: More robust warm-up ---
                logger.info(f"[{self.name}] Warming up the RMVPE model...")
                # Create dummy inputs with the correct shape and type
                dummy_waveform = np.random.randn(1, MODEL_INPUT_LENGTH).astype(np.float32)
                dummy_threshold = np.array([0.3], dtype=np.float32)
                
                # Run inference 3 times to trigger JIT compilation and memory allocation
                for _ in range(3):
                    _ = session.run(["pitchf"], {"waveform": dummy_waveform, "threshold": dummy_threshold})

                logger.info(f"[{self.name}] RMVPE model is warmed up and ready.")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to load or warm up ONNX model: {e}", exc_info=True)
        
        with self._lock: self._onnx_session = session

    def _get_onnx_provider(self, gpu: int):
        providers, opts = ["CPUExecutionProvider"], [{"intra_op_num_threads": 4}]
        if gpu >= 0 and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            try:
                if torch.cuda.is_available() and gpu < torch.cuda.device_count():
                    return ["CUDAExecutionProvider"], [{"device_id": gpu}]
            except Exception: pass
        return providers, opts

    @Slot(str)
    def set_model_path(self, path: str):
        with self._lock:
            if self._is_processing:
                logger.warning(f"[{self.name}] Cannot change model path while processing.")
                return
            self._model_path = path
        self._load_model()

    @Slot(int)
    def set_device(self, device_id: int):
        with self._lock:
            if self._is_processing:
                logger.warning(f"[{self.name}] Cannot change device while processing.")
                return
            if self._device == device_id: return
            self._device = device_id
        self._load_model()
    
    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return {
                "model_path": self._model_path, 
                "device": self._device, 
                "f0_hz": self._last_f0_hz, 
                "is_processing": self._is_processing,
                "deque_size": len(self._audio_deque)
            }

    def process(self, input_data: dict) -> dict:
        audio_in = input_data.get("in")
        
        # FIX: Explicitly handle the case where the input is disconnected (None)
        f0_up_key_input = input_data.get("f0_up_key")
        f0_up_key = 0.0 if f0_up_key_input is None else f0_up_key_input

        # --- Producer part: add audio to deque ---
        if audio_in is not None:
            mono_signal = np.mean(audio_in, axis=1) if audio_in.ndim > 1 else audio_in
            with self._lock:
                if self._onnx_session is not None:
                    self._audio_deque.extend(mono_signal.astype(np.float32).tolist())

        # --- Consumer part: get latest result from worker ---
        latest_result = None
        with self._lock:
            if self._results_deque:
                latest_result = self._results_deque.popleft()

        if latest_result is None:
            # Return None for all array outputs if no new data
            return {"f0_hz": self._last_f0_hz, "f0_coarse": None, "pitchf": None}
        
        pitchf_buffer, num_new_frames = latest_result
        
        # Apply real-time pitch shift to get the final continuous pitch in Hz
        f0 = pitchf_buffer * pow(2, f0_up_key / 12)

        # Post-processing for coarse F0
        f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int64)

        with self._lock:
            voiced_f0 = f0[f0 > 0]
            self._last_f0_hz = np.mean(voiced_f0) if len(voiced_f0) > 0 else 0.0
        
        return {
            "f0_hz": self._last_f0_hz, 
            "f0_coarse": f0_coarse[-num_new_frames:],
            "pitchf": f0[-num_new_frames:] # Return the continuous F0 in Hz
        }
 
    def start(self):
        super().start()
        with self._lock:
            self._audio_deque.clear()
            self._results_deque.clear()
            self._pitchf_buffer.fill(0)
            self._last_f0_hz = 0.0
            self._is_processing = True

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._pitch_estimation_loop, daemon=True)
        self._worker_thread.start()

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

    def remove(self):
        self.stop()
        super().remove()

    def serialize_extra(self) -> dict:
        with self._lock: return {"model_path": self._model_path, "device": self._device}

    def deserialize_extra(self, data: dict):
        self._device = data.get("device", -1)
        self._model_path = data.get("model_path")
        self._load_model()