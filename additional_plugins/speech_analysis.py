import torch
import numpy as np
import threading
import logging
import time
from typing import Dict, Optional, Tuple

# The librosa library is required for the 'PYIN' and 'YIN' methods to function.
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("speech_analysis.py: 'librosa' not found. PYIN/YIN methods will be disabled.")

# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE

# --- Qt Imports ---
from PySide6.QtCore import Qt, Slot, QSignalBlocker
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QComboBox

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Node-Specific Constants ---
REQUIRED_FRAME_LENGTH = 2048
UI_UPDATE_INTERVAL_S = 0.1
VAD_ENERGY_THRESHOLD = 1e-6
F0_SMOOTHING_ALPHA = 0.3


# ==============================================================================
# 1. Custom UI Class (F0EstimatorNodeItem)
# ==============================================================================
class F0EstimatorNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "F0EstimatorNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(5)

        layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItem("PYIN (Accurate)", "pyin")
        self.method_combo.addItem("YIN (Classic)", "yin")
        self.method_combo.addItem("Autocorrelation (Fast)", "autocorr")
        layout.addWidget(self.method_combo)

        self.f0_label = QLabel("F0: ... Hz")
        self.confidence_label = QLabel("Confidence: ...")
        self.f0_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.f0_label)
        layout.addWidget(self.confidence_label)

        if not LIBROSA_AVAILABLE:
            error_label = QLabel("Note: 'PYIN' & 'YIN' require\n(pip install librosa)")
            error_label.setStyleSheet("color: orange;")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setWordWrap(True)
            layout.addWidget(error_label)

        self.setContentWidget(self.container_widget)

        self.method_combo.currentIndexChanged.connect(self._on_method_change)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        self.updateFromLogic()

    @Slot(int)
    def _on_method_change(self, index: int):
        method_key = self.method_combo.itemData(index)
        self.node_logic.set_estimation_method(method_key)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        method = state.get("method", "pyin")
        with QSignalBlocker(self.method_combo):
            index = self.method_combo.findData(method)
            if index != -1:
                self.method_combo.setCurrentIndex(index)

        for key in ["pyin", "yin"]:
            key_index = self.method_combo.findData(key)
            if key_index != -1:
                self.method_combo.model().item(key_index).setEnabled(LIBROSA_AVAILABLE)

        f0 = state.get("f0_hz")
        confidence = state.get("confidence")
        if f0 is not None and f0 > 0:
            self.f0_label.setText(f"F0: {f0:.2f} Hz")
        else:
            self.f0_label.setText("F0: Unvoiced")
        if confidence is not None:
            self.confidence_label.setText(f"Confidence: {confidence:.1%}")
        else:
            self.confidence_label.setText("Confidence: ...")

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ==============================================================================
# 2. Node Logic Class (F0EstimatorNode)
# ==============================================================================
class F0EstimatorNode(Node):
    NODE_TYPE = "Speech F0 Estimator"
    UI_CLASS = F0EstimatorNodeItem
    CATEGORY = "Analysis"
    DESCRIPTION = "Estimates F0 using selectable methods (PYIN, YIN, or Autocorrelation)."

    class _PitchEngine:
        """Helper class to contain the pure DSP for autocorrelation."""
        def __init__(self, sample_rate: int):
            self.sample_rate = sample_rate
            self.min_f0 = 80.0
            self.max_f0 = 800.0
            self.max_lag = int(sample_rate / self.min_f0)
            self.min_lag = int(sample_rate / self.max_f0)
            self.pre_emphasis = 0.97
            self.prev_sample = 0.0

        def _preprocess(self, audio_block: np.ndarray) -> np.ndarray:
            filtered = np.zeros_like(audio_block)
            filtered[0] = audio_block[0] - self.pre_emphasis * self.prev_sample
            filtered[1:] = audio_block[1:] - self.pre_emphasis * audio_block[:-1]
            self.prev_sample = audio_block[-1]
            return filtered

        def autocorrelation_f0(self, audio_block: np.ndarray) -> "Tuple[float, float]":
            processed = self._preprocess(audio_block)
            autocorr = np.correlate(processed, processed, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]

            if autocorr[0] <= 0: return 0.0, 0.0
            autocorr /= autocorr[0]

            search_range = autocorr[self.min_lag : min(self.max_lag, len(autocorr))]
            if len(search_range) == 0: return 0.0, 0.0

            peak_idx = np.argmax(search_range)
            peak_value = search_range[peak_idx]

            confidence = np.clip(peak_value / 0.8, 0.0, 1.0)
            if peak_value < 0.3: return 0.0, confidence

            lag = peak_idx + self.min_lag
            f0 = self.sample_rate / lag
            return f0, confidence

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()

        self.add_input("in", data_type=torch.Tensor)
        self.add_output("f0_hz", data_type=float)
        self.add_output("confidence", data_type=float)

        self._lock = threading.Lock()
        self._method = "pyin" if LIBROSA_AVAILABLE else "autocorr"
        self._buffer = np.array([], dtype=np.float32)
        self._pitch_engine = self._PitchEngine(DEFAULT_SAMPLERATE)
        
        # --- State for UI update thread ---
        self._latest_f0_hz = 0.0
        self._latest_confidence = 0.0
        self._smoothed_f0 = 0.0
        
        # --- UI update thread management ---
        self._ui_update_thread: Optional[threading.Thread] = None
        self._stop_ui_thread_event = threading.Event()

    @Slot(str)
    def set_estimation_method(self, method: str):
        """Thread-safe setter called by the UI."""
        state_to_emit = None
        with self._lock:
            if method != self._method:
                self._method = method
                self._smoothed_f0 = 0.0
                self._latest_f0_hz = 0.0
                self._latest_confidence = 0.0
                state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def get_current_state_snapshot(self, locked: bool = False) -> Dict:
        """Gets a snapshot of the node's state for UI or serialization."""
        state = {
            "method": self._method,
            "f0_hz": self._latest_f0_hz,
            "confidence": self._latest_confidence,
        }
        if locked:
            return state
        with self._lock:
            return state

    def process(self, input_data: dict) -> dict:
        """DSP-only processing method. UI updates are handled by a separate thread."""
        audio_in = input_data.get("in")
        output_f0 = 0.0
        output_confidence = 0.0

        if not isinstance(audio_in, torch.Tensor) or audio_in.numel() == 0:
            return {"f0_hz": 0.0, "confidence": 0.0}

        try:
            mono_tensor = torch.mean(audio_in, dim=0) if audio_in.ndim > 1 else audio_in
            if torch.sum(mono_tensor**2) < VAD_ENERGY_THRESHOLD:
                with self._lock:
                    self._smoothed_f0 = 0.0
                    self._latest_f0_hz = 0.0
                    self._latest_confidence = 0.0
                return {"f0_hz": 0.0, "confidence": 0.0}

            mono_signal_np = mono_tensor.numpy().astype(np.float32)

            with self._lock:
                self._buffer = np.concatenate((self._buffer, mono_signal_np))
                if len(self._buffer) < REQUIRED_FRAME_LENGTH:
                    return {"f0_hz": self._smoothed_f0, "confidence": self._latest_confidence}
                analysis_chunk = self._buffer[:REQUIRED_FRAME_LENGTH]
                self._buffer = self._buffer[DEFAULT_BLOCKSIZE:]
                method = self._method

            raw_f0, confidence = 0.0, 0.0
            if method == "pyin" and LIBROSA_AVAILABLE:
                f0, _, probs = librosa.pyin(analysis_chunk, fmin=80, fmax=800, sr=DEFAULT_SAMPLERATE, frame_length=REQUIRED_FRAME_LENGTH, fill_na=0.0)
                raw_f0 = float(f0[0]) if f0 is not None and len(f0) > 0 else 0.0
                confidence = float(probs[0]) if probs is not None and len(probs) > 0 else 0.0
            elif method == "yin" and LIBROSA_AVAILABLE:
                f0 = librosa.yin(analysis_chunk, fmin=80, fmax=800, sr=DEFAULT_SAMPLERATE, frame_length=REQUIRED_FRAME_LENGTH)
                raw_f0 = float(f0[0]) if f0 is not None and len(f0) > 0 and not np.isnan(f0[0]) else 0.0
                confidence = 1.0 if raw_f0 > 0 else 0.0
            else: # Autocorrelation fallback
                raw_f0, confidence = self._pitch_engine.autocorrelation_f0(analysis_chunk)

            with self._lock:
                if self._smoothed_f0 > 0:
                    output_f0 = F0_SMOOTHING_ALPHA * raw_f0 + (1 - F0_SMOOTHING_ALPHA) * self._smoothed_f0
                else:
                    output_f0 = raw_f0
                self._smoothed_f0 = output_f0 if output_f0 > 0 else 0.0
                output_confidence = confidence
                # Store latest raw values for the UI thread
                self._latest_f0_hz = output_f0
                self._latest_confidence = confidence

        except Exception as e:
            logger.error(f"[{self.name}] Error during F0 estimation: {e}", exc_info=True)
            with self._lock: self._smoothed_f0 = 0.0

        return {"f0_hz": output_f0, "confidence": output_confidence}

    def _ui_updater_loop(self):
        """Dedicated thread loop to send throttled updates to the UI."""
        while not self._stop_ui_thread_event.is_set():
            state_to_emit = self.get_current_state_snapshot()
            self.emitter.stateUpdated.emit(state_to_emit)
            time.sleep(UI_UPDATE_INTERVAL_S)

    def start(self):
        super().start()
        with self._lock:
            self._buffer = np.array([], dtype=np.float32)
            self._smoothed_f0 = 0.0
            self._latest_f0_hz = 0.0
            self._latest_confidence = 0.0
            self._pitch_engine.prev_sample = 0.0
        
        self._stop_ui_thread_event.clear()
        self._ui_update_thread = threading.Thread(target=self._ui_updater_loop, daemon=True)
        self._ui_update_thread.start()

    def stop(self):
        super().stop()
        self._stop_ui_thread_event.set()
        if self._ui_update_thread:
            self._ui_update_thread.join(timeout=0.5)
        
        # Send a final cleared state to the UI
        with self._lock:
            state = {"method": self._method, "f0_hz": 0.0, "confidence": 0.0}
        self.emitter.stateUpdated.emit(state)

    def remove(self):
        self.stop()
        super().remove()

    def serialize_extra(self) -> dict:
        with self._lock:
            return {"method": self._method}

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._method = data.get("method", "pyin" if LIBROSA_AVAILABLE else "autocorr")