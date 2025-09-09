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

try:
    import swift_f0

    SWIFT_F0_AVAILABLE = True
except ImportError:
    SWIFT_F0_AVAILABLE = False
    logging.warning("speech_analysis.py: 'swift_f0' not found. SwiftF0 method will be disabled.")

try:
    import torchcrepe

    TORCHCREPE_AVAILABLE = True
except ImportError:
    TORCHCREPE_AVAILABLE = False
    logging.warning("speech_analysis.py: 'torchcrepe' not found. TorchCrepe method will be disabled.")

# --- Import for Resampling (hard dependency for this plugin) ---
import torchaudio.transforms as T


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
ANALYSIS_SAMPLERATE = 16000
ANALYSIS_FRAME_LENGTH = 1024
REQUIRED_FRAME_LENGTH = int(np.ceil(ANALYSIS_FRAME_LENGTH * DEFAULT_SAMPLERATE / ANALYSIS_SAMPLERATE))

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
        self.method_combo.addItem("SwiftF0 (ML-based)", "swift-f0")
        self.method_combo.addItem("TorchCrepe (ML-based)", "torchcrepe")
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

        if not SWIFT_F0_AVAILABLE:
            error_label = QLabel("Note: 'SwiftF0' requires\n(pip install swift-f0)")
            error_label.setStyleSheet("color: orange;")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setWordWrap(True)
            layout.addWidget(error_label)

        if not TORCHCREPE_AVAILABLE:
            error_label = QLabel("Note: 'TorchCrepe' requires\n(pip install torchcrepe)")
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

        swift_index = self.method_combo.findData("swift-f0")
        if swift_index != -1:
            self.method_combo.model().item(swift_index).setEnabled(SWIFT_F0_AVAILABLE)

        torchcrepe_index = self.method_combo.findData("torchcrepe")
        if torchcrepe_index != -1:
            self.method_combo.model().item(torchcrepe_index).setEnabled(TORCHCREPE_AVAILABLE)

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
    DESCRIPTION = "Estimates F0 using selectable methods (PYIN, YIN, Swift-F0 or Autocorrelation)."

    class _PitchEngine:
        """Helper class to contain the pure DSP for autocorrelation using PyTorch."""

        def __init__(self, sample_rate: int):
            self.sample_rate = sample_rate
            self.min_f0 = 80.0
            self.max_f0 = 800.0
            self.max_lag = int(sample_rate / self.min_f0)
            self.min_lag = int(sample_rate / self.max_f0)
            self.pre_emphasis = 0.97
            self.prev_sample = torch.tensor(0.0, dtype=torch.float32)

        def _preprocess(self, audio_block: torch.Tensor) -> torch.Tensor:
            self.prev_sample = self.prev_sample.to(audio_block.device)
            shifted_block = torch.cat([self.prev_sample.unsqueeze(0), audio_block[:-1]])
            filtered = audio_block - self.pre_emphasis * shifted_block
            self.prev_sample = audio_block[-1]
            return filtered

        def autocorrelation_f0(self, audio_block: torch.Tensor) -> "Tuple[float, float]":
            processed = self._preprocess(audio_block)

            fft_len = 2 * len(processed)
            fft_val = torch.fft.rfft(processed, n=fft_len)
            psd = torch.abs(fft_val) ** 2
            autocorr = torch.fft.irfft(psd, n=fft_len)

            autocorr = autocorr[: len(processed)]

            if autocorr[0] <= 0:
                return 0.0, 0.0

            # --- FIX: Avoid in-place division to resolve memory overlap error ---
            # Add epsilon to prevent division by very small values
            autocorr = autocorr / (autocorr[0] + 1e-10)

            search_range = autocorr[self.min_lag : min(self.max_lag, len(autocorr))]
            if len(search_range) == 0:
                return 0.0, 0.0

            peak_idx = torch.argmax(search_range)
            peak_value = search_range[peak_idx]

            confidence = torch.clamp(peak_value / 0.8, 0.0, 1.0).item()
            if peak_value < 0.3:
                return 0.0, confidence

            lag = peak_idx + self.min_lag
            f0 = self.sample_rate / lag
            return f0.item(), confidence

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()

        self.add_input("in", data_type=torch.Tensor)
        self.add_output("f0_hz", data_type=float)
        self.add_output("confidence", data_type=float)

        self._lock = threading.Lock()
        self._method = "pyin" if LIBROSA_AVAILABLE else "autocorr"

        self._buffer = torch.tensor([], dtype=torch.float32)

        self._resampler = T.Resample(orig_freq=DEFAULT_SAMPLERATE, new_freq=ANALYSIS_SAMPLERATE, dtype=torch.float32)
        logger.info(
            f"F0 Estimator will resample audio from {DEFAULT_SAMPLERATE}Hz to {ANALYSIS_SAMPLERATE}Hz for analysis."
        )

        self._pitch_engine = self._PitchEngine(ANALYSIS_SAMPLERATE)

        if SWIFT_F0_AVAILABLE:
            self._swift_detector = swift_f0.SwiftF0(fmin=80.0, fmax=800.0, confidence_threshold=0.3)

        if TORCHCREPE_AVAILABLE:
            self._crepe_device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"TorchCrepe will use device: {self._crepe_device}")

        self._latest_f0_hz = 0.0
        self._latest_confidence = 0.0
        self._smoothed_f0 = 0.0

        self._ui_update_thread: Optional[threading.Thread] = None
        self._stop_ui_thread_event = threading.Event()

    @Slot(str)
    def set_estimation_method(self, method: str):
        state_to_emit = None
        with self._lock:
            if method != self._method:
                self._method = method
                self._smoothed_f0 = 0.0
                self._latest_f0_hz = 0.0
                self._latest_confidence = 0.0
                state_to_emit = self._get_current_state_snapshot_unlocked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_unlocked()

    def _get_current_state_snapshot_unlocked(self) -> Dict:
        return {
            "method": self._method,
            "f0_hz": self._latest_f0_hz,
            "confidence": self._latest_confidence,
        }

    def process(self, input_data: dict) -> dict:
        audio_in = input_data.get("in")
        output_f0 = 0.0
        output_confidence = 0.0

        if not isinstance(audio_in, torch.Tensor) or audio_in.numel() == 0:
            return {"f0_hz": 0.0, "confidence": 0.0}

        try:
            # Compute hop length for analysis consistency across methods
            analysis_hop_length = int(np.round(DEFAULT_BLOCKSIZE * ANALYSIS_SAMPLERATE / DEFAULT_SAMPLERATE))
            hop_discard_samples = int(np.round(analysis_hop_length * DEFAULT_SAMPLERATE / ANALYSIS_SAMPLERATE))

            mono_tensor = torch.mean(audio_in, dim=0) if audio_in.ndim > 1 else audio_in
            if torch.sum(mono_tensor**2) < VAD_ENERGY_THRESHOLD:
                with self._lock:
                    self._smoothed_f0 = 0.0
                    self._latest_f0_hz = 0.0
                    self._latest_confidence = 0.0
                return {"f0_hz": 0.0, "confidence": 0.0}

            with self._lock:
                self._buffer = torch.cat((self._buffer, mono_tensor))
                if len(self._buffer) < REQUIRED_FRAME_LENGTH:
                    return {"f0_hz": self._smoothed_f0, "confidence": self._latest_confidence}

                analysis_chunk_tensor = self._buffer[:REQUIRED_FRAME_LENGTH]
                self._buffer = self._buffer[hop_discard_samples:]
                method = self._method

            resampled_tensor = self._resampler(analysis_chunk_tensor)

            current_len = resampled_tensor.shape[0]
            if current_len < ANALYSIS_FRAME_LENGTH:
                pad_amount = ANALYSIS_FRAME_LENGTH - current_len
                resampled_tensor = torch.nn.functional.pad(resampled_tensor, (0, pad_amount), "constant", 0)
            elif current_len > ANALYSIS_FRAME_LENGTH:
                resampled_tensor = resampled_tensor[:ANALYSIS_FRAME_LENGTH]

            current_samplerate = ANALYSIS_SAMPLERATE
            analysis_frame_length = ANALYSIS_FRAME_LENGTH
            raw_f0, confidence = 0.0, 0.0

            if method == "autocorr":
                raw_f0, confidence = self._pitch_engine.autocorrelation_f0(resampled_tensor)

            elif method == "torchcrepe" and TORCHCREPE_AVAILABLE:
                try:
                    tensor_input = resampled_tensor.unsqueeze(0).to(torch.float32)

                    with torch.no_grad():
                        f0_hz, f0_conf = torchcrepe.predict(
                            tensor_input,
                            current_samplerate,
                            hop_length=analysis_hop_length,
                            fmin=80,
                            fmax=800,
                            model="tiny",
                            batch_size=1,
                            device=self._crepe_device,
                            return_periodicity=True,
                        )

                    if f0_hz.numel() > 0 and f0_conf.numel() > 0:
                        raw_f0 = float(f0_hz[0, -1].item())
                        confidence = float(f0_conf[0, -1].item())
                        if confidence < 0.5:
                            raw_f0 = 0.0
                    else:
                        raw_f0, confidence = 0.0, 0.0
                except Exception as e:
                    logger.warning(f"TorchCrepe processing failed: {e}")
                    raw_f0, confidence = 0.0, 0.0
            else:
                analysis_chunk_for_f0_np = resampled_tensor.numpy()

                if method == "pyin" and LIBROSA_AVAILABLE:
                    f0, _, probs = librosa.pyin(
                        analysis_chunk_for_f0_np,
                        fmin=80,
                        fmax=800,
                        sr=current_samplerate,
                        frame_length=analysis_frame_length,
                    )
                    raw_f0 = float(f0[0]) if f0 is not None and len(f0) > 0 and not np.isnan(f0[0]) else 0.0
                    confidence = (
                        float(probs[0]) if probs is not None and len(probs) > 0 and not np.isnan(probs[0]) else 0.0
                    )

                elif method == "yin" and LIBROSA_AVAILABLE:
                    f0 = librosa.yin(
                        analysis_chunk_for_f0_np,
                        fmin=80,
                        fmax=800,
                        sr=current_samplerate,
                        frame_length=analysis_frame_length,
                    )
                    raw_f0 = float(f0[0]) if f0 is not None and len(f0) > 0 and not np.isnan(f0[0]) else 0.0
                    confidence = 1.0 if raw_f0 > 0 else 0.0

                elif method == "swift-f0" and SWIFT_F0_AVAILABLE:
                    try:
                        result = self._swift_detector.detect_from_array(
                            audio_array=analysis_chunk_for_f0_np, sample_rate=current_samplerate
                        )
                        if len(result.pitch_hz) > 0 and len(result.confidence) > 0:
                            voiced_indices = np.where(result.voicing)[0]
                            if len(voiced_indices) > 0:
                                # Use last voiced frame for lower latency and to avoid smearing glides
                                last_voiced_idx = voiced_indices[-1]
                                raw_f0 = float(result.pitch_hz[last_voiced_idx])
                                confidence = float(result.confidence[last_voiced_idx])
                                if confidence < 0.5:
                                    raw_f0 = 0.0
                            else:
                                raw_f0, confidence = 0.0, 0.0
                        else:
                            raw_f0, confidence = 0.0, 0.0
                    except Exception as e:
                        logger.warning(f"SwiftF0 processing failed: {e}")
                        raw_f0, confidence = 0.0, 0.0

            with self._lock:
                if raw_f0 > 0:
                    if self._smoothed_f0 > 0:
                        self._smoothed_f0 = F0_SMOOTHING_ALPHA * raw_f0 + (1 - F0_SMOOTHING_ALPHA) * self._smoothed_f0
                    else:
                        self._smoothed_f0 = raw_f0
                    output_f0 = self._smoothed_f0
                else:
                    if self._smoothed_f0 > 0:
                        self._smoothed_f0 *= 1 - F0_SMOOTHING_ALPHA * 2
                        if self._smoothed_f0 < 10:
                            self._smoothed_f0 = 0.0
                    output_f0 = 0.0

                output_confidence = confidence
                self._latest_f0_hz = output_f0
                self._latest_confidence = confidence

        except Exception as e:
            logger.error(f"[{self.name}] Error during F0 estimation: {e}", exc_info=True)
            with self._lock:
                self._smoothed_f0 = 0.0
            output_f0 = 0.0
            output_confidence = 0.0

        return {"f0_hz": output_f0, "confidence": output_confidence}

    def _ui_updater_loop(self):
        while not self._stop_ui_thread_event.is_set():
            state_to_emit = self.get_current_state_snapshot()
            self.emitter.stateUpdated.emit(state_to_emit)
            time.sleep(UI_UPDATE_INTERVAL_S)

    def start(self):
        super().start()
        with self._lock:
            self._buffer = torch.tensor([], dtype=torch.float32)
            self._smoothed_f0 = 0.0
            self._latest_f0_hz = 0.0
            self._latest_confidence = 0.0
            self._pitch_engine.prev_sample = torch.tensor(0.0, dtype=torch.float32)

        self._stop_ui_thread_event.clear()
        self._ui_update_thread = threading.Thread(target=self._ui_updater_loop, daemon=True)
        self._ui_update_thread.start()

    def stop(self):
        super().stop()
        self._stop_ui_thread_event.set()
        if self._ui_update_thread:
            self._ui_update_thread.join(timeout=0.5)

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
