import torch
import numpy as np
import threading
import logging
import time
import queue
from typing import Dict, Optional

# --- Dependency Check ---
try:
    import swift_f0
    from swift_f0 import NoteSegment  # Explicitly import for type hints

    SWIFT_F0_AVAILABLE = True
except ImportError:
    SWIFT_F0_AVAILABLE = False
    logging.warning("swift_f0_node.py: 'swift-f0' library not found. SwiftF0 node will be disabled.")

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import mido

    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    logging.warning("swift_f0_node.py: 'mido' library not found. MIDI message output will be disabled.")


# --- Import for Resampling ---
import torchaudio.transforms as T

# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE

# --- Qt Imports ---
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Optimization Constants ---
ANALYSIS_SAMPLERATE = 16000
ANALYSIS_CHUNK_S = 0.2  # Analyze audio in 0.2-second chunks for efficiency
ANALYSIS_CHUNK_SAMPLES = int(ANALYSIS_CHUNK_S * ANALYSIS_SAMPLERATE)
INPUT_CHUNK_SAMPLES = int(ANALYSIS_CHUNK_S * DEFAULT_SAMPLERATE)

UI_UPDATE_INTERVAL_S = 0.05
VAD_ENERGY_THRESHOLD = 1e-6


# ==============================================================================
# 1. Custom UI Class (SwiftF0NodeItem)
# ==============================================================================
class SwiftF0NodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "SwiftF0Node"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(5)

        self.title_label = QLabel("SwiftF0 Pitch Tracker")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.f0_label = QLabel("F0: ... Hz")
        self.gate_label = QLabel("Gate: OFF")
        self.confidence_label = QLabel("Confidence: ...")

        for label in [self.f0_label, self.gate_label, self.confidence_label]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.f0_label)
        layout.addWidget(self.gate_label)
        layout.addWidget(self.confidence_label)

        if not SWIFT_F0_AVAILABLE or not MIDO_AVAILABLE:
            reqs = []
            if not SWIFT_F0_AVAILABLE:
                reqs.append("swift-f0")
            if not MIDO_AVAILABLE:
                reqs.append("mido")
            error_label = QLabel(f"Note: Requires pip install\n{' '.join(reqs)}")
            error_label.setStyleSheet("color: orange;")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setWordWrap(True)
            layout.addWidget(error_label)

        self.setContentWidget(self.container_widget)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        super()._on_state_updated_from_logic(state)
        f0 = state.get("f0_hz")
        confidence = state.get("confidence")
        gate = state.get("gate", False)

        if f0 is not None and f0 > 0:
            self.f0_label.setText(f"F0: {f0:.2f} Hz")
        else:
            self.f0_label.setText("F0: Unvoiced")

        if confidence is not None:
            self.confidence_label.setText(f"Confidence: {confidence:.1%}")
        else:
            self.confidence_label.setText("Confidence: ...")

        if gate:
            self.gate_label.setText("Gate: ON")
            self.gate_label.setStyleSheet("color: lightgreen;")
        else:
            self.gate_label.setText("Gate: OFF")
            self.gate_label.setStyleSheet("color: lightgray;")


# ==============================================================================
# 2. Node Logic Class (SwiftF0Node) - CORRECTED
# ==============================================================================
class SwiftF0Node(Node):
    NODE_TYPE = "SwiftF0 Pitch/Note"
    UI_CLASS = SwiftF0NodeItem
    CATEGORY = "Analysis"
    DESCRIPTION = "Detects pitch (F0) and musical notes from an audio signal using the SwiftF0 library."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)

        self.add_input("in", data_type=torch.Tensor)
        self.add_output("f0_hz", data_type=float)
        self.add_output("msg_out", data_type=object)

        self._resampler = T.Resample(orig_freq=DEFAULT_SAMPLERATE, new_freq=ANALYSIS_SAMPLERATE, dtype=torch.float32)

        if SWIFT_F0_AVAILABLE:
            self._swift_detector = swift_f0.SwiftF0(fmin=65, fmax=1200, confidence_threshold=0.4)
        else:
            self._swift_detector = None

        # --- Threading and Queues for Optimization ---
        self._analysis_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._audio_in_queue = queue.Queue()
        self._midi_out_queue = queue.Queue()

        # --- Internal State for UI and Logic ---
        self._latest_f0_hz: float = 0.0
        self._latest_confidence: float = 0.0
        self._latest_midi_note: Optional[int] = None
        self._latest_gate: bool = False
        self._ui_update_thread: Optional[threading.Thread] = None
        self._stop_ui_thread_event = threading.Event()

    def _get_current_state_snapshot(self) -> Dict:
        return {
            "f0_hz": self._latest_f0_hz,
            "confidence": self._latest_confidence,
            "midi_note": self._latest_midi_note,  # Kept for potential future use
            "gate": self._latest_gate,
        }

    def _analysis_thread_loop(self):
        """This function runs in a separate thread and does the heavy lifting."""
        analysis_buffer = torch.tensor([], dtype=torch.float32)
        last_emitted_note: Optional[int] = None

        while not self._stop_event.is_set():
            try:
                # Get all available audio chunks from the queue
                while not self._audio_in_queue.empty():
                    chunk = self._audio_in_queue.get_nowait()
                    analysis_buffer = torch.cat((analysis_buffer, chunk))

                # If we have enough data, perform analysis
                if len(analysis_buffer) >= INPUT_CHUNK_SAMPLES:
                    chunk_to_process = analysis_buffer[:INPUT_CHUNK_SAMPLES]
                    analysis_buffer = analysis_buffer[INPUT_CHUNK_SAMPLES:]

                    # --- Core SwiftF0 Analysis ---
                    resampled = self._resampler(chunk_to_process)
                    result = self._swift_detector.detect_from_array(resampled.numpy(), ANALYSIS_SAMPLERATE)
                    notes: list[NoteSegment] = swift_f0.segment_notes(result)

                    # --- State Management and MIDI Message Generation ---
                    detected_note = notes[-1].pitch_midi if notes else None

                    # 1. If a new note is detected that is different from the last one
                    if detected_note is not None and detected_note != last_emitted_note:
                        if last_emitted_note is not None:
                            self._midi_out_queue.put(mido.Message("note_off", note=last_emitted_note))
                        self._midi_out_queue.put(mido.Message("note_on", note=detected_note, velocity=80))
                        last_emitted_note = detected_note

                    # 2. If no note is detected, but there was a previous note on
                    elif detected_note is None and last_emitted_note is not None:
                        self._midi_out_queue.put(mido.Message("note_off", note=last_emitted_note))
                        last_emitted_note = None

                    # --- Update UI State (locked) ---
                    with self._lock:
                        voiced_indices = np.where(result.voicing)[0]
                        if len(voiced_indices) > 0:
                            self._latest_f0_hz = float(np.median(result.pitch_hz[voiced_indices]))
                            self._latest_confidence = float(np.median(result.confidence[voiced_indices]))
                        else:
                            self._latest_f0_hz = 0.0
                            self._latest_confidence = 0.0

                        self._latest_gate = detected_note is not None
                        self._latest_midi_note = detected_note

                else:
                    time.sleep(0.02)

            except Exception as e:
                logger.error(f"[{self.name}] Error in analysis thread: {e}", exc_info=True)
                time.sleep(0.1)

    def process(self, input_data: dict) -> dict:
        """This function runs in the real-time audio thread. It must be FAST."""
        audio_in = input_data.get("in")
        msg_to_send = None

        if isinstance(audio_in, torch.Tensor) and audio_in.numel() > 0:
            mono_tensor = torch.mean(audio_in, dim=0) if audio_in.ndim > 1 else audio_in
            self._audio_in_queue.put(mono_tensor)

        try:
            msg_to_send = self._midi_out_queue.get_nowait()
        except queue.Empty:
            msg_to_send = None

        return {"f0_hz": self._latest_f0_hz, "msg_out": msg_to_send}

    def _ui_updater_loop(self):
        while not self._stop_ui_thread_event.is_set():
            state_to_emit = self.get_current_state_snapshot()
            self.ui_update_callback(state_to_emit)
            time.sleep(UI_UPDATE_INTERVAL_S)

    def start(self):
        super().start()
        with self._lock:
            while not self._audio_in_queue.empty():
                self._audio_in_queue.get()
            while not self._midi_out_queue.empty():
                self._midi_out_queue.get()
            self._latest_f0_hz = 0.0
            self._latest_confidence = 0.0
            self._latest_midi_note = None
            self._latest_gate = False

        self._stop_event.clear()
        self._stop_ui_thread_event.clear()
        if SWIFT_F0_AVAILABLE and MIDO_AVAILABLE:
            self._analysis_thread = threading.Thread(target=self._analysis_thread_loop, daemon=True)
            self._analysis_thread.start()
            self._ui_update_thread = threading.Thread(target=self._ui_updater_loop, daemon=True)
            self._ui_update_thread.start()

    def stop(self):
        super().stop()
        self._stop_event.set()
        self._stop_ui_thread_event.set()

        if self._analysis_thread:
            self._analysis_thread.join(timeout=0.5)
        if self._ui_update_thread:
            self._ui_update_thread.join(timeout=0.5)

        with self._lock:
            state = {"f0_hz": 0.0, "confidence": 0.0, "midi_note": None, "gate": False}
        self.ui_update_callback(state)

    def remove(self):
        self.stop()
        super().remove()

    def serialize_extra(self) -> dict:
        return {}

    def deserialize_extra(self, data: dict):
        pass
