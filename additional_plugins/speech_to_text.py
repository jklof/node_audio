import collections
import logging
import threading
import time
from typing import Deque, Dict, Optional, Tuple, Any

import torch
import torchaudio.transforms as T
import webrtcvad
import whisper

from constants import DEFAULT_SAMPLERATE
from node_system import Node
from ui_elements import ParameterNodeItem
from PySide6.QtWidgets import QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Slot, QObject, Signal
from node_helpers import managed_parameters, Parameter

logger = logging.getLogger(__name__)

# --- VAD & Whisper Constants ---
TARGET_SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
VAD_PADDING_MS = 300
VAD_AGGRESSIVENESS = 3
MAX_SEGMENT_S = 29.0
WHISPER_MODELS = ["tiny.en", "tiny", "base.en", "base", "small.en", "small"]


# ==============================================================================
# Worker Communication Objects
# ==============================================================================
class WorkerCommand:
    PROCESS_AUDIO = "PROCESS_AUDIO"
    SHUTDOWN = "SHUTDOWN"


class WorkerResponse:
    STATUS = "STATUS"
    ERROR = "ERROR"
    TRANSCRIPTION = "TRANSCRIPTION"


# ==============================================================================
# VAD Speech Segment Collector
# ==============================================================================
class SpeechSegmentCollector:
    """Uses WebRTCVAD to collect segments of speech from an audio stream."""

    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.samples_per_frame = int(TARGET_SAMPLE_RATE * VAD_FRAME_MS / 1000)

        num_padding_frames = int(VAD_PADDING_MS / VAD_FRAME_MS)
        self.ring_buffer: Deque[Tuple[torch.Tensor, bool]] = collections.deque(maxlen=num_padding_frames)

        self.triggered = False
        self.voiced_frames: Deque[torch.Tensor] = collections.deque()
        self.segment_duration_s = 0.0

    def _reset_state(self):
        self.ring_buffer.clear()
        self.voiced_frames.clear()
        self.triggered = False
        self.segment_duration_s = 0.0

    def process_frame(self, frame_s16: torch.Tensor) -> Optional[torch.Tensor]:
        if frame_s16.numel() != self.samples_per_frame:
            return None

        try:
            is_speech = self.vad.is_speech(frame_s16.numpy().tobytes(), TARGET_SAMPLE_RATE)
        except Exception:
            is_speech = False

        segment_to_return = None
        frame_duration_s = VAD_FRAME_MS / 1000.0

        if not self.triggered:
            self.ring_buffer.append((frame_s16, is_speech))
            num_voiced = sum(1 for _, speech in self.ring_buffer if speech)
            if num_voiced > 0.9 * self.ring_buffer.maxlen:
                self.triggered = True
                for f, _ in self.ring_buffer:
                    self.voiced_frames.append(f)
                    self.segment_duration_s += frame_duration_s
                self.ring_buffer.clear()
        else:
            self.voiced_frames.append(frame_s16)
            self.segment_duration_s += frame_duration_s
            self.ring_buffer.append((frame_s16, is_speech))
            num_unvoiced = sum(1 for _, speech in self.ring_buffer if not speech)

            end_of_speech = num_unvoiced > 0.9 * self.ring_buffer.maxlen
            max_duration_reached = self.segment_duration_s >= MAX_SEGMENT_S

            if end_of_speech or max_duration_reached:
                segment_to_return = torch.cat(list(self.voiced_frames))
                self._reset_state()

        return segment_to_return

    def flush(self) -> Optional[torch.Tensor]:
        if not self.voiced_frames:
            return None
        segment = torch.cat(list(self.voiced_frames))
        self._reset_state()
        return segment


# ==============================================================================
# Worker Thread for Whisper Transcription
# ==============================================================================
class WhisperWorker(threading.Thread):
    def __init__(self, node_name: str, req_q: Deque, res_q: Deque, model_name: str):
        super().__init__(name=f"WhisperWorker-{node_name}", daemon=True)
        self.node_name = node_name
        self.req_q = req_q
        self.res_q = res_q
        self.model_name = model_name

        self.model = None
        self.resampler = None
        self.segment_collector = None
        self.audio_buffer = torch.tensor([], dtype=torch.int16)

    def _send_response(self, type: str, data: any):
        self.res_q.append((type, data))

    def _load_resources(self):
        try:
            self._send_response(WorkerResponse.STATUS, f"Loading model: {self.model_name}...")
            self.model = whisper.load_model(self.model_name)
            logger.info(f"[{self.node_name}] Whisper model '{self.model_name}' loaded.")

            self.resampler = T.Resample(orig_freq=DEFAULT_SAMPLERATE, new_freq=TARGET_SAMPLE_RATE, dtype=torch.float32)
            self.segment_collector = SpeechSegmentCollector()

            self._send_response(WorkerResponse.STATUS, "Ready")
            return True
        except Exception as e:
            logger.error(f"[{self.node_name}] Worker failed to load resources: {e}", exc_info=True)
            self._send_response(WorkerResponse.ERROR, str(e))
            return False

    def _process_audio(self, audio_chunk: torch.Tensor):
        if self.resampler is None or self.segment_collector is None:
            return

        mono_chunk = torch.mean(audio_chunk, dim=0, keepdim=True)
        resampled_f32 = self.resampler(mono_chunk).squeeze(0)
        resampled_s16 = (resampled_f32 * 32767).to(torch.int16)

        self.audio_buffer = torch.cat((self.audio_buffer, resampled_s16))

        samples_per_frame = self.segment_collector.samples_per_frame
        while self.audio_buffer.numel() >= samples_per_frame:
            frame = self.audio_buffer[:samples_per_frame]
            self.audio_buffer = self.audio_buffer[samples_per_frame:]

            segment = self.segment_collector.process_frame(frame)
            # CORRECTED: Explicitly check for None instead of truthiness of a Tensor
            if segment is not None:
                self._transcribe_segment(segment)

    def _transcribe_segment(self, segment_s16: torch.Tensor):
        if not self.model:
            return
        self._send_response(WorkerResponse.STATUS, "Transcribing...")
        try:
            audio_f32 = segment_s16.float() / 32768.0
            result = self.model.transcribe(audio_f32, fp16=torch.cuda.is_available())
            text = result["text"].strip()
            if text:
                logger.info(f"[{self.node_name}] Transcription: '{text}'")
                self._send_response(WorkerResponse.TRANSCRIPTION, text)
            self._send_response(WorkerResponse.STATUS, "Ready")
        except Exception as e:
            logger.error(f"[{self.node_name}] Transcription failed: {e}", exc_info=True)
            self._send_response(WorkerResponse.ERROR, str(e))

    def run(self):
        if not self._load_resources():
            return

        while True:
            try:
                command, data = self.req_q.popleft()
                if command == WorkerCommand.SHUTDOWN:
                    if self.segment_collector:
                        final_segment = self.segment_collector.flush()
                        # CORRECTED: Explicitly check for None instead of truthiness of a Tensor
                        if final_segment is not None:
                            self._transcribe_segment(final_segment)
                    break
                elif command == WorkerCommand.PROCESS_AUDIO:
                    self._process_audio(data)
            except IndexError:
                time.sleep(0.005)
        logger.info(f"[{self.node_name}] Worker thread finished.")


# ==============================================================================
# Node UI Class
# ==============================================================================
class SpeechToTextNodeItem(ParameterNodeItem):
    """UI for the Speech-to-Text node, allowing model selection and status display."""

    def __init__(self, node_logic: "SpeechToTextNode"):
        parameters = [
            {
                "key": "model_name",
                "name": "Whisper Model",
                "type": "combobox",
                "items": [(name, name) for name in WHISPER_MODELS],
            }
        ]

        super().__init__(node_logic, parameters, width=220)

        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setWordWrap(True)
        self.reload_button = QPushButton("Reload Model")

        self.container_widget.layout().addWidget(self.status_label)
        self.container_widget.layout().addWidget(self.reload_button)

        self.reload_button.clicked.connect(node_logic.reload_model)
        node_logic.statusUpdated.updated.connect(self._on_status_updated)

        self._on_status_updated(node_logic.get_current_status())

    @Slot(str)
    def _on_status_updated(self, status: str):
        self.status_label.setText(f"Status: {status}")
        if "Error" in status:
            self.status_label.setStyleSheet("color: red;")
        elif "Transcribing" in status or "Loading" in status:
            self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setStyleSheet("color: lightgreen;")


# ==============================================================================
# Node Logic Class
# ==============================================================================
@managed_parameters
class SpeechToTextNode(Node):
    NODE_TYPE = "Speech to Text"
    UI_CLASS = SpeechToTextNodeItem
    CATEGORY = "Speech"
    DESCRIPTION = "Transcribes speech from audio using VAD and Whisper."

    class StatusSignal(QObject):
        updated = Signal(str)

    model_name = Parameter(default="base.en")

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("audio_in", data_type=torch.Tensor)
        self.add_output("text_out", data_type=str)

        self.statusUpdated = self.StatusSignal()
        self._current_status = "Uninitialized"

        self._req_q: Deque[Tuple[str, Any]] = collections.deque()
        self._res_q: Deque[Tuple[str, Any]] = collections.deque()
        self._worker: Optional[WhisperWorker] = None
        self._transcribed_text: Deque[str] = collections.deque(maxlen=10)

    def get_current_status(self) -> str:
        with self._lock:
            return self._current_status

    def _set_status(self, status: str):
        with self._lock:
            if self._current_status != status:
                self._current_status = status
                self.statusUpdated.updated.emit(status)

    def _poll_worker_responses(self):
        while True:
            try:
                res_type, data = self._res_q.popleft()
                if res_type == WorkerResponse.STATUS:
                    self._set_status(data)
                elif res_type == WorkerResponse.ERROR:
                    self._set_status(f"Error: {data}")
                elif res_type == WorkerResponse.TRANSCRIPTION:
                    with self._lock:
                        self._transcribed_text.append(data)
            except IndexError:
                break

    @Slot()
    def reload_model(self):
        logger.info(f"[{self.name}] User requested model reload.")
        self.stop()
        self.start()

    def start(self):
        if self._worker and self._worker.is_alive():
            return
        with self._lock:
            model = self._model_name
        self._set_status("Starting worker...")
        self._req_q.clear()
        self._res_q.clear()
        with self._lock:
            self._transcribed_text.clear()
        self._worker = WhisperWorker(self.name, self._req_q, self._res_q, model)
        self._worker.start()

    def stop(self):
        if self._worker and self._worker.is_alive():
            self._set_status("Stopping worker...")
            self._req_q.append((WorkerCommand.SHUTDOWN, None))
            self._worker.join(timeout=3.0)
            if self._worker.is_alive():
                logger.warning(f"[{self.name}] Worker failed to stop cleanly.")
        self._worker = None
        self._set_status("Stopped")

    def remove(self):
        self.stop()
        super().remove()

    def process(self, input_data: dict) -> dict:
        self._poll_worker_responses()
        audio = input_data.get("audio_in")

        if self._worker and self._worker.is_alive() and isinstance(audio, torch.Tensor):
            self._req_q.append((WorkerCommand.PROCESS_AUDIO, audio.clone()))

        output_text = None
        with self._lock:
            if self._transcribed_text:
                output_text = self._transcribed_text.popleft()

        return {"text_out": output_text}
