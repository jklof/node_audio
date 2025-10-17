import os
import time
import asyncio
import threading
import logging
import urllib.request
import hashlib
from collections import deque
from enum import Enum, auto
from typing import Deque, Optional, Any, List, Tuple

# --- Third-Party & Project Imports ---
import torch
import torchaudio.transforms as T
from kokoro_onnx import Kokoro
from misaki import en, espeak

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from PySide6.QtWidgets import QLabel, QComboBox, QDoubleSpinBox, QVBoxLayout, QWidget
from PySide6.QtCore import Slot, QSignalBlocker

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
KOKORO_MODELS_SUBDIR = "../extras/kokoro"
KOKORO_BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"

# --- Simplified model list ---
KOKORO_MODELS = [
    {
        "name": "kokoro-v1.0.fp16-gpu.onnx",  # kokoro-v1.0.fp16.onnx is identical
        "sha256": "c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204",
    },
    {
        "name": "kokoro-v1.0.int8.onnx",
        "sha256": "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb",
    },
]
VOICES_FILENAME = "voices-v1.0.bin"
VOICES_SHA256 = "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"


# =============================================================================
# Worker (Simplified and Corrected)
# =============================================================================
class WorkerCommand(Enum):
    PROCESS_TEXT = auto()
    SHUTDOWN = auto()


class WorkerResponse(Enum):
    STATUS = auto()
    ERROR = auto()
    VOICES = auto()
    AUDIO = auto()


class KokoroWorker(threading.Thread):
    def __init__(self, node_name: str, req_q: Deque, res_q: Deque, target_sr: int, model_name: str):
        super().__init__(name=f"KokoroWorker-{node_name}", daemon=True)
        self.node_name, self.req_q, self.res_q, self.target_sr = node_name, req_q, res_q, target_sr
        self.model_name = model_name  # The specific model to use is now passed in
        self.model_path: Optional[str] = None
        self.voices_path: Optional[str] = None
        self._resamplers, self._kokoro, self._g2p = {}, None, None

    def _send_response(self, type: WorkerResponse, data: Any):
        self.res_q.append((type, data))

    class DownloadProgressBar:
        def __init__(self, worker_instance: "KokoroWorker", file_name: str):
            self.worker = worker_instance
            self.file_name = file_name
            self._last_percent_reported = -1

        def __call__(self, block_num: int, block_size: int, total_size: int):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = int((downloaded / total_size) * 100)
                if percent != self._last_percent_reported:
                    self.worker._send_response(WorkerResponse.STATUS, f"Downloading {self.file_name} ({percent}%)")
                    self._last_percent_reported = percent

    def _verify_sha256(self, file_path: str, expected_hash: str) -> bool:
        file_name = os.path.basename(file_path)
        self._send_response(WorkerResponse.STATUS, f"Verifying {file_name}...")
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                chunk_size = 64 * 1024  # 64KB
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    sha256_hash.update(chunk)
            calculated_hash = sha256_hash.hexdigest()

            if calculated_hash == expected_hash:
                logger.info(f"[{self.node_name}] SHA256 hash verified for {file_name}.")
                return True
            else:
                logger.critical(f"[{self.node_name}] FATAL: SHA256 HASH MISMATCH for {file_name}!")
                logger.critical(f"  - Expected:   {expected_hash}")
                logger.critical(f"  - Calculated: {calculated_hash}")
                self._send_response(WorkerResponse.ERROR, f"Aborting: Hash mismatch for {file_name}")
                return False
        except Exception as e:
            logger.error(f"[{self.node_name}] Could not verify hash for {file_path}: {e}", exc_info=True)
            self._send_response(WorkerResponse.ERROR, f"File verification failed for {file_name}")
            return False

    def _download_file(self, url: str, dest_path: str, expected_hash: str) -> bool:
        file_name = os.path.basename(dest_path)
        self._send_response(WorkerResponse.STATUS, f"Downloading {file_name}...")
        try:
            progress_hook = self.DownloadProgressBar(self, file_name)
            urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
            self._send_response(WorkerResponse.STATUS, f"{file_name} downloaded.")

            if not self._verify_sha256(dest_path, expected_hash):
                raise ValueError(f"Hash verification failed for downloaded file: {file_name}")

            return True
        except Exception as e:
            logger.error(f"[{self.node_name}] Failed to download or verify {url}: {e}")
            return False

    def _ensure_model_files_exist(self) -> bool:
        model_dir = os.path.join(_PLUGIN_DIR, KOKORO_MODELS_SUBDIR)
        os.makedirs(model_dir, exist_ok=True)

        # --- Step 1: Ensure voices.bin is valid ---
        self.voices_path = os.path.join(model_dir, VOICES_FILENAME)
        is_voices_file_valid = os.path.exists(self.voices_path) and self._verify_sha256(self.voices_path, VOICES_SHA256)
        if not is_voices_file_valid:
            logger.info(f"[{self.node_name}] Voices file invalid or missing. Downloading...")
            voices_url = KOKORO_BASE_URL + VOICES_FILENAME
            if not self._download_file(voices_url, self.voices_path, VOICES_SHA256):
                return False

        # --- Step 2: Find the user-selected model info ---
        target_model_info = next((m for m in KOKORO_MODELS if m["name"] == self.model_name), None)
        if not target_model_info:
            self._send_response(WorkerResponse.ERROR, f"Model '{self.model_name}' is not a valid choice.")
            return False

        # --- Step 3: Ensure the selected model file is valid ---
        target_model_path = os.path.join(model_dir, target_model_info["name"])
        is_model_file_valid = os.path.exists(target_model_path) and self._verify_sha256(
            target_model_path, target_model_info["sha256"]
        )
        if not is_model_file_valid:
            logger.info(f"[{self.node_name}] Model '{self.model_name}' invalid or missing. Downloading...")
            model_url = KOKORO_BASE_URL + target_model_info["name"]
            if not self._download_file(model_url, target_model_path, target_model_info["sha256"]):
                return False

        self.model_path = target_model_path
        return True

    def _load_resources(self) -> bool:
        try:
            if not self._ensure_model_files_exist():
                logger.error(f"[{self.node_name}] Aborting resource load due to missing or corrupt model files.")
                return False

            self._send_response(WorkerResponse.STATUS, "Loading models...")
            self._kokoro = Kokoro(self.model_path, self.voices_path)
            fallback = espeak.EspeakFallback(british=True)
            self._g2p = en.G2P(trf=False, fallback=fallback)
            self._send_response(WorkerResponse.STATUS, "Ready")
            self._send_response(WorkerResponse.VOICES, self._kokoro.get_voices())
            return True
        except Exception as e:
            logger.error(f"[{self.node_name}] Worker failed to load models: {e}", exc_info=True)
            self._send_response(WorkerResponse.ERROR, f"Model load failed: {e}")
            return False

    def _resample(self, t: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.target_sr:
            return t
        if sr not in self._resamplers:
            self._resamplers[sr] = T.Resample(sr, self.target_sr, dtype=DEFAULT_DTYPE)
        return self._resamplers[sr](t)

    def _process_text(self, text: str, voice_id: Any, speed: float):
        try:
            self._send_response(WorkerResponse.STATUS, "Synthesizing...")
            phonemes, _ = self._g2p(text)

            async def gen():
                s = self._kokoro.create_stream(phonemes, voice=voice_id, speed=speed, is_phonemes=True, trim=True)
                async for chunk, sr in s:
                    tensor = torch.atleast_2d(torch.from_numpy(chunk.T.copy()).to(DEFAULT_DTYPE))
                    self._send_response(WorkerResponse.AUDIO, self._resample(tensor, sr))

            self._loop.run_until_complete(gen())
            self._send_response(WorkerResponse.STATUS, "Ready")
        except Exception as e:
            logger.error(f"[{self.node_name}] TTS error: {e}", exc_info=True)
            self._send_response(WorkerResponse.ERROR, f"Synthesis error: {e}")

    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        if not self._load_resources():
            return
        while True:
            try:
                cmd, data = self.req_q.popleft()
                if cmd == WorkerCommand.SHUTDOWN:
                    break
                elif cmd == WorkerCommand.PROCESS_TEXT:
                    self._process_text(*data)
            except IndexError:
                time.sleep(0.005)
        self._loop.close()
        logger.info(f"[{self.node_name}] Worker thread finished.")


# =============================================================================
# REVISED Node UI Class (Adds model selection)
# =============================================================================
class TTSKokoroNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "TTSKokoroNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container = QWidget()
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(5)

        self.model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        for model in KOKORO_MODELS:
            self.model_combo.addItem(model["name"], model["name"])

        self.voice_label = QLabel("Voice:")
        self.voice_combo = QComboBox()
        self.voice_combo.addItem("Loading...")

        self.speed_label = QLabel("Speed:")
        self.speed_box = QDoubleSpinBox()
        self.speed_box.setRange(0.5, 2.0)
        self.speed_box.setSingleStep(0.1)
        self.speed_box.setDecimals(1)

        self.status_label = QLabel("Status: ...")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addWidget(self.voice_label)
        layout.addWidget(self.voice_combo)
        layout.addWidget(self.speed_label)
        layout.addWidget(self.speed_box)
        layout.addWidget(self.status_label)
        self.setContentWidget(self.container)

        self.model_combo.textActivated.connect(self.node_logic.set_model_name)
        self.voice_combo.currentIndexChanged.connect(self._on_voice_selected)
        self.speed_box.valueChanged.connect(self.node_logic.set_speed)
        self.updateFromLogic()

    @Slot(int)
    def _on_voice_selected(self, index: int):
        voice_data = self.voice_combo.itemData(index)
        if voice_data:
            self.node_logic.set_voice(voice_data)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        super()._on_state_updated_from_logic(state)
        status = state.get("status", "N/A")
        self.status_label.setText(f"Status: {status}")
        if "Error" in status or "failed" in status.lower() or "Aborting" in status:
            self.status_label.setStyleSheet("color: red;")
        elif "Synthesizing" in status or "Loading" in status or "Downloading" in status or "Verifying" in status:
            self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setStyleSheet("color: lightgreen;")

        model_name = state.get("model_name")
        if model_name and self.model_combo.currentText() != model_name:
            with QSignalBlocker(self.model_combo):
                self.model_combo.setCurrentText(model_name)

        speed_value = state.get("speed")
        if speed_value is not None and self.speed_box.value() != speed_value:
            with QSignalBlocker(self.speed_box):
                self.speed_box.setValue(speed_value)

        available_voices = state.get("available_voices", [])
        if available_voices:
            current_items = [self.voice_combo.itemData(i) for i in range(self.voice_combo.count())]
            if available_voices != current_items:
                with QSignalBlocker(self.voice_combo):
                    self.voice_combo.clear()
                    for v in available_voices:
                        self.voice_combo.addItem(str(v), v)

        selected_voice = state.get("voice")
        if selected_voice:
            idx = self.voice_combo.findData(selected_voice)
            if idx != -1 and self.voice_combo.currentIndex() != idx:
                with QSignalBlocker(self.voice_combo):
                    self.voice_combo.setCurrentIndex(idx)


# =============================================================================
# REVISED Node Logic Class (Adds model selection state)
# =============================================================================
class TTSKokoroNode(Node):
    NODE_TYPE = "TTS Kokoro"
    UI_CLASS = TTSKokoroNodeItem
    CATEGORY = "Generators"
    DESCRIPTION = "Generates speech from text using the Kokoro TTS engine."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("text_in", data_type=str)
        self.add_output("out", data_type=torch.Tensor)
        self._current_status = "Stopped"
        self._speed: float = 1.0
        self._voice: Optional[Any] = None
        self._model_name: str = KOKORO_MODELS[0]["name"]  # Default to gpu fp16
        self._available_voices: List[Any] = []
        self._saved_voice_name: Optional[str] = None
        self._audio_buffer = torch.empty((1, 0), dtype=DEFAULT_DTYPE)
        self._last_processed_text = ""
        self._req_q: Deque[Tuple[Enum, Any]] = deque()
        self._res_q: Deque[Tuple[Enum, Any]] = deque()
        self._worker: Optional[KokoroWorker] = None

    def _set_status(self, status: str):
        state_to_emit = None
        with self._lock:
            if self._current_status != status:
                self._current_status = status
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_speed(self, speed: float):
        state_to_emit = None
        with self._lock:
            clamped_speed = max(0.5, min(speed, 2.0))
            if self._speed != clamped_speed:
                self._speed = clamped_speed
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(object)
    def set_voice(self, voice: Any):
        state_to_emit = None
        with self._lock:
            if self._voice != voice:
                self._voice = voice
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(str)
    def set_model_name(self, model_name: str):
        should_restart = False
        state_to_emit = None
        with self._lock:
            if self._model_name != model_name:
                self._model_name = model_name
                state_to_emit = self._get_state_snapshot_locked()
                if self._worker and self._worker.is_alive():
                    should_restart = True
        if state_to_emit:
            self.ui_update_callback(state_to_emit)
        if should_restart:
            self.stop()
            self.start()

    def _get_state_snapshot_locked(self) -> dict:
        return {
            "speed": self._speed,
            "voice": self._voice,
            "available_voices": self._available_voices,
            "status": self._current_status,
            "model_name": self._model_name,
        }

    def _poll_worker_responses(self):
        ui_params_update_needed = False
        while self._res_q:
            res_type, data = self._res_q.popleft()
            if res_type == WorkerResponse.STATUS:
                self._set_status(str(data))
            elif res_type == WorkerResponse.ERROR:
                self._set_status(f"Error: {data}")
            elif res_type == WorkerResponse.AUDIO:
                with self._lock:
                    self._audio_buffer = torch.cat((self._audio_buffer, data), dim=1)
            elif res_type == WorkerResponse.VOICES:
                with self._lock:
                    self._available_voices = data
                    if self._saved_voice_name:
                        for v in data:
                            if str(v) == self._saved_voice_name:
                                self._voice = v
                                break
                        self._saved_voice_name = None
                    if not self._voice and data:
                        self._voice = data[0]
                ui_params_update_needed = True
        if ui_params_update_needed:
            self.ui_update_callback(self.get_current_state_snapshot())

    def start(self):
        if self._worker and self._worker.is_alive():
            return
        self._set_status("Starting worker...")
        self._req_q.clear()
        self._res_q.clear()
        with self._lock:
            self._audio_buffer = torch.empty((1, 0), dtype=DEFAULT_DTYPE)
            model_to_use = self._model_name
        self._worker = KokoroWorker(self.name, self._req_q, self._res_q, DEFAULT_SAMPLERATE, model_to_use)
        self._worker.start()

    def stop(self):
        if self._worker and self._worker.is_alive():
            self._set_status("Stopping worker...")
            self._req_q.append((WorkerCommand.SHUTDOWN, None))
            self._worker.join(timeout=2.0)
            if self._worker.is_alive():
                logger.warning(f"[{self.name}] Worker failed to stop cleanly.")
        self._worker = None
        self._set_status("Stopped")

    def remove(self):
        self.stop()
        super().remove()

    def process(self, input_data: dict) -> dict:
        self._poll_worker_responses()
        text = input_data.get("text_in")
        if self._worker and self._worker.is_alive() and isinstance(text, str):
            stripped_text = text.strip()
            if stripped_text and stripped_text != self._last_processed_text:
                self._last_processed_text = stripped_text
                with self._lock:
                    self._audio_buffer = torch.empty((1, 0), dtype=DEFAULT_DTYPE)
                    current_voice, current_speed = self._voice, self._speed
                if current_voice:
                    self._req_q.append((WorkerCommand.PROCESS_TEXT, (stripped_text, current_voice, current_speed)))
        output_block = torch.zeros((1, DEFAULT_BLOCKSIZE), dtype=DEFAULT_DTYPE)
        with self._lock:
            if self._audio_buffer.shape[1] >= DEFAULT_BLOCKSIZE:
                output_block = self._audio_buffer[:, :DEFAULT_BLOCKSIZE]
                self._audio_buffer = self._audio_buffer[:, DEFAULT_BLOCKSIZE:]
        return {"out": output_block}

    def serialize_extra(self) -> dict:
        with self._lock:
            return {
                "speed": self._speed,
                "voice": str(self._voice) if self._voice else None,
                "model_name": self._model_name,
            }

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._speed = data.get("speed", 1.0)
            self._saved_voice_name = data.get("voice")
            # Ensure the loaded model name is valid, otherwise fall back to default
            loaded_model_name = data.get("model_name")
            valid_model_names = [m["name"] for m in KOKORO_MODELS]
            if loaded_model_name in valid_model_names:
                self._model_name = loaded_model_name
            else:
                self._model_name = KOKORO_MODELS[1]["name"]  # Default
