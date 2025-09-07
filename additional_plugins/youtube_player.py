import logging
import threading
import time
import subprocess
import sys
from collections import deque
from typing import Optional, Dict, Tuple

import numpy as np
import torch
from pytubefix import YouTube, Stream

from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE
from ui_elements import NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QSlider
from PySide6.QtCore import Qt, Slot, Signal, QObject, QTimer
from PySide6.QtGui import QImage, QPixmap

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Constants for the Player ---
INTERNAL_BUFFER_MAX_BLOCKS = 50
VIDEO_WIDTH = 280
VIDEO_HEIGHT = 158
VIDEO_FRAME_BYTES = VIDEO_WIDTH * VIDEO_HEIGHT * 4
FFMPEG_AUDIO_CHUNK_SIZE = DEFAULT_BLOCKSIZE * 2 * 2
UI_UPDATE_INTERVAL_MS = 250


class PlaybackState:
    STOPPED = "STOPPED"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    LOADING = "LOADING"
    BUFFERING = "BUFFERING"
    ERROR = "ERROR"


class VideoSignalEmitter(QObject):
    """A dedicated emitter for video frames to keep them separate from state dicts."""
    newVideoFrame = Signal(QImage)

    def emit_video_frame(self, frame: QImage):
        try:
            self.newVideoFrame.emit(frame)
        except RuntimeError as e:
            logger.debug(f"YouTube Emitter (video) caught RuntimeError: {e}")


class YouTubePlayerNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 300

    def __init__(self, node_logic: "YouTubePlayerNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(5)

        self.video_display = QLabel()
        self.video_display.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT)
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setStyleSheet("background-color: black; color: gray;")
        self.video_display.setText("Video stream will appear here")
        main_layout.addWidget(self.video_display)

        url_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("YouTube URL")
        self.load_button = QPushButton("Load")
        url_layout.addWidget(self.url_input)
        url_layout.addWidget(self.load_button)
        main_layout.addLayout(url_layout)

        self.title_label = QLabel("No stream loaded")
        self.title_label.setWordWrap(True)
        self.status_label = QLabel("Status: STOPPED")
        self.status_label.setStyleSheet("color: lightgray;")
        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.status_label)

        time_layout = QHBoxLayout()
        self.time_label = QLabel("00:00")
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.setEnabled(False)
        self.duration_label = QLabel("00:00")

        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.seek_slider, stretch=1)
        time_layout.addWidget(self.duration_label)
        main_layout.addLayout(time_layout)

        self.play_pause_button = QPushButton("▶")
        self.play_pause_button.setFixedSize(25, 25)
        self.play_pause_button.setEnabled(False)
        main_layout.addWidget(self.play_pause_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setContentWidget(self.container_widget)

        self.load_button.clicked.connect(self._on_load_clicked)
        self.url_input.returnPressed.connect(self._on_load_clicked)
        self.play_pause_button.clicked.connect(self._on_play_pause_clicked)
        self.seek_slider.sliderReleased.connect(self._on_seek_slider_released)
        self.seek_slider.sliderMoved.connect(self._on_seek_slider_moved)

        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        self.node_logic.video_emitter.newVideoFrame.connect(self._on_new_video_frame)

        QTimer.singleShot(0, self.node_logic.load_url_if_present)
        self.updateFromLogic()

    def _format_time(self, seconds: float) -> str:
        if seconds < 0: return "00:00"
        mins, secs = int(seconds // 60), int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    @Slot()
    def _on_load_clicked(self):
        url = self.url_input.text()
        if url:
            self.video_display.setText("Loading...")
            self.node_logic.load_url(url)

    @Slot()
    def _on_play_pause_clicked(self):
        self.node_logic.toggle_playback()

    @Slot(int)
    def _on_seek_slider_moved(self, value: int):
        duration = self.node_logic.get_current_state_snapshot().get("duration_s", 0.0)
        if duration > 0:
            self.time_label.setText(self._format_time((value / 1000.0) * duration))

    @Slot()
    def _on_seek_slider_released(self):
        self.node_logic.seek(self.seek_slider.value() / 1000.0)

    @Slot(QImage)
    def _on_new_video_frame(self, frame: QImage):
        if not frame.isNull():
            self.video_display.setPixmap(QPixmap.fromImage(frame))

    @Slot(dict)
    def _on_state_updated(self, state: Dict):
        playback_state = state.get("state")
        stream_info = state.get("stream_info", {})
        error_message = state.get("error_message", "")
        current_position = state.get("position_s", 0.0)
        total_duration = state.get("duration_s", 0.0)

        has_stream = stream_info.get("title") is not None and total_duration > 0
        self.play_pause_button.setEnabled(has_stream)
        self.seek_slider.setEnabled(has_stream)

        self.play_pause_button.setText("⏸" if playback_state == PlaybackState.PLAYING else "▶")
        self.title_label.setText(stream_info.get("title", "No stream loaded"))
        if not self.url_input.hasFocus(): self.url_input.setText(state.get("url", ""))

        status_text = f"Status: {playback_state}"
        if playback_state == PlaybackState.ERROR:
            self.status_label.setStyleSheet("color: red;")
            status_text = "Status: ERROR"
            self.video_display.setText(f"Error:\n{error_message[:100]}")
        elif playback_state == PlaybackState.PLAYING: self.status_label.setStyleSheet("color: lightgreen;")
        elif playback_state == PlaybackState.STOPPED:
            self.video_display.setText("Video stream will appear here")
            self.video_display.setPixmap(QPixmap())
        else: self.status_label.setStyleSheet("color: lightgray;")
        self.status_label.setText(status_text)

        self.time_label.setText(self._format_time(current_position))
        self.duration_label.setText(self._format_time(total_duration))

        if total_duration > 0 and not self.seek_slider.isSliderDown():
            self.seek_slider.setValue(int((current_position / total_duration) * 1000))
        elif total_duration == 0: self.seek_slider.setValue(0)

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


class YouTubePlayerNode(Node):
    NODE_TYPE = "YouTube Player"
    UI_CLASS = YouTubePlayerNodeItem
    CATEGORY = "Generators"
    DESCRIPTION = "Streams audio and video from a YouTube video URL."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_output("out", data_type=torch.Tensor)
        self.emitter = NodeStateEmitter()
        self.video_emitter = VideoSignalEmitter()
        self._lock = threading.Lock()

        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._audio_buffer = deque()

        # --- Explicit State Variables ---
        self._playback_state = PlaybackState.STOPPED
        self._url: str = ""
        self._stream_info: Dict = {}
        self._error_message: str = ""
        self._position_s: float = 0.0
        self._duration_s: float = 0.0
        self._seek_request_s: float = -1.0

    def get_current_state_snapshot(self, locked: bool = False) -> Dict:
        """Constructs a state dictionary from internal variables."""
        state = {
            "state": self._playback_state, "url": self._url, "stream_info": self._stream_info,
            "error_message": self._error_message, "position_s": self._position_s, "duration_s": self._duration_s,
        }
        if locked: return state
        with self._lock:
            return state

    def load_url_if_present(self):
        with self._lock: url = self._url
        if url:
            logger.info(f"[{self.name}] Auto-loading URL: {url}")
            self.load_url(url)
            with self._lock: initial_seek = self._position_s
            if initial_seek > 0: self._seek_request_s = initial_seek

    def load_url(self, url: str):
        self._stop_worker()
        state_to_emit = None
        with self._lock:
            self._playback_state = PlaybackState.LOADING
            self._url, self._stream_info, self._error_message = url, {}, ""
            self._position_s, self._duration_s = 0.0, 0.0
            state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit: self.emitter.stateUpdated.emit(state_to_emit)

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._stream_reader_loop, daemon=True)
        self._worker_thread.start()

    def toggle_playback(self):
        state_to_emit = None
        with self._lock:
            if not self._stream_info.get("title"): return
            if self._playback_state in [PlaybackState.PAUSED, PlaybackState.STOPPED]:
                if self._position_s >= self._duration_s and self._duration_s > 0:
                    self._seek_request_s = 0.0
                self._playback_state = PlaybackState.PLAYING
            else: self._playback_state = PlaybackState.PAUSED
            state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit: self.emitter.stateUpdated.emit(state_to_emit)

    def seek(self, proportion: float):
        state_to_emit = None
        with self._lock:
            if self._duration_s <= 0: return
            self._seek_request_s = max(0.0, min(self._duration_s, proportion * self._duration_s))
            self._position_s = self._seek_request_s
            self._playback_state = PlaybackState.BUFFERING
            state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit: self.emitter.stateUpdated.emit(state_to_emit)

    def process(self, input_data: dict) -> dict:
        state_to_emit = None
        output_block = None
        with self._lock:
            if self._audio_buffer:
                output_block = self._audio_buffer.popleft()
                if self._playback_state == PlaybackState.BUFFERING:
                    self._playback_state = PlaybackState.PLAYING
                    state_to_emit = self.get_current_state_snapshot(locked=True)
            else:
                if self._playback_state == PlaybackState.PLAYING:
                    self._playback_state = PlaybackState.BUFFERING
                    state_to_emit = self.get_current_state_snapshot(locked=True)
                output_block = torch.zeros((2, DEFAULT_BLOCKSIZE), dtype=DEFAULT_DTYPE)
        if state_to_emit: self.emitter.stateUpdated.emit(state_to_emit)
        return {"out": output_block}

    def remove(self):
        self._stop_worker()
        super().remove()

    def _stop_worker(self):
        if self._worker_thread and self._worker_thread.is_alive():
            self._stop_event.set()
            if self._ffmpeg_process:
                self._ffmpeg_process.terminate()
                try: self._ffmpeg_process.wait(timeout=1.0)
                except subprocess.TimeoutExpired: self._ffmpeg_process.kill()
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None
        self._ffmpeg_process = None
        with self._lock: self._audio_buffer.clear(); self._seek_request_s = -1.0

    def _get_youtube_info(self, url: str) -> Optional[Tuple[str, float, str, Stream]]:
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
            if not stream: stream = yt.streams.get_highest_resolution()
            if not stream: raise RuntimeError("No suitable MP4 stream found.")
            return yt.title or "Unknown", yt.length or 0.0, stream.url, stream
        except Exception as e:
            logger.error(f"[{self.name}] Error fetching YouTube info for {url}: {e}", exc_info=True)
            with self._lock: self._error_message = str(e)
            return None

    def _audio_reader_thread(self, pipe, start_time_s: float):
        frames_read = 0
        last_ui_update = time.monotonic()
        while not self._stop_event.is_set():
            with self._lock: is_paused_or_full = self._playback_state == PlaybackState.PAUSED or len(self._audio_buffer) >= INTERNAL_BUFFER_MAX_BLOCKS
            if is_paused_or_full: time.sleep(0.01); continue
            raw_audio = pipe.read(FFMPEG_AUDIO_CHUNK_SIZE)
            if not raw_audio: break
            
            # --- FIX: Create a writable copy of the numpy array ---
            audio_array_np = np.frombuffer(raw_audio, dtype=np.int16).copy()
            
            audio_tensor = torch.from_numpy(audio_array_np).to(DEFAULT_DTYPE) / 32768.0
            num_samples = audio_tensor.shape[0] // 2
            if num_samples == 0: continue
            reshaped = audio_tensor.reshape(num_samples, 2).T

            for i in range(0, num_samples, DEFAULT_BLOCKSIZE):
                block = reshaped[:, i : i + DEFAULT_BLOCKSIZE]
                if block.shape[1] == DEFAULT_BLOCKSIZE:
                    with self._lock: self._audio_buffer.append(block)
                    frames_read += DEFAULT_BLOCKSIZE
            
            now = time.monotonic()
            if now - last_ui_update > UI_UPDATE_INTERVAL_MS / 1000.0:
                with self._lock:
                    self._position_s = start_time_s + (frames_read / DEFAULT_SAMPLERATE)
                    state_to_emit = self.get_current_state_snapshot(locked=True)
                self.emitter.stateUpdated.emit(state_to_emit)
                last_ui_update = now
        
        with self._lock: final_pos = start_time_s + (frames_read / DEFAULT_SAMPLERATE)
        self._end_of_stream_actions(final_pos)
        logger.info(f"[{self.name}] Audio reader thread finished.")

    def _video_reader_thread(self, pipe):
        while not self._stop_event.is_set():
            raw_frame = pipe.read(VIDEO_FRAME_BYTES)
            if len(raw_frame) < VIDEO_FRAME_BYTES: break
            self.video_emitter.emit_video_frame(QImage(raw_frame, VIDEO_WIDTH, VIDEO_HEIGHT, QImage.Format.Format_ARGB32).copy())
            time.sleep(1 / 20)
        logger.info(f"[{self.name}] Video reader thread finished.")

    def _end_of_stream_actions(self, final_playback_time_s: float):
        state_to_emit = None
        with self._lock:
            if abs(final_playback_time_s - self._duration_s) <= 0.5 and self._duration_s > 0:
                logger.info(f"[{self.name}] End of video reached, looping.")
                self._seek_request_s, self._position_s = 0.0, 0.0
                self._playback_state = PlaybackState.PLAYING
            elif not self._stop_event.is_set():
                self._playback_state, self._position_s = PlaybackState.STOPPED, final_playback_time_s
            else: self._playback_state = PlaybackState.STOPPED
            state_to_emit = self.get_current_state_snapshot(locked=True)
        if state_to_emit: self.emitter.stateUpdated.emit(state_to_emit)

    def _stream_reader_loop(self):
        with self._lock: url = self._url
        info = self._get_youtube_info(url)
        if not info:
            with self._lock: state_to_emit = self.get_current_state_snapshot(locked=True)
            self.emitter.stateUpdated.emit(state_to_emit)
            return
        
        title, duration_s, stream_url, _ = info
        with self._lock:
            self._stream_info, self._duration_s = {"title": title, "length": duration_s}, duration_s
            state_to_emit = self.get_current_state_snapshot(locked=True)
        self.emitter.stateUpdated.emit(state_to_emit)

        while not self._stop_event.is_set():
            start_time = 0.0
            with self._lock:
                if self._seek_request_s != -1.0:
                    start_time, self._seek_request_s = self._seek_request_s, -1.0
                    self._audio_buffer.clear()
                elif not self._ffmpeg_process:
                    start_time = self._position_s
                    if start_time >= self._duration_s and self._duration_s > 0: start_time = 0.0
                if self._playback_state != PlaybackState.PAUSED:
                    self._playback_state = PlaybackState.BUFFERING
                    state_to_emit = self.get_current_state_snapshot(locked=True)
            self.emitter.stateUpdated.emit(state_to_emit)
            
            ffmpeg_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
            if start_time > 0: ffmpeg_cmd.extend(["-ss", str(start_time)])
            ffmpeg_cmd.extend([
                "-i", stream_url, "-f", "s16le", "-ac", "2", "-ar", str(DEFAULT_SAMPLERATE), "-c:a", "pcm_s16le", "pipe:1",
                "-f", "rawvideo", "-pix_fmt", "bgra", "-vf", f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}", "-r", "15", "-c:v", "rawvideo", "pipe:2",
            ])
            
            try:
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                with self._lock: self._ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
                if not self._ffmpeg_process: raise RuntimeError("Failed to start FFMPEG.")
                
                audio_thread = threading.Thread(target=self._audio_reader_thread, args=(self._ffmpeg_process.stdout, start_time), daemon=True)
                video_thread = threading.Thread(target=self._video_reader_thread, args=(self._ffmpeg_process.stderr,), daemon=True)
                audio_thread.start(); video_thread.start()
                
                while self._ffmpeg_process.poll() is None and not self._stop_event.is_set():
                    with self._lock:
                        if self._seek_request_s != -1.0: break
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"[{self.name}] FFmpeg error: {e}", exc_info=True)
                with self._lock: self._playback_state, self._error_message = PlaybackState.ERROR, f"Stream error: {e}"
                break
            finally:
                if self._ffmpeg_process: self._ffmpeg_process.terminate()
                if 'audio_thread' in locals() and audio_thread.is_alive(): audio_thread.join(timeout=1.0)
                if 'video_thread' in locals() and video_thread.is_alive(): video_thread.join(timeout=1.0)
                with self._lock: self._ffmpeg_process = None
                if self._stop_event.is_set(): break
        
        logger.info(f"[{self.name}] Main worker loop finished.")
        with self._lock:
            if self._playback_state != PlaybackState.ERROR: self._playback_state = PlaybackState.STOPPED
            state_to_emit = self.get_current_state_snapshot(locked=True)
        self.emitter.stateUpdated.emit(state_to_emit)

    def serialize_extra(self) -> dict:
        with self._lock:
            return {"youtube_url": self._url, "saved_position_s": self._position_s}

    def deserialize_extra(self, data: dict):
        url, pos = data.get("youtube_url"), data.get("saved_position_s", 0.0)
        if url:
            with self._lock: self._url, self._position_s = url, pos
