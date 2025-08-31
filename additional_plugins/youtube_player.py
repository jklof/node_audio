import logging
import threading
import time
import subprocess
import sys
from collections import deque
from typing import Optional, Dict, Tuple, Any

import numpy as np
from pytubefix import YouTube, Stream
from pytubefix.exceptions import AgeRestrictedError, VideoUnavailable, LiveStreamError

from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE
from ui_elements import NodeItem, NODE_CONTENT_PADDING

from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QSlider
)
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

class YouTubePlayerSignalEmitter(QObject):
    playbackStateChanged = Signal(dict)
    newVideoFrame = Signal(QImage)

    def emit_state_change(self, state_dict: Dict):
        try:
            self.playbackStateChanged.emit(state_dict.copy())
        except RuntimeError as e:
            logger.debug(f"YouTube Emitter caught RuntimeError: {e}")

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
        main_layout.setContentsMargins(NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING)
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

        self.node_logic.emitter.playbackStateChanged.connect(self._on_playback_state_changed)
        self.node_logic.emitter.newVideoFrame.connect(self._on_new_video_frame)
        
        # Defer loading from a saved file until after the main event loop has started
        QTimer.singleShot(0, self.node_logic.load_url_if_present)

        self.updateFromLogic()

    def _format_time(self, seconds: float) -> str:
        if seconds < 0: return "00:00"
        mins = int(seconds // 60)
        secs = int(seconds % 60)
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
        if not self.node_logic: return
        duration = self.node_logic.get_current_state_snapshot().get("duration_s", 0.0)
        if duration > 0:
            current_s = (value / 1000.0) * duration
            self.time_label.setText(self._format_time(current_s))

    @Slot()
    def _on_seek_slider_released(self):
        if not self.node_logic: return
        seek_proportion = self.seek_slider.value() / 1000.0
        self.node_logic.seek(seek_proportion)

    @Slot(QImage)
    def _on_new_video_frame(self, frame: QImage):
        if not frame.isNull():
            self.video_display.setPixmap(QPixmap.fromImage(frame))

    @Slot(dict)
    def _on_playback_state_changed(self, state: Dict):
        playback_state = state.get("state")
        stream_info = state.get("stream_info", {})
        error_message = state.get("error_message", "")
        current_position = state.get("position_s", 0.0)
        total_duration = state.get("duration_s", 0.0)
        
        has_stream = stream_info.get("title") is not None and total_duration > 0
        self.play_pause_button.setEnabled(has_stream)
        self.seek_slider.setEnabled(has_stream)

        if playback_state == PlaybackState.PLAYING:
            self.play_pause_button.setText("⏸")
        else:
            self.play_pause_button.setText("▶")
        
        self.title_label.setText(stream_info.get("title", "No stream loaded"))
        if not self.url_input.hasFocus():
             self.url_input.setText(state.get("url", ""))

        status_text = f"Status: {playback_state}"
        if playback_state == PlaybackState.ERROR:
            self.status_label.setStyleSheet("color: red;")
            status_text = f"Status: ERROR"
            self.video_display.setText(f"Error:\n{error_message[:100]}")
        elif playback_state == PlaybackState.PLAYING:
            self.status_label.setStyleSheet("color: lightgreen;")
        elif playback_state == PlaybackState.STOPPED:
             self.video_display.setText("Video stream will appear here")
             self.video_display.setPixmap(QPixmap())
        else:
            self.status_label.setStyleSheet("color: lightgray;")
        
        self.status_label.setText(status_text)

        self.time_label.setText(self._format_time(current_position))
        self.duration_label.setText(self._format_time(total_duration))

        if total_duration > 0 and not self.seek_slider.isSliderDown():
            slider_value = int((current_position / total_duration) * 1000)
            self.seek_slider.setValue(slider_value)
        elif total_duration == 0:
            self.seek_slider.setValue(0)
    
    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_playback_state_changed(state)
        super().updateFromLogic()

class YouTubePlayerNode(Node):
    NODE_TYPE = "YouTube Player"
    UI_CLASS = YouTubePlayerNodeItem
    CATEGORY = "Generators"
    DESCRIPTION = "Streams audio and video from a YouTube video URL."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_output("out", data_type=np.ndarray)
        self.emitter = YouTubePlayerSignalEmitter()
        self._lock = threading.Lock()

        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        
        self._audio_buffer = deque()
        self._state_snapshot: Dict = {
            "state": PlaybackState.STOPPED, "url": "", "stream_info": {},
            "error_message": "", "position_s": 0.0, "duration_s": 0.0
        }
        self._playback_state = PlaybackState.STOPPED
        self._seek_request_s: float = -1.0

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._state_snapshot.copy()

    def load_url_if_present(self):
        """Called by the UI after init to load a URL from a saved file."""
        with self._lock:
            url = self._state_snapshot.get("url")
        if url:
            logger.info(f"[{self.name}] Auto-loading URL from saved state: {url}")
            self.load_url(url)
            # Restore position after loading info
            with self._lock:
                 initial_seek_pos = self._state_snapshot.get("position_s", 0.0)
            if initial_seek_pos > 0:
                 self._seek_request_s = initial_seek_pos


    def load_url(self, url: str):
        self._stop_worker()
        state_to_emit = None
        with self._lock:
            self._playback_state = PlaybackState.LOADING
            self._state_snapshot.update(
                state=self._playback_state, url=url, stream_info={}, 
                error_message="", position_s=0.0, duration_s=0.0
            )
            state_to_emit = self._state_snapshot.copy()
        if state_to_emit:
            self.emitter.emit_state_change(state_to_emit)
        
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._stream_reader_loop, daemon=True)
        self._worker_thread.start()

    def toggle_playback(self):
        state_to_emit = None
        with self._lock:
            if not self._state_snapshot.get("stream_info").get("title"):
                return
            
            if self._playback_state in [PlaybackState.PAUSED, PlaybackState.STOPPED]:
                if self._state_snapshot.get("position_s", 0) >= self._state_snapshot.get("duration_s", 0) and self._state_snapshot.get("duration_s", 0) > 0:
                    self._seek_request_s = 0.0
                self._playback_state = PlaybackState.PLAYING
            else:
                self._playback_state = PlaybackState.PAUSED
            state_to_emit = self._update_state_snapshot_locked(state=self._playback_state)

        if state_to_emit:
            self.emitter.emit_state_change(state_to_emit)

    def seek(self, proportion: float):
        state_to_emit = None
        with self._lock:
            duration = self._state_snapshot.get("duration_s", 0.0)
            if duration <= 0: return
            
            target_s = max(0.0, min(duration, proportion * duration))
            self._seek_request_s = target_s
            self._playback_state = PlaybackState.BUFFERING
            state_to_emit = self._update_state_snapshot_locked(state=self._playback_state, position_s=target_s)
        
        if state_to_emit:
            self.emitter.emit_state_change(state_to_emit)

    def process(self, input_data: dict) -> dict:
        state_to_emit = None
        output_block = None
        with self._lock:
            if len(self._audio_buffer) > 0:
                output_block = self._audio_buffer.popleft()
                if self._playback_state == PlaybackState.BUFFERING:
                    self._playback_state = PlaybackState.PLAYING
                    state_to_emit = self._update_state_snapshot_locked(state=self._playback_state)
            else:
                if self._playback_state == PlaybackState.PLAYING:
                    self._playback_state = PlaybackState.BUFFERING
                    state_to_emit = self._update_state_snapshot_locked(state=self._playback_state)
                output_block = np.zeros((DEFAULT_BLOCKSIZE, 2), dtype=DEFAULT_DTYPE)
        
        if state_to_emit:
            self.emitter.emit_state_change(state_to_emit)
            
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
        with self._lock:
            self._audio_buffer.clear()
            self._seek_request_s = -1.0

    def _update_state_snapshot_locked(self, **kwargs) -> Dict:
        self._state_snapshot.update(kwargs)
        return self._state_snapshot.copy()

    def _get_youtube_info(self, url: str) -> Optional[Tuple[str, float, str, Stream]]:
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if not stream:
                stream = yt.streams.get_highest_resolution()
            if not stream:
                raise RuntimeError("No suitable MP4 stream found.")
            
            stream_url = stream.url
            title = yt.title or "Unknown Title"
            duration_s = yt.length or 0.0
            
            return title, duration_s, stream_url, stream
        except Exception as e:
            logger.error(f"[{self.name}] Error fetching YouTube info for {url}: {e}", exc_info=True)
            return None

    def _audio_reader_thread(self, pipe, start_time_s: float):
        frames_read_since_start = 0
        current_playback_time_s = start_time_s
        last_ui_update_time = time.monotonic()

        while not self._stop_event.is_set():
            with self._lock:
                is_paused = self._playback_state == PlaybackState.PAUSED
                is_buffer_full = len(self._audio_buffer) >= INTERNAL_BUFFER_MAX_BLOCKS
            
            if is_paused or is_buffer_full:
                time.sleep(0.01)
                continue

            raw_audio = pipe.read(FFMPEG_AUDIO_CHUNK_SIZE)
            if not raw_audio: break

            audio_array = np.frombuffer(raw_audio, dtype=np.int16).astype(DEFAULT_DTYPE) / 32768.0
            num_samples_in_chunk = audio_array.shape[0] // 2
            if num_samples_in_chunk == 0: continue
            
            reshaped_audio = audio_array.reshape(num_samples_in_chunk, 2)
            
            for i in range(0, num_samples_in_chunk, DEFAULT_BLOCKSIZE):
                block = reshaped_audio[i:i + DEFAULT_BLOCKSIZE]
                if block.shape[0] == DEFAULT_BLOCKSIZE:
                    with self._lock: self._audio_buffer.append(block)
                    frames_read_since_start += DEFAULT_BLOCKSIZE
            
            current_playback_time_s = start_time_s + (frames_read_since_start / DEFAULT_SAMPLERATE)
            
            now = time.monotonic()
            if now - last_ui_update_time > UI_UPDATE_INTERVAL_MS / 1000.0:
                state_to_emit = None
                with self._lock:
                    state_to_emit = self._update_state_snapshot_locked(position_s=current_playback_time_s)
                if state_to_emit: self.emitter.emit_state_change(state_to_emit)
                last_ui_update_time = now
        
        state_to_emit = None
        with self._lock:
            if current_playback_time_s >= self._state_snapshot.get("duration_s", 0) and self._state_snapshot.get("duration_s", 0) > 0:
                self._playback_state = PlaybackState.STOPPED
                self._state_snapshot["position_s"] = self._state_snapshot.get("duration_s", 0)
            else:
                self._playback_state = PlaybackState.STOPPED
            state_to_emit = self._update_state_snapshot_locked()
        if state_to_emit: self.emitter.emit_state_change(state_to_emit)

        logger.info(f"[{self.name}] Audio reader thread finished.")

    def _video_reader_thread(self, pipe):
        while not self._stop_event.is_set():
            raw_frame = pipe.read(VIDEO_FRAME_BYTES)
            if len(raw_frame) < VIDEO_FRAME_BYTES: break
            image = QImage(raw_frame, VIDEO_WIDTH, VIDEO_HEIGHT, QImage.Format.Format_ARGB32)
            self.emitter.emit_video_frame(image.copy())
            time.sleep(1/20)
        logger.info(f"[{self.name}] Video reader thread finished.")

    def _stream_reader_loop(self):
        url = self._state_snapshot["url"]
        title, duration_s, stream_url = (None, 0.0, None)
        
        try:
            info = self._get_youtube_info(url)
            if not info: raise RuntimeError("Failed to get YouTube video information.")
            title, duration_s, stream_url, _ = info
            state_to_emit = None
            with self._lock:
                state_to_emit = self._update_state_snapshot_locked(
                    stream_info={"title": title, "length": duration_s}, 
                    duration_s=duration_s
                )
            if state_to_emit: self.emitter.emit_state_change(state_to_emit)
        except Exception as e:
            state_to_emit = None
            with self._lock:
                state_to_emit = self._update_state_snapshot_locked(state=PlaybackState.ERROR, error_message=str(e))
            if state_to_emit: self.emitter.emit_state_change(state_to_emit)
            return

        current_stream_start_time = 0.0
        
        while not self._stop_event.is_set():
            audio_thread, video_thread = None, None
            
            with self._lock:
                if self._seek_request_s != -1.0:
                    current_stream_start_time = self._seek_request_s
                    self._seek_request_s = -1.0
                    self._audio_buffer.clear()
                elif self._ffmpeg_process is None:
                    current_stream_start_time = self._state_snapshot.get("position_s", 0.0)
                    if current_stream_start_time >= duration_s and duration_s > 0:
                        current_stream_start_time = 0.0
                
                if self._playback_state != PlaybackState.PAUSED:
                    self._playback_state = PlaybackState.BUFFERING
                    state_to_emit = self._update_state_snapshot_locked(state=self._playback_state)
            
            if state_to_emit: self.emitter.emit_state_change(state_to_emit)

            ffmpeg_command = [ 'ffmpeg', '-hide_banner', '-loglevel', 'error' ]
            if current_stream_start_time > 0:
                ffmpeg_command.extend(['-ss', str(current_stream_start_time)])
            
            ffmpeg_command.extend([
                '-i', stream_url,
                '-f', 's16le', '-ac', '2', '-ar', str(DEFAULT_SAMPLERATE), '-c:a', 'pcm_s16le', 'pipe:1',
                '-f', 'rawvideo', '-pix_fmt', 'bgra', '-vf', f'scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}', '-r', '15', '-c:v', 'rawvideo', 'pipe:2'
            ])
            
            try:
                creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                with self._lock:
                    self._ffmpeg_process = subprocess.Popen(
                        ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags
                    )
                if not self._ffmpeg_process: raise RuntimeError("Failed to start FFMPEG process.")

                audio_thread = threading.Thread(target=self._audio_reader_thread, args=(self._ffmpeg_process.stdout, current_stream_start_time), daemon=True)
                video_thread = threading.Thread(target=self._video_reader_thread, args=(self._ffmpeg_process.stderr,), daemon=True)
                audio_thread.start()
                video_thread.start()
                
                # --- THIS IS THE FIX ---
                # This loop is now interruptible by a seek request.
                while self._ffmpeg_process.poll() is None and not self._stop_event.is_set():
                    with self._lock:
                        if self._seek_request_s != -1.0:
                            logger.info(f"[{self.name}] Seek detected, interrupting current ffmpeg process.")
                            break # Exit the waiting loop to handle the seek
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"[{self.name}] FFmpeg pipeline error: {e}", exc_info=True)
                state_to_emit = None
                with self._lock:
                    state_to_emit = self._update_state_snapshot_locked(state=PlaybackState.ERROR, error_message=f"Stream error: {e}")
                if state_to_emit: self.emitter.emit_state_change(state_to_emit)
                break

            finally:
                if self._ffmpeg_process: self._ffmpeg_process.terminate()
                if audio_thread and audio_thread.is_alive(): audio_thread.join(timeout=1.0)
                if video_thread and video_thread.is_alive(): video_thread.join(timeout=1.0)
                with self._lock: self._ffmpeg_process = None
                
                if self._stop_event.is_set(): break
                with self._lock:
                    # If we broke out of the loop but it wasn't a seek, it was end-of-stream
                    if self._seek_request_s == -1.0:
                        self._playback_state = PlaybackState.STOPPED
                        state_to_emit = self._update_state_snapshot_locked()
                if state_to_emit: self.emitter.emit_state_change(state_to_emit)

                # If the stream just ended naturally, break out of the main while loop
                # instead of trying to restart it. The user must press play again.
                with self._lock:
                    if self._seek_request_s == -1.0:
                        break

        logger.info(f"[{self.name}] Main worker loop finished.")
        state_to_emit = None
        with self._lock:
            if self._playback_state != PlaybackState.ERROR:
                self._playback_state = PlaybackState.STOPPED
                state_to_emit = self._update_state_snapshot_locked(state=self._playback_state)
        if state_to_emit:
            self.emitter.emit_state_change(state_to_emit)

    def serialize_extra(self) -> dict:
        with self._lock:
            return {
                "youtube_url": self._state_snapshot.get("url", ""),
                "saved_position_s": self._state_snapshot.get("position_s", 0.0)
            }
            
    def deserialize_extra(self, data: dict):
        url = data.get("youtube_url")
        saved_position = data.get("saved_position_s", 0.0)
        if url:
            with self._lock:
                self._state_snapshot["url"] = url
                self._state_snapshot["position_s"] = saved_position