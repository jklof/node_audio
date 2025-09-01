import os
import threading
import time
import logging
from collections import deque
from typing import Optional, Dict

import numpy as np
import soundfile as sf
import resampy

from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE
from ui_elements import NodeItem, NODE_CONTENT_PADDING

from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout, QHBoxLayout, QFileDialog, QPushButton
from PySide6.QtCore import Qt, Slot, Signal, QObject

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Constants for the Player ---
CHUNK_SIZE_FRAMES = 4096
INTERNAL_BUFFER_MAX_BLOCKS = 50
UI_UPDATE_INTERVAL_MS = 100


class PlaybackState:
    """Enum-like class for playback states."""

    STOPPED = "STOPPED"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    LOADING = "LOADING"
    ERROR = "ERROR"


class PlayerNodeSignalEmitter(QObject):
    """A dedicated QObject to safely emit signals to the UI thread."""

    playbackStateChanged = Signal(dict)

    def emit_state_change(self, state_dict: Dict):
        try:
            self.playbackStateChanged.emit(state_dict.copy())
        except RuntimeError as e:
            logger.debug(f"Signal emitter caught RuntimeError: {e}")


class AudioFilePlayerNodeItem(NodeItem):
    """Simplified UI for the AudioFilePlayerNode, with controls removed."""

    NODE_SPECIFIC_WIDTH = 250

    def __init__(self, node_logic: "AudioFilePlayerNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(5)

        self.load_button = QPushButton("Load Audio File...")
        main_layout.addWidget(self.load_button)

        self.filename_label = QLabel("No file loaded")
        self.filename_label.setWordWrap(True)
        self.fileinfo_label = QLabel("Info: N/A")
        main_layout.addWidget(self.filename_label)
        main_layout.addWidget(self.fileinfo_label)

        time_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("▶")
        self.play_pause_button.setFixedSize(25, 25)
        self.play_pause_button.setEnabled(False)

        self.time_label = QLabel("00:00")
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.duration_label = QLabel("00:00")

        time_layout.addWidget(self.play_pause_button)
        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.seek_slider, stretch=1)
        time_layout.addWidget(self.duration_label)
        main_layout.addLayout(time_layout)

        self.setContentWidget(self.container_widget)

        self.load_button.clicked.connect(self._on_load_button_clicked)
        self.play_pause_button.clicked.connect(self._on_play_pause_clicked)
        self.seek_slider.sliderReleased.connect(self._on_seek)
        self.node_logic.emitter.playbackStateChanged.connect(self._on_playback_state_changed)

        self.updateFromLogic()

    def _format_time(self, seconds: float) -> str:
        if seconds < 0:
            return "00:00"
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    @Slot()
    def _on_load_button_clicked(self):
        """Opens a file dialog to load an audio file."""
        parent_widget = self.scene().views()[0] if self.scene() and self.scene().views() else None
        file_path, _ = QFileDialog.getOpenFileName(
            parent_widget, "Open Audio File", "", "Audio Files (*.wav *.flac *.mp3 *.ogg *.aiff)"
        )
        if file_path:
            self.node_logic.load_file(file_path)

    @Slot()
    def _on_play_pause_clicked(self):
        """Signals the logic node to toggle its playback state."""
        self.node_logic.toggle_playback()

    @Slot()
    def _on_seek(self):
        seek_proportion = self.seek_slider.value() / 1000.0
        self.node_logic.seek(seek_proportion)

    @Slot(dict)
    def _on_playback_state_changed(self, state: Dict):
        playback_state = state.get("state")
        has_file = state.get("file_path") is not None

        self.play_pause_button.setEnabled(has_file)
        if playback_state == PlaybackState.PLAYING:
            self.play_pause_button.setText("⏸")
            self.play_pause_button.setToolTip("Pause")
        else:
            self.play_pause_button.setText("▶")
            self.play_pause_button.setToolTip("Play")

        file_path = state.get("file_path")
        self.filename_label.setText(os.path.basename(file_path) if file_path else "No file loaded")
        self.filename_label.setToolTip(file_path or "")

        info = state.get("file_info", {})
        channels = info.get("channels", "N/A")
        samplerate = info.get("samplerate", "N/A")
        self.fileinfo_label.setText(f"Info: {channels} ch @ {samplerate} Hz")

        position = state.get("position", 0.0)
        duration = state.get("duration", 0.0)
        self.time_label.setText(self._format_time(position))
        self.duration_label.setText(self._format_time(duration))

        if duration > 0:
            self.seek_slider.setEnabled(True)
            slider_pos = int((position / duration) * 1000.0)
            if not self.seek_slider.isSliderDown():
                self.seek_slider.setValue(slider_pos)
        else:
            self.seek_slider.setEnabled(False)
            self.seek_slider.setValue(0)

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_playback_state_changed(state)
        super().updateFromLogic()


class AudioFilePlayerNode(Node):
    NODE_TYPE = "Audio File Player"
    UI_CLASS = AudioFilePlayerNodeItem
    CATEGORY = "Generators"
    DESCRIPTION = "Streams audio from a file. Playback is controlled by its own play/pause button."
    IS_CLOCK_SOURCE = False

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_output("out", data_type=np.ndarray)
        self.emitter = PlayerNodeSignalEmitter()
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state_snapshot: Dict = {
            "state": PlaybackState.STOPPED,
            "file_path": None,
            "file_info": {},
            "position": 0.0,
            "duration": 0.0,
        }
        self._file_path: Optional[str] = None
        self._seek_request_proportion: float = -1.0
        self._playback_state = PlaybackState.STOPPED
        self._user_intended_state = PlaybackState.STOPPED
        self._audio_buffer = deque()
        self._output_channels = 1
        self._initial_seek_seconds: float = -1.0

    def load_file(self, file_path: str):
        logger.info(f"[{self.name}] Load requested for: {file_path}")
        self._stop_worker()

        state_to_emit = None
        with self._lock:
            self._file_path = file_path
            self._playback_state = PlaybackState.LOADING
            state_to_emit = self._update_state_snapshot_locked(
                state=self._playback_state, file_path=self._file_path, file_info={}, duration=0.0, position=0.0
            )

        if state_to_emit:
            self.emitter.emit_state_change(state_to_emit)

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._file_reader_loop, daemon=True)
        self._worker_thread.start()

    def toggle_playback(self):
        state_to_emit = None
        with self._lock:
            if not self._file_path or self._playback_state in [PlaybackState.LOADING, PlaybackState.ERROR]:
                return

            if self._playback_state == PlaybackState.PLAYING:
                new_state = PlaybackState.PAUSED
                logger.info(f"[{self.name}] Playback PAUSED by UI toggle.")
            else:
                new_state = PlaybackState.PLAYING
                logger.info(f"[{self.name}] Playback RESUMED by UI toggle.")

            self._playback_state = new_state
            self._user_intended_state = new_state
            state_to_emit = self._update_state_snapshot_locked(state=self._playback_state)

        if state_to_emit:
            self.emitter.emit_state_change(state_to_emit)

    def seek(self, proportion: float):
        with self._lock:
            if 0.0 <= proportion <= 1.0:
                self._seek_request_proportion = proportion
                logger.info(f"[{self.name}] Seek requested to {proportion*100:.1f}%")

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._state_snapshot.copy()

    def start(self):
        """
        Called when graph processing starts.
        This method restores the node's effective playback state to match
        the user's intended state, allowing playback to resume automatically.
        """
        super().start()
        state_to_emit = None
        with self._lock:
            if self._file_path and self._user_intended_state == PlaybackState.PLAYING:
                self._playback_state = PlaybackState.PLAYING
                logger.info(f"[{self.name}] Graph started. Resuming playback as per user intent.")
                state_to_emit = self._update_state_snapshot_locked(state=self._playback_state)
            else:
                logger.info(f"[{self.name}] Graph started. Node is active but will remain paused.")

        if state_to_emit:
            self.emitter.emit_state_change(state_to_emit)

    def stop(self):
        """
        Called when graph processing stops.
        Even though no ticks are processed when the graph is stopped,
        we explicitly pause the worker thread to prevent it from
        needlessly reading the file into the buffer. The user's
        intended state is preserved in a separate variable.
        """
        super().stop()
        state_to_emit = None
        with self._lock:
            if self._playback_state == PlaybackState.PLAYING:
                self._playback_state = PlaybackState.PAUSED
                logger.info(f"[{self.name}] Graph stopped. Forcing effective state to PAUSED to idle worker thread.")
                state_to_emit = self._update_state_snapshot_locked(state=self._playback_state)

        if state_to_emit:
            self.emitter.emit_state_change(state_to_emit)

    def remove(self):
        logger.info(f"[{self.name}] Remove called. Stopping worker thread.")
        self._stop_worker()
        super().remove()

    def process(self, input_data: dict) -> dict:
        with self._lock:
            if len(self._audio_buffer) > 0:
                return {"out": self._audio_buffer.popleft()}
            else:
                return {"out": np.zeros((DEFAULT_BLOCKSIZE, self._output_channels), dtype=DEFAULT_DTYPE)}

    def _stop_worker(self):
        if self._worker_thread and self._worker_thread.is_alive():
            self._stop_event.set()
            logger.debug(f"[{self.name}] Waiting for worker thread to join...")
            self._worker_thread.join(timeout=1.0)
            if self._worker_thread.is_alive():
                logger.warning(f"[{self.name}] Worker thread did not terminate in time.")
        self._worker_thread = None

    def _update_state_snapshot_locked(self, **kwargs) -> Dict:
        """
        Helper to update the state dictionary and return a copy for emission.
        MUST be called with the lock already held.
        """
        self._state_snapshot.update(kwargs)
        return self._state_snapshot.copy()

    def _file_reader_loop(self):
        current_frame = 0
        state_to_emit = None
        try:
            with sf.SoundFile(self._file_path, "r") as sf_file:
                file_info = {"samplerate": sf_file.samplerate, "channels": sf_file.channels, "frames": sf_file.frames}
                duration = sf_file.frames / sf_file.samplerate if sf_file.samplerate > 0 else 0.0
                # This buffer holds resampled audio data before it's chunked into blocks.
                resampling_buffer = np.zeros((0, file_info["channels"]), dtype="float32")
                with self._lock:
                    self._output_channels = file_info["channels"]
                    self._playback_state = PlaybackState.PLAYING
                    self._user_intended_state = PlaybackState.PLAYING
                    initial_seek_seconds = self._initial_seek_seconds
                    self._initial_seek_seconds = -1.0
                    state_to_emit = self._update_state_snapshot_locked(
                        state=self._playback_state,
                        file_path=self._file_path,
                        file_info=file_info,
                        duration=duration,
                        position=0.0,
                    )

                if state_to_emit:
                    self.emitter.emit_state_change(state_to_emit)

                if initial_seek_seconds > 0 and duration > 0:
                    seek_proportion = min(1.0, initial_seek_seconds / duration)
                    target_frame = int(seek_proportion * file_info["frames"])
                    sf_file.seek(target_frame)
                    current_frame = target_frame
                    logger.info(f"[{self.name}] Worker performed initial seek to {initial_seek_seconds:.2f}s")

                    state_to_emit = None
                    with self._lock:
                        state_to_emit = self._update_state_snapshot_locked(
                            position=(current_frame / file_info["samplerate"])
                        )

                    if state_to_emit:
                        self.emitter.emit_state_change(state_to_emit)

                last_ui_update_time = 0

                while not self._stop_event.is_set():
                    with self._lock:
                        current_playback_state = self._playback_state
                        seek_prop = self._seek_request_proportion
                        self._seek_request_proportion = -1.0

                    if seek_prop != -1.0:
                        target_frame = int(seek_prop * file_info["frames"])
                        sf_file.seek(target_frame)
                        current_frame = target_frame
                        with self._lock:
                            self._audio_buffer.clear()
                        # Also clear the intermediate resampling buffer on seek
                        resampling_buffer = np.zeros((0, file_info["channels"]), dtype="float32")
                        logger.info(f"[{self.name}] Worker seeked to frame {target_frame}")

                    if (
                        current_playback_state == PlaybackState.PLAYING
                        and len(self._audio_buffer) < INTERNAL_BUFFER_MAX_BLOCKS
                    ):
                        raw_chunk = sf_file.read(CHUNK_SIZE_FRAMES, dtype="float32", always_2d=True)

                        if raw_chunk is None or raw_chunk.shape[0] == 0:
                            logger.info(f"[{self.name}] End of file, looping back to start.")
                            sf_file.seek(0)
                            current_frame = 0
                            continue

                        current_frame += raw_chunk.shape[0]

                        if file_info["samplerate"] != DEFAULT_SAMPLERATE:
                            resampled_chunk = resampy.resample(
                                raw_chunk, file_info["samplerate"], DEFAULT_SAMPLERATE, axis=0, filter="kaiser_fast"
                            )
                        else:
                            resampled_chunk = raw_chunk

                        # Add new resampled data to the intermediate buffer
                        if resampled_chunk.shape[0] > 0:
                            resampling_buffer = np.vstack((resampling_buffer, resampled_chunk))

                        # Create as many fixed-size blocks as possible from the buffer
                        while resampling_buffer.shape[0] >= DEFAULT_BLOCKSIZE:
                            # Slice one block from the start
                            block = resampling_buffer[:DEFAULT_BLOCKSIZE]
                            # Remove the sliced block from the intermediate buffer
                            resampling_buffer = resampling_buffer[DEFAULT_BLOCKSIZE:]
                            # Add the complete block to the output buffer for consumption
                            with self._lock:
                                self._audio_buffer.append(block)
                    else:
                        time.sleep(0.01)

                    now = time.monotonic()
                    if now - last_ui_update_time > UI_UPDATE_INTERVAL_MS / 1000.0:
                        position = current_frame / file_info["samplerate"]
                        state_to_emit = None
                        with self._lock:
                            state_to_emit = self._update_state_snapshot_locked(
                                position=position, state=current_playback_state
                            )

                        if state_to_emit:
                            self.emitter.emit_state_change(state_to_emit)

                        last_ui_update_time = now

        except Exception as e:
            logger.error(f"[{self.name}] Error in file reader thread: {e}", exc_info=True)
            state_to_emit = None
            with self._lock:
                state_to_emit = self._update_state_snapshot_locked(state=PlaybackState.ERROR, file_path=self._file_path)

            if state_to_emit:
                self.emitter.emit_state_change(state_to_emit)

        logger.info(f"[{self.name}] File reader thread finished.")
        state_to_emit = None
        with self._lock:
            if self._playback_state != PlaybackState.ERROR:
                state_to_emit = self._update_state_snapshot_locked(state=PlaybackState.STOPPED, position=0.0)

        if state_to_emit:
            self.emitter.emit_state_change(state_to_emit)

    def serialize_extra(self) -> dict:
        with self._lock:
            # We can use the getter here as a shortcut
            state = self._state_snapshot
            return {"file_path": state.get("file_path"), "position_seconds": state.get("position", 0.0)}

    def deserialize_extra(self, data: dict):
        file_path = data.get("file_path")
        position = data.get("position_seconds", 0.0)

        if file_path and os.path.exists(file_path):
            with self._lock:
                self._initial_seek_seconds = position
            self.load_file(file_path)
