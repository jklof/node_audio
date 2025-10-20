import os
import threading
import logging
import time
import queue
from enum import Enum
from typing import Dict, Optional

import torch
import soundfile as sf

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE
from ui_elements import NodeItem, NODE_CONTENT_PADDING

# --- Qt Imports ---
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout, QComboBox
from PySide6.QtCore import Qt, Slot, QSignalBlocker, QTimer

# Configure logging for this plugin
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Configuration and Enums
# ==============================================================================
class FileFormat(Enum):
    """Enum for supported audio file formats."""

    WAV = "WAV"
    FLAC = "FLAC"


# Map our enum to the string identifiers that the soundfile library expects.
SOUNDFILE_FORMAT_MAP = {
    FileFormat.WAV: "WAV",
    FileFormat.FLAC: "FLAC",
}

# Map our enum to the default file extensions.
FILE_EXTENSION_MAP = {
    FileFormat.WAV: "wav",
    FileFormat.FLAC: "flac",
}

# Subtypes for high-quality audio. PCM_24 is a good standard.
SOUNDFILE_SUBTYPE_MAP = {
    FileFormat.WAV: "PCM_24",
    FileFormat.FLAC: "PCM_24",
}


# ==============================================================================
# 2. UI Class for the Audio File Writer
# ==============================================================================
class AudioFileWriterNodeItem(NodeItem):
    """Custom UI for the AudioFileWriterNode."""

    NODE_SPECIFIC_WIDTH = 240

    def __init__(self, node_logic: "AudioFileWriterNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )

        # File Selection
        self.select_file_button = QPushButton("Select Output File...")
        self.file_path_label = QLabel("File: Not selected")
        self.file_path_label.setWordWrap(True)

        # Format Selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        for fmt in FileFormat:
            self.format_combo.addItem(fmt.value, userData=fmt)
        format_layout.addWidget(self.format_combo)

        # Controls and Status
        self.record_button = QPushButton("Record")
        self.record_button.setCheckable(True)
        self.status_label = QLabel("Status: Idle")
        self.time_label = QLabel("Time: 00:00.0")

        # Layout Assembly
        layout.addWidget(self.select_file_button)
        layout.addWidget(self.file_path_label)
        layout.addLayout(format_layout)
        layout.addWidget(self.record_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.time_label)
        self.setContentWidget(self.container_widget)

        # Connect Signals
        self.select_file_button.clicked.connect(self._on_select_file_clicked)
        self.record_button.toggled.connect(self.node_logic.set_recording_state)
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)

    @Slot()
    def _on_select_file_clicked(self):
        """Opens a file dialog to choose the output file path."""
        current_format = self.format_combo.currentData()
        extension = FILE_EXTENSION_MAP.get(current_format, "wav")
        filter_str = f"{current_format.value} Files (*.{extension})"

        parent_widget = self.scene().views()[0] if self.scene() and self.scene().views() else None
        file_path, _ = QFileDialog.getSaveFileName(parent_widget, "Save Audio As", "", filter_str)

        if file_path:
            self.node_logic.set_file_path(file_path)

    @Slot(int)
    def _on_format_changed(self, index: int):
        """Notifies the logic node when the user selects a new format."""
        selected_format = self.format_combo.itemData(index)
        if selected_format:
            self.node_logic.set_format(selected_format)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        """Updates the entire UI based on a state dictionary from the logic node."""
        super()._on_state_updated_from_logic(state)

        # Update file path
        file_path = state.get("file_path")
        self.file_path_label.setText(f"File: {os.path.basename(file_path) if file_path else 'Not selected'}")
        self.file_path_label.setToolTip(file_path or "")

        # Update format combo
        file_format = state.get("file_format")
        if file_format:
            with QSignalBlocker(self.format_combo):
                index = self.format_combo.findData(file_format)
                if index != -1:
                    self.format_combo.setCurrentIndex(index)

        # Update record button and status
        is_recording = state.get("is_recording", False)
        status = state.get("status", "Idle")
        with QSignalBlocker(self.record_button):
            self.record_button.setChecked(is_recording)
        self.record_button.setText("Stop" if is_recording else "Record")
        self.record_button.setEnabled(bool(file_path))

        self.status_label.setText(f"Status: {status}")
        if "Error" in status:
            self.status_label.setStyleSheet("color: red;")
        elif is_recording:
            self.status_label.setStyleSheet("color: lightgreen;")
        else:
            self.status_label.setStyleSheet("color: lightgray;")

        # Update recording time
        rec_time_s = state.get("recording_time_s", 0.0)
        mins, secs = divmod(rec_time_s, 60)
        self.time_label.setText(f"Time: {int(mins):02d}:{secs:04.1f}")


# ==============================================================================
# 3. Logic Class for the Audio File Writer
# ==============================================================================
class AudioFileWriterNode(Node):
    NODE_TYPE = "Audio File Writer"
    UI_CLASS = AudioFileWriterNodeItem
    CATEGORY = "Input / Output"
    DESCRIPTION = "Records an incoming audio signal to a WAV or FLAC file."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("audio_in", data_type=torch.Tensor)
        self.add_output("monitor_out", data_type=torch.Tensor)  # <-- NEW: Monitor Output

        # --- Internal State ---
        self._file_path: Optional[str] = None
        self._file_format: FileFormat = FileFormat.WAV
        self._is_recording: bool = False
        self._status: str = "Idle"
        self._frames_written: int = 0

        # --- Threading for Non-Blocking I/O ---
        self._writer_thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self._stop_writer_event = threading.Event()

        # --- Timer for UI updates ---
        self._ui_update_timer = QTimer()
        self._ui_update_timer.setInterval(200)  # Update UI 5 times a second
        self._ui_update_timer.timeout.connect(self._emit_ui_update)
        self._ui_update_timer.start()

    def _get_state_snapshot_locked(self) -> Dict:
        """Returns a snapshot of the node's state for UI updates or serialization."""
        return {
            "file_path": self._file_path,
            "file_format": self._file_format,
            "is_recording": self._is_recording,
            "status": self._status,
            "recording_time_s": self._frames_written / DEFAULT_SAMPLERATE,
        }

    @Slot()
    def _emit_ui_update(self):
        """Periodically sends the latest state to the UI thread."""
        state = self.get_current_state_snapshot()
        self.ui_update_callback(state)

    @Slot(str)
    def set_file_path(self, path: str):
        with self._lock:
            if self._is_recording:
                return
            self._file_path = path

    @Slot(object)  # Using 'object' for the enum type
    def set_format(self, fmt: FileFormat):
        with self._lock:
            if self._is_recording:
                return
            self._file_format = fmt

    @Slot(bool)
    def set_recording_state(self, should_record: bool):
        """Starts or stops the recording process."""
        with self._lock:
            if should_record == self._is_recording:
                return

            if should_record and not self._file_path:
                self._status = "Error: No file selected"
                return

            self._is_recording = should_record

            if self._is_recording:
                # --- Start Recording ---
                self._stop_writer_event.clear()
                self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
                self._writer_thread.start()
                self._status = "Recording"
            else:
                # --- Stop Recording ---
                self._stop_writer_event.set()
                # Send a sentinel value to unblock the queue if it's waiting
                self._audio_queue.put(None)
                self._status = "Finalizing..."

    def process(self, input_data: dict) -> dict:
        """Real-time audio processing part. Must be fast and non-blocking."""
        audio_chunk = input_data.get("audio_in")

        # If no audio is connected, output None (silence)
        if not isinstance(audio_chunk, torch.Tensor):
            return {"monitor_out": None}

        with self._lock:
            is_recording_now = self._is_recording

        if is_recording_now:
            # --- IF RECORDING ---
            # 1. Queue the audio for writing to the file
            try:
                # The clone is important so the writer thread gets its own copy
                self._audio_queue.put_nowait(audio_chunk.clone())
            except queue.Full:
                logger.warning(f"[{self.name}] Audio buffer full, dropping data.")

            # 2. Pass the original audio through to the monitor output
            return {"monitor_out": audio_chunk}
        else:
            # --- IF NOT RECORDING ---
            # 1. Do not queue audio.
            # 2. Output silence with the same shape as the input.
            return {"monitor_out": torch.zeros_like(audio_chunk)}

    def _writer_loop(self):
        """
        Runs in a separate thread. Pulls audio from the queue and writes it to disk.
        This is where all the slow file I/O happens.
        """
        sf_file = None
        try:
            with self._lock:
                path = self._file_path
                fmt = self._file_format
                self._frames_written = 0

            sf_format = SOUNDFILE_FORMAT_MAP[fmt]
            sf_subtype = SOUNDFILE_SUBTYPE_MAP[fmt]

            # The channel count is determined by the first chunk of audio.
            first_chunk: Optional[torch.Tensor] = self._audio_queue.get()
            if first_chunk is None:  # Stopped before receiving any audio
                return

            channels = first_chunk.shape[0]

            with sf.SoundFile(path, "w", DEFAULT_SAMPLERATE, channels, sf_subtype, format=sf_format) as sf_file:
                logger.info(f"[{self.name}] Started writing to {path} ({channels}ch, {fmt.value})")

                # Write the first chunk that we already retrieved
                sf_file.write(first_chunk.numpy().T)
                with self._lock:
                    self._frames_written += first_chunk.shape[1]

                while not self._stop_writer_event.is_set():
                    # Block until a new chunk is available or the sentinel is received
                    chunk = self._audio_queue.get()
                    if chunk is None:  # Sentinel value means stop
                        break

                    # soundfile expects (samples, channels), so we transpose
                    sf_file.write(chunk.numpy().T)
                    with self._lock:
                        self._frames_written += chunk.shape[1]

        except Exception as e:
            logger.error(f"[{self.name}] Error in writer thread: {e}", exc_info=True)
            with self._lock:
                self._status = f"Error: {e}"
        finally:
            logger.info(f"[{self.name}] Writer thread finished for {self._file_path}.")
            with self._lock:
                # Clear the queue of any remaining data
                while not self._audio_queue.empty():
                    self._audio_queue.get_nowait()
                self._is_recording = False
                if "Error" not in self._status:
                    self._status = "Idle"

    def remove(self):
        """Ensures the writer thread is stopped when the node is deleted."""
        self._ui_update_timer.stop()
        self.set_recording_state(False)
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=1.0)
        super().remove()

    def serialize_extra(self) -> dict:
        with self._lock:
            return {
                "file_path": self._file_path,
                "file_format": self._file_format.name,  # Save enum by name
            }

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._file_path = data.get("file_path")
            format_name = data.get("file_format", "WAV")
            try:
                self._file_format = FileFormat[format_name]
            except KeyError:
                self._file_format = FileFormat.WAV
