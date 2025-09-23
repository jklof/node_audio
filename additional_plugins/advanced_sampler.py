import os
import threading
import logging
import time
from typing import Dict, Optional

import torch
import numpy as np
import soundfile as sf
import torchaudio.transforms as T

from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE
from ui_elements import NodeItem, NODE_CONTENT_PADDING

from PySide6.QtCore import Qt, Signal, Slot, QObject, QRunnable, QThreadPool, QPointF, QSize, QRectF, QSignalBlocker
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QPaintEvent, QPainterPath, QCursor
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QFileDialog,
    QSizePolicy,
    QHBoxLayout,
    QDoubleSpinBox,
)

logger = logging.getLogger(__name__)


class SampleLoadSignaller(QObject):
    load_finished = Signal(tuple)


class SampleLoadRunnable(QRunnable):
    def __init__(self, file_path: str, target_sr: int, signaller: SampleLoadSignaller):
        super().__init__()
        self.file_path, self.target_sr, self.signaller = file_path, target_sr, signaller

    def run(self):
        try:
            audio_data_np, source_sr = sf.read(self.file_path, dtype="float32", always_2d=True)
            audio_data = torch.from_numpy(audio_data_np.T)
            if source_sr != self.target_sr:
                resampler = T.Resample(orig_freq=source_sr, new_freq=self.target_sr, dtype=audio_data.dtype)
                audio_data = resampler(audio_data)
            max_val = torch.max(torch.abs(audio_data))
            if max_val > 0:
                audio_data /= max_val
            self.signaller.load_finished.emit(("success", audio_data, self.file_path))
        except Exception as e:
            self.signaller.load_finished.emit(("failure", f"Failed to load sample: {e}", self.file_path))


class WaveformDisplayWidget(QWidget):
    clipRangeChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(100)
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self.setMouseTracking(True)  # Required for hover events
        self._full_waveform: Optional[torch.Tensor] = None
        self._downsampled_waveform: Optional[np.ndarray] = None
        self._view_start_norm, self._view_end_norm = 0.0, 1.0
        self._clip_start_norm, self._clip_end_norm = 0.0, 1.0
        self._playhead_norm: Optional[float] = None
        self._is_panning = False
        self._last_pan_pos = QPointF()
        self._drag_mode = "none"
        self._drag_start_x_norm = 0.0
        self._drag_orig_clip_start, self._drag_orig_clip_end = 0.0, 0.0

    def sizeHint(self) -> QSize:
        return QSize(300, 100)

    def set_waveform(self, waveform_data: Optional[torch.Tensor]):
        self._full_waveform = waveform_data
        self._view_start_norm, self._view_end_norm = 0.0, 1.0
        self._clip_start_norm, self._clip_end_norm = 0.0, 1.0
        self._playhead_norm = None
        self._resample_for_display()
        self.update()

    def set_clip_range(self, start_norm: float, end_norm: float):
        if self._clip_start_norm != start_norm or self._clip_end_norm != end_norm:
            self._clip_start_norm, self._clip_end_norm = start_norm, end_norm
            self.update()

    def set_playhead(self, playhead_norm: Optional[float]):
        if self._playhead_norm != playhead_norm:
            self._playhead_norm = playhead_norm
            self.update()

    def is_dragging(self) -> bool:
        return self._drag_mode != "none"

    def _resample_for_display(self):
        if self._full_waveform is None or self.width() <= 0:
            self._downsampled_waveform = None
            return
        total_frames = self._full_waveform.shape[1]
        view_start_frame, view_end_frame = int(self._view_start_norm * total_frames), int(
            self._view_end_norm * total_frames
        )
        visible_frames = view_end_frame - view_start_frame
        if visible_frames <= 0:
            self._downsampled_waveform = None
            return
        step = max(1.0, visible_frames / self.width())
        indices = np.arange(0, visible_frames, step).astype(int) + view_start_frame
        indices = np.clip(indices, 0, total_frames - 1)
        self._downsampled_waveform = self._full_waveform[0, indices].numpy()
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        if self._downsampled_waveform is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a sample to begin")
            return
        clip_start_x, clip_end_x = self.norm_to_pixel(self._clip_start_norm), self.norm_to_pixel(self._clip_end_norm)
        painter.fillRect(QRectF(clip_start_x, 0, clip_end_x - clip_start_x, h), QColor(80, 60, 0, 100))
        path = QPainterPath()
        num_points = len(self._downsampled_waveform)
        if num_points > 0:
            path.moveTo(0, h / 2 * (1 - self._downsampled_waveform[0]))
            for i in range(1, num_points):
                x = (i / (num_points - 1)) * w
                y = h / 2 * (1 - self._downsampled_waveform[i])
                path.lineTo(x, y)
            painter.setPen(QPen(QColor("orange"), 1.5))
            painter.drawPath(path)

        # --- Draw visual handles on top of the selection lines ---
        painter.setPen(QPen(QColor("yellow"), 2))
        painter.drawLine(int(clip_start_x), 0, int(clip_start_x), h)
        painter.drawLine(int(clip_end_x), 0, int(clip_end_x), h)

        # Draw small triangles at the top as grabber handles
        handle_width = 8
        handle_height = 8
        painter.setBrush(QColor("yellow"))
        painter.setPen(Qt.PenStyle.NoPen)

        # Start handle triangle
        start_handle_path = QPainterPath()
        start_handle_path.moveTo(clip_start_x, 0)
        start_handle_path.lineTo(clip_start_x + handle_width, 0)
        start_handle_path.lineTo(clip_start_x, handle_height)
        start_handle_path.closeSubpath()
        painter.drawPath(start_handle_path)

        # End handle triangle
        end_handle_path = QPainterPath()
        end_handle_path.moveTo(clip_end_x, 0)
        end_handle_path.lineTo(clip_end_x - handle_width, 0)
        end_handle_path.lineTo(clip_end_x, handle_height)
        end_handle_path.closeSubpath()
        painter.drawPath(end_handle_path)

        if self._playhead_norm is not None and self._clip_start_norm <= self._playhead_norm <= self._clip_end_norm:
            playhead_x = self.norm_to_pixel(self._playhead_norm)
            painter.setPen(QPen(QColor(100, 255, 100), 2))
            painter.drawLine(int(playhead_x), 0, int(playhead_x), h)

    def mousePressEvent(self, event):
        if self._full_waveform is None:
            return

        # --- Right-click panning logic ---
        if event.button() == Qt.MouseButton.RightButton:
            self._is_panning = True
            self._last_pan_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            pos_norm = self.pixel_to_norm(event.position().x())

            # --- SELECTION LOGIC ---
            clip_start_x, clip_end_x = self.norm_to_pixel(self._clip_start_norm), self.norm_to_pixel(
                self._clip_end_norm
            )
            handle_width = 10
            is_on_start = abs(event.position().x() - clip_start_x) < handle_width
            is_on_end = abs(event.position().x() - clip_end_x) < handle_width
            is_inside = clip_start_x < event.position().x() < clip_end_x

            if is_on_start:
                self._drag_mode = "start"
            elif is_on_end:
                self._drag_mode = "end"
            elif is_inside:
                self._drag_mode = "move_both"
            else:
                self._drag_mode = "select"
                self._clip_start_norm = pos_norm
                self._clip_end_norm = pos_norm

            self._drag_start_x_norm = pos_norm
            self._drag_orig_clip_start, self._drag_orig_clip_end = self._clip_start_norm, self._clip_end_norm
            self.update()

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._full_waveform is None:
            return

        # --- cursor feedback when not dragging ---
        if not self._is_panning and self._drag_mode == "none":
            is_in_selection_zone = event.position().y() < self.height() / 2
            if is_in_selection_zone:
                clip_start_x = self.norm_to_pixel(self._clip_start_norm)
                clip_end_x = self.norm_to_pixel(self._clip_end_norm)
                handle_width = 10
                is_on_start = abs(event.position().x() - clip_start_x) < handle_width
                is_on_end = abs(event.position().x() - clip_end_x) < handle_width
                is_inside = clip_start_x < event.position().x() < clip_end_x

                if is_on_start or is_on_end:
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                elif is_inside:
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                else:
                    self.setCursor(Qt.CursorShape.IBeamCursor)
            else:
                self.setCursor(Qt.CursorShape.IBeamCursor)

        pos_norm = self.pixel_to_norm(event.position().x())
        if self._is_panning:
            delta_x = event.position().x() - self._last_pan_pos.x()
            self._last_pan_pos = event.position()
            view_range = self._view_end_norm - self._view_start_norm
            pan_amount = -(delta_x / self.width()) * view_range
            new_start, new_end = self._view_start_norm + pan_amount, self._view_end_norm + pan_amount
            if new_start < 0:
                new_end, new_start = new_end - new_start, 0
            if new_end > 1.0:
                new_start, new_end = new_start - (new_end - 1.0), 1.0
            self._view_start_norm, self._view_end_norm = max(0.0, new_start), min(1.0, new_end)
            self._resample_for_display()
            event.accept()
        elif self._drag_mode != "none":
            delta_norm = pos_norm - self._drag_start_x_norm
            if self._drag_mode == "start":
                self._clip_start_norm = min(max(pos_norm, self._view_start_norm), self._drag_orig_clip_end)
            elif self._drag_mode == "end":
                self._clip_end_norm = max(min(pos_norm, self._view_end_norm), self._drag_orig_clip_start)
            elif self._drag_mode == "move_both":
                new_start, new_end = self._drag_orig_clip_start + delta_norm, self._drag_orig_clip_end + delta_norm
                if new_start >= self._view_start_norm and new_end <= self._view_end_norm:
                    self._clip_start_norm, self._clip_end_norm = new_start, new_end
            elif self._drag_mode == "select":
                self._clip_start_norm, self._clip_end_norm = min(pos_norm, self._drag_start_x_norm), max(
                    pos_norm, self._drag_start_x_norm
                )
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # --- Reset cursor on release ---
        if event.button() == Qt.MouseButton.RightButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.IBeamCursor)
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton and self._drag_mode != "none":
            if self._clip_start_norm > self._clip_end_norm:
                self._clip_start_norm, self._clip_end_norm = self._clip_end_norm, self._clip_start_norm
            self.clipRangeChanged.emit(self._clip_start_norm, self._clip_end_norm)
            self._drag_mode = "none"
            self.setCursor(Qt.CursorShape.IBeamCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if self._full_waveform is None:
            return
        zoom_factor = 1.25 if event.angleDelta().y() < 0 else 1 / 1.25
        mouse_pos_norm = self.pixel_to_norm(event.position().x())
        view_range = self._view_end_norm - self._view_start_norm
        new_range = view_range * zoom_factor
        self._view_start_norm = mouse_pos_norm - (mouse_pos_norm - self._view_start_norm) * zoom_factor
        self._view_end_norm = self._view_start_norm + new_range
        self._clamp_view()
        self._resample_for_display()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._view_start_norm, self._view_end_norm = 0.0, 1.0
            self._resample_for_display()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def _clamp_view(self):
        self._view_start_norm, self._view_end_norm = max(0.0, self._view_start_norm), min(1.0, self._view_end_norm)
        if self._view_end_norm - self._view_start_norm < 1e-6:
            self._view_end_norm = self._view_start_norm + 1e-6

    def pixel_to_norm(self, x: float) -> float:
        view_range = self._view_end_norm - self._view_start_norm
        if abs(view_range) < 1e-9:
            return self._view_start_norm
        return self._view_start_norm + (x / self.width()) * view_range

    def norm_to_pixel(self, norm_val: float) -> float:
        view_range = self._view_end_norm - self._view_start_norm
        if abs(view_range) < 1e-9:
            return 0
        return ((norm_val - self._view_start_norm) / view_range) * self.width()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resample_for_display()


class AdvancedSamplePlayerNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 350

    def __init__(self, node_logic: "AdvancedSamplePlayerNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)
        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        top_row_layout = QHBoxLayout()
        self.load_button, self.filename_label = QPushButton("Load Sample"), QLabel("File: None")
        self.filename_label.setWordWrap(True)
        top_row_layout.addWidget(self.load_button)
        top_row_layout.addWidget(self.filename_label, stretch=1)
        main_layout.addLayout(top_row_layout)
        self.waveform_widget = WaveformDisplayWidget()
        main_layout.addWidget(self.waveform_widget)
        bottom_row_layout = QHBoxLayout()
        bottom_row_layout.addWidget(QLabel("Root Pitch (Hz):"))
        self.root_pitch_spinbox = QDoubleSpinBox()
        self.root_pitch_spinbox.setRange(20.0, 20000.0)
        self.root_pitch_spinbox.setDecimals(2)
        self.root_pitch_spinbox.setToolTip("The original pitch of the sample file.")
        bottom_row_layout.addWidget(self.root_pitch_spinbox)
        bottom_row_layout.addStretch()
        main_layout.addLayout(bottom_row_layout)
        self.setContentWidget(self.container_widget)
        self.load_button.clicked.connect(self._on_load_clicked)
        self.waveform_widget.clipRangeChanged.connect(self.node_logic.set_clip_range)
        self.root_pitch_spinbox.valueChanged.connect(self.node_logic.set_root_pitch)

    @Slot()
    def _on_load_clicked(self):
        parent = self.scene().views()[0] if self.scene() and self.scene().views() else None
        file_path, _ = QFileDialog.getOpenFileName(parent, "Load Sample", "", "Audio Files (*.wav *.flac *.aiff *.mp3)")
        if file_path:
            self.node_logic.load_new_file_and_reset(file_path)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        super()._on_state_updated_from_logic(state)
        if "waveform_data" in state and state["waveform_data"] is not None:
            self.waveform_widget.set_waveform(state["waveform_data"])
        filepath = state.get("filepath")
        self.filename_label.setText(f"File: {os.path.basename(filepath) if filepath else 'None'}")
        self.filename_label.setToolTip(filepath or "")
        if not self.waveform_widget.is_dragging():
            self.waveform_widget.set_clip_range(state.get("start_pos_norm", 0.0), state.get("end_pos_norm", 1.0))
        self.waveform_widget.set_playhead(state.get("playhead_norm"))
        root_pitch = state.get("root_pitch_hz", 261.63)
        with QSignalBlocker(self.root_pitch_spinbox):
            self.root_pitch_spinbox.setValue(root_pitch)


class AdvancedSamplePlayerNode(Node):
    NODE_TYPE = "Advanced Sample Player"
    CATEGORY = "Generators"
    DESCRIPTION = "Plays a selected clip from an audio sample with zoom, pan, and pitch control."
    UI_CLASS = AdvancedSamplePlayerNodeItem

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("trigger", data_type=bool)
        self.add_input("pitch", data_type=float)
        self.add_output("out", data_type=torch.Tensor)
        self.add_output("on_end", data_type=bool)
        self._filepath: Optional[str] = None
        self._status: str = "No Sample"
        self._audio_data: Optional[torch.Tensor] = None
        self._play_pos: float = 0.0
        self._is_playing: bool = False
        self._prev_trigger: bool = False
        self._root_pitch_hz: float = 261.63
        self._start_pos_norm: float = 0.0
        self._end_pos_norm: float = 1.0
        self._last_playhead_update_time: float = 0.0
        self._last_known_pitch_hz: float = self._root_pitch_hz
        self._silence_block = torch.zeros((1, DEFAULT_BLOCKSIZE), dtype=DEFAULT_DTYPE)
        self.loader_signaller = SampleLoadSignaller()
        self.loader_signaller.load_finished.connect(self._on_load_finished)

    def _get_state_snapshot_locked(self) -> Dict:
        playhead_norm = None
        if self._is_playing and self._audio_data is not None and self._audio_data.shape[1] > 0:
            playhead_norm = self._play_pos / self._audio_data.shape[1]
        return {
            "status": self._status,
            "filepath": self._filepath,
            "root_pitch_hz": self._root_pitch_hz,
            "start_pos_norm": self._start_pos_norm,
            "end_pos_norm": self._end_pos_norm,
            "playhead_norm": playhead_norm,
        }

    def load_new_file_and_reset(self, file_path: str):
        with self._lock:
            self._start_pos_norm, self._end_pos_norm = 0.0, 1.0
        self.load_file(file_path)

    def load_file(self, file_path: str):
        state_to_emit = None
        with self._lock:
            self._status, self._filepath = "Loading...", file_path
            state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)
        runnable = SampleLoadRunnable(file_path, DEFAULT_SAMPLERATE, self.loader_signaller)
        QThreadPool.globalInstance().start(runnable)

    @Slot(tuple)
    def _on_load_finished(self, result: tuple):
        status, data, filepath = result
        state_to_emit = None
        with self._lock:
            if filepath != self._filepath:
                return
            if status == "success":
                self._audio_data, self._status = data, "Ready"
                self._is_playing, self._play_pos = False, 0.0
                self._silence_block = torch.zeros((data.shape[0], DEFAULT_BLOCKSIZE), dtype=DEFAULT_DTYPE)
                state_to_emit = self._get_state_snapshot_locked()
                state_to_emit["waveform_data"] = self._audio_data.clone()
            else:
                self._audio_data, self._status = None, f"Error: {data}"
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float, float)
    def set_clip_range(self, start_norm: float, end_norm: float):
        state_to_emit = None
        with self._lock:
            self._start_pos_norm, self._end_pos_norm = start_norm, end_norm
            state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(float)
    def set_root_pitch(self, pitch_hz: float):
        state_to_emit = None
        with self._lock:
            self._root_pitch_hz = pitch_hz
            self._last_known_pitch_hz = pitch_hz
            state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    def process(self, input_data: dict) -> dict:
        trigger_socket_val = input_data.get("trigger")
        trigger = bool(trigger_socket_val) if trigger_socket_val is not None else False
        pitch_socket_val = input_data.get("pitch")
        output_block, on_end_signal = self._silence_block.clone(), False
        should_process, start_pos_samples, end_pos_samples, current_play_pos = False, 0, 0, 0
        speed_ratio = 1.0

        with self._lock:
            local_audio_data = self._audio_data
            if local_audio_data is None:
                self._prev_trigger = trigger
                return {"out": output_block, "on_end": on_end_signal}

            # --- PITCH LATCHING LOGIC ---
            if trigger:
                if pitch_socket_val is not None:
                    self._last_known_pitch_hz = float(pitch_socket_val)
            speed_ratio = self._last_known_pitch_hz / self._root_pitch_hz if self._root_pitch_hz != 0 else 1.0
            total_frames = local_audio_data.shape[1]
            start_pos_samples, end_pos_samples = int(self._start_pos_norm * total_frames), int(
                self._end_pos_norm * total_frames
            )
            if trigger and not self._prev_trigger:
                self._is_playing, self._play_pos = True, float(start_pos_samples)
            self._prev_trigger = trigger
            should_process = self._is_playing and total_frames > 1
            if should_process:
                current_play_pos = self._play_pos
                self._play_pos += DEFAULT_BLOCKSIZE * speed_ratio
                if self._play_pos >= end_pos_samples:
                    self._is_playing, on_end_signal = False, True
        if should_process:
            num_channels_sample, _ = local_audio_data.shape
            indices = current_play_pos + torch.arange(DEFAULT_BLOCKSIZE, dtype=torch.float32) * speed_ratio
            valid_mask = (indices < end_pos_samples - 1) & (indices >= start_pos_samples)
            valid_float_indices = indices[valid_mask]
            num_valid = len(valid_float_indices)
            if num_valid > 0:
                indices_floor = valid_float_indices.long()
                indices_ceil = indices_floor + 1
                fraction = (valid_float_indices - indices_floor).unsqueeze(0)
                sample_floor = local_audio_data.gather(1, indices_floor.expand(num_channels_sample, -1))
                sample_ceil = local_audio_data.gather(1, indices_ceil.expand(num_channels_sample, -1))
                interpolated_samples = sample_floor * (1.0 - fraction) + sample_ceil * fraction
                output_block[:, valid_mask] = interpolated_samples
        now = time.monotonic()
        if now > self._last_playhead_update_time + 0.05:
            self._last_playhead_update_time = now
            state_to_emit = self._get_state_snapshot_locked()
            if state_to_emit:
                self.ui_update_callback(state_to_emit)
        return {"out": output_block, "on_end": on_end_signal}

    def serialize_extra(self) -> dict:
        with self._lock:
            return {
                "filepath": self._filepath,
                "root_pitch_hz": self._root_pitch_hz,
                "start_pos_norm": self._start_pos_norm,
                "end_pos_norm": self._end_pos_norm,
            }

    def deserialize_extra(self, data: dict):
        filepath = data.get("filepath")
        if filepath and os.path.exists(filepath):
            self.load_file(filepath)
        state_to_emit = None
        with self._lock:
            self._root_pitch_hz = data.get("root_pitch_hz", 261.63)
            self._last_known_pitch_hz = self._root_pitch_hz
            self._start_pos_norm = data.get("start_pos_norm", 0.0)
            self._end_pos_norm = data.get("end_pos_norm", 1.0)
            state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)
