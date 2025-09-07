import torch
import numpy as np
import threading
import logging
from collections import deque
from typing import Dict

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE
from ui_elements import NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING

# --- UI and Qt Imports ---
from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout
from PySide6.QtCore import Qt, Slot, QSignalBlocker

EPSILON = 1e-9

# Configure logging for this plugin
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. UI Class for the AutoGain Node
# ==============================================================================
class AutoGainNodeItem(NodeItem):
    """
    UI for the AutoGainNode with intuitive controls for professional leveling.
    """

    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "AutoGainNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(4)

        # --- Create Intuitive Slider Controls ---
        self.target_slider, self.target_label = self._create_slider_control("Target Level", -40.0, 0.0, "{:.1f} dB")
        self.averaging_slider, self.averaging_label = self._create_slider_control(
            "Averaging Time", 0.5, 10.0, "{:.1f} s"
        )
        self.smoothing_slider, self.smoothing_label = self._create_slider_control(
            "Gain Smoothing", 50.0, 2000.0, "{:.0f} ms", is_log=True
        )

        for label, slider in [
            (self.target_label, self.target_slider),
            (self.averaging_label, self.averaging_slider),
            (self.smoothing_label, self.smoothing_slider),
        ]:
            main_layout.addWidget(label)
            main_layout.addWidget(slider)

        self.setContentWidget(self.container_widget)

        # --- Connect Signals ---
        self.target_slider.valueChanged.connect(self._on_target_changed)
        self.averaging_slider.valueChanged.connect(self._on_averaging_changed)
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_changed)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

    def _create_slider_control(
        self, name: str, min_val: float, max_val: float, fmt: str, is_log: bool = False
    ) -> tuple[QSlider, QLabel]:
        """Helper factory to create a labeled slider."""
        label = QLabel(f"{name}: ...")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)
        slider.setProperty("min_val", min_val)
        slider.setProperty("max_val", max_val)
        slider.setProperty("name", name)
        slider.setProperty("format", fmt)
        slider.setProperty("is_log", is_log)
        return slider, label

    def _map_slider_to_logical(self, slider: QSlider) -> float:
        min_val = slider.property("min_val")
        max_val = slider.property("max_val")
        norm = slider.value() / 1000.0
        if slider.property("is_log"):
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            return 10 ** (log_min + norm * (log_max - log_min))
        else:
            return min_val + norm * (max_val - min_val)

    def _map_logical_to_slider(self, slider: QSlider, value: float) -> int:
        min_val = slider.property("min_val")
        max_val = slider.property("max_val")
        if slider.property("is_log"):
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            safe_val = np.clip(value, min_val, max_val)
            norm = (np.log10(safe_val) - log_min) / (log_max - log_min)
            return int(norm * 1000.0)
        else:
            range_val = max_val - min_val
            if range_val == 0:
                return 0
            norm = (value - min_val) / range_val
            return int(np.clip(norm, 0.0, 1.0) * 1000.0)

    @Slot()
    def _on_target_changed(self):
        self.node_logic.set_target_db(self._map_slider_to_logical(self.target_slider))

    @Slot()
    def _on_averaging_changed(self):
        self.node_logic.set_averaging_time_s(self._map_slider_to_logical(self.averaging_slider))

    @Slot()
    def _on_smoothing_changed(self):
        self.node_logic.set_gain_smoothing_ms(self._map_slider_to_logical(self.smoothing_slider))

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        sliders_map = {
            "target_db": (self.target_slider, self.target_label),
            "averaging_time_s": (self.averaging_slider, self.averaging_label),
            "gain_smoothing_ms": (self.smoothing_slider, self.smoothing_label),
        }
        for key, (slider, label) in sliders_map.items():
            value = state.get(key, slider.property("min_val"))
            is_connected = key in self.node_logic.inputs and self.node_logic.inputs[key].connections
            slider.setEnabled(not is_connected)
            with QSignalBlocker(slider):
                slider.setValue(self._map_logical_to_slider(slider, value))
            label_text = f"{slider.property('name')}: {slider.property('format').format(value)}"
            if is_connected:
                label_text += " (ext)"
            label.setText(label_text)

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ==============================================================================
# 2. Logic Class for the AutoGain Node (Professional Leveler Algorithm)
# ==============================================================================
class AutoGainNode(Node):
    NODE_TYPE = "Auto Gain"
    UI_CLASS = AutoGainNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Automatically calculates gain to match a target signal level (RMS)."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("target_db", data_type=float)
        self.add_input("averaging_time_s", data_type=float)
        self.add_input("gain_smoothing_ms", data_type=float)
        self.add_output("gain_out", data_type=float)

        self._lock = threading.Lock()

        # --- Internal state parameters ---
        self._target_db: float = -14.0
        self._averaging_time_s: float = 3.0
        self._gain_smoothing_ms: float = 500.0

        # --- DSP State ---
        self._current_gain_db: float = -70.0  # Start silent
        self._rms_history: deque = deque(maxlen=1)
        self._recalculate_deque_size()

    def _recalculate_deque_size(self):
        """Calculates the required size of the history buffer based on averaging time."""
        num_blocks = int((self._averaging_time_s * DEFAULT_SAMPLERATE) / (DEFAULT_BLOCKSIZE + EPSILON))
        new_maxlen = max(1, num_blocks)
        if self._rms_history.maxlen != new_maxlen:
            self._rms_history = deque(self._rms_history, maxlen=new_maxlen)

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {
            "target_db": self._target_db,
            "averaging_time_s": self._averaging_time_s,
            "gain_smoothing_ms": self._gain_smoothing_ms,
        }

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    # --- Thread-safe setters ---
    @Slot(float)
    def set_target_db(self, value: float):
        state_to_emit = None
        with self._lock:
            if self._target_db != value:
                self._target_db = value
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_averaging_time_s(self, value: float):
        state_to_emit = None
        with self._lock:
            if self._averaging_time_s != value:
                self._averaging_time_s = value
                self._recalculate_deque_size()
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_gain_smoothing_ms(self, value: float):
        state_to_emit = None
        with self._lock:
            if self._gain_smoothing_ms != value:
                self._gain_smoothing_ms = value
                state_to_emit = self._get_current_state_snapshot_locked()
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"gain_out": 1.0}

        state_snapshot_to_emit = None
        with self._lock:
            ui_update_needed = False
            # Update parameters from sockets
            target_db_socket = input_data.get("target_db")
            if target_db_socket is not None and self._target_db != float(target_db_socket):
                self._target_db = float(target_db_socket)
                ui_update_needed = True
            avg_time_socket = input_data.get("averaging_time_s")
            if avg_time_socket is not None and self._averaging_time_s != float(avg_time_socket):
                self._averaging_time_s = float(avg_time_socket)
                self._recalculate_deque_size()
                ui_update_needed = True
            gain_smooth_socket = input_data.get("gain_smoothing_ms")
            if gain_smooth_socket is not None and self._gain_smoothing_ms != float(gain_smooth_socket):
                self._gain_smoothing_ms = float(gain_smooth_socket)
                ui_update_needed = True

            if ui_update_needed:
                state_snapshot_to_emit = self._get_current_state_snapshot_locked()

            target_db = self._target_db
            gain_smoothing_ms = self._gain_smoothing_ms

        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

        # --- STAGE 1: Long-Term Loudness Measurement ---
        mono_signal = torch.mean(signal, dim=0)
        rms_linear = torch.sqrt(torch.mean(torch.square(mono_signal)) + EPSILON)
        self._rms_history.append(rms_linear.item())

        long_term_rms_linear = np.mean(list(self._rms_history))
        long_term_rms_db = 20 * np.log10(long_term_rms_linear + EPSILON)

        # --- STAGE 2: Gain Correction ---
        gain_needed_db = target_db - long_term_rms_db

        # --- STAGE 3: Gain Smoothing ---
        samples_per_block = signal.shape[1]
        smoothing_samples = (gain_smoothing_ms / 1000.0) * DEFAULT_SAMPLERATE
        alpha = 1 - torch.exp(torch.tensor(-samples_per_block / (smoothing_samples + EPSILON)))
        self._current_gain_db += alpha * (gain_needed_db - self._current_gain_db)

        output_gain_linear = 10.0 ** (self._current_gain_db / 20.0)

        return {"gain_out": float(output_gain_linear)}

    def start(self):
        """Reset DSP state when processing starts."""
        with self._lock:
            # --- THE FIX: Start with a very low gain (effectively silent) ---
            self._current_gain_db = -70.0

            self._recalculate_deque_size()
            self._rms_history.clear()
        super().start()

    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()

    def deserialize_extra(self, data: dict):
        self.set_target_db(data.get("target_db", -14.0))
        self.set_averaging_time_s(data.get("averaging_time_s", 3.0))
        self.set_gain_smoothing_ms(data.get("gain_smoothing_ms", 500.0))
