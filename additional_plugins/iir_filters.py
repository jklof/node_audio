import torch
import numpy as np
import scipy.signal
import threading
import logging
from typing import Dict, Optional

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_DTYPE
from ui_elements import NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING

# --- UI and Qt Imports ---
from PySide6.QtWidgets import QWidget, QLabel, QComboBox, QSlider, QVBoxLayout
from PySide6.QtCore import Qt, Slot, QSignalBlocker

# --- Logging ---
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. UI Class for the Biquad (IIR) Filter Node (Unchanged)
# ==============================================================================
class BiquadFilterNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "BiquadFilterNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(5)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["Low Pass", "High Pass", "Band Pass", "Notch", "Peaking", "Low Shelf", "High Shelf"])
        main_layout.addWidget(self.type_combo)

        self.freq_slider, self.freq_label = self._create_slider_control("Freq", 20.0, 20000.0, "{:.0f} Hz", is_log=True)
        self.q_slider, self.q_label = self._create_slider_control("Q", 0.1, 10.0, "{:.2f}")
        self.gain_slider, self.gain_label = self._create_slider_control("Gain", -24.0, 24.0, "{:+.1f} dB")

        main_layout.addWidget(self.freq_label)
        main_layout.addWidget(self.freq_slider)
        main_layout.addWidget(self.q_label)
        main_layout.addWidget(self.q_slider)
        main_layout.addWidget(self.gain_label)
        main_layout.addWidget(self.gain_slider)

        self.setContentWidget(self.container_widget)

        self.type_combo.currentTextChanged.connect(self.node_logic.set_filter_type)
        self.freq_slider.valueChanged.connect(self._on_freq_changed)
        self.q_slider.valueChanged.connect(self._on_q_changed)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

        self.updateFromLogic()

    def _create_slider_control(self, name: str, min_val: float, max_val: float, fmt: str, is_log: bool = False):
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
        min_val, max_val = slider.property("min_val"), slider.property("max_val")
        norm = slider.value() / 1000.0
        if slider.property("is_log"):
            log_min, log_max = np.log10(min_val), np.log10(max_val)
            return 10 ** (log_min + norm * (log_max - log_min))
        return min_val + norm * (max_val - min_val)

    def _map_logical_to_slider(self, slider: QSlider, value: float) -> int:
        min_val, max_val = slider.property("min_val"), slider.property("max_val")
        safe_value = np.clip(value, min_val, max_val)
        if slider.property("is_log"):
            log_min, log_max = np.log10(min_val), np.log10(max_val)
            norm = (np.log10(safe_value) - log_min) / (log_max - log_min)
            return int(norm * 1000.0)
        range_val = max_val - min_val
        norm = (safe_value - min_val) / (range_val or 1.0)
        return int(np.clip(norm, 0.0, 1.0) * 1000.0)

    @Slot()
    def _on_freq_changed(self):
        self.node_logic.set_cutoff_freq(self._map_slider_to_logical(self.freq_slider))

    @Slot()
    def _on_q_changed(self):
        self.node_logic.set_q(self._map_slider_to_logical(self.q_slider))

    @Slot()
    def _on_gain_changed(self):
        self.node_logic.set_gain_db(self._map_slider_to_logical(self.gain_slider))

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        filter_type = state.get("filter_type", "Low Pass")
        with QSignalBlocker(self.type_combo):
            self.type_combo.setCurrentText(filter_type)
        for key, slider, label in [
            ("freq", self.freq_slider, self.freq_label),
            ("q", self.q_slider, self.q_label),
            ("gain_db", self.gain_slider, self.gain_label),
        ]:
            val = state.get(key)
            is_ext = state.get(f"is_{key}_ext", False)
            slider.setEnabled(not is_ext)
            with QSignalBlocker(slider):
                slider.setValue(self._map_logical_to_slider(slider, val))
            label_text = f"{slider.property('name')}: {slider.property('format').format(val)}"
            if is_ext:
                label_text += " (ext)"
            label.setText(label_text)
        q_visible = filter_type in ["Low Pass", "High Pass", "Band Pass", "Notch", "Peaking"]
        gain_visible = filter_type in ["Peaking", "Low Shelf", "High Shelf"]
        self.q_label.setVisible(q_visible)
        self.q_slider.setVisible(q_visible)
        self.gain_label.setVisible(gain_visible)
        self.gain_slider.setVisible(gain_visible)
        self.container_widget.adjustSize()
        self.update_geometry()

    @Slot()
    def updateFromLogic(self):
        self._on_state_updated(self.node_logic.get_current_state_snapshot())
        super().updateFromLogic()


# ==============================================================================
# 2. Logic Class for the Biquad (IIR) Filter Node
# ==============================================================================
class BiquadFilterNode(Node):
    NODE_TYPE = "Biquad Filter (IIR)"
    UI_CLASS = BiquadFilterNodeItem
    CATEGORY = "Filters"
    DESCRIPTION = "Applies a highly efficient IIR filter (EQ)."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self._lock = threading.Lock()
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("cutoff_freq", data_type=float)
        self.add_input("q", data_type=float)
        self.add_input("gain_db", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._filter_type: str = "Low Pass"
        self._cutoff_freq: float = 1000.0
        self._q: float = 0.707
        self._gain_db: float = 0.0
        self._params_dirty: bool = True

        self._b_coeffs: Optional[np.ndarray] = None
        self._a_coeffs: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None
        self._expected_channels: Optional[int] = None

    def _recalculate_coeffs(self):
        sr = DEFAULT_SAMPLERATE
        freq, q, gain_db = self._cutoff_freq, self._q, self._gain_db
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * freq / sr
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * q)
        b0, b1, b2, a0, a1, a2 = 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
        if self._filter_type == "Low Pass":
            b0 = (1 - cos_w0) / 2
            b1 = 1 - cos_w0
            b2 = (1 - cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
        elif self._filter_type == "High Pass":
            b0 = (1 + cos_w0) / 2
            b1 = -(1 + cos_w0)
            b2 = (1 + cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
        elif self._filter_type == "Band Pass":
            b0 = alpha
            b1 = 0
            b2 = -alpha
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
        elif self._filter_type == "Notch":
            b0 = 1
            b1 = -2 * cos_w0
            b2 = 1
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
        elif self._filter_type == "Peaking":
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A
        elif self._filter_type == "Low Shelf":
            beta = np.sqrt(A) * alpha * 2
            b0 = A * ((A + 1) - (A - 1) * cos_w0 + beta)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
            b2 = A * ((A + 1) - (A - 1) * cos_w0 - beta)
            a0 = (A + 1) + (A - 1) * cos_w0 + beta
            a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
            a2 = (A + 1) + (A - 1) * cos_w0 - beta
        elif self._filter_type == "High Shelf":
            beta = np.sqrt(A) * alpha * 2
            b0 = A * ((A + 1) + (A - 1) * cos_w0 + beta)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - beta)
            a0 = (A + 1) - (A - 1) * cos_w0 + beta
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
            a2 = (A + 1) - (A - 1) * cos_w0 - beta
        self._b_coeffs = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float32)
        self._a_coeffs = np.array([a0 / a0, a1 / a0, a2 / a0], dtype=np.float32)
        self._params_dirty = False
        logger.info(f"[{self.name}] Recalculated IIR coefficients for {self._filter_type}.")

    @Slot(str)
    def set_filter_type(self, f_type: str):
        with self._lock:
            if self._filter_type != f_type:
                self._filter_type = f_type
                self._params_dirty = True
        self.emitter.stateUpdated.emit(self.get_current_state_snapshot())

    @Slot(float)
    def set_cutoff_freq(self, freq: float):
        with self._lock:
            new_freq = max(20.0, min(float(freq), DEFAULT_SAMPLERATE / 2 - 1))
            if self._cutoff_freq != new_freq:
                self._cutoff_freq = new_freq
                self._params_dirty = True
        self.emitter.stateUpdated.emit(self.get_current_state_snapshot())

    @Slot(float)
    def set_q(self, q: float):
        with self._lock:
            new_q = max(0.1, float(q))
            if self._q != new_q:
                self._q = new_q
                self._params_dirty = True
        self.emitter.stateUpdated.emit(self.get_current_state_snapshot())

    @Slot(float)
    def set_gain_db(self, gain: float):
        with self._lock:
            if self._gain_db != float(gain):
                self._gain_db = float(gain)
                self._params_dirty = True
        self.emitter.stateUpdated.emit(self.get_current_state_snapshot())

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {
            "filter_type": self._filter_type,
            "freq": self._cutoff_freq,
            "q": self._q,
            "gain_db": self._gain_db,
            "is_freq_ext": "cutoff_freq" in self.inputs and self.inputs["cutoff_freq"].connections,
            "is_q_ext": "q" in self.inputs and self.inputs["q"].connections,
            "is_gain_db_ext": "gain_db" in self.inputs and self.inputs["gain_db"].connections,
        }

    def start(self):
        with self._lock:
            self._zi = None
            self._expected_channels = None
            self._params_dirty = True

    def process(self, input_data: dict) -> dict:
        signal_tensor = input_data.get("in")
        if not isinstance(signal_tensor, torch.Tensor):
            return {"out": None}

        num_channels, _ = signal_tensor.shape
        state_snapshot_to_emit = None

        with self._lock:
            freq_s = input_data.get("cutoff_freq")
            q_s = input_data.get("q")
            gain_s = input_data.get("gain_db")
            if freq_s is not None and self._cutoff_freq != float(freq_s):
                self._cutoff_freq = float(freq_s)
                self._params_dirty = True
            if q_s is not None and self._q != float(q_s):
                self._q = float(q_s)
                self._params_dirty = True
            if gain_s is not None and self._gain_db != float(gain_s):
                self._gain_db = float(gain_s)
                self._params_dirty = True

            if self._params_dirty:
                self._recalculate_coeffs()
                self._zi = None
                state_snapshot_to_emit = self._get_current_state_snapshot_locked()

            if self._expected_channels != num_channels:
                self._expected_channels = num_channels
                self._zi = None

            if self._zi is None:
                zi_single_channel = scipy.signal.lfilter_zi(self._b_coeffs, self._a_coeffs)
                self._zi = np.tile(zi_single_channel, (num_channels, 1)).astype(np.float32)

            b, a, zi_current = self._b_coeffs, self._a_coeffs, self._zi

        if b is None or a is None or zi_current is None:
            return {"out": signal_tensor}

        signal_np = signal_tensor.numpy()
        filtered_signal_np, zf_next_np = scipy.signal.lfilter(b, a, signal_np, axis=-1, zi=zi_current)

        with self._lock:
            self._zi = zf_next_np
        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

        return {"out": torch.from_numpy(filtered_signal_np.astype(np.float32))}

    def serialize_extra(self) -> Dict:
        with self._lock:
            return {"filter_type": self._filter_type, "freq": self._cutoff_freq, "q": self._q, "gain_db": self._gain_db}

    def deserialize_extra(self, data: Dict):
        with self._lock:
            self._filter_type = data.get("filter_type", "Low Pass")
            self._cutoff_freq = data.get("freq", 1000.0)
            self._q = data.get("q", 0.707)
            self._gain_db = data.get("gain_db", 0.0)
            self._params_dirty = True
