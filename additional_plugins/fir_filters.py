import torch
import torch.nn.functional as F
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

# --- Filter Constants ---
NUM_TAPS = 511
FILTER_LATENCY_SAMPLES = (NUM_TAPS - 1) // 2


# ==============================================================================
# 1. UI Class for the Linear Phase EQ Node (Unchanged)
# ==============================================================================
class LinearPhaseEQNodeItem(NodeItem):
    """Custom UI for the Linear Phase EQ, with controls for type, frequency, and Q."""

    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "LinearPhaseEQNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        main_layout.setSpacing(5)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["Low Pass", "High Pass", "Band Pass", "Band Stop (Notch)"])
        main_layout.addWidget(self.type_combo)

        self.freq_slider, self.freq_label = self._create_slider_control("Freq", 20.0, 20000.0, "{:.0f} Hz", is_log=True)
        self.q_slider, self.q_label = self._create_slider_control("Q", 0.1, 10.0, "{:.2f}")

        main_layout.addWidget(self.freq_label)
        main_layout.addWidget(self.freq_slider)
        main_layout.addWidget(self.q_label)
        main_layout.addWidget(self.q_slider)

        self.setContentWidget(self.container_widget)

        self.type_combo.currentTextChanged.connect(self.node_logic.set_filter_type)
        self.freq_slider.valueChanged.connect(self._on_freq_changed)
        self.q_slider.valueChanged.connect(self._on_q_changed)
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
        else:
            return min_val + norm * (max_val - min_val)

    def _map_logical_to_slider(self, slider: QSlider, value: float) -> int:
        min_val, max_val = slider.property("min_val"), slider.property("max_val")
        safe_value = np.clip(value, min_val, max_val)
        if slider.property("is_log"):
            log_min, log_max = np.log10(min_val), np.log10(max_val)
            norm = (np.log10(safe_value) - log_min) / (log_max - log_min)
            return int(norm * 1000.0)
        else:
            range_val = max_val - min_val
            norm = (safe_value - min_val) / (range_val or 1.0)
            return int(np.clip(norm, 0.0, 1.0) * 1000.0)

    @Slot()
    def _on_freq_changed(self):
        self.node_logic.set_cutoff_freq(self._map_slider_to_logical(self.freq_slider))

    @Slot()
    def _on_q_changed(self):
        self.node_logic.set_q(self._map_slider_to_logical(self.q_slider))

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        filter_type = state.get("filter_type", "Low Pass")
        freq = state.get("freq", 1000.0)
        q = state.get("q", 1.0)

        with QSignalBlocker(self.type_combo):
            self.type_combo.setCurrentText(filter_type)

        for key, slider, label in [("freq", self.freq_slider, self.freq_label), ("q", self.q_slider, self.q_label)]:
            val = state.get(key)
            is_ext = state.get(f"is_{key}_ext", False)
            slider.setEnabled(not is_ext)
            with QSignalBlocker(slider):
                slider.setValue(self._map_logical_to_slider(slider, val))
            label_text = f"{slider.property('name')}: {slider.property('format').format(val)}"
            if is_ext:
                label_text += " (ext)"
            label.setText(label_text)

        q_visible = filter_type in ["Band Pass", "Band Stop (Notch)"]
        self.q_label.setVisible(q_visible)
        self.q_slider.setVisible(q_visible)
        self.container_widget.adjustSize()
        self.update_geometry()

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ==============================================================================
# 2. Logic Class for the Linear Phase EQ Node (CORRECTED)
# ==============================================================================
class LinearPhaseEQNode(Node):
    NODE_TYPE = "Linear Phase EQ"
    UI_CLASS = LinearPhaseEQNodeItem
    CATEGORY = "Filters"
    DESCRIPTION = "Applies a linear-phase (FIR) filter to an audio signal."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self._lock = threading.Lock()
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("cutoff_freq", data_type=float)
        self.add_input("q", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._filter_type: str = "Low Pass"
        self._cutoff_freq: float = 1000.0
        self._q: float = 1.0
        self._params_dirty: bool = True
        self._coeffs: Optional[torch.Tensor] = None
        self._history_buffer: Optional[torch.Tensor] = None
        self._expected_channels: Optional[int] = None

    def _recalculate_coeffs(self):
        nyquist = DEFAULT_SAMPLERATE / 2.0
        taps = None
        if self._filter_type in ["Low Pass", "High Pass"]:
            cutoff_norm = self._cutoff_freq / nyquist
            taps = scipy.signal.firwin(
                NUM_TAPS, cutoff=cutoff_norm, window="hann", pass_zero=(self._filter_type == "Low Pass")
            )
        elif self._filter_type in ["Band Pass", "Band Stop (Notch)"]:
            bandwidth = self._cutoff_freq / self._q
            f_low = self._cutoff_freq - (bandwidth / 2)
            f_high = self._cutoff_freq + (bandwidth / 2)
            f_low = max(20.0, f_low)
            f_high = min(nyquist - 1, f_high)
            if f_low >= f_high:
                f_low = f_high - 1.0
            cutoff_norm = [f_low / nyquist, f_high / nyquist]
            bp_taps = scipy.signal.firwin(NUM_TAPS, cutoff=cutoff_norm, window="hann", pass_zero=False)
            if self._filter_type == "Band Pass":
                taps = bp_taps
            else:
                impulse = np.zeros(NUM_TAPS)
                impulse[FILTER_LATENCY_SAMPLES] = 1.0
                taps = impulse - bp_taps
        if taps is not None:
            coeffs_tensor = torch.from_numpy(taps.astype(np.float32)).view(1, 1, -1)
            if self._expected_channels is not None and self._expected_channels > 0:
                self._coeffs = coeffs_tensor.repeat(self._expected_channels, 1, 1)
            else:
                self._coeffs = coeffs_tensor
        self._params_dirty = False
        logger.info(f"[{self.name}] Recalculated FIR coefficients for {self._filter_type}.")

    # --- Public methods for UI interaction ---
    @Slot(str)
    def set_filter_type(self, f_type: str):
        state_snapshot_to_emit = None
        with self._lock:
            if self._filter_type != f_type:
                self._filter_type = f_type
                self._params_dirty = True
                state_snapshot_to_emit = self._get_current_state_snapshot_locked()
        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

    @Slot(float)
    def set_cutoff_freq(self, freq: float):
        state_snapshot_to_emit = None
        with self._lock:
            new_freq = max(20.0, min(float(freq), DEFAULT_SAMPLERATE / 2 - 1))
            if self._cutoff_freq != new_freq:
                self._cutoff_freq = new_freq
                self._params_dirty = True
                state_snapshot_to_emit = self._get_current_state_snapshot_locked()
        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

    @Slot(float)
    def set_q(self, q: float):
        with self._lock:
            new_q = max(0.1, float(q))
            if self._q != new_q:
                self._q = new_q
                self._params_dirty = True
        self.emitter.stateUpdated.emit(self.get_current_state_snapshot())

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    def _get_current_state_snapshot_locked(self) -> Dict:
        """Gathers the current state. ASSUMES THE CALLER HOLDS THE LOCK."""
        is_freq_ext = "cutoff_freq" in self.inputs and self.inputs["cutoff_freq"].connections
        is_q_ext = "q" in self.inputs and self.inputs["q"].connections
        return {
            "filter_type": self._filter_type,
            "freq": self._cutoff_freq,
            "q": self._q,
            "is_freq_ext": is_freq_ext,
            "is_q_ext": is_q_ext,
        }

    # --- Processing methods ---
    def start(self):
        with self._lock:
            self._history_buffer = None
            self._expected_channels = None
            self._params_dirty = True

    def process(self, input_data: dict) -> dict:
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}

        num_channels, _ = signal.shape
        state_snapshot_to_emit = None

        with self._lock:
            freq_socket = input_data.get("cutoff_freq")
            if freq_socket is not None and self._cutoff_freq != float(freq_socket):
                self._cutoff_freq = max(20.0, min(float(freq_socket), DEFAULT_SAMPLERATE / 2 - 1))
                self._params_dirty = True
            q_socket = input_data.get("q")
            if q_socket is not None and self._q != float(q_socket):
                self._q = max(0.1, float(q_socket))
                self._params_dirty = True

            if self._expected_channels != num_channels:
                self._expected_channels = num_channels
                self._history_buffer = torch.zeros((num_channels, NUM_TAPS - 1), dtype=DEFAULT_DTYPE)
                self._params_dirty = True

            if self._params_dirty:
                self._recalculate_coeffs()
                state_snapshot_to_emit = self._get_current_state_snapshot_locked()

            if self._coeffs is None or self._history_buffer is None:
                return {"out": signal}

            combined_input = torch.cat([self._history_buffer, signal], dim=1)
            batched_input = combined_input.unsqueeze(0)
            filtered_batched = F.conv1d(batched_input, self._coeffs, padding="valid", groups=num_channels)
            filtered_signal = filtered_batched.squeeze(0)
            self._history_buffer = signal[:, -(NUM_TAPS - 1) :]

        if state_snapshot_to_emit:
            self.emitter.stateUpdated.emit(state_snapshot_to_emit)

        return {"out": filtered_signal}

    def serialize_extra(self) -> Dict:
        with self._lock:
            return {"filter_type": self._filter_type, "freq": self._cutoff_freq, "q": self._q}

    def deserialize_extra(self, data: Dict):
        with self._lock:
            self._filter_type = data.get("filter_type", "Low Pass")
            self._cutoff_freq = data.get("freq", 1000.0)
            self._q = data.get("q", 1.0)
            self._params_dirty = True
