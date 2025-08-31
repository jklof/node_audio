import numpy as np
import threading
import logging
import time
from typing import Dict  # <-- THIS IS THE FIX

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE

# --- UI and Qt Imports ---
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from PySide6.QtWidgets import (
    QWidget, QSlider, QLabel, QVBoxLayout, QGridLayout
)
from PySide6.QtCore import Qt, Signal, Slot, QObject, QSignalBlocker

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants for Compressor ---
MIN_THRESHOLD_DB = -80.0
MAX_THRESHOLD_DB = 0.0
MIN_RATIO = 1.0
MAX_RATIO = 20.0
MIN_ATTACK_MS = 0.1
MAX_ATTACK_MS = 500.0
MIN_RELEASE_MS = 10.0
MAX_RELEASE_MS = 3000.0
MIN_KNEE_DB = 0.0
MAX_KNEE_DB = 12.0
MIN_MAKEUP_GAIN_DB = 0.0
MAX_MAKEUP_GAIN_DB = 24.0
UI_UPDATE_THROTTLE_S = 0.05
EPSILON = 1e-12

# ==============================================================================
# 1. State Emitter for UI Communication (Updated)
# ==============================================================================
class CompressorEmitter(QObject):
    """A dedicated QObject for thread-safe UI communication."""
    stateUpdated = Signal(dict)
    gainReductionUpdated = Signal(float)

# ==============================================================================
# 2. Custom UI Class (Updated for New Architecture)
# ==============================================================================
class DynamicRangeCompressorNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "DynamicRangeCompressorNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)

        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING)
        main_layout.setSpacing(5)

        controls_layout = QGridLayout()
        controls_layout.setSpacing(8)
        self.controls = {}

        param_configs = [
            ("threshold_db", "Threshold", "{:.1f} dB", MIN_THRESHOLD_DB, MAX_THRESHOLD_DB),
            ("ratio", "Ratio", "{:.1f}:1", MIN_RATIO, MAX_RATIO),
            ("attack_ms", "Attack", "{:.1f} ms", MIN_ATTACK_MS, MAX_ATTACK_MS),
            ("release_ms", "Release", "{:.1f} ms", MIN_RELEASE_MS, MAX_RELEASE_MS),
            ("knee_db", "Knee", "{:.1f} dB", MIN_KNEE_DB, MAX_KNEE_DB),
            ("makeup_gain_db", "Makeup", "{:.1f} dB", MIN_MAKEUP_GAIN_DB, MAX_MAKEUP_GAIN_DB),
        ]

        for i, (key, name, fmt, p_min, p_max) in enumerate(param_configs):
            param_label = QLabel(name + ":")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 1000)
            value_label = QLabel(fmt.format(0.0))
            value_label.setMinimumWidth(65)
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            controls_layout.addWidget(param_label, i, 0)
            controls_layout.addWidget(slider, i, 1)
            controls_layout.addWidget(value_label, i, 2)
            
            self.controls[key] = {
                "slider": slider, "value_label": value_label, 
                "format": fmt, "min_val": p_min, "max_val": p_max
            }
            slider.valueChanged.connect(lambda val, k=key: self._handle_slider_change(k, val))

        main_layout.addLayout(controls_layout)
        
        self.gr_label = QLabel("GR: 0.0 dB")
        self.gr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gr_label.setStyleSheet("font-weight: bold; color: orange;")
        main_layout.addWidget(self.gr_label)
        
        self.setContentWidget(self.container_widget)

        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        self.node_logic.emitter.gainReductionUpdated.connect(self._update_gr_display)
        self.updateFromLogic()

    def _map_slider_to_logical(self, key: str, value: int) -> float:
        info = self.controls[key]
        norm = value / 1000.0
        return info["min_val"] + norm * (info["max_val"] - info["min_val"])

    def _map_logical_to_slider(self, key: str, value: float) -> int:
        info = self.controls[key]
        range_val = info["max_val"] - info["min_val"]
        if abs(range_val) < EPSILON: return 0
        norm = (np.clip(value, info["min_val"], info["max_val"]) - info["min_val"]) / range_val
        return int(round(norm * 1000.0))

    @Slot(str, int)
    def _handle_slider_change(self, key: str, value: int):
        logical_val = self._map_slider_to_logical(key, value)
        setter = getattr(self.node_logic, f"set_{key}", None)
        if setter: setter(logical_val)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        for key, control_info in self.controls.items():
            value = state.get(key, control_info["min_val"])
            
            is_connected = key in self.node_logic.inputs and self.node_logic.inputs[key].connections
            control_info["slider"].setEnabled(not is_connected)
            
            with QSignalBlocker(control_info["slider"]):
                control_info["slider"].setValue(self._map_logical_to_slider(key, value))
            
            label_text = control_info["format"].format(value)
            if is_connected: label_text += " (ext)"
            control_info["value_label"].setText(label_text)

    @Slot(float)
    def _update_gr_display(self, gr_db: float):
        self.gr_label.setText(f"GR: {gr_db:.1f} dB")

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        last_gr = self.node_logic.get_current_gain_reduction_db()
        self._update_gr_display(float(last_gr))
        super().updateFromLogic()

# ==============================================================================
# 3. Logic Class for Dynamic Range Compressor (Updated)
# ==============================================================================
class DynamicRangeCompressorNode(Node):
    NODE_TYPE = "Dynamic Range Compressor"
    UI_CLASS = DynamicRangeCompressorNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Reduces audio dynamic range by attenuating loud sounds."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.emitter = CompressorEmitter()

        # --- Define Sockets ---
        self.add_input("in", data_type=np.ndarray)
        self.add_input("threshold_db", data_type=float)
        self.add_input("ratio", data_type=float)
        self.add_input("attack_ms", data_type=float)
        self.add_input("release_ms", data_type=float)
        self.add_input("knee_db", data_type=float)
        self.add_input("makeup_gain_db", data_type=float)
        self.add_output("out", data_type=np.ndarray)
        self.add_output("gain_reduction_db", data_type=float)

        # --- Internal State ---
        self._lock = threading.Lock()
        self._threshold_db = -20.0
        self._ratio = 2.0
        self._attack_ms = 20.0
        self._release_ms = 100.0
        self._knee_db = 0.0
        self._makeup_gain_db = 0.0

        # --- DSP State ---
        self._samplerate = float(DEFAULT_SAMPLERATE)
        self._detector_envelope = 0.0
        self._compressor_gain_db = 0.0 # Gain reduction is stored as a negative dB value
        self._last_ui_update_time = 0
        self._update_coefficients()
        
    def _calculate_coeff(self, time_ms: float) -> float:
        if time_ms <= 0.0: return 0.0
        return np.exp(-1.0 / (time_ms * 0.001 * self._samplerate))

    def _update_coefficients(self):
        self._attack_coeff = self._calculate_coeff(self._attack_ms)
        self._release_coeff = self._calculate_coeff(self._release_ms)
        logger.debug(f"[{self.name}] Coeffs updated: Attack={self._attack_coeff:.4f}, Release={self._release_coeff:.4f}")

    def get_current_state_snapshot(self, locked: bool = False) -> Dict:
        state = {
            "threshold_db": self._threshold_db, "ratio": self._ratio,
            "attack_ms": self._attack_ms, "release_ms": self._release_ms,
            "knee_db": self._knee_db, "makeup_gain_db": self._makeup_gain_db,
        }
        if locked: return state
        with self._lock: return state

    def process(self, input_data: dict) -> dict:
        signal_in = input_data.get("in")
        if not isinstance(signal_in, np.ndarray):
            # If no signal, smoothly release the compressor
            with self._lock:
                self._compressor_gain_db = self._release_coeff * self._compressor_gain_db
            return {"out": None, "gain_reduction_db": self._compressor_gain_db}

        state_update_needed = False
        with self._lock:
            # --- Update state from sockets, setting flag if changed ---
            params = ["threshold_db", "ratio", "attack_ms", "release_ms", "knee_db", "makeup_gain_db"]
            for p in params:
                socket_val = input_data.get(p)
                if socket_val is not None:
                    if getattr(self, f"_{p}") != float(socket_val):
                        setattr(self, f"_{p}", float(socket_val))
                        state_update_needed = True
            
            if state_update_needed and ("attack_ms" in [p for p in params if input_data.get(p) is not None] or "release_ms" in [p for p in params if input_data.get(p) is not None]):
                self._update_coefficients()

            # --- Copy state to local variables for processing ---
            threshold = self._threshold_db; ratio = self._ratio; knee = self._knee_db
            attack_c = self._attack_coeff; release_c = self._release_coeff
            makeup_gain = self._makeup_gain_db
            detector_env = self._detector_envelope
            comp_gain = self._compressor_gain_db

        # --- Emit state update AFTER releasing lock ---
        if state_update_needed:
            self.emitter.stateUpdated.emit(self.get_current_state_snapshot())
        
        # --- Audio Processing ---
        mono_signal = np.mean(np.abs(signal_in), axis=1) if signal_in.ndim > 1 else np.abs(signal_in)
        output_block = np.zeros_like(signal_in)

        for i in range(len(mono_signal)):
            detector_env = max(mono_signal[i], detector_env * release_c) # Simplified envelope follower
            level_db = 20 * np.log10(detector_env + EPSILON)
            
            over_db = level_db - threshold
            gain_reduction_db = 0.0
            
            if knee > 0 and over_db > -knee/2:
                if over_db < knee/2: # Inside knee
                    gain_reduction_db = (1/ratio - 1) * (over_db + knee/2)**2 / (2 * knee)
                else: # Above knee
                    gain_reduction_db = (1/ratio - 1) * over_db
            elif over_db > 0: # Hard knee
                gain_reduction_db = (1/ratio - 1) * over_db
            
            target_gain = gain_reduction_db
            if target_gain < comp_gain:
                comp_gain = attack_c * comp_gain + (1 - attack_c) * target_gain
            else:
                comp_gain = release_c * comp_gain + (1 - release_c) * target_gain
            
            linear_gain = 10**((comp_gain + makeup_gain) / 20.0)
            output_block[i] = signal_in[i] * linear_gain

        with self._lock:
            self._detector_envelope = detector_env
            self._compressor_gain_db = comp_gain

        current_time = time.monotonic()
        if current_time >= self._last_ui_update_time + UI_UPDATE_THROTTLE_S:
            self.emitter.gainReductionUpdated.emit(comp_gain)
            self._last_ui_update_time = current_time

        return {"out": output_block.astype(DEFAULT_DTYPE), "gain_reduction_db": comp_gain}

    def _create_setter(param_name: str, min_val: float, max_val: float):
        def setter(self, value: float):
            needs_coeff_update = False
            state_to_emit = None
            with self._lock:
                clipped_value = float(np.clip(value, min_val, max_val))
                if getattr(self, f"_{param_name}") != clipped_value:
                    setattr(self, f"_{param_name}", clipped_value)
                    if param_name in ["attack_ms", "release_ms"]:
                        needs_coeff_update = True
                    state_to_emit = self.get_current_state_snapshot(locked=True)
            if needs_coeff_update: self._update_coefficients()
            if state_to_emit: self.emitter.stateUpdated.emit(state_to_emit)
        return setter

    set_threshold_db = _create_setter("threshold_db", MIN_THRESHOLD_DB, MAX_THRESHOLD_DB)
    set_ratio = _create_setter("ratio", MIN_RATIO, MAX_RATIO)
    set_attack_ms = _create_setter("attack_ms", MIN_ATTACK_MS, MAX_ATTACK_MS)
    set_release_ms = _create_setter("release_ms", MIN_RELEASE_MS, MAX_RELEASE_MS)
    set_knee_db = _create_setter("knee_db", MIN_KNEE_DB, MAX_KNEE_DB)
    set_makeup_gain_db = _create_setter("makeup_gain_db", MIN_MAKEUP_GAIN_DB, MAX_MAKEUP_GAIN_DB)

    def get_current_gain_reduction_db(self) -> float:
        with self._lock: return self._compressor_gain_db

    def start(self):
        with self._lock:
            self._detector_envelope = 0.0
            self._compressor_gain_db = 0.0
        logger.debug(f"[{self.name}] State reset on start.")

    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()

    def deserialize_extra(self, data: dict):
        with self._lock:
            self._threshold_db = data.get("threshold_db", -20.0)
            self._ratio = data.get("ratio", 2.0)
            self._attack_ms = data.get("attack_ms", 20.0)
            self._release_ms = data.get("release_ms", 100.0)
            self._knee_db = data.get("knee_db", 0.0)
            self._makeup_gain_db = data.get("makeup_gain_db", 0.0)
        self._update_coefficients()