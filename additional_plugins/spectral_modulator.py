import numpy as np
import threading
import logging
from typing import Dict, Optional

# --- Node System Imports ---
from node_system import Node
from constants import SpectralFrame, DEFAULT_COMPLEX_DTYPE
from ui_elements import NodeItem, NODE_CONTENT_PADDING

# --- Qt Imports ---
from PySide6.QtCore import Qt, Signal, Slot, QObject, QSignalBlocker
from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout

# --- Logging ---
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. State Emitter for UI Communication
# ==============================================================================
class SpectralModulatorEmitter(QObject):
    stateUpdated = Signal(dict)

# ==============================================================================
# 2. Custom UI Class (SpectralModulatorNodeItem)
# ==============================================================================
class SpectralModulatorNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "SpectralModulatorNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)
        
        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING)
        main_layout.setSpacing(5)

        # --- Create Slider Controls ---
        self.rate_slider, self.rate_label = self._create_slider_control("Rate", 0.1, 10.0, "{:.2f} Hz")
        self.depth_slider, self.depth_label = self._create_slider_control("Depth", 0.0, 20.0, "{:.1f} ms")
        self.mix_slider, self.mix_label = self._create_slider_control("Mix", 0.0, 1.0, "{:.0%}")
        
        for label, slider in [
            (self.rate_label, self.rate_slider),
            (self.depth_label, self.depth_slider),
            (self.mix_label, self.mix_slider)
        ]:
            main_layout.addWidget(label)
            main_layout.addWidget(slider)

        self.setContentWidget(self.container_widget)

        # --- Connect Signals ---
        self.rate_slider.valueChanged.connect(self._on_rate_changed)
        self.depth_slider.valueChanged.connect(self._on_depth_changed)
        self.mix_slider.valueChanged.connect(self._on_mix_changed)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        
        self.updateFromLogic()

    def _create_slider_control(self, name: str, min_val: float, max_val: float, fmt: str) -> tuple[QSlider, QLabel]:
        label = QLabel(f"{name}: ...")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)
        slider.setProperty("min_val", min_val)
        slider.setProperty("max_val", max_val)
        slider.setProperty("name", name)
        slider.setProperty("format", fmt)
        return slider, label

    def _map_slider_to_logical(self, slider: QSlider) -> float:
        min_val, max_val = slider.property("min_val"), slider.property("max_val")
        norm = slider.value() / 1000.0
        return min_val + norm * (max_val - min_val)

    def _map_logical_to_slider(self, slider: QSlider, value: float) -> int:
        min_val, max_val = slider.property("min_val"), slider.property("max_val")
        range_val = max_val - min_val
        if range_val == 0: return 0
        norm = (value - min_val) / range_val
        return int(np.clip(norm, 0.0, 1.0) * 1000.0)
            
    @Slot()
    def _on_rate_changed(self):
        self.node_logic.set_rate(self._map_slider_to_logical(self.rate_slider))
    @Slot()
    def _on_depth_changed(self):
        self.node_logic.set_depth(self._map_slider_to_logical(self.depth_slider))
    @Slot()
    def _on_mix_changed(self):
        self.node_logic.set_mix(self._map_slider_to_logical(self.mix_slider))

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        sliders_map = {
            "rate": (self.rate_slider, self.rate_label),
            "depth": (self.depth_slider, self.depth_label),
            "mix": (self.mix_slider, self.mix_label),
        }
        
        # --- NEW: Check for external modulation connection ---
        is_mod_ext = self.node_logic.inputs["mod_in"].connections

        for key, (slider, label) in sliders_map.items():
            value = state.get(key, slider.property("min_val"))
            is_param_ext = key in self.node_logic.inputs and self.node_logic.inputs[key].connections
            
            # --- NEW: Disable rate slider if mod_in is connected ---
            if key == "rate":
                slider.setEnabled(not is_mod_ext)
            else:
                slider.setEnabled(not is_param_ext)

            with QSignalBlocker(slider):
                slider.setValue(self._map_logical_to_slider(slider, value))
            
            label_text = f"{slider.property('name')}: {slider.property('format').format(value)}"

            if (key == "rate" and is_mod_ext) or is_param_ext:
                 label_text += " (ext)"

            label.setText(label_text)

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()


# ==============================================================================
# 3. Node Logic Class (SpectralModulatorNode)
# ==============================================================================
class SpectralModulatorNode(Node):
    NODE_TYPE = "Spectral Modulator"
    UI_CLASS = SpectralModulatorNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "Applies phase modulation to a spectral frame, with internal or external LFO."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = SpectralModulatorEmitter()

        # --- Setup Sockets ---
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("mod_in", data_type=float) # <-- NEW SOCKET
        self.add_input("rate", data_type=float)
        self.add_input("depth", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        self._lock = threading.Lock()
        
        # --- Internal State ---
        self._rate_hz: float = 1.5
        self._depth_ms: float = 5.0
        self._mix: float = 0.5
        
        # --- DSP State ---
        self._lfo_phase: float = 0.0

    def _emit_state_update_locked(self):
        state = self.get_current_state_snapshot(locked=True)
        self.emitter.stateUpdated.emit(state)

    # --- Parameter Setter Slots ---
    @Slot(float)
    def set_rate(self, value: float):
        with self._lock:
            self._rate_hz = value
            self._emit_state_update_locked()
    @Slot(float)
    def set_depth(self, value: float):
        with self._lock:
            self._depth_ms = value
            self._emit_state_update_locked()
    @Slot(float)
    def set_mix(self, value: float):
        with self._lock:
            self._mix = np.clip(value, 0.0, 1.0)
            self._emit_state_update_locked()

    def get_current_state_snapshot(self, locked: bool = False) -> Dict:
        state = {"rate": self._rate_hz, "depth": self._depth_ms, "mix": self._mix}
        if locked: return state
        with self._lock: return state

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        with self._lock:
            # --- Update state from parameter sockets if connected ---
            if input_data.get("rate") is not None: self._rate_hz = input_data.get("rate")
            if input_data.get("depth") is not None: self._depth_ms = input_data.get("depth")
            if input_data.get("mix") is not None: self._mix = np.clip(input_data.get("mix"), 0.0, 1.0)
            
            lfo_value = 0.0
            lfo_mod_input = input_data.get("mod_in")

            # --- NEW: Core Logic Update ---
            if lfo_mod_input is not None:
                # 1. Use external modulation signal if it's connected
                lfo_value = float(lfo_mod_input)
            else:
                # 2. Fall back to internal LFO if no external signal
                frame_duration_s = frame.hop_size / frame.sample_rate
                phase_increment = 2 * np.pi * self._rate_hz * frame_duration_s
                self._lfo_phase = np.mod(self._lfo_phase + phase_increment, 2 * np.pi)
                lfo_value = np.sin(self._lfo_phase)

            # --- The rest of the algorithm remains the same ---
            frame_data = frame.data
            delay_s = (self._depth_ms / 1000.0) * lfo_value
            freqs = np.fft.rfftfreq(frame.fft_size, d=1.0 / frame.sample_rate)
            phase_shifts_rad = 2 * np.pi * freqs * delay_s
            phasor = np.exp(1j * phase_shifts_rad).astype(DEFAULT_COMPLEX_DTYPE)
            
            wet_signal = frame_data * phasor[:, np.newaxis]
            output_fft = (frame_data * (1.0 - self._mix)) + (wet_signal * self._mix)

        # --- Create and return the output frame ---
        output_frame = SpectralFrame(
            data=output_fft,
            fft_size=frame.fft_size, hop_size=frame.hop_size,
            window_size=frame.window_size, sample_rate=frame.sample_rate,
            analysis_window=frame.analysis_window
        )
        return {"spectral_frame_out": output_frame}

    def start(self):
        with self._lock:
            self._lfo_phase = 0.0
            
    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()
    
    def deserialize_extra(self, data: dict):
        with self._lock:
            self._rate_hz = data.get("rate", 1.5)
            self._depth_ms = data.get("depth", 5.0)
            self._mix = data.get("mix", 0.5)