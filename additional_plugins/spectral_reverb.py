import numpy as np
import threading
import logging
from typing import Dict, Optional
from collections import deque

# --- Node System Imports ---
from node_system import Node
from constants import SpectralFrame, DEFAULT_DTYPE, DEFAULT_COMPLEX_DTYPE
from ui_elements import NodeItem, NODE_CONTENT_PADDING

# --- Qt Imports ---
from PySide6.QtCore import Qt, Signal, Slot, QObject, QSignalBlocker
from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout

# --- Logging ---
logger = logging.getLogger(__name__)

# A small value to prevent log(0) errors or division by zero
EPSILON = 1e-9

# ==============================================================================
# 1. State Emitter for UI Communication
# ==============================================================================
class SpectralReverbEmitter(QObject):
    stateUpdated = Signal(dict)

# ==============================================================================
# 2. Custom UI Class (SpectralReverbNodeItem)
# ==============================================================================
class SpectralReverbNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "SpectralReverbNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)
        
        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING)
        main_layout.setSpacing(5)

        # --- Create Slider Controls ---
        self.pre_delay_slider, self.pre_delay_label = self._create_slider_control("Pre-delay", 0.0, 250.0, "{:.0f} ms")
        self.decay_slider, self.decay_label = self._create_slider_control("Decay Time", 0.1, 15.0, "{:.1f} s")
        self.damp_slider, self.damp_label = self._create_slider_control("HF Damp", 500.0, 20000.0, "{:.0f} Hz", is_log=True)
        self.lf_damp_slider, self.lf_damp_label = self._create_slider_control("LF Damp", 20.0, 2000.0, "{:.0f} Hz", is_log=True) # <-- NEW
        self.diffusion_slider, self.diffusion_label = self._create_slider_control("Diffusion", 0.0, 1.0, "{:.0%}")
        self.width_slider, self.width_label = self._create_slider_control("Stereo Width", 0.0, 1.0, "{:.0%}") # <-- NEW
        self.mix_slider, self.mix_label = self._create_slider_control("Mix", 0.0, 1.0, "{:.0%}")
        
        for label, slider in [
            (self.pre_delay_label, self.pre_delay_slider),
            (self.decay_label, self.decay_slider),
            (self.damp_label, self.damp_slider),
            (self.lf_damp_label, self.lf_damp_slider), # <-- NEW
            (self.diffusion_label, self.diffusion_slider),
            (self.width_label, self.width_slider), # <-- NEW
            (self.mix_label, self.mix_slider)
        ]:
            main_layout.addWidget(label)
            main_layout.addWidget(slider)

        self.setContentWidget(self.container_widget)

        # --- Connect Signals ---
        self.pre_delay_slider.valueChanged.connect(self._on_pre_delay_changed)
        self.decay_slider.valueChanged.connect(self._on_decay_changed)
        self.damp_slider.valueChanged.connect(self._on_damp_changed)
        self.lf_damp_slider.valueChanged.connect(self._on_lf_damp_changed) # <-- NEW
        self.diffusion_slider.valueChanged.connect(self._on_diffusion_changed)
        self.width_slider.valueChanged.connect(self._on_width_changed) # <-- NEW
        self.mix_slider.valueChanged.connect(self._on_mix_changed)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
        
        self.updateFromLogic()

    def _create_slider_control(self, name: str, min_val: float, max_val: float, fmt: str, is_log: bool = False) -> tuple[QSlider, QLabel]:
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
        if slider.property("is_log"):
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            norm = slider.value() / 1000.0
            return 10**(log_min + norm * (log_max - log_min))
        else:
            norm = slider.value() / 1000.0
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
            if range_val == 0: return 0
            norm = (value - min_val) / range_val
            return int(np.clip(norm, 0.0, 1.0) * 1000.0)
            
    @Slot()
    def _on_pre_delay_changed(self):
        self.node_logic.set_pre_delay_ms(self._map_slider_to_logical(self.pre_delay_slider))
    @Slot()
    def _on_decay_changed(self):
        self.node_logic.set_decay_time(self._map_slider_to_logical(self.decay_slider))
    @Slot()
    def _on_damp_changed(self):
        self.node_logic.set_hf_damp(self._map_slider_to_logical(self.damp_slider))
    @Slot() # <-- NEW
    def _on_lf_damp_changed(self):
        self.node_logic.set_lf_damp(self._map_slider_to_logical(self.lf_damp_slider))
    @Slot()
    def _on_diffusion_changed(self):
        self.node_logic.set_diffusion(self._map_slider_to_logical(self.diffusion_slider))
    @Slot() # <-- NEW
    def _on_width_changed(self):
        self.node_logic.set_width(self._map_slider_to_logical(self.width_slider))
    @Slot()
    def _on_mix_changed(self):
        self.node_logic.set_mix(self._map_slider_to_logical(self.mix_slider))

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        sliders_map = {
            "pre_delay_ms": (self.pre_delay_slider, self.pre_delay_label),
            "decay_time": (self.decay_slider, self.decay_label),
            "hf_damp": (self.damp_slider, self.damp_label),
            "lf_damp": (self.lf_damp_slider, self.lf_damp_label), # <-- NEW
            "diffusion": (self.diffusion_slider, self.diffusion_label),
            "width": (self.width_slider, self.width_label), # <-- NEW
            "mix": (self.mix_slider, self.mix_label),
        }
        
        for key, (slider, label) in sliders_map.items():
            value = state.get(key, slider.property("min_val"))
            is_connected = key in self.node_logic.inputs and self.node_logic.inputs[key].connections
            slider.setEnabled(not is_connected)
            
            with QSignalBlocker(slider):
                slider.setValue(self._map_logical_to_slider(slider, value))
            
            label_text = f"{slider.property('name')}: {slider.property('format').format(value)}"
            if is_connected: label_text += " (ext)"
            label.setText(label_text)

    @Slot()
    def updateFromLogic(self):
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()

# ==============================================================================
# 3. Node Logic Class (SpectralReverbNode)
# ==============================================================================
class SpectralReverbNode(Node):
    NODE_TYPE = "Spectral Reverb"
    UI_CLASS = SpectralReverbNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "Algorithmic reverb that applies frequency-dependent decay to spectral frames."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = SpectralReverbEmitter()

        # --- Setup Sockets ---
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("pre_delay_ms", data_type=float)
        self.add_input("decay_time", data_type=float)
        self.add_input("hf_damp", data_type=float)
        self.add_input("lf_damp", data_type=float)
        self.add_input("diffusion", data_type=float)
        self.add_input("width", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        self._lock = threading.Lock()
        
        # --- Internal State ---
        self._pre_delay_ms: float = 20.0
        self._decay_time_s: float = 2.5
        self._hf_damp_hz: float = 4000.0
        self._lf_damp_hz: float = 150.0 # <-- NEW
        self._diffusion: float = 1.0
        self._width: float = 1.0 # <-- NEW
        self._mix: float = 0.5
        
        # --- DSP Buffers & State ---
        self._reverb_fft_buffer: Optional[np.ndarray] = None
        self._decay_factors: Optional[np.ndarray] = None
        self._pre_delay_buffer: deque = deque()
        self._pre_delay_frames: int = 0
        self._last_frame_params: tuple = (0, 0)
        self._params_dirty: bool = True

    def _emit_state_update_locked(self):
        state = self.get_current_state_snapshot(locked=True)
        self.emitter.stateUpdated.emit(state)

    # --- Parameter Setter Slots ---
    @Slot(float)
    def set_pre_delay_ms(self, value: float):
        with self._lock:
            self._pre_delay_ms = value; self._params_dirty = True
            self._emit_state_update_locked()
    @Slot(float)
    def set_decay_time(self, value: float):
        with self._lock:
            self._decay_time_s = value; self._params_dirty = True
            self._emit_state_update_locked()
    @Slot(float)
    def set_hf_damp(self, value: float):
        with self._lock:
            self._hf_damp_hz = value; self._params_dirty = True
            self._emit_state_update_locked()
    @Slot(float)
    def set_lf_damp(self, value: float):
        with self._lock:
            self._lf_damp_hz = value; self._params_dirty = True
            self._emit_state_update_locked()
    @Slot(float)
    def set_diffusion(self, value: float):
        with self._lock:
            self._diffusion = np.clip(value, 0.0, 1.0); self._params_dirty = True
            self._emit_state_update_locked()
    @Slot(float)
    def set_width(self, value: float):
        with self._lock:
            self._width = np.clip(value, 0.0, 1.0); self._params_dirty = True
            self._emit_state_update_locked()
    @Slot(float)
    def set_mix(self, value: float):
        with self._lock:
            self._mix = np.clip(value, 0.0, 1.0)
            self._emit_state_update_locked()

    def get_current_state_snapshot(self, locked: bool = False) -> Dict:
        state = {
            "pre_delay_ms": self._pre_delay_ms,
            "decay_time": self._decay_time_s,
            "hf_damp": self._hf_damp_hz,
            "lf_damp": self._lf_damp_hz, # <-- NEW
            "diffusion": self._diffusion,
            "width": self._width, # <-- NEW
            "mix": self._mix
        }
        if locked: return state
        with self._lock: return state

    def _recalculate_params(self, frame: SpectralFrame):
        # --- Recalculate Decay Factors ---
        num_bins, _ = frame.data.shape
        t60 = self._decay_time_s
        freqs = np.fft.rfftfreq(frame.fft_size, d=1.0/frame.sample_rate)
        
        # --- Damping Logic ---
        # HF Damp: values are 1.0 below the cutoff, then fall off
        hf_damp_factor = np.clip((self._hf_damp_hz - freqs) / self._hf_damp_hz, 0.1, 1.0)
        hf_damp_factor[freqs < self._hf_damp_hz] = 1.0
        # LF Damp: values are 1.0 above the cutoff, then fall off
        lf_damp_factor = np.clip(freqs / self._lf_damp_hz, 0.1, 1.0)
        lf_damp_factor[freqs > self._lf_damp_hz] = 1.0
        # Combine factors
        damped_t60 = t60 * hf_damp_factor * lf_damp_factor

        frames_per_second = frame.sample_rate / frame.hop_size
        magnitudes = 10.0**(-3.0 / (damped_t60 * frames_per_second + EPSILON))

        # --- Stereo Width Logic ---
        phase_shift_amount = self._diffusion * np.pi
        # Generate two independent sets of random phases
        random_phases_1 = np.random.uniform(-phase_shift_amount, phase_shift_amount, size=num_bins)
        random_phases_2 = np.random.uniform(-phase_shift_amount, phase_shift_amount, size=num_bins)
        
        # Blend between mono and stereo phase randomization based on width
        phases_L = random_phases_1
        phases_R = random_phases_1 * (1.0 - self._width) + random_phases_2 * self._width

        decay_L = (magnitudes * np.exp(1j * phases_L)).astype(DEFAULT_COMPLEX_DTYPE)
        decay_R = (magnitudes * np.exp(1j * phases_R)).astype(DEFAULT_COMPLEX_DTYPE)
        self._decay_factors = np.column_stack((decay_L, decay_R))

        # --- Recalculate Pre-delay Buffer ---
        pre_delay_s = self._pre_delay_ms / 1000.0
        frame_duration_s = frame.hop_size / frame.sample_rate
        self._pre_delay_frames = int(round(pre_delay_s / (frame_duration_s + EPSILON)))
        
        if self._pre_delay_buffer.maxlen != self._pre_delay_frames:
            self._pre_delay_buffer = deque(self._pre_delay_buffer, maxlen=self._pre_delay_frames)

        self._params_dirty = False
        logger.info(f"[{self.name}] Recalculated reverb params: Pre-delay={self._pre_delay_frames} frames.")

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        with self._lock:
            # --- Update state from sockets if connected ---
            socket_params = {
                "pre_delay_ms": input_data.get("pre_delay_ms"), "decay_time": input_data.get("decay_time"),
                "hf_damp": input_data.get("hf_damp"), "lf_damp": input_data.get("lf_damp"),
                "diffusion": input_data.get("diffusion"), "width": input_data.get("width"),
                "mix": input_data.get("mix"),
            }
            if socket_params["pre_delay_ms"] is not None: self._pre_delay_ms = socket_params["pre_delay_ms"]; self._params_dirty = True
            if socket_params["decay_time"] is not None: self._decay_time_s = socket_params["decay_time"]; self._params_dirty = True
            if socket_params["hf_damp"] is not None: self._hf_damp_hz = socket_params["hf_damp"]; self._params_dirty = True
            if socket_params["lf_damp"] is not None: self._lf_damp_hz = socket_params["lf_damp"]; self._params_dirty = True
            if socket_params["diffusion"] is not None: self._diffusion = np.clip(socket_params["diffusion"], 0.0, 1.0); self._params_dirty = True
            if socket_params["width"] is not None: self._width = np.clip(socket_params["width"], 0.0, 1.0); self._params_dirty = True
            if socket_params["mix"] is not None: self._mix = np.clip(socket_params["mix"], 0.0, 1.0)
            
            # --- Initialize / re-initialize buffers on format change ---
            num_bins, num_channels = frame.data.shape
            current_frame_params = (frame.fft_size, frame.hop_size)
            if self._last_frame_params != current_frame_params:
                self._last_frame_params = current_frame_params
                self._params_dirty = True
                self._reverb_fft_buffer = np.zeros((num_bins, 2), dtype=DEFAULT_COMPLEX_DTYPE)
                self._pre_delay_buffer.clear()

            if self._params_dirty:
                self._recalculate_params(frame)
            
            # Ensure reverb buffer is always stereo
            if self._reverb_fft_buffer.shape[1] != 2:
                 self._reverb_fft_buffer = np.zeros((num_bins, 2), dtype=DEFAULT_COMPLEX_DTYPE)

            # --- Pre-delay Logic ---
            self._pre_delay_buffer.append(frame.data)
            if len(self._pre_delay_buffer) < self._pre_delay_frames or self._pre_delay_frames == 0:
                delayed_frame_data = np.zeros_like(frame.data)
            else:
                delayed_frame_data = self._pre_delay_buffer[0]

            # --- Handle Mono Input ---
            dry_signal = frame.data
            if num_channels == 1:
                # Ensure delayed frame is also mono before stacking
                if delayed_frame_data.shape[1] > 1:
                    delayed_frame_data = delayed_frame_data[:, 0:1]
                delayed_frame_data = np.hstack([delayed_frame_data, delayed_frame_data])
                dry_signal = np.hstack([dry_signal, dry_signal])

            # --- STABLE DSP ALGORITHM ---
            wet_signal = self._reverb_fft_buffer * self._decay_factors
            self._reverb_fft_buffer = delayed_frame_data + wet_signal
            output_fft = (dry_signal * (1.0 - self._mix)) + (wet_signal * self._mix)
            
        output_frame = SpectralFrame(
            data=output_fft,
            fft_size=frame.fft_size, hop_size=frame.hop_size,
            window_size=frame.window_size, sample_rate=frame.sample_rate,
            analysis_window=frame.analysis_window
        )
        return {"spectral_frame_out": output_frame}

    def start(self):
        with self._lock:
            self._reverb_fft_buffer = None
            self._last_frame_params = (0, 0)
            self._pre_delay_buffer.clear()
            self._params_dirty = True
            
    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()