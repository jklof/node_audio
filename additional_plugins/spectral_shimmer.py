import numpy as np
import threading
import logging
from typing import Dict, Optional

# --- Node System Imports ---
from node_system import Node
from constants import SpectralFrame, DEFAULT_DTYPE, DEFAULT_COMPLEX_DTYPE
from ui_elements import NodeItem, NODE_CONTENT_PADDING

# --- Qt Imports ---
from PySide6.QtCore import Qt, Signal, Slot, QObject, QSignalBlocker
from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout

# --- Logging ---
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. State Emitter for UI Communication
# ==============================================================================
class SpectralShimmerEmitter(QObject):
    stateUpdated = Signal(dict)

# ==============================================================================
# 2. Custom UI Class (SpectralShimmerNodeItem)
# ==============================================================================
class SpectralShimmerNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "SpectralShimmerNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)
        
        self.container_widget = QWidget()
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING)
        main_layout.setSpacing(5)

        # --- Create Slider Controls ---
        self.pitch_slider, self.pitch_label = self._create_slider_control("Pitch Shift", -12.0, 24.0, "{:+.1f} st")
        self.feedback_slider, self.feedback_label = self._create_slider_control("Feedback", 0.0, 1.0, "{:.0%}")
        self.mix_slider, self.mix_label = self._create_slider_control("Mix", 0.0, 1.0, "{:.0%}")
        
        for label, slider in [
            (self.pitch_label, self.pitch_slider),
            (self.feedback_label, self.feedback_slider),
            (self.mix_label, self.mix_slider)
        ]:
            main_layout.addWidget(label)
            main_layout.addWidget(slider)

        self.setContentWidget(self.container_widget)

        # --- Connect Signals ---
        self.pitch_slider.valueChanged.connect(self._on_pitch_changed)
        self.feedback_slider.valueChanged.connect(self._on_feedback_changed)
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
    def _on_pitch_changed(self):
        self.node_logic.set_pitch_shift(self._map_slider_to_logical(self.pitch_slider))
    @Slot()
    def _on_feedback_changed(self):
        self.node_logic.set_feedback(self._map_slider_to_logical(self.feedback_slider))
    @Slot()
    def _on_mix_changed(self):
        self.node_logic.set_mix(self._map_slider_to_logical(self.mix_slider))

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        sliders_map = {
            "pitch_shift": (self.pitch_slider, self.pitch_label),
            "feedback": (self.feedback_slider, self.feedback_label),
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
# 3. Node Logic Class (SpectralShimmerNode)
# ==============================================================================
class SpectralShimmerNode(Node):
    NODE_TYPE = "Spectral Shimmer"
    UI_CLASS = SpectralShimmerNodeItem
    CATEGORY = "Spectral"
    DESCRIPTION = "A pitch-shifting feedback effect for creating ethereal textures."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.emitter = SpectralShimmerEmitter()

        # --- Setup Sockets ---
        self.add_input("spectral_frame_in", data_type=SpectralFrame)
        self.add_input("pitch_shift", data_type=float)
        self.add_input("feedback", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("spectral_frame_out", data_type=SpectralFrame)

        self._lock = threading.Lock()
        
        # --- Internal State ---
        self._pitch_shift_st: float = 12.0
        self._feedback: float = 0.75
        self._mix: float = 0.5
        
        # --- DSP Buffers & State ---
        self._shimmer_buffer: Optional[np.ndarray] = None
        self._last_frame_params: tuple = (0, 0)

    def _emit_state_update_locked(self):
        state = self.get_current_state_snapshot(locked=True)
        self.emitter.stateUpdated.emit(state)

    # --- Parameter Setter Slots ---
    @Slot(float)
    def set_pitch_shift(self, value: float):
        with self._lock:
            self._pitch_shift_st = value
            self._emit_state_update_locked()
    @Slot(float)
    def set_feedback(self, value: float):
        with self._lock:
            self._feedback = np.clip(value, 0.0, 1.0)
            self._emit_state_update_locked()
    @Slot(float)
    def set_mix(self, value: float):
        with self._lock:
            self._mix = np.clip(value, 0.0, 1.0)
            self._emit_state_update_locked()

    def get_current_state_snapshot(self, locked: bool = False) -> Dict:
        state = {"pitch_shift": self._pitch_shift_st, "feedback": self._feedback, "mix": self._mix}
        if locked: return state
        with self._lock: return state

    def _pitch_shift_frame(self, frame_data: np.ndarray, ratio: float) -> np.ndarray:
        """
        Shifts pitch by resampling the frequency bins.
        Works on a per-channel basis.
        """
        num_bins, num_channels = frame_data.shape
        shifted_frame = np.zeros_like(frame_data)
        
        original_indices = np.arange(num_bins)
        # Create new indices by "stretching" or "squashing" the original index space
        shifted_indices = original_indices / ratio

        for i in range(num_channels):
            magnitudes = np.abs(frame_data[:, i])
            phases = np.angle(frame_data[:, i])
            
            # Interpolate the magnitudes at the new index locations
            # Frequencies shifted beyond the nyquist are faded to zero
            shifted_magnitudes = np.interp(original_indices, shifted_indices, magnitudes, left=0, right=0)
            
            # Reconstruct the complex number with the new magnitudes and original phases
            shifted_frame[:, i] = shifted_magnitudes * np.exp(1j * phases)
            
        return shifted_frame

    def process(self, input_data: dict) -> dict:
        frame = input_data.get("spectral_frame_in")
        if not isinstance(frame, SpectralFrame):
            return {"spectral_frame_out": None}

        with self._lock:
            # --- Update state from sockets if connected ---
            socket_params = {
                "pitch_shift": input_data.get("pitch_shift"),
                "feedback": input_data.get("feedback"),
                "mix": input_data.get("mix"),
            }
            if socket_params["pitch_shift"] is not None: self._pitch_shift_st = socket_params["pitch_shift"]
            if socket_params["feedback"] is not None: self._feedback = np.clip(socket_params["feedback"], 0.0, 1.0)
            if socket_params["mix"] is not None: self._mix = np.clip(socket_params["mix"], 0.0, 1.0)
            
            # --- Initialize / re-initialize buffers on format change ---
            num_bins, num_channels = frame.data.shape
            current_frame_params = (frame.fft_size, frame.hop_size, num_channels)
            if self._last_frame_params != current_frame_params:
                self._last_frame_params = current_frame_params
                self._shimmer_buffer = np.zeros((num_bins, num_channels), dtype=DEFAULT_COMPLEX_DTYPE)

            # --- Shimmer DSP Algorithm ---
            pitch_ratio = 2**(self._pitch_shift_st / 12.0)
            
            # 1. Pitch-shift the contents of the feedback buffer from the last frame
            shifted_tail = self._pitch_shift_frame(self._shimmer_buffer, pitch_ratio)
            
            # 2. Apply feedback and create the wet signal for this frame's output
            wet_signal = shifted_tail * self._feedback
            
            # 3. Create the new buffer content by mixing the current input with the processed tail
            self._shimmer_buffer = frame.data + wet_signal
            
            # 4. Mix the dry input with the wet signal for the final output
            output_fft = (frame.data * (1.0 - self._mix)) + (wet_signal * self._mix)
            
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
            self._shimmer_buffer = None
            self._last_frame_params = (0, 0, 0)
            
    def serialize_extra(self) -> dict:
        return self.get_current_state_snapshot()
    
    def deserialize_extra(self, data: dict):
        with self._lock:
            self._pitch_shift_st = data.get("pitch_shift", 12.0)
            self._feedback = data.get("feedback", 0.75)
            self._mix = data.get("mix", 0.5)