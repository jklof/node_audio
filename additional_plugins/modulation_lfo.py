import numpy as np
import threading
import logging
from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING

from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout, QSizePolicy
from PySide6.QtCore import Qt, Slot, QSignalBlocker

# --- FIX: Added DEFAULT_BLOCKSIZE to constants import ---
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE

logger = logging.getLogger(__name__)


# ===============================
# UI CLASS (No changes needed here, it was correct)
# ===============================
class LFONodeItem(NodeItem):
    """UI for LFO node: just one slider for frequency control."""

    def __init__(self, node_logic: "LFONode"):
        super().__init__(node_logic)

        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING)
        layout.setSpacing(4)

        self.freq_label = QLabel(f"Frequency: {node_logic.get_frequency_hz():.2f} Hz")
        self.freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.freq_label)

        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.slider_min_int, self.slider_max_int = 1, 2000  # integer steps
        self.logical_min_freq, self.logical_max_freq = 0.01, 20.0  # Hz range
        self.freq_slider.setRange(self.slider_min_int, self.slider_max_int)
        self.freq_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.freq_slider)

        self.container_widget.setLayout(layout)
        self.setContentWidget(self.container_widget)

        # Connect slider
        self.freq_slider.valueChanged.connect(self._on_slider_change)

        # Initial sync
        self.updateFromLogic()

    def _map_slider_to_logical(self, slider_value: int) -> float:
        # Map integer slider to float freq
        norm = (slider_value - self.slider_min_int) / (self.slider_max_int - self.slider_min_int)
        return self.logical_min_freq + norm * (self.logical_max_freq - self.logical_min_freq)

    def _map_logical_to_slider(self, logical_value: float) -> int:
        # Clamp logical value to ensure it's within the expected range before mapping
        clamped_logical = max(self.logical_min_freq, min(logical_value, self.logical_max_freq))
        norm = (clamped_logical - self.logical_min_freq) / (self.logical_max_freq - self.logical_min_freq)
        return int(round(self.slider_min_int + norm * (self.slider_max_int - self.slider_min_int)))

    @Slot(int)
    def _on_slider_change(self, slider_value: int):
        freq = self._map_slider_to_logical(slider_value)
        self.node_logic.set_frequency_hz(freq)
        self.freq_label.setText(f"Frequency: {freq:.2f} Hz")

    @Slot()
    def updateFromLogic(self):
        current_freq = self.node_logic.get_frequency_hz()
        with QSignalBlocker(self.freq_slider):
            self.freq_slider.setValue(self._map_logical_to_slider(current_freq))
        self.freq_label.setText(f"Frequency: {current_freq:.2f} Hz")
        super().updateFromLogic()


# ===============================
# LOGIC CLASS
# ===============================
class LFONode(Node):
    NODE_TYPE = "LFO"
    CATEGORY = "Modulation"
    DESCRIPTION = "Low-frequency oscillator for modulation (sine, square, saw)."
    UI_CLASS = LFONodeItem
    IS_CLOCK_SOURCE = False  # Driven by graph ticks

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        
        self.add_input("sync_control", data_type=bool)
        self.add_output("sine_out", data_type=float)
        self.add_output("square_out", data_type=float)
        self.add_output("saw_out", data_type=float)

        self.samplerate = DEFAULT_SAMPLERATE
        self.blocksize = DEFAULT_BLOCKSIZE
        self.lock = threading.Lock()

        self._frequency_hz = 1.0
        self._phase = 0.0  # [0, 1) range

        logger.debug(f"LFO [{self.name}] initialized at {self._frequency_hz} Hz")

    # -----------------
    # UI thread methods (thread-safe)
    # -----------------
    @Slot(float)
    def set_frequency_hz(self, freq: float):
        with self.lock:
            self._frequency_hz = max(0.001, float(freq))  # Avoid 0 Hz

    def get_frequency_hz(self) -> float:
        with self.lock:
            return self._frequency_hz

    # -----------------
    # Worker thread method
    # -----------------
    def process(self, input_data: dict) -> dict:
        sync_trigger = input_data.get("sync_control")
        with self.lock:
            freq = self._frequency_hz

        # phase increment calculation.
        # The phase must advance by the number of samples in one processing block (tick).
        phase_increment = (freq / self.samplerate) * self.blocksize
        
        # --- sync trigger ---
        if sync_trigger is not None and sync_trigger:
            self._phase = 0.0
        else:
            self._phase = (self._phase + phase_increment) % 1.0

        phase = self._phase

        # Calculate waveforms
        sine_val = float(np.sin(2 * np.pi * phase))
        square_val = 1.0 if phase < 0.5 else -1.0
        saw_val = (2.0 * phase) - 1.0  # Ramps from -1.0 to 1.0

        return {
            "sine_out": sine_val,
            "square_out": square_val,
            "saw_out": saw_val
        }

    def serialize_extra(self):
        with self.lock:
            return {"frequency_hz": self._frequency_hz, "phase": self._phase}

    def deserialize_extra(self, data):
        with self.lock:
            self._frequency_hz = float(data.get("frequency_hz", 1.0))
            self._phase = float(data.get("phase", 0.0))