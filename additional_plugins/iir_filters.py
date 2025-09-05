import numpy as np
import threading
import logging

# --- Core Dependencies ---
try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- Node System Imports ---
from node_system import Node
from ui_elements import NodeItem, NodeStateEmitter, NODE_CONTENT_PADDING
from constants import DEFAULT_SAMPLERATE, DEFAULT_DTYPE, torch

# --- Qt Imports ---
from PySide6.QtCore import Qt, Slot, QSignalBlocker
from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants for Filters ---
DEFAULT_FILTER_ORDER = 4
MIN_CUTOFF_HZ = 20.0
MAX_CUTOFF_HZ = 20000.0
MIN_Q = 0.1
MAX_Q = 20.0


# ==============================================================================
# 1. Custom UI Class (IIRFilterNodeItem)
# ==============================================================================
class IIRFilterNodeItem(NodeItem):
    """Custom UI for all IIR Filter nodes."""

    def __init__(self, node_logic: "BaseIIRFilterNode"):
        super().__init__(node_logic)
        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(5)

        if not SCIPY_AVAILABLE:
            error_label = QLabel("Missing dependency:\n(scipy)")
            error_label.setStyleSheet("color: orange;")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(error_label)
            self.setContentWidget(self.container_widget)
            return

        self.cutoff_label = QLabel("Cutoff Freq: ... Hz")
        self.cutoff_slider = QSlider(Qt.Orientation.Horizontal)
        self.cutoff_slider.setRange(0, 1000)
        layout.addWidget(self.cutoff_label)
        layout.addWidget(self.cutoff_slider)

        self.q_widget = QWidget()
        q_layout = QVBoxLayout(self.q_widget)
        q_layout.setContentsMargins(0, 0, 0, 0)
        self.q_label = QLabel("Q: ...")
        self.q_slider = QSlider(Qt.Orientation.Horizontal)
        self.q_slider.setRange(int(MIN_Q * 100), int(MAX_Q * 100))
        q_layout.addWidget(self.q_label)
        q_layout.addWidget(self.q_slider)
        layout.addWidget(self.q_widget)

        self.setContentWidget(self.container_widget)

        self.cutoff_slider.valueChanged.connect(self._on_cutoff_change)
        self.q_slider.valueChanged.connect(self._on_q_change)
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)

    def _map_slider_to_freq(self, value):
        min_log = np.log10(MIN_CUTOFF_HZ)
        max_log = np.log10(MAX_CUTOFF_HZ)
        return 10 ** (min_log + (value / 1000.0) * (max_log - min_log))

    def _map_freq_to_slider(self, freq):
        min_log = np.log10(MIN_CUTOFF_HZ)
        max_log = np.log10(MAX_CUTOFF_HZ)
        norm = (np.log10(max(MIN_CUTOFF_HZ, freq)) - min_log) / (max_log - min_log)
        return int(np.clip(norm, 0, 1) * 1000)

    @Slot(int)
    def _on_cutoff_change(self, value: int):
        freq = self._map_slider_to_freq(value)
        self.node_logic.set_cutoff_hz(freq)

    @Slot(int)
    def _on_q_change(self, value: int):
        q = value / 100.0
        self.node_logic.set_q(q)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        cutoff = state.get("cutoff_hz", 1000.0)
        q = state.get("Q", 1.0)
        filter_type = state.get("filter_type", "lowpass")

        is_cutoff_ext = "cutoff_hz" in self.node_logic.inputs and self.node_logic.inputs["cutoff_hz"].connections
        is_q_ext = "Q" in self.node_logic.inputs and self.node_logic.inputs["Q"].connections

        # Update Cutoff control
        with QSignalBlocker(self.cutoff_slider):
            self.cutoff_slider.setValue(self._map_freq_to_slider(cutoff))
        self.cutoff_slider.setEnabled(not is_cutoff_ext)
        cutoff_label_text = f"Cutoff Freq: {cutoff:.0f} Hz"
        if is_cutoff_ext:
            cutoff_label_text += " (ext)"
        self.cutoff_label.setText(cutoff_label_text)

        # Update Q control
        with QSignalBlocker(self.q_slider):
            self.q_slider.setValue(int(q * 100))
        self.q_slider.setEnabled(not is_q_ext)
        q_label_text = f"Q: {q:.2f}"
        if is_q_ext:
            q_label_text += " (ext)"
        self.q_label.setText(q_label_text)

        self.q_widget.setVisible(filter_type == "bandpass")

    def updateFromLogic(self):
        if SCIPY_AVAILABLE:
            state = self.node_logic.get_state_snapshot()
            self._on_state_updated(state)
        super().updateFromLogic()


# ==============================================================================
# 2. Base Class for IIR Filters
# ==============================================================================
class BaseIIRFilterNode(Node):
    NODE_TYPE = None
    CATEGORY = "Filters"
    UI_CLASS = IIRFilterNodeItem
    FILTER_ORDER = DEFAULT_FILTER_ORDER
    filter_type = None

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.emitter = NodeStateEmitter()
        self._lock = threading.Lock()

        self.add_input("in", data_type=torch.Tensor)
        self.add_input("cutoff_hz", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._samplerate = float(DEFAULT_SAMPLERATE)
        self._cutoff_hz = 1000.0
        self._Q = 1.0 / np.sqrt(2)

        self._sos = None
        self._filter_state_zi = None
        self._coefficients_dirty = True

    def _calculate_wn(self, cutoff, q):
        return cutoff

    def _update_coefficients_locked(self):
        """
        --- CORRECTED ---
        Internal helper to recalculate filter coefficients.
        This method MUST be called with the node's lock already acquired.
        It does not acquire the lock itself, preventing deadlocks.
        """
        if not SCIPY_AVAILABLE or not self.filter_type:
            self._sos = None
            return

        cutoff_to_use = self._cutoff_hz
        q_to_use = self._Q
        nyquist = self._samplerate / 2.0
        clamped_cutoff = np.clip(cutoff_to_use, MIN_CUTOFF_HZ, nyquist * 0.99)
        clamped_Q = np.clip(q_to_use, MIN_Q, MAX_Q)

        try:
            Wn = self._calculate_wn(clamped_cutoff, clamped_Q)
            self._sos = scipy.signal.iirfilter(
                N=self.FILTER_ORDER, Wn=Wn, btype=self.filter_type,
                analog=False, ftype="butter", output="sos", fs=self._samplerate
            )
            self._filter_state_zi = None  # Invalidate state so it gets re-initialized
            self._coefficients_dirty = False
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.error(f"[{self.name}] Error calculating SOS coefficients: {e}. Filter disabled.")
            self._sos = None

    @Slot(float)
    def set_cutoff_hz(self, freq: float):
        state_to_emit = None
        with self._lock:
            new_freq = np.clip(float(freq), MIN_CUTOFF_HZ, MAX_CUTOFF_HZ).item()
            if self._cutoff_hz != new_freq:
                self._cutoff_hz = new_freq
                self._coefficients_dirty = True
                state_to_emit = self.get_state_snapshot(locked=True)
        # Emit signal AFTER releasing the lock
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    @Slot(float)
    def set_q(self, q: float):
        state_to_emit = None
        with self._lock:
            new_q = np.clip(float(q), MIN_Q, MAX_Q).item()
            if self._Q != new_q:
                self._Q = new_q
                self._coefficients_dirty = True
                state_to_emit = self.get_state_snapshot(locked=True)
        # Emit signal AFTER releasing the lock
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)

    def get_state_snapshot(self, locked: bool = False):
        def get_it():
            return {"cutoff_hz": self._cutoff_hz, "Q": self._Q, "filter_type": self.filter_type}
        if locked:
            return get_it()
        with self._lock:
            return get_it()

    def process(self, input_data: dict) -> dict:
        signal_in = input_data.get("in")
        if not isinstance(signal_in, torch.Tensor) or not SCIPY_AVAILABLE:
            return {"out": signal_in}

        state_to_emit = None
        with self._lock:
            ui_update_needed = False
            
            cutoff_in = input_data.get("cutoff_hz")
            if cutoff_in is not None:
                new_cutoff = np.clip(float(cutoff_in), MIN_CUTOFF_HZ, MAX_CUTOFF_HZ).item()
                if self._cutoff_hz != new_cutoff:
                    self._cutoff_hz = new_cutoff
                    self._coefficients_dirty = True
                    ui_update_needed = True

            q_in = input_data.get("Q")
            if q_in is not None and self.filter_type == "bandpass":
                new_q = np.clip(float(q_in), MIN_Q, MAX_Q).item()
                if self._Q != new_q:
                    self._Q = new_q
                    self._coefficients_dirty = True
                    ui_update_needed = True

            if ui_update_needed:
                state_to_emit = self.get_state_snapshot(locked=True)

            if self._coefficients_dirty:
                # --- CORRECTED ---
                # Call the helper method that assumes the lock is already held.
                self._update_coefficients_locked()

            if self._sos is None:
                output_tensor = signal_in
            else:
                # Transpose to (samples, channels) for scipy
                signal_in_np = signal_in.T.numpy()
                num_channels = signal_in_np.shape[1]

                if self._filter_state_zi is None or self._filter_state_zi.shape[-1] != num_channels:
                    zi_init_single = scipy.signal.sosfilt_zi(self._sos)
                    self._filter_state_zi = np.repeat(zi_init_single[:, :, np.newaxis], num_channels, axis=2)
                try:
                    signal_out_np, self._filter_state_zi = scipy.signal.sosfilt(
                        self._sos, signal_in_np, axis=0, zi=self._filter_state_zi
                    )
                    # Transpose back to (channels, samples) for torch
                    output_tensor = torch.from_numpy(signal_out_np.T.copy()).to(DEFAULT_DTYPE)
                except Exception as e:
                    logger.error(f"[{self.name}] Error during sosfilt: {e}. Resetting state.")
                    self._filter_state_zi = None
                    output_tensor = torch.zeros_like(signal_in)

        # Emit signal AFTER releasing the lock
        if state_to_emit:
            self.emitter.stateUpdated.emit(state_to_emit)
        
        return {"out": output_tensor}

    def start(self):
        with self._lock:
            self._filter_state_zi = None
            self._coefficients_dirty = True

    def stop(self):
        with self._lock:
            self._filter_state_zi = None

    def serialize_extra(self) -> dict:
        return self.get_state_snapshot()

    def deserialize_extra(self, data: dict):
        self.set_cutoff_hz(data.get("cutoff_hz", 1000.0))
        self.set_q(data.get("Q", 1.0 / np.sqrt(2)))

# ==============================================================================
# 3. Concrete Filter Node Implementations
# ==============================================================================
class LowpassFilterNode(BaseIIRFilterNode):
    NODE_TYPE = "IIR Lowpass Filter"
    DESCRIPTION = f"Applies a {DEFAULT_FILTER_ORDER}th order Butterworth lowpass filter."
    filter_type = "lowpass"

class HighpassFilterNode(BaseIIRFilterNode):
    NODE_TYPE = "IIR Highpass Filter"
    DESCRIPTION = f"Applies a {DEFAULT_FILTER_ORDER}th order Butterworth highpass filter."
    filter_type = "highpass"

class BandpassFilterNode(BaseIIRFilterNode):
    NODE_TYPE = "IIR Bandpass Filter"
    DESCRIPTION = f"Applies a {DEFAULT_FILTER_ORDER}th order Butterworth bandpass filter (constant Q)."
    filter_type = "bandpass"

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("Q", data_type=float)

    def _calculate_wn(self, cutoff, q):
        center_freq, Q, nyquist = cutoff, q, self._samplerate / 2.0
        bandwidth = center_freq / Q
        low_cutoff = max(MIN_CUTOFF_HZ, center_freq - (bandwidth / 2.0))
        high_cutoff = min(nyquist * 0.99, center_freq + (bandwidth / 2.0))
        return [low_cutoff, high_cutoff] if low_cutoff < high_cutoff else [center_freq - 1, center_freq]