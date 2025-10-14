import torch
import numpy as np
import scipy.signal
import threading
import logging
from typing import Dict, Optional

# --- Node System Imports ---
from node_system import Node
from constants import DEFAULT_SAMPLERATE, DEFAULT_DTYPE
from ui_elements import ParameterNodeItem
from node_helpers import with_parameters, Parameter

# --- UI and Qt Imports ---
from PySide6.QtCore import Slot

# --- Logging ---
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. UI Class for the Biquad (IIR) Filter Node (NO CHANGES NEEDED)
# ==============================================================================
class BiquadFilterNodeItem(ParameterNodeItem):
    """
    The UI for this node does not need any changes. It relies on a state dictionary
    with specific keys, which the refactored logic node will continue to provide.
    """

    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "BiquadFilterNode"):
        parameters = [
            {
                "key": "filter_type",
                "name": "Filter Type",
                "type": "combobox",
                "items": [
                    ("Low Pass", "Low Pass"),
                    ("High Pass", "High Pass"),
                    ("Band Pass", "Band Pass"),
                    ("Notch", "Notch"),
                    ("Peaking", "Peaking"),
                    ("Low Shelf", "Low Shelf"),
                    ("High Shelf", "High Shelf"),
                ],
            },
            {
                "key": "cutoff_freq",
                "name": "Freq",
                "min": 20.0,
                "max": 20000.0,
                "format": "{:.0f} Hz",
                "is_log": True,
            },
            {
                "key": "q",
                "name": "Q",
                "min": 0.1,
                "max": 10.0,
                "format": "{:.2f}",
            },
            {
                "key": "gain_db",
                "name": "Gain",
                "min": -24.0,
                "max": 24.0,
                "format": "{:+.1f} dB",
            },
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        super()._on_state_updated_from_logic(state)
        filter_type = state.get("filter_type", "Low Pass")
        q_visible = filter_type in ["Low Pass", "High Pass", "Band Pass", "Notch", "Peaking"]
        gain_visible = filter_type in ["Peaking", "Low Shelf", "High Shelf"]
        self._controls["q"]["label"].setVisible(q_visible)
        self._controls["q"]["widget"].setVisible(q_visible)
        self._controls["gain_db"]["label"].setVisible(gain_visible)
        self._controls["gain_db"]["widget"].setVisible(gain_visible)
        self.container_widget.adjustSize()
        self.update_geometry()


# ==============================================================================
# 2. Logic Class for the Biquad (IIR) Filter Node (REFACTORED)
# ==============================================================================
@with_parameters
class BiquadFilterNode(Node):
    NODE_TYPE = "Biquad Filter (IIR)"
    UI_CLASS = BiquadFilterNodeItem
    CATEGORY = "Filters"
    DESCRIPTION = "Applies a highly efficient IIR filter (EQ)."

    # --- Declarative  parameters ---
    # The decorator automatically creates thread-safe setters, serialization, and state management.
    # The on_change hook cleanly handles the logic for dirtying the filter coefficients.
    filter_type = Parameter(default="Low Pass", on_change="_mark_params_dirty")
    cutoff_freq = Parameter(default=1000.0, clip=(20.0, 20000.0), on_change="_mark_params_dirty")
    q = Parameter(default=0.707, clip=(0.1, 10.0), on_change="_mark_params_dirty")
    gain_db = Parameter(default=0.0, clip=(-24.0, 24.0), on_change="_mark_params_dirty")

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)

        self._init_parameters()

        self.add_input("in", data_type=torch.Tensor)
        # Sockets now match parameter names for automatic modulation updates.
        self.add_input("cutoff_freq", data_type=float)
        self.add_input("q", data_type=float)
        self.add_input("gain_db", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        # --- DSP State ---
        self._params_dirty: bool = True
        self._b_coeffs: Optional[np.ndarray] = None
        self._a_coeffs: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None
        self._expected_channels: Optional[int] = None

    def _get_state_snapshot_locked(self) -> dict:
        return self._get_parameters_state()

    def serialize_extra(self) -> dict:
        return self._serialize_parameters()

    def deserialize_extra(self, data: dict):
        self._deserialize_parameters(data)

    def _mark_params_dirty(self):
        """Callback for the decorator to invalidate coefficient cache."""
        self._params_dirty = True
        self._zi = None  # Force re-initialization of filter state

    def _recalculate_coeffs(self):
        """(Unchanged) Calculates the filter coefficients based on internal state."""
        sr = DEFAULT_SAMPLERATE
        freq, q, gain_db = self._cutoff_freq, self._q, self._gain_db
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * freq / sr
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * q)
        b0, b1, b2, a0, a1, a2 = 0.0, 0.0, 0.0, 1.0, 0.0, 0.0

        filter_type = self._filter_type  # Read the enum member directly
        if filter_type == "Low Pass":
            b0 = (1 - cos_w0) / 2
            b1 = 1 - cos_w0
            b2 = (1 - cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
        elif filter_type == "High Pass":
            b0 = (1 + cos_w0) / 2
            b1 = -(1 + cos_w0)
            b2 = (1 + cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
        elif filter_type == "Band Pass":
            b0 = alpha
            b1 = 0
            b2 = -alpha
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
        elif filter_type == "Notch":
            b0 = 1
            b1 = -2 * cos_w0
            b2 = 1
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
        elif filter_type == "Peaking":
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A
        elif filter_type == "Low Shelf":
            beta = np.sqrt(A) * alpha * 2
            b0 = A * ((A + 1) - (A - 1) * cos_w0 + beta)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
            b2 = A * ((A + 1) - (A - 1) * cos_w0 - beta)
            a0 = (A + 1) + (A - 1) * cos_w0 + beta
            a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
            a2 = (A + 1) + (A - 1) * cos_w0 - beta
        elif filter_type == "High Shelf":
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
        logger.info(f"[{self.name}] Recalculated IIR coefficients for {filter_type}.")

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

        self._update_parameters_from_sockets(input_data)

        with self._lock:
            if self._params_dirty:
                self._recalculate_coeffs()

            if self._expected_channels != num_channels:
                self._expected_channels = num_channels
                self._zi = None  # Force re-initialization of filter state for new channel count

            if self._zi is None and self._b_coeffs is not None and self._a_coeffs is not None:
                zi_single_channel = scipy.signal.lfilter_zi(self._b_coeffs, self._a_coeffs)
                self._zi = np.tile(zi_single_channel, (num_channels, 1)).astype(np.float32)

            b, a, zi_current = self._b_coeffs, self._a_coeffs, self._zi

        if b is None or a is None or zi_current is None:
            return {"out": signal_tensor}

        signal_np = signal_tensor.numpy()
        filtered_signal_np, zf_next_np = scipy.signal.lfilter(b, a, signal_np, axis=-1, zi=zi_current)

        with self._lock:
            self._zi = zf_next_np

        return {"out": torch.from_numpy(filtered_signal_np.astype(np.float32))}

    # serialize_extra and deserialize_extra are now handled by the decorator.
