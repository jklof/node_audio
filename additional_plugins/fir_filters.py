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
from ui_elements import ParameterNodeItem, NODE_CONTENT_PADDING
from node_helpers import with_parameters, Parameter

# --- UI and Qt Imports ---
from PySide6.QtWidgets import QWidget, QLabel, QComboBox, QSlider, QVBoxLayout
from PySide6.QtCore import Qt, Slot, QSignalBlocker

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Filter Constants ---
NUM_TAPS = 511
FILTER_LATENCY_SAMPLES = (NUM_TAPS - 1) // 2


# ==============================================================================
# 1. UI Class for the Linear Phase EQ Node
# ==============================================================================
class LinearPhaseEQNodeItem(ParameterNodeItem):
    """
    Refactored UI for the Linear Phase EQ, now using ParameterNodeItem for
    simpler, declarative UI construction.
    """

    NODE_SPECIFIC_WIDTH = 200

    def __init__(self, node_logic: "LinearPhaseEQNode"):
        # Define the UI controls declaratively
        parameters = [
            {
                "key": "filter_type",
                "name": "Filter Type",
                "type": "combobox",
                "items": [
                    ("Low Pass", "Low Pass"),
                    ("High Pass", "High Pass"),
                    ("Band Pass", "Band Pass"),
                    ("Band Stop (Notch)", "Band Stop (Notch)"),
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
        ]
        # The parent class constructor handles all the heavy lifting
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        """
        Override the parent method to add custom logic after standard updates.
        """
        # 1. Let the parent class handle all standard widget updates first
        super()._on_state_updated_from_logic(state)

        # 2. Add custom logic: Show/hide the 'Q' control based on filter type
        filter_type = state.get("filter_type", "Low Pass")
        q_control = self._controls.get("q")

        if q_control:
            q_visible = filter_type in ["Band Pass", "Band Stop (Notch)"]
            q_control["label"].setVisible(q_visible)
            q_control["widget"].setVisible(q_visible)
            # Request a geometry update since visibility changed
            self.container_widget.adjustSize()
            self.update_geometry()


# ==============================================================================
# 2. Logic Class for the Linear Phase EQ Node
# ==============================================================================
@with_parameters
class LinearPhaseEQNode(Node):
    NODE_TYPE = "Linear Phase EQ"
    UI_CLASS = LinearPhaseEQNodeItem
    CATEGORY = "Filters"
    DESCRIPTION = "Applies a linear-phase (FIR) filter to an audio signal."

    # --- Declarative managed parameters ---
    # The decorator will create setters (e.g., set_filter_type) and handle state.
    # The on_change hook is used to set the dirty flag, preserving the optimization.
    filter_type = Parameter(default="Low Pass", on_change="_mark_params_dirty")
    cutoff_freq = Parameter(default=1000.0, on_change="_mark_params_dirty", clip=(20.0, 20000.0))
    q = Parameter(default=1.0, on_change="_mark_params_dirty", clip=(0.1, 10.0))

    def __init__(self, name: str, node_id: Optional[str] = None):
        # The decorator's __init__ wrapper will initialize self._filter_type,
        # self._cutoff_freq, and self._q before this is called.
        super().__init__(name, node_id)

        self._init_parameters()

        self.add_input("in", data_type=torch.Tensor)
        self.add_input("cutoff_freq", data_type=float)
        self.add_input("q", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._params_dirty: bool = True
        self._coeffs: Optional[torch.Tensor] = None
        self._history_buffer: Optional[torch.Tensor] = None
        self._expected_channels: Optional[int] = None

    def _get_state_snapshot_locked(self) -> dict:
        return self._get_parameters_state()

    def serialize_extra(self) -> dict:
        return self._serialize_parameters()

    def deserialize_extra(self, data: dict):
        self._deserialize_parameters(data)

    def _mark_params_dirty(self):
        """Callback for the decorator to set the dirty flag when a parameter changes."""
        self._params_dirty = True

    def _recalculate_coeffs(self):
        """
        Calculates the FIR filter coefficients based on the current parameters.
        """
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

        self._update_parameters_from_sockets(input_data)

        with self._lock:
            if self._expected_channels != num_channels:
                self._expected_channels = num_channels
                self._history_buffer = torch.zeros((num_channels, NUM_TAPS - 1), dtype=DEFAULT_DTYPE)
                self._params_dirty = True

            if self._params_dirty:
                self._recalculate_coeffs()

            if self._coeffs is None or self._history_buffer is None:
                return {"out": signal}

            combined_input = torch.cat([self._history_buffer, signal], dim=1)
            batched_input = combined_input.unsqueeze(0)
            filtered_batched = F.conv1d(batched_input, self._coeffs, padding="valid", groups=num_channels)
            filtered_signal = filtered_batched.squeeze(0)
            self._history_buffer = signal[:, -(NUM_TAPS - 1) :]

        return {"out": filtered_signal}
