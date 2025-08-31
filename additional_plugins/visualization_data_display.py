import numpy as np
import time
import logging
import threading
from node_system import Node
from ui_elements import NodeItem

# --- Qt Imports ---
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy
from PySide6.QtCore import Qt, Slot, Signal, QObject

# --- Custom Type Imports ---
from constants import SpectralFrame

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================
# UI Item Class for Data Analyze Node
# ============================================================
class DataDisplayNodeItem(NodeItem):
    """Custom NodeItem UI for the Data Display Node."""

    def __init__(self, node_logic):
        super().__init__(node_logic)

        self.controls_widget = QWidget()
        main_layout = QVBoxLayout(self.controls_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(2)
        self.stats_label = QLabel("Stats: N/A")
        self.stats_label.setWordWrap(True)
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.stats_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.stats_label.setMinimumWidth(250)
        main_layout.addWidget(self.stats_label)
        self.controls_widget.setLayout(main_layout)
        self.setContentWidget(self.controls_widget)

        node_logic.statsUpdated.connect(self._update_stats_display)

    def _format_stats(self, stats: dict | None) -> str:
        """Formats the statistics dictionary into a display string."""
        if not stats:
            return "Stats: N/A"
        if "None" in stats:
            return "Type: None"

        lines = []
        # --- NEW: Handle SpectralFrame ---
        if stats.get("is_spectral_frame"):
            lines.append("Type: SpectralFrame")
            if "error" in stats:
                lines.append(f"Error: {stats['error']}")
            else:
                lines.append(f"Data Shape: {stats.get('data_shape', 'N/A')}")
                lines.append(f"Data DType: {stats.get('data_dtype', 'N/A')}")
                lines.append(f"FFT/Hop/Win: {stats.get('fft_size', 'N/A')} / {stats.get('hop_size', 'N/A')} / {stats.get('window_size', 'N/A')}")
                # New stats for the complex data
                lines.append(f"Mag. Min: {stats.get('mag_min', 'N/A'):.3f}")
                lines.append(f"Mag. Max: {stats.get('mag_max', 'N/A'):.3f}")
                lines.append(f"Mag. Mean: {stats.get('mag_mean', 'N/A'):.3f}")
        # Check for scalar value
        elif "value" in stats and "dtype" in stats:
            lines.append(f"Type: {stats['dtype']}")
            lines.append(f"Value: {stats['value']:.3f}")
        # Check for array stats
        elif "shape" in stats and "dtype" in stats:
            lines.append(f"Shape: {stats['shape']}")
            lines.append(f"Type: {stats['dtype']}")
            if "min" in stats:
                lines.append(f"Min: {stats['min']:.3f}, Max: {stats['max']:.3f}")
            if "mean" in stats:
                lines.append(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            if "error" in stats:
                lines.append(f"Error: {stats['error']}")
        else:
            lines.append("Type: Unknown")

        return "\n".join(lines)

    @Slot(dict)
    def _update_stats_display(self, stats_dict: dict):
        """Updates the statistics label when the logic node emits new stats."""
        display_string = self._format_stats(stats_dict)
        self.stats_label.setText(display_string)

    @Slot()
    def updateFromLogic(self):
        """Synchronizes the UI with the current state of the logic node."""
        latest_stats = self.node_logic.get_latest_stats()
        display_string = self._format_stats(latest_stats)
        self.stats_label.setText(display_string)
        super().updateFromLogic()


# ============================================================
# Logic Node Class for Data Analyze
# ============================================================
class DataDisplayNode(Node):
    NODE_TYPE = "Data Display"
    CATEGORY = "Visualization"
    DESCRIPTION = "Displays statistics for numpy arrays, floats, or spectral frames."
    UI_CLASS = DataDisplayNodeItem

    class WrappedSignal(QObject):
        _s = Signal(dict)
        def emit(self, data):
            data_copy = data.copy() if data is not None else None
            self._s.emit(data_copy)
        def connect(self, x): self._s.connect(x)

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.statsUpdated = self.WrappedSignal()

        self.add_input("in", data_type=None)
        self.add_output("out", data_type=None)

        self._update_interval = 0.5
        self._next_update_time = 0
        self._latest_stats = None
        self._lock = threading.Lock()

    def update_latest_stats(self, new_stats):
        with self._lock:
            if self._latest_stats != new_stats:
                self._latest_stats = new_stats.copy() if new_stats is not None else None
                return True
        return False

    def get_latest_stats(self) -> dict | None:
        with self._lock:
            return self._latest_stats.copy() if self._latest_stats is not None else None

    def process(self, input_data):
        signal = input_data.get("in")
        output_signal = signal

        current_time = time.monotonic()
        if current_time >= self._next_update_time:
            self._next_update_time = current_time + self._update_interval
            new_stats = None

            if signal is None:
                new_stats = {"None": 0}
            # --- NEW: Handle SpectralFrame ---
            elif isinstance(signal, SpectralFrame):
                try:
                    mag = np.abs(signal.data) # Get magnitude
                    new_stats = {
                        "is_spectral_frame": True,
                        "data_shape": str(signal.data.shape),
                        "data_dtype": str(signal.data.dtype),
                        "fft_size": signal.fft_size,
                        "hop_size": signal.hop_size,
                        "window_size": signal.window_size,
                        # New: add stats about the magnitude
                        "mag_min": np.min(mag),
                        "mag_max": np.max(mag),
                        "mag_mean": np.mean(mag),
                    }
                except Exception as e:
                    logger.error(f"DataDisplayNode [{self.name}]: Error analyzing SpectralFrame: {e}", exc_info=False)
                    new_stats = {"is_spectral_frame": True, "error": str(e)}
            # Handle NumPy arrays
            elif isinstance(signal, np.ndarray) and signal.size > 0:
                try:
                    new_stats = {
                        "shape": str(signal.shape),
                        "dtype": str(signal.dtype),
                        "mean": np.mean(signal),
                        "std": np.std(signal),
                        "min": np.min(signal),
                        "max": np.max(signal),
                    }
                except Exception as e:
                    new_stats = {"shape": str(signal.shape), "dtype": str(signal.dtype), "error": str(e)}
            # Handle Floats, Integers
            elif isinstance(signal, (float, int, np.number)):
                try:
                    new_stats = {"value": float(signal), "dtype": type(signal).__name__}
                except Exception as e:
                    new_stats = {"value": "Error", "dtype": type(signal).__name__}

            try:
                if self.update_latest_stats(new_stats):
                    self.statsUpdated.emit(new_stats)
            except RuntimeError as e:
                logger.error(f"DataDisplayNode [{self.name}]: Error emitting statsUpdated signal: {e}")

        return {"out": output_signal}