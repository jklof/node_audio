import numpy as np
from dataclasses import dataclass
from PySide6.QtGui import QColor
from typing import Any

# ==============================================================================
# Core Data Structure for Spectral Processing
# ==============================================================================

@dataclass
class SpectralFrame:
    """A data object holding FFT data and all necessary metadata for perfect reconstruction."""
    data: np.ndarray
    fft_size: int
    hop_size: int
    window_size: int
    sample_rate: int
    analysis_window: np.ndarray

# --- Configuration Defaults ---
DEFAULT_SAMPLERATE = 44100
DEFAULT_BLOCKSIZE = 512
DEFAULT_CHANNELS = 2
DEFAULT_BUFFER_SIZE_BLOCKS = 10
DEFAULT_DTYPE = np.float32

# FFT Default parameters
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_HOP_SIZE = 512  # Corresponds to 50% overlap for window_size=1024
DEFAULT_FFT_SIZE = DEFAULT_WINDOW_SIZE  # Usually same as window size
DEFAULT_COMPLEX_DTYPE = np.complex64

# --- Performance Monitoring ---
TICK_DURATION_S = DEFAULT_BLOCKSIZE / DEFAULT_SAMPLERATE
TICK_DURATION_NS = int(TICK_DURATION_S * 1_000_000_000)

# --- NEW: Central dictionary for socket type colors ---
SOCKET_TYPE_COLORS = {
    np.ndarray: QColor("#2274A5"),      # Muted Blue for Audio/Arrays
    float: QColor("#57A773"),           # Green for Floats
    bool: QColor("#F45B69"),            # Red for Bools/Triggers
    int: QColor("#F45B69"),             # Also red for ints (often used as triggers)
    SpectralFrame: QColor("#9A44B2"),   # Purple for Spectral Data
    Any: QColor("#E6E6E6"),             # White/Light Gray for Universal
    None: QColor("#E6E6E6"),            # Treat None as universal as well
    "default": QColor("#888888")        # A default gray for unregistered types
}