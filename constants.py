import numpy as np
from dataclasses import dataclass

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