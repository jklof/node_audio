import numpy as np
import sounddevice as sd
from collections import deque
import threading
import logging

from node_system import Node, IClockProvider
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_DTYPE, DEFAULT_CHANNELS

logger = logging.getLogger(__name__)

class AudioOutNode(Node, IClockProvider):
    NODE_TYPE = "Audio Output"
    CATEGORY = "Input / Output"
    DESCRIPTION = "Sends audio to an output device and can drive the graph clock."

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=np.ndarray)
        self._stream = None
        self._buffer = deque(maxlen=50)
        self._lock = threading.Lock()
        self._tick_callback: callable | None = None

    def _audio_callback(self, outdata: np.ndarray, frames: int, time, status):
        """High-priority audio thread callback."""
        # Only call the tick callback if this node is the active clock source
        if self._tick_callback:
            self._tick_callback()

        if status:
            logger.warning(f"Sounddevice status: {status}")
        try:
            with self._lock:
                if len(self._buffer) > 0:
                    data = self._buffer.popleft()
                    if data.shape == outdata.shape:
                        outdata[:] = data
                    else:
                        logger.warning(f"Shape mismatch in audio callback. Expected {outdata.shape}, got {data.shape}. Outputting silence.")
                        outdata.fill(0)
                else:
                    logger.warning("Audio buffer underrun!")
                    outdata.fill(0)
        except Exception as e:
            logger.error(f"Error in audio callback: {e}", exc_info=True)
            outdata.fill(0)

    def start(self):
        """
        Generic start method. Opens the audio stream in a passive state.
        This is called for all AudioSinkNodes when the graph starts processing.
        """
        if self._stream is not None:
            return # Already running
            
        logger.info(f"[{self.name}] Opening audio stream...")
        with self._lock:
            self._buffer.clear()
            num_prefill_blocks = 5
            for _ in range(num_prefill_blocks):
                self._buffer.append(np.zeros((DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS), dtype=DEFAULT_DTYPE))

        try:
            self._stream = sd.OutputStream(
                samplerate=DEFAULT_SAMPLERATE,
                blocksize=DEFAULT_BLOCKSIZE,
                channels=DEFAULT_CHANNELS,
                dtype=DEFAULT_DTYPE,
                latency='low',
                callback=self._audio_callback
            )
            self._stream.start()
            logger.info(f"[{self.name}] Audio stream started successfully.")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to open audio stream: {e}", exc_info=True)
            self._stream = None

    def stop(self):
        """Generic stop method. Closes the audio stream completely."""
        logger.info(f"[{self.name}] Stopping audio stream...")
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error(f"[{self.name}] Error closing audio stream: {e}", exc_info=True)
            finally:
                self._stream = None
        
        self._tick_callback = None # Ensure callback is cleared
        with self._lock:
            self._buffer.clear()
        logger.info(f"[{self.name}] Audio stream stopped.")

    # --- IClockProvider Implementation ---
    def start_clock(self, tick_callback: callable):
        """Promotes this node to be the ACTIVE clock source."""
        logger.info(f"[{self.name}] Promoting to ACTIVE clock source.")
        self.start() # Ensure the stream is running
        self._tick_callback = tick_callback

    def stop_clock(self):
        """Demotes this node from its active clock role, making it passive."""
        logger.info(f"[{self.name}] Demoting from active clock source to passive.")
        self._tick_callback = None
        # The stream remains open, continuing to function as a passive output.

    def process(self, input_data):
        """Receives audio from the graph and adds it to the buffer."""
        signal_block = input_data.get("in")
        
        if signal_block is None:
            signal_block = np.zeros((DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS), dtype=DEFAULT_DTYPE)

        processed_block = None
        if isinstance(signal_block, np.ndarray):
            if signal_block.shape == (DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS):
                processed_block = signal_block
            elif signal_block.ndim == 1 and signal_block.shape[0] == DEFAULT_BLOCKSIZE:
                processed_block = np.column_stack([signal_block, signal_block])
            
        if processed_block is not None:
            with self._lock:
                self._buffer.append(processed_block.astype(DEFAULT_DTYPE))
        else:
            with self._lock:
                self._buffer.append(np.zeros((DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS), dtype=DEFAULT_DTYPE))

        return {}