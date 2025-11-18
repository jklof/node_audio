import torch
import torchaudio.transforms as T
import math
import warnings
from typing import Optional


class ResamplingStream:
    """
    A single-directional, streaming sample rate converter with a simplified
    FIFO-like API. Designed to minimize memory allocations during steady-state
    processing.
    """

    def __init__(
        self,
        orig_sr: int,
        target_sr: int,
        num_channels: int = 1,
        max_buffer_s: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ):
        if orig_sr <= 0 or target_sr <= 0:
            raise ValueError(f"Sample rates must be positive. Got orig_sr={orig_sr}, target_sr={target_sr}")
        if num_channels < 1:
            raise ValueError(f"Number of channels must be at least 1. Got {num_channels}")

        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.num_channels = num_channels
        self.dtype = dtype

        # FIFO stores audio already resampled to target_sr
        self.max_buffer_samples = int(max_buffer_s * target_sr)

        # Main Output Buffer
        self._output_fifo: torch.Tensor = torch.empty((num_channels, self.max_buffer_samples), dtype=self.dtype)
        self._fifo_write_pos: int = 0
        self._fifo_read_pos: int = 0

        self._resampler: Optional[T.Resample] = None
        self.resampling_is_needed = orig_sr != target_sr
        self.conversion_ratio = self.target_sr / self.orig_sr

        # --- History & Scratch Buffers (Optimization) ---
        # Amount of input history to keep for filter continuity.
        # 10ms + 1 sample is generally sufficient for standard resampling kernels.
        self._history_len = int(orig_sr * 0.01) + 1

        # Fixed buffer to store the tail of the previous input.
        self._input_history: torch.Tensor = torch.zeros((num_channels, self._history_len), dtype=self.dtype)

        # Reusable scratch buffer to assemble (History + New Input) without 'torch.cat'.
        # We initialize it with size 0 and resize lazily in push().
        self._scratch_input: torch.Tensor = torch.empty((num_channels, 0), dtype=self.dtype)

        if self.resampling_is_needed:
            self._resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr, dtype=dtype)

    def reset(self):
        """Resets pointers and history without deallocating main buffers."""
        self._fifo_write_pos = 0
        self._fifo_read_pos = 0
        self._input_history.zero_()

    def get_available_samples(self) -> int:
        return self._fifo_write_pos - self._fifo_read_pos

    def _realloc_for_channel_change(self, new_channels: int):
        """Handles rare reallocation if channel count changes dynamically."""
        warnings.warn(
            f"Channel count changed from {self.num_channels} to {new_channels}. " f"Reallocating buffers.",
            RuntimeWarning,
        )
        self.num_channels = new_channels
        self._output_fifo = torch.empty((new_channels, self.max_buffer_samples), dtype=self.dtype)
        self._input_history = torch.zeros((new_channels, self._history_len), dtype=self.dtype)
        self._scratch_input = torch.empty((new_channels, 0), dtype=self.dtype)
        self.reset()

    def push(self, audio_chunk: torch.Tensor) -> int:
        """
        Pushes input audio (at orig_sr). Resamples immediately and stores in FIFO.
        Optimized to avoid allocations in the hot path.
        """
        if audio_chunk.numel() == 0:
            return self.get_available_samples()

        # Ensure 2D (channels, samples)
        if audio_chunk.ndim == 1:
            audio_chunk = audio_chunk.unsqueeze(0)

        current_channels, input_samples = audio_chunk.shape

        # Handle channel change
        if current_channels != self.num_channels:
            self._realloc_for_channel_change(current_channels)

        # Ensure dtype matches (usually no-op if pipeline is consistent)
        if audio_chunk.dtype != self.dtype:
            audio_chunk = audio_chunk.to(self.dtype)

        resampled_chunk = audio_chunk

        if self.resampling_is_needed:
            # --- Zero-Alloc Assembly Strategy ---
            # We need to feed [History | New Input] to the resampler.
            # Instead of torch.cat (which allocates), we copy into a pre-sized scratch buffer.

            required_scratch_len = self._history_len + input_samples

            # Resize scratch buffer only if the incoming block size grows
            if self._scratch_input.shape[1] < required_scratch_len:
                self._scratch_input = torch.empty((self.num_channels, required_scratch_len), dtype=self.dtype)

            # View into scratch buffer for the active region
            active_scratch = self._scratch_input[:, :required_scratch_len]

            # 1. Copy history to start
            active_scratch[:, : self._history_len].copy_(self._input_history)

            # 2. Copy new input after history
            active_scratch[:, self._history_len :].copy_(audio_chunk)

            # 3. Run Resampler (This is the unavoidable allocation step,
            #    as T.Resample output size varies)
            resampled_full = self._resampler(active_scratch)

            # 4. Discard startup transients (samples generated mostly by history)
            #    Linear approximation of output length mapping
            samples_to_discard = int(self._history_len * self.conversion_ratio)

            if samples_to_discard < resampled_full.shape[1]:
                resampled_chunk = resampled_full[:, samples_to_discard:]
            else:
                resampled_chunk = torch.empty((self.num_channels, 0), dtype=self.dtype)

            # 5. Update History for next time using copy_ (no allocation)
            #    We take the last N samples from the *assembled* input (active_scratch)
            #    This correctly handles cases where input_samples < history_len
            self._input_history.copy_(active_scratch[:, -self._history_len :])

        # --- Buffer Writing Logic ---
        chunk_samples = resampled_chunk.shape[1]
        available_space = self.max_buffer_samples - self._fifo_write_pos

        if chunk_samples > available_space:
            # Buffer is full at the tail. We must compact (move data to index 0).
            current_size = self.get_available_samples()

            if current_size > 0:
                # SAFE COPY: Use .clone() to prevent "memory overlap" RuntimeError.
                # This allocates, but only happens on buffer wrap-around/overflow (rare).
                self._output_fifo[:, :current_size] = self._output_fifo[
                    :, self._fifo_read_pos : self._fifo_write_pos
                ].clone()

            self._fifo_read_pos = 0
            self._fifo_write_pos = current_size

            # Recalculate space
            available_space = self.max_buffer_samples - current_size

            # If still not enough space, we must drop data (Overflow condition)
            if chunk_samples > available_space:
                # Just fit what we can
                samples_to_write = available_space
                # (Optional) could warn here, but dropping samples in realtime is standard
            else:
                samples_to_write = chunk_samples
        else:
            samples_to_write = chunk_samples

        if samples_to_write > 0:
            self._output_fifo[:, self._fifo_write_pos : self._fifo_write_pos + samples_to_write] = resampled_chunk[
                :, :samples_to_write
            ]
            self._fifo_write_pos += samples_to_write

        return self.get_available_samples()

    def pull(self, num_samples: Optional[int] = None) -> torch.Tensor:
        """
        Pulls resampled audio. Returns a cloned tensor to ensure the consumer
        owns the memory and the FIFO can be safely overwritten.
        """
        available_samples = self.get_available_samples()

        # Pull All
        if num_samples is None:
            if available_samples == 0:
                return torch.empty((self.num_channels, 0), dtype=self.dtype)

            # Must clone because we are about to reset pointers which conceptually
            # "frees" this memory region for writing.
            chunk = self._output_fifo[:, self._fifo_read_pos : self._fifo_write_pos].clone()
            self.reset()
            return chunk

        # Validation
        if num_samples <= 0:
            return torch.empty((self.num_channels, 0), dtype=self.dtype)

        if available_samples < num_samples:
            # Underrun
            return torch.empty((self.num_channels, 0), dtype=self.dtype)

        # Pull Specific Amount
        chunk = self._output_fifo[:, self._fifo_read_pos : self._fifo_read_pos + num_samples].clone()
        self._fifo_read_pos += num_samples

        # Optimization: If buffer empty, reset pointers to 0 to avoid needless compaction later
        if self._fifo_read_pos == self._fifo_write_pos:
            self._fifo_read_pos = 0
            self._fifo_write_pos = 0

        return chunk

    def can_pull(self, num_samples: int) -> bool:
        return self.get_available_samples() >= num_samples
