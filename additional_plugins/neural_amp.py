import os
import json
import logging
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn

from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from constants import DEFAULT_DTYPE, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog
from PySide6.QtCore import Slot

# --- Import the new ResamplingStream wrapper ---
try:
    from resampler import ResamplingStream
except ImportError:
    # Fallback or error handling if resampler.py isn't present yet in the environment
    logging.error("Could not import ResamplingStream from resampler.py")

logger = logging.getLogger(__name__)


# ==============================================================================
# JIT-COMPATIBLE STREAMING NAM IMPLEMENTATION
# ==============================================================================
def _get_activation(name: str) -> nn.Module:
    if name.lower() == "tanh":
        return nn.Tanh()
    if name.lower() == "relu":
        return nn.ReLU()
    logger.warning(f"Unsupported activation '{name}', falling back to Identity.")
    return nn.Identity()


class NAMConv1d(nn.Conv1d):
    def import_weights(self, weights: torch.Tensor, i: int) -> int:
        if self.weight is not None:
            n = self.weight.numel()
            self.weight.data = weights[i : i + n].view(self.weight.shape)
            i += n
        if self.bias is not None:
            n = self.bias.numel()
            self.bias.data = weights[i : i + n].view(self.bias.shape)
            i += n
        return i


class Layer(nn.Module):
    def __init__(
        self, condition_size: int, channels: int, kernel_size: int, dilation: int, activation: str, gated: bool
    ):
        super().__init__()
        self._channels = channels
        self._gated = gated

        mid_channels = 2 * channels if gated else channels
        self._conv = NAMConv1d(channels, mid_channels, kernel_size, dilation=dilation)
        self._input_mixer = NAMConv1d(condition_size, mid_channels, 1, bias=False)
        self._activation = _get_activation(activation)
        self._1x1 = NAMConv1d(channels, channels, 1)

        buffer_size = (kernel_size - 1) * dilation
        self.register_buffer("_buffer", torch.zeros(1, self._channels, buffer_size))

    def reset(self):
        self._buffer.zero_()

    def import_weights(self, weights: torch.Tensor, i: int) -> int:
        i = self._conv.import_weights(weights, i)
        i = self._input_mixer.import_weights(weights, i)
        return self._1x1.import_weights(weights, i)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = torch.cat([self._buffer, x], dim=2)

        if self._buffer.shape[2] > 0:
            self._buffer = block_input[:, :, -self._buffer.shape[2] :].detach()

        zconv = self._conv(block_input)
        z1 = zconv + self._input_mixer(h)[:, :, -zconv.shape[2] :]

        if self._gated:
            post_activation = self._activation(z1[:, : self._channels]) * torch.sigmoid(z1[:, self._channels :])
        else:
            post_activation = self._activation(z1)

        return x + self._1x1(post_activation), post_activation


class LayerStack(nn.Module):
    def __init__(
        self, input_size, condition_size, head_size, channels, kernel_size, dilations, activation, gated, head_bias
    ):
        super().__init__()
        self._receptive_field = 1 + (kernel_size - 1) * sum(dilations)
        self._rechannel = NAMConv1d(input_size, channels, 1, bias=False)
        self._layers = nn.ModuleList(
            [Layer(condition_size, channels, kernel_size, d, activation, gated) for d in dilations]
        )
        self._head_rechannel = NAMConv1d(channels, head_size, 1, bias=head_bias)

    def reset(self):
        for layer in self._layers:
            layer.reset()

    def import_weights(self, weights: torch.Tensor, i: int) -> int:
        i = self._rechannel.import_weights(weights, i)
        for layer in self._layers:
            i = layer.import_weights(weights, i)
        return self._head_rechannel.import_weights(weights, i)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, head_input: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._rechannel(x)
        for layer in self._layers:
            x, head_term = layer(x, c)
            if head_input is None:
                head_input = head_term
            else:
                head_input = head_input + head_term
        return self._head_rechannel(head_input), x


class Head(nn.Module):
    def __init__(self, in_channels, channels, activation, num_layers, out_channels):
        super().__init__()
        self._layers = nn.ModuleList()
        cin = in_channels
        for i in range(num_layers):
            cout = channels if i != num_layers - 1 else out_channels
            self._layers.append(nn.Sequential(_get_activation(activation), NAMConv1d(cin, cout, 1)))
            cin = channels

    def import_weights(self, weights: torch.Tensor, i: int) -> int:
        for layer_block in self._layers:
            i = layer_block[1].import_weights(weights, i)
        return i

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
        return x


class NAMWaveNet(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        net_config = config.get("config", config.get("architecture", config))
        self._layers = nn.ModuleList([LayerStack(**lc) for lc in net_config["layers"]])

        if net_config.get("head"):
            self._head: Optional[Head] = Head(in_channels=net_config["layers"][-1]["head_size"], **net_config["head"])
        else:
            self._head: Optional[Head] = None

        self._head_scale = net_config.get("head_scale", 1.0)
        self.receptive_field = 1 + sum([(l._receptive_field - 1) for l in self._layers])

    def reset(self):
        for layer_stack in self._layers:
            layer_stack.reset()

    def import_weights(self, weights: torch.Tensor):
        i = 0
        for ls in self._layers:
            i = ls.import_weights(weights, i)
        if self._head is not None:
            i = self._head.import_weights(weights, i)

        # The training code appends `head_scale` as the last value in the weights array.
        # We must read it here to consume all weights and prevent the mismatch warning.
        if i < len(weights):
            self._head_scale = weights[i]
            i += 1
        if i != len(weights):
            logger.warning(f"NAM weight mismatch: Used {i} of {len(weights)} weights.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, head_input = x, None
        for layer_stack in self._layers:
            head_input, y = layer_stack(y, x, head_input)
        assert head_input is not None
        result = head_input * self._head_scale
        if self._head is not None:
            result = self._head(result)
        return result


def load_nam_model(filepath: str) -> Tuple[nn.Module, int]:
    with open(filepath, "r") as f:
        config = json.load(f)
    if config.get("architecture", "WaveNet") != "WaveNet":
        raise ValueError("Unsupported NAM architecture.")
    model_config = config.get("config", {})
    model_samplerate = model_config.get("sample_rate", model_config.get("fs", 48000))
    model = NAMWaveNet(config)
    model.eval()
    model.import_weights(torch.tensor(config["weights"], dtype=DEFAULT_DTYPE))
    return model, int(model_samplerate)


# ==============================================================================
# UI & NODE IMPLEMENTATION
# ==============================================================================
class NAMNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, n):
        super().__init__(n, width=self.NODE_SPECIFIC_WIDTH)
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(5, 5, 5, 5)
        l.setSpacing(5)
        self.b = QPushButton("Load .nam Model...")
        self.s = QLabel("No model loaded.")
        self.s.setWordWrap(True)
        self.s.setStyleSheet("color: lightgray;")
        l.addWidget(self.b)
        l.addWidget(self.s)
        self.setContentWidget(w)
        self.b.clicked.connect(self._on_load)

    @Slot()
    def _on_load(self):
        p = self.scene().views()[0] if self.scene() and self.scene().views() else None
        fp, _ = QFileDialog.getOpenFileName(p, "Open NAM Model", "", "NAM Files (*.nam)")
        if fp:
            self.node_logic.load_model(fp)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        super()._on_state_updated_from_logic(state)
        fp, err = state.get("file_path"), state.get("error_message")
        if err:
            self.s.setText(f"Error: {err}")
            self.s.setStyleSheet("color: red;")
            self.s.setToolTip(err)
        elif fp:
            self.s.setText(f"Loaded: {os.path.basename(fp)}")
            self.s.setStyleSheet("color: lightgreen;")
            self.s.setToolTip(fp)
        else:
            self.s.setText("No model loaded.")
            self.s.setStyleSheet("color: lightgray;")
            self.s.setToolTip("")


class NAMNode(Node):
    NODE_TYPE = "Neural Amp Modeler"
    UI_CLASS = NAMNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Processes audio using a .nam file. Multi-channel audio is mixed to mono before processing."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_output("out", data_type=torch.Tensor)
        self._model: Optional[torch.jit.ScriptModule | NAMWaveNet] = None
        self._file_path: Optional[str] = None
        self._error_message: Optional[str] = None
        self._model_samplerate: int = DEFAULT_SAMPLERATE

        # --- Updated: Use ResamplingStream for clean continuity ---
        self._resampler_stream_in: Optional[ResamplingStream] = None
        self._resampler_stream_out: Optional[ResamplingStream] = None

        try:
            torch.set_flush_denormal(True)
        except Exception:
            pass

    @Slot(str)
    def load_model(self, file_path: str):
        state_to_emit = None
        with self._lock:
            # Reset streams
            self._resampler_stream_in = None
            self._resampler_stream_out = None

            try:
                model, model_samplerate = load_nam_model(file_path)
                self._model_samplerate = model_samplerate
                model.reset()

                if self._model_samplerate != DEFAULT_SAMPLERATE:
                    logger.info(
                        f"[{self.name}] Resampling configured: App ({DEFAULT_SAMPLERATE} Hz) <-> Model ({self._model_samplerate} Hz)"
                    )
                    # Initialize streaming resamplers (1 channel for NAM)
                    self._resampler_stream_in = ResamplingStream(
                        orig_sr=DEFAULT_SAMPLERATE,
                        target_sr=self._model_samplerate,
                        num_channels=1,
                        dtype=DEFAULT_DTYPE,
                    )
                    self._resampler_stream_out = ResamplingStream(
                        orig_sr=self._model_samplerate,
                        target_sr=DEFAULT_SAMPLERATE,
                        num_channels=1,
                        dtype=DEFAULT_DTYPE,
                    )

                try:
                    self._model = torch.jit.script(model)
                except Exception:
                    self._model = model
                self._file_path, self._error_message = file_path, None
                self.clear_error_state()
            except Exception as e:
                self._model, self._file_path, self._error_message, self.error_state = None, None, str(e), str(e)
                logger.error(f"[{self.name}] Failed to load model: {e}", exc_info=True)
            state_to_emit = self._get_state_snapshot_locked()
        self.ui_update_callback(state_to_emit)

    def start(self):
        with self._lock:
            if hasattr(self._model, "reset"):
                self._model.reset()
            if self._resampler_stream_in:
                self._resampler_stream_in.reset()
            if self._resampler_stream_out:
                self._resampler_stream_out.reset()

    def process(self, input_data: Dict) -> Dict:
        audio_in = input_data.get("in")
        if self._model is None or not isinstance(audio_in, torch.Tensor):
            return {"out": audio_in}

        # NAM processes mono. Mix down if necessary.
        mono_signal = torch.mean(audio_in, dim=0, keepdim=True) if audio_in.shape[0] > 1 else audio_in

        with torch.inference_mode():
            # --- 1. Input Resampling (if needed) ---
            input_to_model = None

            if self._resampler_stream_in:
                # Push current block into the input resampler stream
                self._resampler_stream_in.push(mono_signal)
                # Pull ALL available samples in the model's sample rate
                input_to_model = self._resampler_stream_in.pull()
            else:
                input_to_model = mono_signal

            # --- 2. Model Inference ---
            # If resampling yielded 0 samples (rare, but possible with tiny buffers), skip inference
            if input_to_model.shape[1] > 0:
                # NAM models generally preserve input length
                model_output = self._model(input_to_model.unsqueeze(0)).squeeze(0)
            else:
                model_output = torch.zeros_like(input_to_model)

            # --- 3. Output Resampling (if needed) ---
            final_output = None

            if self._resampler_stream_out:
                # Push model output (at model_sr) into the output resampler stream
                self._resampler_stream_out.push(model_output)

                # Pull exactly one block size for the application output.
                # ResamplingStream's internal buffer handles the rate conversion continuity.
                if self._resampler_stream_out.can_pull(DEFAULT_BLOCKSIZE):
                    final_output = self._resampler_stream_out.pull(DEFAULT_BLOCKSIZE)
                else:
                    # Buffer underflow handling: Output silence.
                    # This might happen briefly at start-up or if rates drift significantly.
                    final_output = torch.zeros((1, DEFAULT_BLOCKSIZE), dtype=DEFAULT_DTYPE)
            else:
                # If no resampling, the model output *should* match the blocksize
                # assuming 1:1 processing.
                final_output = model_output

            return {"out": final_output}

    def _get_state_snapshot_locked(self):
        return {"file_path": self._file_path, "error_message": self._error_message}

    def serialize_extra(self):
        with self._lock:
            return {"file_path": self._file_path}

    def deserialize_extra(self, data: Dict):
        fp = data.get("file_path")
        if fp and os.path.exists(fp):
            self.load_model(fp)
