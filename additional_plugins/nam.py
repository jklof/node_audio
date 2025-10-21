import os
import json
import logging
from typing import Optional, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from node_system import Node
from ui_elements import NodeItem, NODE_CONTENT_PADDING
from constants import DEFAULT_DTYPE

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog
from PySide6.QtCore import Slot

logger = logging.getLogger(__name__)

# ==============================================================================
# GROUND TRUTH NAM INFERENCE ENGINE
# Ported directly from official NeuralAmpModelerCore (models/wavenet.py)
# ==============================================================================

def _get_activation(name: str) -> nn.Module:
    """Matches models/_activations.py"""
    if name is None:
        return nn.Identity()
    try:
        return getattr(nn, name)()
    except AttributeError:
        # Fallback for some older models or varying naming conventions
        if name.lower() == 'tanh': return nn.Tanh()
        if name.lower() == 'relu': return nn.ReLU()
        if name.lower() == 'sigmoid': return nn.Sigmoid()
        if name.lower() == 'identity': return nn.Identity()
        raise ValueError(f"Unknown activation: {name}")

class NamConv1d(nn.Conv1d):
    """Custom Conv1d that knows how to import its weights from a flat array."""
    def import_weights(self, weights: torch.Tensor, i: int) -> int:
        if self.weight is not None:
            n = self.weight.numel()
            self.weight.data = weights[i : i + n].reshape(self.weight.shape).to(self.weight.device)
            i += n
        if self.bias is not None:
            n = self.bias.numel()
            self.bias.data = weights[i : i + n].reshape(self.bias.shape).to(self.bias.device)
            i += n
        return i

class _Layer(nn.Module):
    def __init__(self, condition_size: int, channels: int, kernel_size: int, dilation: int, activation: str, gated: bool):
        super().__init__()
        mid_channels = 2 * channels if gated else channels
        self._conv = NamConv1d(channels, mid_channels, kernel_size, dilation=dilation)
        self._input_mixer = NamConv1d(condition_size, mid_channels, 1, bias=False)
        self._activation = _get_activation(activation)
        self._1x1 = NamConv1d(channels, channels, 1)
        self._gated = gated
        self._channels_val = channels 

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor], out_length: int) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        zconv = self._conv(x)
        # If h is provided (conditioning), mix it in. Standard NAM often has condition_size=1 but doesn't use it differently than input.
        z1 = zconv
        if h is not None:
             z1 = z1 + self._input_mixer(h)[:, :, -zconv.shape[2] :]

        if self._gated:
            post_activation = self._activation(z1[:, : self._channels_val]) * torch.sigmoid(z1[:, self._channels_val :])
        else:
            post_activation = self._activation(z1)

        return (
            x[:, :, -post_activation.shape[2] :] + self._1x1(post_activation),
            post_activation[:, :, -out_length:],
        )

    def import_weights(self, weights: torch.Tensor, i: int) -> int:
        i = self._conv.import_weights(weights, i)
        i = self._input_mixer.import_weights(weights, i)
        i = self._1x1.import_weights(weights, i)
        return i

class _Layers(nn.Module):
    def __init__(self, input_size: int, condition_size: int, head_size, channels: int, kernel_size: int, dilations: Sequence[int], activation: str = "Tanh", gated: bool = True, head_bias: bool = True):
        super().__init__()
        self._rechannel = NamConv1d(input_size, channels, 1, bias=False)
        self._layers = nn.ModuleList([
            _Layer(condition_size, channels, kernel_size, dilation, activation, gated)
            for dilation in dilations
        ])
        self._head_rechannel = NamConv1d(channels, head_size, 1, bias=head_bias)
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)

    def forward(self, x: torch.Tensor, c: torch.Tensor, head_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out_length = x.shape[2] - (self.receptive_field - 1)
        x = self._rechannel(x)
        for layer in self._layers:
            x, head_term = layer(x, c, out_length)
            head_input = head_term if head_input is None else head_input[:, :, -out_length:] + head_term
        return self._head_rechannel(head_input), x

    def import_weights(self, weights: torch.Tensor, i: int) -> int:
        i = self._rechannel.import_weights(weights, i)
        for layer in self._layers:
            i = layer.import_weights(weights, i)
        i = self._head_rechannel.import_weights(weights, i)
        return i

class _Head(nn.Module):
    def __init__(self, channels: int, activation: str, num_layers: int, out_channels: int):
        super().__init__()
        layers = nn.Sequential()
        cin = channels
        for i in range(num_layers):
            layers.add_module(f"act_{i}", _get_activation(activation))
            layers.add_module(f"conv_{i}", NamConv1d(cin, channels if i != num_layers - 1 else out_channels, 1))
            cin = channels
        self._layers = layers

    def forward(self, x):
        return self._layers(x)

    def import_weights(self, weights: torch.Tensor, i: int) -> int:
        for module in self._layers:
            if isinstance(module, NamConv1d):
                i = module.import_weights(weights, i)
        return i

class WaveNet(nn.Module):
    def __init__(self, layers_configs: Sequence[Dict], head_config: Optional[Dict] = None, head_scale: float = 1.0, sample_rate: Optional[float] = None):
        super().__init__()
        self._layers = nn.ModuleList([_Layers(**lc) for lc in layers_configs])
        self._head = _Head(in_channels=layers_configs[-1]['head_size'], **head_config) if head_config is not None else None
        self._head_scale = head_scale
        self.receptive_field = 1 + sum([(layer.receptive_field - 1) for layer in self._layers])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is (B, C, L)
        y, head_input = x, None
        for layer in self._layers:
            head_input, y = layer(y, x, head_input=head_input) # Passing x as condition c
        
        head_input = self._head_scale * head_input
        if self._head is None:
            return head_input
        return self._head(head_input)

    def import_weights(self, weights: torch.Tensor):
        i = 0
        for layer in self._layers:
            i = layer.import_weights(weights, i)
        if self._head is not None:
             i = self._head.import_weights(weights, i)
        # Confirm we used all weights (optional, but good for debugging)
        if i != len(weights):
             logger.warning(f"NAM import_weights: Used {i} of {len(weights)} weights.")

def init_from_nam_json(nam_path: str) -> nn.Module:
    """
    Strictly follows models/_from_nam.py logic to initialize the correct model.
    """
    with open(nam_path, 'r') as fp:
        config = json.load(fp)
    
    architecture = config.get("architecture")
    weights = torch.tensor(config.get("weights"))
    model_config = config.get("config")
    
    if architecture == "WaveNet":
        model = WaveNet(
            layers_configs=model_config["layers"],
            head_config=model_config.get("head"),
            head_scale=model_config.get("head_scale", 1.0),
            sample_rate=config.get("sample_rate")
        )
    elif architecture == "Linear":
         # Minimal Linear implementation if needed
         raise NotImplementedError("Linear NAM models not yet supported in this plugin.")
    elif architecture == "LSTM":
         # Minimal LSTM implementation if needed
         raise NotImplementedError("LSTM NAM models not yet supported in this plugin.")
    else:
        raise ValueError(f"Unsupported NAM architecture: {architecture}")

    # Load weights using the custom flattened loader
    model.import_weights(weights)
    model.eval()
    return model

# ==============================================================================
# 1. UI Class for the NAM Node
# ==============================================================================
class NAMNodeItem(NodeItem):
    NODE_SPECIFIC_WIDTH = 220
    def __init__(self, node_logic: "NAMNode"):
        super().__init__(node_logic, width=self.NODE_SPECIFIC_WIDTH)
        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING)
        layout.setSpacing(5)
        self.load_button = QPushButton("Load .nam Model...")
        self.status_label = QLabel("No model loaded.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: lightgray;")
        layout.addWidget(self.load_button)
        layout.addWidget(self.status_label)
        self.setContentWidget(self.container_widget)
        self.load_button.clicked.connect(self._on_load_button_clicked)

    @Slot()
    def _on_load_button_clicked(self):
        parent_widget = self.scene().views()[0] if self.scene() and self.scene().views() else None
        file_path, _ = QFileDialog.getOpenFileName(parent_widget, "Open NAM Model", "", "NAM Files (*.nam)")
        if file_path:
            self.node_logic.load_model(file_path)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        super()._on_state_updated_from_logic(state)
        file_path = state.get("file_path")
        error_message = state.get("error_message")
        if error_message:
            self.status_label.setText(f"Error: {error_message}")
            self.status_label.setStyleSheet("color: red;")
            self.status_label.setToolTip(error_message)
        elif file_path:
            filename = os.path.basename(file_path)
            self.status_label.setText(f"Loaded: {filename}")
            self.status_label.setStyleSheet("color: lightgreen;")
            self.status_label.setToolTip(file_path)
        else:
            self.status_label.setText("No model loaded.")
            self.status_label.setStyleSheet("color: lightgray;")
            self.status_label.setToolTip("")

# ==============================================================================
# 2. Logic Class for the NAM Node
# ==============================================================================
class NAMNode(Node):
    NODE_TYPE = "Neural Amp Modeler"
    UI_CLASS = NAMNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "Processes audio using a trained Neural Amp Model (.nam) file."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_output("out", data_type=torch.Tensor)
        self._model: Optional[WaveNet] = None
        self._file_path: Optional[str] = None
        self._error_message: Optional[str] = None
        # Buffer for warm-up/overlap-add if needed, though standard NAM is stateless between blocks 
        # if we ignore the receptive field tail. For simplicity in this bespoke generic node, 
        # we might accept slight discontinuities at block boundaries or implement overlap later.
        # The official plugin usually handles this by keeping a small history buffer.
        self._input_history = None 

    @Slot(str)
    def load_model(self, file_path: str):
        logger.info(f"[{self.name}] Attempting to load model from: {file_path}")
        state_to_emit = None
        with self._lock:
            try:
                self._model = init_from_nam_json(file_path)
                self._file_path = file_path
                self._error_message = None
                self._input_history = None # Reset history on new model
                self.clear_error_state()
                logger.info(f"[{self.name}] Successfully loaded NAM model.")
            except Exception as e:
                self._model = None
                self._file_path = None
                self._error_message = str(e)
                self.error_state = self._error_message
                logger.error(f"[{self.name}] Failed to load NAM model: {e}", exc_info=True)
            state_to_emit = self._get_state_snapshot_locked()
        self.ui_update_callback(state_to_emit)

    def process(self, input_data: Dict) -> Dict:
        audio_in = input_data.get("in")
        if self._model is None or not isinstance(audio_in, torch.Tensor) or audio_in.numel() == 0:
             return {"out": audio_in}

        # Ensure mono for standard NAM models
        if audio_in.ndim == 2 and audio_in.shape[0] > 1:
             mono_in = torch.mean(audio_in, dim=0, keepdim=True)
        elif audio_in.ndim == 1:
             mono_in = audio_in.unsqueeze(0)
        else:
             mono_in = audio_in

        # --- Handle Receptive Field Padding ---
        # To avoid clicks at block boundaries, we need to pad the current block 
        # with the end of the previous block equal to (receptive_field - 1).
        rf = self._model.receptive_field
        
        with self._lock:
             if self._input_history is None:
                  # First block: pad with zeros
                  self._input_history = torch.zeros((1, rf - 1), dtype=mono_in.dtype, device=mono_in.device)
             
             # Prepend history to current input
             padded_input = torch.cat((self._input_history, mono_in), dim=1)
             # Save new history for next block
             self._input_history = padded_input[:, - (rf - 1):].detach()

        # Add batch dimension: [1, channels, samples]
        input_tensor = padded_input.unsqueeze(0)

        with torch.no_grad():
             output_tensor = self._model(input_tensor)

        # The model output will be shorter than the padded input by exactly (rf - 1) samples,
        # which means it should be exactly the length of our original 'mono_in'.
        return {"out": output_tensor.squeeze(0).to(DEFAULT_DTYPE)}

    def _get_state_snapshot_locked(self) -> Dict:
        return {
            "file_path": self._file_path,
            "error_message": self._error_message,
        }

    def serialize_extra(self) -> Dict:
        with self._lock:
            return {"file_path": self._file_path}

    def deserialize_extra(self, data: Dict):
        file_path = data.get("file_path")
        if file_path and os.path.exists(file_path):
            self.load_model(file_path)