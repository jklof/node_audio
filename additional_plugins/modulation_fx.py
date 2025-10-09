import ctypes
import torch
from enum import Enum
from ffi_node import FFINodeBase
from ui_elements import ParameterNodeItem
from PySide6.QtCore import Slot
from constants import DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS

# Define a C-style pointer to a float pointer for our 2D audio data
float_pp = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))


# --- MODIFIED: Added VIBRATO ---
class ModulationEffectType(Enum):
    CHORUS = "Chorus"
    FLANGER = "Flanger"
    PHASER = "Phaser"
    VIBRATO = "Vibrato"


# UI Class (can reuse the existing one)
class ModulationFXNodeItem(ParameterNodeItem):
    NODE_SPECIFIC_WIDTH = 220

    def __init__(self, node_logic: "ModulationFXNode"):
        parameters = [
            {"key": "mode", "name": "Mode", "type": "combobox", "items": [(m.value, m) for m in ModulationEffectType]},
            {
                "key": "rate",
                "name": "Rate",
                "type": "dial",
                "min": 0.05,
                "max": 20.0,
                "format": "{:.2f} Hz",
                "is_log": True,
            },
            {"key": "depth", "name": "Depth", "type": "slider", "min": 0.0, "max": 1.0, "format": "{:.0%}"},
            {"key": "feedback", "name": "Feedback", "type": "slider", "min": 0.0, "max": 0.98, "format": "{:.0%}"},
            {"key": "mix", "name": "Mix", "type": "slider", "min": 0.0, "max": 1.0, "format": "{:.0%}"},
        ]
        super().__init__(node_logic, parameters, width=self.NODE_SPECIFIC_WIDTH)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):
        super()._on_state_updated_from_logic(state)
        mode = state.get("mode")

        # --- MODIFIED: Update visibility logic ---
        is_feedback_visible = mode in [ModulationEffectType.FLANGER, ModulationEffectType.PHASER]
        is_mix_visible = mode not in [ModulationEffectType.VIBRATO]

        feedback_control = self._controls.get("feedback")
        if feedback_control:
            feedback_control["widget"].setVisible(is_feedback_visible)
            feedback_control["label"].setVisible(is_feedback_visible)

        mix_control = self._controls.get("mix")
        if mix_control:
            mix_control["widget"].setVisible(is_mix_visible)
            mix_control["label"].setVisible(is_mix_visible)

        self.container_widget.adjustSize()
        self.update_geometry()


# Logic Class
class ModulationFXNode(FFINodeBase):
    NODE_TYPE = "Modulation FX"
    UI_CLASS = ModulationFXNodeItem
    CATEGORY = "Effects"
    DESCRIPTION = "High-performance Chorus, Flanger, Phaser, and Vibrato effects."

    LIB_NAME = "modulation_processor"  # Assumes the compiled library is named modulation_processor.dll/so/dylib
    API = {
        "prepare": {"argtypes": [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]},
        "reset": {"argtypes": [ctypes.c_void_p]},
        "set_parameters": {
            "argtypes": [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        },
        "process_block": {"argtypes": [ctypes.c_void_p, float_pp, float_pp, ctypes.c_int, ctypes.c_int]},
    }
    MAX_CHANNELS = DEFAULT_CHANNELS  # Define max channels for buffer allocation

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=torch.Tensor)
        self.add_input("rate", data_type=float)
        self.add_input("depth", data_type=float)
        self.add_input("feedback", data_type=float)
        self.add_input("mix", data_type=float)
        self.add_output("out", data_type=torch.Tensor)

        self._mode = ModulationEffectType.CHORUS
        self._rate_hz = 0.5
        self._depth = 0.5
        self._feedback = 0.0
        self._mix = 0.5

        self._output_buffer = torch.zeros((self.MAX_CHANNELS, DEFAULT_BLOCKSIZE), dtype=torch.float32)

        if self.dsp_handle:
            self.lib.prepare(self.dsp_handle, DEFAULT_SAMPLERATE, DEFAULT_BLOCKSIZE, self.MAX_CHANNELS)
            self._update_cpp_params()

    def _get_state_snapshot_locked(self) -> dict:
        return {
            "mode": self._mode,
            "rate": self._rate_hz,
            "depth": self._depth,
            "feedback": self._feedback,
            "mix": self._mix,
        }

    def _update_cpp_params(self):
        if not self.dsp_handle:
            return

        mode_map = {
            ModulationEffectType.CHORUS: 0,
            ModulationEffectType.FLANGER: 1,
            ModulationEffectType.PHASER: 2,
            ModulationEffectType.VIBRATO: 3,
        }
        mode_int = mode_map.get(self._mode, 0)

        self.lib.set_parameters(self.dsp_handle, mode_int, self._rate_hz, self._depth, self._feedback, self._mix)

    def _update_and_emit(self, **kwargs):
        state_to_emit = None
        with self._lock:
            changed = False
            for key, value in kwargs.items():
                attr_name = f"_{key}"
                if getattr(self, attr_name) != value:
                    setattr(self, attr_name, value)
                    changed = True
            if changed:
                self._update_cpp_params()
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    @Slot(ModulationEffectType)
    def set_mode(self, value: ModulationEffectType):
        self._update_and_emit(mode=value)

    @Slot(float)
    def set_rate(self, value: float):
        self._update_and_emit(rate_hz=value)

    @Slot(float)
    def set_depth(self, value: float):
        self._update_and_emit(depth=value)

    @Slot(float)
    def set_feedback(self, value: float):
        self._update_and_emit(feedback=value)

    @Slot(float)
    def set_mix(self, value: float):
        self._update_and_emit(mix=value)

    def start(self):
        if self.dsp_handle:
            self.lib.reset(self.dsp_handle)

    def process(self, input_data: dict) -> dict:
        if not self.dsp_handle or self.error_state:
            return {"out": None}

        # Update parameters from sockets
        if (val := input_data.get("rate")) is not None and self._rate_hz != val:
            self.set_rate(val)
        if (val := input_data.get("depth")) is not None and self._depth != val:
            self.set_depth(val)
        if (val := input_data.get("feedback")) is not None and self._feedback != val:
            self.set_feedback(val)
        if (val := input_data.get("mix")) is not None and self._mix != val:
            self.set_mix(val)

        # Process audio
        signal = input_data.get("in")
        if not isinstance(signal, torch.Tensor):
            return {"out": None}

        in_tensor = signal.contiguous().to(torch.float32)
        out_tensor = self._output_buffer
        num_channels, num_samples = in_tensor.shape

        in_channel_pointers = (ctypes.POINTER(ctypes.c_float) * num_channels)()
        out_channel_pointers = (ctypes.POINTER(ctypes.c_float) * num_channels)()
        for i in range(num_channels):
            in_channel_pointers[i] = ctypes.cast(in_tensor[i].data_ptr(), ctypes.POINTER(ctypes.c_float))
            out_channel_pointers[i] = ctypes.cast(out_tensor[i].data_ptr(), ctypes.POINTER(ctypes.c_float))

        self.lib.process_block(self.dsp_handle, in_channel_pointers, out_channel_pointers, num_channels, num_samples)

        return {"out": out_tensor[:num_channels, :].clone()}

    def serialize_extra(self) -> dict:
        with self._lock:
            state = self._get_state_snapshot_locked()
            state["mode"] = state["mode"].name  # Serialize enum as string
            return state

    def deserialize_extra(self, data: dict):
        """
        Loads the node's state from a dictionary, following the efficient pattern
        of setting all internal state before a single FFI update.
        """
        with self._lock:
            # Directly set internal Python attributes from the data dictionary.
            # This avoids multiple FFI calls and redundant UI updates.
            mode_name = data.get("mode", ModulationEffectType.CHORUS.name)
            try:
                # Safely convert the saved string back to an enum member
                self._mode = ModulationEffectType[mode_name]
            except KeyError:
                self._mode = ModulationEffectType.CHORUS  # Default on failure

            self._rate_hz = data.get("rate", 0.5)
            self._depth = data.get("depth", 0.5)
            self._feedback = data.get("feedback", 0.0)
            self._mix = data.get("mix", 0.5)

            # After setting all Python-side state, update the C++ side in a single batch.
            self._update_cpp_params()

        # No UI update callback is needed here. The engine's post-load sync
        # will handle updating the UI with the final, fully loaded state.
