from PySide6.QtCore import Slot
import numpy as np
from enum import Enum


class Parameter:
    """Defines a parameter for a Node, used by both decorators."""

    def __init__(self, default, on_change: str | None = None, clip: tuple | None = None):
        self.default = default
        self.on_change = on_change
        self.clip = clip


# ==============================================================================
# : Decorator to reduce boilerplate code @with_parameters ===
# ==============================================================================
def with_parameters(cls):
    """
    A class decorator that injects HELPER methods for thread-safe parameter
    handling, promoting an explicit and extensible pattern. This is the new,
    recommended way to manage parameters.
    """
    # This dictionary is attached to the class itself for introspection
    cls._managed_params_info = {name: attr for name, attr in cls.__dict__.items() if isinstance(attr, Parameter)}

    # --- Generate public @Slot setters ---
    for name, param in cls._managed_params_info.items():

        def make_setter(param_name, param_info):
            def setter(instance, value):
                if param_info.clip:
                    value = np.clip(value, param_info.clip[0], param_info.clip[1]).item()
                state_to_emit = None
                with instance._lock:
                    is_enum = isinstance(getattr(instance, f"_{param_name}"), Enum)
                    current_val = getattr(instance, f"_{param_name}")
                    if (is_enum and current_val != value) or (not is_enum and current_val != value):
                        setattr(instance, f"_{param_name}", value)
                        if param_info.on_change and hasattr(instance, param_info.on_change):
                            getattr(instance, param_info.on_change)()
                        state_to_emit = instance._get_state_snapshot_locked()
                if state_to_emit:
                    instance.ui_update_callback(state_to_emit)

            return setter

        slot_type = type(param.default)
        slot_setter = Slot(slot_type)(make_setter(name, param))
        setattr(cls, f"set_{name}", slot_setter)

    # --- INJECT HELPER METHODS (with clearer names) ---
    def _init_parameters(self):
        """HELPER: Initializes internal attributes for all managed parameters."""
        for name, param in self.__class__._managed_params_info.items():
            setattr(self, f"_{name}", param.default)

    def _get_parameters_state(self) -> dict:
        """HELPER: Returns a dict of the current state of managed parameters."""
        return {name: getattr(self, f"_{name}") for name in self.__class__._managed_params_info}

    def _serialize_parameters(self) -> dict:
        """HELPER: Serializes only managed parameters, handling Enums."""
        with self._lock:
            state = self._get_parameters_state()
            for key, value in state.items():
                if isinstance(value, Enum):
                    state[key] = value.name
            return state

    def _deserialize_parameters(self, data: dict):
        """HELPER: Deserializes and sets managed parameters from a dict."""
        with self._lock:
            for name, param in self.__class__._managed_params_info.items():
                value = data.get(name)
                if value is None:
                    value = param.default
                elif isinstance(param.default, Enum) and isinstance(value, str):
                    enum_class = type(param.default)
                    try:
                        value = enum_class[value]
                    except KeyError:
                        value = param.default
                if param.clip:
                    value = np.clip(value, param.clip[0], param.clip[1]).item()
                setattr(self, f"_{name}", value)
                if param.on_change and hasattr(self, param.on_change):
                    getattr(self, param.on_change)()

    def _update_parameters_from_sockets(self, input_data: dict):
        """HELPER: Updates managed parameters from input sockets."""
        state_to_emit = None
        with self._lock:
            ui_update_needed = False
            for name, param in self.__class__._managed_params_info.items():
                socket_val = input_data.get(name)
                if socket_val is not None:
                    if param.clip:
                        socket_val = np.clip(socket_val, param.clip[0], param.clip[1]).item()
                    if getattr(self, f"_{name}") != socket_val:
                        setattr(self, f"_{name}", float(socket_val))
                        ui_update_needed = True
                        if param.on_change and hasattr(self, param.on_change):
                            getattr(self, param.on_change)()
            if ui_update_needed:
                state_to_emit = self._get_state_snapshot_locked()
        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    # Attach the public helper methods to the user's class
    setattr(cls, "_init_parameters", _init_parameters)
    setattr(cls, "_get_parameters_state", _get_parameters_state)
    setattr(cls, "_serialize_parameters", _serialize_parameters)
    setattr(cls, "_deserialize_parameters", _deserialize_parameters)
    setattr(cls, "_update_parameters_from_sockets", _update_parameters_from_sockets)
    return cls
