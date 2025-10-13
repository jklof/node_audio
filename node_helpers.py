from PySide6.QtCore import Slot
import numpy as np
from enum import Enum


class Parameter:
    def __init__(self, default, on_change: str | None = None, clip: tuple | None = None):
        self.default = default
        self.on_change = on_change
        self.clip = clip


def managed_parameters(cls):
    """
    A class decorator that injects boilerplate for thread-safe parameter handling,
    with corrected serialization logic that is robust to plugin reloads.
    """
    managed_params = {name: attr for name, attr in cls.__dict__.items() if isinstance(attr, Parameter)}

    # --- 1. Generate public @Slot setters ---
    for name, param in managed_params.items():

        def make_setter(param_name, param_info):
            def setter(instance, value):
                # Apply clipping if defined
                if param_info.clip:
                    value = np.clip(value, param_info.clip[0], param_info.clip[1]).item()

                state_to_emit = None
                with instance._lock:
                    # Special handling for Enum comparison
                    is_enum = isinstance(getattr(instance, f"_{param_name}"), Enum)
                    current_val = getattr(instance, f"_{param_name}")

                    if (is_enum and current_val != value) or (not is_enum and current_val != value):
                        setattr(instance, f"_{param_name}", value)
                        # Call on_change callback if it exists
                        if param_info.on_change and hasattr(instance, param_info.on_change):
                            getattr(instance, param_info.on_change)()
                        state_to_emit = instance._get_state_snapshot_locked()
                if state_to_emit:
                    instance.ui_update_callback(state_to_emit)

            return setter

        # Determine the type hint for the slot from the parameter's default value
        slot_type = type(param.default)
        # For enums, the slot receives the enum member itself, not a string
        slot_setter = Slot(slot_type)(make_setter(name, param))
        setattr(cls, f"set_{name}", slot_setter)

    # --- 2. Override __init__ to create internal attributes (e.g., self._gain_db) ---
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        for name, param in managed_params.items():
            setattr(self, f"_{name}", param.default)
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init

    # --- 3. Inject boilerplate methods ---
    def _get_state_snapshot_locked(self):
        return {name: getattr(self, f"_{name}") for name in managed_params}

    # =========================================================================
    # --- serialize_extra ---
    # =========================================================================
    def serialize_extra(self):
        """
        CORRECTED: Serializes Enum members to their string names to ensure
        the state is reload-safe.
        """
        with self._lock:
            state = self._get_state_snapshot_locked()
            for key, value in state.items():
                if isinstance(value, Enum):
                    state[key] = value.name  # Store the name (e.g., 'SINE')
            return state

    # =========================================================================
    # --- deserialize_extra ---
    # =========================================================================
    def deserialize_extra(self, data: dict):
        """
        CORRECTED: Correctly deserializes Enum members from their string names,
        making the process robust to plugin reloads.
        """
        with self._lock:
            for name, param in managed_params.items():
                value = data.get(name)

                # If value is not in saved data, use the default
                if value is None:
                    value = param.default

                # Check if this parameter is an Enum
                elif isinstance(param.default, Enum) and isinstance(value, str):
                    enum_class = type(param.default)
                    try:
                        # Look up the member in the (potentially new) Enum class
                        value = enum_class[value]
                    except KeyError:
                        # Fall back to default if the name no longer exists
                        value = param.default

                # Apply clipping logic directly, just like the setter
                if param.clip:
                    value = np.clip(value, param.clip[0], param.clip[1]).item()

                # Set the internal attribute, bypassing the public setter
                setattr(self, f"_{name}", value)

                # Still call the on_change hook for side effects
                if param.on_change and hasattr(self, param.on_change):
                    getattr(self, param.on_change)()

    # =========================================================================
    # --- _update_params_from_sockets ---
    # =========================================================================
    def _update_params_from_sockets(self, input_data: dict):
        """
        IMPROVED: Now correctly handles clipping and on_change side effects,
        while still batching the UI update into a single signal.
        """
        state_to_emit = None
        with self._lock:
            ui_update_needed = False
            for name, param in managed_params.items():
                socket_val = input_data.get(name)
                if socket_val is not None:
                    # Apply clipping from the Parameter definition
                    if param.clip:
                        socket_val = np.clip(socket_val, param.clip[0], param.clip[1]).item()

                    if getattr(self, f"_{name}") != socket_val:
                        setattr(self, f"_{name}", float(socket_val))
                        ui_update_needed = True

                        # Trigger side effects if the value changed
                        if param.on_change and hasattr(self, param.on_change):
                            getattr(self, param.on_change)()

            if ui_update_needed:
                state_to_emit = self._get_state_snapshot_locked()

        if state_to_emit:
            self.ui_update_callback(state_to_emit)

    # Attach the injected methods to the class
    setattr(cls, "_get_state_snapshot_locked", _get_state_snapshot_locked)
    setattr(cls, "serialize_extra", serialize_extra)
    setattr(cls, "deserialize_extra", deserialize_extra)
    setattr(cls, "_update_params_from_sockets", _update_params_from_sockets)

    return cls
