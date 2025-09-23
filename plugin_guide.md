### `plugin_guide.md`

# Re-Node Processor: Plugin Implementation Guide

## I. Introduction

A plugin is a single Python file that defines a new audio or data processing unit for the node graph. It consists of two main classes:

1.  **The Logic Class (`Node` subclass):** This is the "engine" of your plugin. It contains the actual processing code that runs in a high-priority, real-time audio thread. Performance and thread safety are critical here.
2.  **The UI Class (`NodeItem` subclass):** This defines the visual representation and user interface for your node in the graph editor. It runs in the main UI thread and should never block.

These two components communicate using Qt's signals and slots, ensuring a clean separation between processing and presentation.

**File Location:** Place your plugin file in the `plugins/` or `additional_plugins/` directory. The application will automatically discover and register it on startup.

## II. The Logic Class (`Node` subclass)

The Logic class is where all data processing happens. It must inherit from `node_system.Node`.

### A. Boilerplate and Registration

Every logic class requires specific class-level attributes that tell the system how to register it.

```python
# plugins/my_awesome_node.py
import torch
from node_system import Node
from ui_elements import ParameterNodeItem # Usually the best choice for UI

# --- Logic Class Definition ---
class MyAwesomeNode(Node):
    # A unique identifier for serialization. MUST be unique across all nodes.
    NODE_TYPE = "My Awesome Node"

    # The custom UI class that will represent this node.
    UI_CLASS = MyAwesomeNodeItem # We will define this later

    # The category under which this node appears in the "Add Node" menu.
    CATEGORY = "Effects"

    # A short, helpful description of what the node does.
    DESCRIPTION = "Applies an awesome effect to an audio signal."
```

### B. The Constructor `__init__`

The constructor is where you define your node's inputs, outputs, and initial internal state.

```python
# In your MyAwesomeNode class:
    def __init__(self, name, node_id=None):
        # 1. Always call the parent constructor first. This also creates self._lock.
        super().__init__(name, node_id)

        # 2. Define the node's sockets (inputs and outputs).
        self.add_input("audio_in", data_type=torch.Tensor)
        self.add_input("intensity", data_type=float) # Input for external control
        self.add_output("audio_out", data_type=torch.Tensor)

        # 3. Initialize internal state. This is automatically thread-safe
        #    because the base class provides self._lock.
        self._intensity = 0.5 # Default value
```
**Supported Data Types:** `torch.Tensor`, `float`, `int`, `bool`, `SpectralFrame`, `MIDIPacket`, or `None` for a universal socket.

### C. The `process()` Method

This is the heart of your node. The engine calls this method on every "tick" of the processing loop. This code runs in a separate, real-time thread, so it must be **fast, non-blocking, and thread-safe.**

```python
# In your MyAwesomeNode class:
    def process(self, input_data: dict) -> dict:
        audio_signal = input_data.get("audio_in")
        if not isinstance(audio_signal, torch.Tensor):
            return {"audio_out": None} # Return silence/None if input is missing

        # --- Thread-Safe Parameter Handling ---
        ui_update_needed = False
        with self._lock:
            # Check for external control from an input socket first.
            intensity_socket_val = input_data.get("intensity")
            
            # If a socket is connected, its value overrides the internal state.
            if intensity_socket_val is not None:
                new_intensity = float(intensity_socket_val)
                if self._intensity != new_intensity:
                    self._intensity = new_intensity
                    ui_update_needed = True # Flag that the UI needs to be synced
            
            # Use the (potentially updated) internal state for processing.
            current_intensity = self._intensity

        # Emit the UI update AFTER releasing the lock.
        if ui_update_needed:
            self.ui_update_callback(self.get_current_state_snapshot())
        
        # --- Perform your DSP/processing logic here using PyTorch ---
        processed_signal = audio_signal * current_intensity

        # Return a dictionary where keys are your output socket names.
        return {"audio_out": processed_signal}
```

### D. Saving and Loading State

To ensure your node's settings are saved with the graph, implement `serialize_extra` and `deserialize_extra`.

```python
# In your MyAwesomeNode class:
    def serialize_extra(self) -> dict:
        """Return a dictionary of the node's state to be saved."""
        with self._lock:
            return {"intensity": self._intensity}

    def deserialize_extra(self, data: dict):
        """Load the node's state from the given dictionary."""
        # Use the public setter to ensure the UI is updated on load.
        self.set_intensity(data.get("intensity", 0.5))
```

## III. The UI Class (`NodeItem` subclass)

You have two main ways to build your UI class. **`ParameterNodeItem` is strongly preferred for most cases.**

### A. The Easy Way: `ParameterNodeItem`

This is the best method for most nodes. It is simple, powerful, and enforces a consistent look and feel.

**Use this when:** Your node's UI consists of standard controls (sliders, dials, combo boxes) that map directly to parameters in your logic node.

```python
from ui_elements import ParameterNodeItem

class MyAwesomeNodeItem(ParameterNodeItem):
    def __init__(self, node_logic: "MyAwesomeNode"):
        # 1. Define your UI controls declaratively.
        parameters = [
            {
                "key": "intensity",      # Must match a key in the state snapshot dict
                "name": "Intensity",     # Label text
                "type": "slider",        # 'slider', 'dial', or 'combobox'
                "min": 0.0,              # Min logical value
                "max": 1.0,              # Max logical value
                "format": "{:.1%}",      # String format for the value display
            }
        ]
        
        # 2. The base class constructor does all the hard work.
        super().__init__(node_logic, parameters)
```

The `ParameterNodeItem` base class will automatically:
- Create all the specified widgets and labels.
- Connect widget signals (e.g., `valueChanged`) to the appropriate setter on your logic node (e.g., `set_intensity`).
- Handle state updates from the logic to keep the UI perfectly in sync.
- Automatically disable a control and add an `(ext)` suffix to its label if its corresponding input socket is connected.

### B. The Custom Way: `NodeItem`

This method provides maximum flexibility for complex or unique user interfaces.

**Use this when:** Your node requires a custom-drawn widget (like a piano or spectrum display), a complex layout, or non-standard controls.

*For an example, see the implementation of `plugins/visualization_waveform.py`.*

## IV. The Unidirectional Data Flow Pattern

This is the most important concept for creating robust, deadlock-free plugins.

1.  **UI Event → Logic Setter:** User interaction in the `NodeItem` calls a public `@Slot()` method on the `Node` logic (e.g., `set_intensity(0.5)`).
2.  **Logic Updates State (Under Lock):**
    *   The setter method acquires `self._lock`.
    *   It updates its internal state variable (e.g., `self._intensity = 0.5`).
    *   A flag is set if the state changed.
    *   The lock is released.
3.  **Logic Emits State:** If the state changed, the node calls `self.ui_update_callback(self.get_current_state_snapshot())`. The base class handles getting the snapshot and emitting the signal **after the lock is released**.
4.  **UI Updates Itself:** The `NodeItem`'s `_on_state_updated_from_logic` slot receives the state dictionary and updates all its widgets to match. This ensures the UI is always a perfect reflection of the logic's state.

**Example Logic Setter (The Correct Pattern):**
```python
# In your MyAwesomeNode (Logic) class:

    def _get_state_snapshot_locked(self) -> dict:
        """Helper to create a state snapshot. The base class handles the lock."""
        return {"intensity": self._intensity}

    @Slot(float)
    def set_intensity(self, value: float):
        state_changed = False
        with self._lock:
            new_value = float(value)
            if self._intensity != new_value:
                self._intensity = new_value
                state_changed = True
        
        # Call the public snapshot method AFTER releasing the lock.
        if state_changed:
            self.ui_update_callback(self.get_current_state_snapshot())
```

## V. Writing Real-Time Safe Code in `process()`

The `process()` method runs in a high-priority thread where performance is critical. Slow operations can cause audible glitches ("dropouts").

- **Rule 1: No Memory Allocations:** Avoid creating new Python objects or PyTorch tensors inside `process()`. Pre-allocate all necessary buffers in `__init__` or when parameters change.
- **Rule 2: Use Vectorized Operations:** Never use Python `for` loops to iterate over samples in a tensor. A single vectorized PyTorch operation is thousands of times faster.
- **Rule 3: Minimize Time Under Lock:** The `threading.Lock` is essential, but it can block the real-time thread. Keep the code inside the `with self._lock:` block as short as possible. Only copy shared parameters to local variables, then release the lock before doing heavy calculations.