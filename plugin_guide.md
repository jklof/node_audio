Re-Node Processor: Plugin Implementation Guide
I. Introduction

A plugin consists of two main parts:

    The Logic Class: A subclass of Node that contains the actual processing code. This is the "engine" of your plugin, running in the real-time thread.

    The UI Class: A subclass of NodeItem that defines the visual representation and user interface for your node in the graph editor. This runs in the main UI thread.

These two components communicate using Qt's signals and slots to ensure a clean separation between the processing logic and the user interface, which is a core principle of this application's architecture.

File Location: Your plugin should be a single Python file placed in the plugins/ or additional_plugins/ directory. The application will automatically discover and register it on startup.
II. Part 1: The Logic Class (Node Subclass)

The Logic class is where all the data processing happens. It must inherit from node_system.Node.

Every logic class requires specific class-level attributes that tell the system how to register and display it.

```Python
# plugins/my_awesome_node.py
import torch
import threading
from node_system import Node
from ui_elements import NodeItem, ParameterNodeItem, NodeStateEmitter

# --- Logic Class Definition ---
class MyAwesomeNode(Node):
    # A unique identifier for serialization. MUST be unique across all nodes.
    NODE_TYPE = "My Awesome Node"

    # The custom UI class that will represent this node.
    UI_CLASS = MyAwesomeNodeItem # We will define this later

    # The category under which this node will appear in the "Add Node" menu.
    CATEGORY = "Effects"

    # A short description of what the node does.
    DESCRIPTION = "Applies an awesome effect to an audio signal."

```

The constructor is where you define your node's inputs, outputs, and initial internal state.
```Python

    
# In your MyAwesomeNode class:
def __init__(self, name, node_id=None):
    # 1. Always call the parent constructor first.
    super().__init__(name, node_id)

    # 2. Initialize the emitter for UI communication. This is crucial.
    self.emitter = NodeStateEmitter()

    # 3. Define the node's sockets (inputs and outputs).
    #    The name is the key used to access data.
    self.add_input("audio_in", data_type=torch.Tensor)
    self.add_input("intensity", data_type=float) # Input for external control
    self.add_output("audio_out", data_type=torch.Tensor)

    # 4. Initialize internal state and a lock for thread safety.
    self._lock = threading.Lock()
    self._intensity = 0.5 # Default value
```

Supported Data Types: torch.Tensor, float, int, bool, SpectralFrame, MIDIPacket, or None for a universal socket.

This is the heart of your node. The engine calls this method on every "tick" of the processing loop. This code runs in a separate, real-time thread, so it must be fast, non-blocking, and thread-safe.
```Python
# In your MyAwesomeNode class:
def process(self, input_data: dict) -> dict:
    audio_signal = input_data.get("audio_in")
    if audio_signal is None:
        return {"audio_out": None}

    # --- Thread-Safe Parameter Handling ---
    state_to_emit = None
    with self._lock:
        # Check for external control from an input socket first.
        intensity_socket = input_data.get("intensity")
        if intensity_socket is not None:
            # If the socket value differs from our internal state, update it.
            new_intensity = float(intensity_socket)
            if self._intensity != new_intensity:
                self._intensity = new_intensity
                # Prepare to notify the UI of the change.
                state_to_emit = self._get_current_state_snapshot_locked()
        
        # Use the (potentially updated) internal state for processing.
        current_intensity = self._intensity

    # Emit the UI update AFTER releasing the lock.
    if state_to_emit:
        self.emitter.stateUpdated.emit(state_to_emit)
    
    # --- Perform your DSP/processing logic here ---
    processed_signal = audio_signal * current_intensity

    # Return a dictionary where keys are your output socket names.
    return {"audio_out": processed_signal}
```
  

To ensure your node's settings are saved with the graph, you must implement serialize_extra and deserialize_extra.
```Python

    
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
  

III. Part 2: The UI Class - Choosing Your Path

You have two ways to build your UI class, depending on its complexity.

This is the best method for most nodes. It is simple, powerful, and enforces consistency.

Use this when: Your node's UI consists of standard controls (dials, sliders, combo boxes, spinboxes) that map directly to parameters in your logic node.

Example Implementation:
```Python

    
from ui_elements import ParameterNodeItem
from PySide6.QtCore import Slot

class MyAwesomeNodeItem(ParameterNodeItem):
    def __init__(self, node_logic: "MyAwesomeNode"):
        # 1. Define your UI controls declaratively.
        parameters = [
            {
                "key": "intensity",      # Must match the logic's state key
                "name": "Intensity",     # Label text
                "type": "slider",        # 'slider', 'dial', 'combobox', or 'spinbox'
                "min": 0.0,              # Min logical value
                "max": 1.0,              # Max logical value
                "format": "{:.1%}",      # String format for the value display
            }
        ]
        
        # 2. The base class constructor does all the work.
        super().__init__(node_logic, parameters)
```
  

The ParameterNodeItem base class will automatically:

    Create all the specified widgets and their labels.

    Connect widget signals (e.g., valueChanged) to the appropriate setter on your logic node (e.g., set_intensity).

    Connect the logic's stateUpdated signal to a built-in slot that updates all widgets.

    Automatically disable a control and add an (ext) suffix to its label if its corresponding input socket is connected.

This method provides maximum flexibility for complex or unique user interfaces.

Use this when: Your node requires a unique, custom-drawn widget (like a piano or spectrum display), a complex layout, or non-standard controls (like momentary buttons).

Example Implementation:
```Python

    
from ui_elements import NodeItem
from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout
from PySide6.QtCore import Qt, Slot, QSignalBlocker

class MyComplexNodeItem(NodeItem):
    def __init__(self, node_logic: MyComplexNode):
        super().__init__(node_logic)

        # 1. Manually create a container and all your widgets.
        self.container = QWidget()
        layout = QVBoxLayout(self.container)
        self.intensity_label = QLabel("Intensity: ...")
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        layout.addWidget(self.intensity_label)
        layout.addWidget(self.intensity_slider)

        # 2. Embed the container widget.
        self.setContentWidget(self.container)

        # 3. Manually connect UI widget signals to the logic's setters.
        self.intensity_slider.valueChanged.connect(self._on_slider_changed)

        # 4. Manually connect the logic's state emitter to the UI update slot.
        #    !!! CRITICAL: Use QueuedConnection for thread safety !!!
        self.node_logic.emitter.stateUpdated.connect(
            self._on_state_updated, Qt.ConnectionType.QueuedConnection
        )
    
    @Slot(int)
    def _on_slider_changed(self, slider_value: int):
        # Your custom logic to convert UI value to logical value
        logical_value = slider_value / 100.0
        self.node_logic.set_intensity(logical_value)

    @Slot(dict)
    def _on_state_updated(self, state: dict):
        # Your custom logic to update all widgets from the state dictionary
        intensity = state.get("intensity", 0.5)
        self.intensity_label.setText(f"Intensity: {intensity:.2f}")
        with QSignalBlocker(self.intensity_slider):
            self.intensity_slider.setValue(int(intensity * 100))

    @Slot()
    def updateFromLogic(self):
        """Pulls the initial state from the logic node to initialize the UI."""
        state = self.node_logic.get_current_state_snapshot()
        self._on_state_updated(state)
        super().updateFromLogic()
```
#### IV. Part 3: The Unidirectional Data Flow Pattern

This is the most important concept for creating robust, deadlock-free plugins.

1.  **UI Event → Logic Setter:** User interaction in the `NodeItem` calls a public `@Slot()` method on the `Node` logic (e.g., `set_intensity(0.5)`).
2.  **Logic Updates State (Under Lock):**
    *   The setter method acquires a `threading.Lock`.
    *   It updates its internal state variable (e.g., `self._intensity = 0.5`).
    *   It creates a *copy* of its complete state as a dictionary.
    *   The lock is released.
3.  **Logic Emits State (Lock Released):** **Crucially, *after* the lock is released**, the `stateUpdated` signal is emitted with the state dictionary as its payload.
4.  **UI Updates Itself:** The `NodeItem`'s `_on_state_updated` slot receives the state dictionary and updates all its widgets to match. This ensures the UI is always a perfect reflection of the logic's state.

**Example Logic Setter (The Correct Pattern):**
```python
# In your MyAwesomeNode (Logic) class:

def _get_current_state_snapshot_locked(self) -> dict:
    """Helper to create a state snapshot. ASSUMES LOCK IS ALREADY HELD."""
    return {"intensity": self._intensity}

def get_current_state_snapshot(self) -> dict:
    """Public, thread-safe method to get the current state."""
    with self._lock:
        return self._get_current_state_snapshot_locked()

@Slot(float)
def set_intensity(self, value: float):
    state_to_emit = None # <- Variable to hold the state copy
    with self._lock:
        new_value = float(value)
        if self._intensity != new_value:
            self._intensity = new_value
            # Get a snapshot of the NEW state while still under lock.
            state_to_emit = self._get_current_state_snapshot_locked()
    
    # Emit the signal AFTER releasing the lock.
    if state_to_emit:
        self.emitter.stateUpdated.emit(state_to_emit)
```
  

V. Writing Real-Time Safe Code in process()

The process() method runs in a high-priority, real-time thread where performance and predictability are critical. Slow or unpredictable operations can cause audible glitches ("dropouts").

    Rule 1: No Memory Allocations: Avoid creating new Python objects or PyTorch tensors inside process(). Pre-allocate all necessary buffers in __init__.

    Rule 2: Use Vectorized Operations: Never use Python for loops to iterate over samples in a tensor. A single PyTorch operation is thousands of times faster.

    Rule 3: Minimize Time Spent Under Lock: The threading.Lock is essential, but it can block the real-time thread. Keep the code inside the with self._lock: block as short as humanly possible. Only copy shared parameters to local variables, then release the lock before doing any heavy calculations.
