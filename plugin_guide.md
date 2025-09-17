Re-Node Processor: Plugin Implementation Guide
I. Introduction


A plugin consists of two main parts:

    The Logic Class: A subclass of Node that contains the actual processing code. This is the "engine" of your plugin.

    The UI Class: A subclass of NodeItem that defines the visual representation and user interface for your node in the graph editor.

These two components communicate using Qt's signals and slots to ensure a clean separation between the real-time processing logic and the user interface, which is a core principle of this application's architecture.

File Location: Your plugin should be a single Python file placed in the plugins/ or additional_plugins/ directory. The application will automatically discover and register it on startup.
II. Part 1: The Logic Class (Node Subclass)

The Logic class is where all the data processing happens. It must inherit from node_system.Node.
2.1. Boilerplate and Metadata

Every logic class requires specific class-level attributes that tell the system how to register and display it.

```Python
# plugins/my_awesome_node.py
import torch
import threading
from node_system import Node
from ui_elements import NodeItem, NodeStateEmitter # Import necessary components

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

    def __init__(self, name, node_id=None):
        # ... implementation ...
```

2.2. The __init__ Method

The constructor is where you define your node's inputs, outputs, and initial state.

```Python
# In your MyAwesomeNode class:
def __init__(self, name, node_id=None):
    # 1. Always call the parent constructor first.
    super().__init__(name, node_id)

    # 2. Initialize the emitter for UI communication. This is crucial.
    self.emitter = NodeStateEmitter()

    # 3. Define the node's sockets (inputs and outputs).
    #    The name is the key used to access data.
    #    The data_type helps the engine validate connections.
    self.add_input("audio_in", data_type=torch.Tensor)
    self.add_input("intensity", data_type=float)
    self.add_output("audio_out", data_type=torch.Tensor)

    # 4. Initialize internal state and a lock for thread safety.
    self._lock = threading.Lock()
    self._intensity = 0.5 # Default value
```

Supported Data Types: torch.Tensor, float, int, bool, SpectralFrame, or None for a universal socket.
2.3. The process() Method

This is the heart of your node. The engine calls this method on every "tick" of the processing loop. This code runs in a separate, real-time thread, so it must be fast and non-blocking.

```Python
# In your MyAwesomeNode class:
def process(self, input_data: dict) -> dict:
    # `input_data` is a dictionary where keys are your input socket names.
    audio_signal = input_data.get("audio_in")
    
    # If a crucial input is missing, it's good practice to return early.
    if audio_signal is None:
        return {"audio_out": None}

    #
    # --- IMPORTANT: Get parameters safely ---
    #
    state_to_emit = None
    with self._lock:
        # Check for external control from an input socket first.
        intensity_socket = input_data.get("intensity")
        if intensity_socket is not None:
            # If the socket value is different from our internal state, update it.
            # This allows the UI to reflect the change from an external connection.
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
    
    #
    # --- Perform your DSP/processing logic here ---
    # See Section V for critical real-time guidelines.
    #
    processed_signal = audio_signal * current_intensity

    # Return a dictionary where keys are your output socket names.
    return {"audio_out": processed_signal}
```

2.4. Saving and Loading State

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
  

III. Part 2: The UI Class (NodeItem Subclass)

The UI class defines how your node looks and feels. It inherits from ui_elements.NodeItem and uses standard PySide6 widgets.
3.1. Boilerplate and UI Construction
```Python
from PySide6.QtWidgets import QWidget, QLabel, QSlider, QVBoxLayout
from PySide6.QtCore import Qt, Slot, QSignalBlocker

# --- UI Class Definition ---
class MyAwesomeNodeItem(NodeItem):
    def __init__(self, node_logic: MyAwesomeNode):
        super().__init__(node_logic)

        # 1. Create a container widget that will hold all your controls.
        self.container = QWidget()
        layout = QVBoxLayout(self.container)

        # 2. Create your UI controls (sliders, dials, labels, etc.).
        self.intensity_label = QLabel("Intensity: ...")
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 100)
        
        layout.addWidget(self.intensity_label)
        layout.addWidget(self.intensity_slider)

        # 3. Embed the container widget into the NodeItem.
        self.setContentWidget(self.container)

        # 4. Connect UI widget signals to handler slots within this class.
        self.intensity_slider.valueChanged.connect(self._on_slider_changed)

        # 5. Connect the logic node's state emitter to the UI update slot.
        self.node_logic.emitter.stateUpdated.connect(self._on_state_updated)
```
  

3.2. Initializing the UI State

When your node is first created or loaded from a file, its UI needs to be synchronized with the logic's state. The updateFromLogic method is called automatically for this purpose.

```Python
# In your MyAwesomeNodeItem class:
@Slot()
def updateFromLogic(self):
    """Pulls the current state from the logic node to initialize the UI."""
    state = self.node_logic.get_current_state_snapshot()
    self._on_state_updated(state)
    super().updateFromLogic()
```
  

IV. Part 3: Connecting Logic and UI (The Emitter Pattern)

This is the most important concept for creating robust plugins. The data flow is strictly unidirectional:

    UI Event: A user interacts with a widget (e.g., moves a slider).

    UI Handler: The NodeItem's slot is triggered. It calls a public "setter" method on the logic node.

    Logic Update: The Node logic class updates its internal state.

    Emit State: The logic class emits a stateUpdated signal containing a dictionary snapshot of its entire current state.

    UI Update: The NodeItem's update slot receives the state dictionary and updates all of its widgets to reflect that state.

This pattern ensures the logic class is the single source of truth and prevents the UI from ever getting out of sync.
4.1. Step-by-Step Implementation

1. Create a public setter and state snapshot method in the Logic Class:
```Python
# In your MyAwesomeNode (Logic) class:

# A helper to create the state snapshot.
def _get_current_state_snapshot_locked(self) -> dict:
    return {"intensity": self._intensity}

def get_current_state_snapshot(self) -> dict:
    with self._lock:
        return self._get_current_state_snapshot_locked()

# Public setter for the UI to call.
@Slot(float)
def set_intensity(self, value: float):
    state_to_emit = None
    with self._lock:
        new_value = float(value)
        if self._intensity != new_value:
            self._intensity = new_value
            # Get a snapshot of the NEW state to emit.
            state_to_emit = self._get_current_state_snapshot_locked()
    
    # Emit the signal AFTER releasing the lock.
    if state_to_emit:
        self.emitter.stateUpdated.emit(state_to_emit)
```
  

2. Implement the handler and update slots in the UI Class:
code Python

```Python
# In your MyAwesomeNodeItem (UI) class:

@Slot(int)
def _on_slider_changed(self, slider_value: int):
    """UI handler: Converts UI value to logical value and calls the logic's setter."""
    logical_value = slider_value / 100.0
    self.node_logic.set_intensity(logical_value)

@Slot(dict)
def _on_state_updated(self, state: dict):
    """UI update: Receives the full state and updates all relevant widgets."""
    intensity = state.get("intensity", 0.5)
    
    # Update the label text.
    self.intensity_label.setText(f"Intensity: {intensity:.2f}")

    # Update the slider position. Use QSignalBlocker to prevent an infinite loop.
    with QSignalBlocker(self.intensity_slider):
        self.intensity_slider.setValue(int(intensity * 100))
        
    # Also, disable the slider if the corresponding input is connected.
    is_ext_controlled = "intensity" in self.node_logic.inputs and \
                        self.node_logic.inputs["intensity"].connections
    self.intensity_slider.setEnabled(not is_ext_controlled)
    if is_ext_controlled:
         self.intensity_label.setText(f"Intensity: {intensity:.2f} (ext)")
```

V. Writing Real-Time Safe Code in process()

The process() method is special. It runs in a high-priority, real-time thread where performance and predictability are critical. An operation that is slow or unpredictable can cause audible glitches ("dropouts"). Follow these rules to ensure your node is well-behaved.
Rule 1: No Memory Allocations

Avoid creating new Python objects or PyTorch tensors inside process(). Allocations can trigger the Garbage Collector (GC), which may pause the audio thread unpredictably.

    DO NOT create new tensors with torch.tensor(), torch.zeros(), torch.empty(), etc.

    DO NOT create lists, dictionaries, or other Python objects inside the process() method.

Solution: Pre-allocation.
Allocate all necessary buffers once in the __init__ method. If their size depends on the input signal, create a _resize_buffers(shape) method that re-allocates them only when the input shape changes. The CompressorNode is the canonical example of this pattern.
Rule 2: Use Vectorized Operations

Never use Python loops (for, while) to iterate over samples in a tensor. A single PyTorch operation is thousands of times faster.

    BAD (Slow):
```Python
# NEVER DO THIS in process()
output = torch.empty_like(signal)
for i in range(signal.shape[1]):
    output[0, i] = signal[0, i] * 0.5
```
  
GOOD (Fast):
```Python
    # This is a single, highly-optimized operation.
    output = signal * 0.5
```
      

Rule 3: Use In-Place Operations to Avoid Temporaries

Even simple vectorized operations can create hidden "temporary" tensors. Use in-place functions to write results directly into pre-allocated buffers.

    OK, but creates a temporary tensor:
    code Python

```Python
# `signal / 20.0` creates a new tensor to hold the result
# before passing it to torch.pow.
torch.pow(10.0, signal / 20.0, out=self.output_buffer)
```
  

BETTER (No temporary tensor):
```Python
# 1. First, divide in-place into the original buffer.
torch.div(signal, 20.0, out=signal)
# 2. Then, use the modified buffer for the next step.
torch.pow(10.0, signal, out=self.output_buffer)
```
  

BEST (Chained in-place):
```Python
    # The `out=` argument returns a reference to the output buffer,
    # allowing you to chain an in-place method call (`mul_`).
    torch.log10(signal, out=self.output_buffer).mul_(10.0)
```
      
    (Note: Look for functions with a trailing underscore like mul_, add_, sub_ for in-place versions).

Rule 4: Minimize Time Spent Under Lock

The threading.Lock is essential for safe communication with the UI thread, but it can block the real-time thread. Keep the code inside the with self._lock: block as short as possible.

    BAD (DSP logic is locked):
```Python
with self._lock:
    current_intensity = self._intensity
    # This complex DSP is unnecessarily blocking the UI from setting a new intensity.
    processed_signal = some_very_slow_torch_function(signal, current_intensity)
```  

GOOD (Only state copy is locked):
```Python        
    with self._lock:
        # Quickly copy the shared parameter to a local variable.
        current_intensity = self._intensity

    # Perform all heavy calculations AFTER the lock is released.
    processed_signal = some_very_slow_torch_function(signal, current_intensity)
```

