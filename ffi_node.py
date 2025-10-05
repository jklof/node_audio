import ctypes
import sys
import os
from node_system import Node

class FFINodeBase(Node):
    """
    An abstract base class for nodes that wrap a compiled C-style shared library.
    """
    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.lib = None
        self.dsp_handle = None

    def _load_library(self, lib_name: str, subclass_file_path: str):
        # Determine the correct library file extension for the OS
        if sys.platform == "win32":
            lib_filename = f"{lib_name}.dll"
        elif sys.platform == "darwin":
            lib_filename = f"lib{lib_name}.dylib"
        else:
            lib_filename = f"lib{lib_name}.so"

        # Look for the library in the same directory as the node's Python file
        # This assumes the plugin structure keeps the .py and .dll/.so together
        #lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), lib_filename)

        plugin_dir = os.path.dirname(os.path.abspath(subclass_file_path))
        lib_path = os.path.join(plugin_dir, lib_filename)


        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Compiled library not found: {lib_path}")

        self.lib = ctypes.CDLL(lib_path)

        # --- Define function signatures for type safety ---
        self.lib.create_processor.restype = ctypes.c_void_p

        self.lib.destroy_processor.argtypes = [ctypes.c_void_p]

        # Get a handle to our C++ object instance
        self.dsp_handle = self.lib.create_processor()
        if not self.dsp_handle:
            raise RuntimeError("Failed to create DSP processor instance in compiled library.")

    def remove(self):
        """Ensures the C++ object is destroyed when the node is removed."""
        if self.lib and self.dsp_handle:
            self.lib.destroy_processor(self.dsp_handle)
            self.dsp_handle = None
        super().remove()

    def process(self, input_data: dict) -> dict:
        # This method must be implemented by the subclass
        raise NotImplementedError("Subclass must implement the process method.")