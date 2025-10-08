import ctypes
import sys
import os
import torch
from node_system import Node
import logging

logger = logging.getLogger(__name__)


class FFINodeBase(Node):
    """
    An abstract base class for nodes that wrap a compiled C-style shared library.

    Its sole responsibility is to find and load the C++ library, and use the
    declarative `API` dictionary to bind the C functions (setting their
    argument and return types).

    The subclass is given the fully configured `self.lib` object and is
    responsible for implementing its own `process` method to call the
    appropriate C functions.
    """

    # --- Subclasses MUST override these ---
    LIB_NAME: str | None = None
    API: dict = {}

    def __init__(self, name: str, node_id: str | None = None):
        super().__init__(name, node_id)
        self.lib = None
        self.dsp_handle = None

        try:
            self._load_and_bind_library()
        except (FileNotFoundError, AttributeError, RuntimeError) as e:
            self.error_state = str(e)
            logger.error(f"[{self.name}] Failed to initialize FFI library: {e}")

    def _load_and_bind_library(self):
        if not self.LIB_NAME or not self.API:
            raise RuntimeError("Subclass must define LIB_NAME and API.")

        if sys.platform == "win32":
            lib_filename = f"{self.LIB_NAME}.dll"
        elif sys.platform == "darwin":
            lib_filename = f"lib{self.LIB_NAME}.dylib"
        else:
            lib_filename = f"lib{self.LIB_NAME}.so"

        subclass_file = sys.modules[self.__class__.__module__].__file__
        plugin_dir = os.path.dirname(os.path.abspath(subclass_file))
        lib_path = os.path.join(plugin_dir, lib_filename)

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Compiled library not found: {lib_path}")

        self.lib = ctypes.CDLL(lib_path)

        # create and remove are required for lifecycle management
        API = dict(self.API)
        API.update(
            {
                "create_handle": {"restype": ctypes.c_void_p},
                "delete_handle": {"argtypes": [ctypes.c_void_p]},
            }
        )

        for func_name, signature in API.items():
            try:
                c_func = getattr(self.lib, func_name)
                c_func.restype = signature.get("restype")
                c_func.argtypes = signature.get("argtypes", [])
            except AttributeError:
                raise RuntimeError(f"Function '{func_name}' not found in library '{self.LIB_NAME}'.")

        # create the handle
        self.dsp_handle = self.lib.create_handle()
        if not self.dsp_handle:
            raise RuntimeError(f"Failed to create DSP handle in library '{self.LIB_NAME}'.")

    def remove(self):
        """Ensures the C++ object is destroyed when the node is removed."""
        if self.lib and self.dsp_handle:
            self.lib.delete_handle(self.dsp_handle)
            self.dsp_handle = None
        super().remove()

    def process(self, input_data: dict) -> dict:
        """
        Subclasses must implement this method to call their specific
        C++ processing functions.
        """
        raise NotImplementedError("FFINodeBase subclass must implement its own process method.")
