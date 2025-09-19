import os
import importlib.util
import importlib
import inspect
import logging
import sys
import gc
from types import ModuleType
from node_system import Node

logger = logging.getLogger(__name__)

# This module-level list will track all modules loaded by our scanner.
_loaded_plugin_modules = []


class NodePluginRegistry:
    """A registry for discovering and storing node plugin classes."""

    def __init__(self):
        self.node_types: dict[str, type[Node]] = {}
        self.node_categories: dict[str, list[str]] = {}

    def clear(self):
        """Clears all registered node types and categories."""
        self.node_types.clear()
        self.node_categories.clear()
        logger.info("Node plugin registry cleared.")

    def register_node_type(self, node_class: type[Node]):
        """Registers a node class if it meets the criteria."""
        if not inspect.isclass(node_class) or not issubclass(node_class, Node) or node_class is Node:
            return

        node_type = getattr(node_class, "NODE_TYPE", None)
        category = getattr(node_class, "CATEGORY", None)

        if not node_type or not category:
            return

        self.node_types[node_type] = node_class
        if category not in self.node_categories:
            self.node_categories[category] = []
        self.node_categories[category].append(node_type)
        logger.info(f"Registered node type: '{node_type}' in category '{category}'")

    def get_node_class(self, node_type_name: str) -> type[Node] | None:
        return self.node_types.get(node_type_name)

    def get_categories(self) -> list[str]:
        return sorted(list(self.node_categories.keys()))

    def get_node_types_in_category(self, category: str) -> list[str]:
        return sorted(self.node_categories.get(category, []))


# Global registry instance
registry = NodePluginRegistry()


def _ensure_package(package_name: str, package_path: str):
    """
    Ensures a package exists in sys.modules, creating a dummy one if needed.
    This is crucial for importlib.reload() to work on submodules.
    """
    if package_name not in sys.modules:
        # Create a dummy module to represent the package
        module = ModuleType(package_name)
        module.__path__ = [package_path]  # The path is important for a package
        sys.modules[package_name] = module
        logger.info(f"Created virtual package '{package_name}' in sys.modules.")


def scan_and_load_plugins(plugin_dirs: list[str], clear_registry: bool = True) -> list[str]:
    """
    Scans specified directories for modules, loads/reloads them, and registers their Node classes.

    Args:
        plugin_dirs: A list of directory names to scan for plugins.
        clear_registry: If True, the existing registry is cleared before loading.
                        This should be True for startup and reloads.

    Returns:
        A list of the module names that were successfully loaded or reloaded.
    """
    global _loaded_plugin_modules
    logger.info(f"Scanning plugins from {plugin_dirs} (Clear Registry: {clear_registry})...")

    if clear_registry:
        registry.clear()
        _loaded_plugin_modules.clear()

    for plugins_dir in plugin_dirs:
        plugins_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), plugins_dir)

        if not os.path.isdir(plugins_path):
            logger.warning(f"Plugin directory not found: {plugins_path}")
            continue

        _ensure_package(plugins_dir, plugins_path)

        for filename in os.listdir(plugins_path):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_path = os.path.join(plugins_path, filename)
                module_name = f"{plugins_dir}.{filename[:-3]}"

                try:
                    # Check if module exists; reload it or load it for the first time.
                    if module_name in sys.modules:
                        module = importlib.reload(sys.modules[module_name])
                        logger.debug(f"Reloaded module: {module_name}")
                    else:
                        spec = importlib.util.spec_from_file_location(module_name, module_path)
                        if not spec or not spec.loader:
                            logger.warning(f"Could not create module spec for {filename}")
                            continue
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        logger.info(f"Loaded new plugin: {module_name}")

                    # Inspect and register node classes from the module.
                    for _, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, Node) and obj is not Node:
                            registry.register_node_type(obj)

                    if module_name not in _loaded_plugin_modules:
                        _loaded_plugin_modules.append(module_name)

                except Exception as e:
                    logger.error(f"Error processing plugin {filename}: {e}", exc_info=True)

    logger.info(f"Plugin scan complete. Total modules processed: {len(_loaded_plugin_modules)}")
    return _loaded_plugin_modules


def finalize_plugins():
    """
    Should be called at interpreter exit via atexit. Clears the plugin registry and unloads
    modules to allow C++ extensions to clean up their static data gracefully.
    """
    logger.info("Finalizing plugin system...")

    # Clear the registry to break reference cycles.
    registry.clear()

    # Unload only the top-level plugin modules we loaded.
    for module_name in _loaded_plugin_modules:
        if module_name in sys.modules:
            try:
                del sys.modules[module_name]
            except Exception as e:
                logger.warning(f"Finalizer: Error unloading module {module_name}: {e}")

    # Force garbage collection to run the cleanup/deallocation code
    gc.collect()
    logger.info("Plugin system finalization complete.")
