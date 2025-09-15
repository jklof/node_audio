import os
import importlib.util
import importlib
import inspect
import logging
import sys
from types import ModuleType
from node_system import Node

logger = logging.getLogger(__name__)


class NodePluginRegistry:
    """A registry for discovering and storing node plugin classes."""

    def __init__(self):
        self.node_types: dict[str, type[Node]] = {}
        self.node_categories: dict[str, list[str]] = {}

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


def load_plugins(plugins_dir: str = "plugins") -> list[str]:
    """
    Loads all valid Python modules from the specified plugins directory
    and returns a list of the loaded module names.
    """
    plugins_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), plugins_dir)
    loaded_modules = []

    if not os.path.isdir(plugins_path):
        logger.error(f"Plugins directory not found: {plugins_path}")
        return loaded_modules

    # Ensure the top-level directory is registered as a package in sys.modules.
    _ensure_package(plugins_dir, plugins_path)

    for filename in os.listdir(plugins_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_path = os.path.join(plugins_path, filename)
            module_name = f"{plugins_dir}.{filename[:-3]}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    for _, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, Node) and obj is not Node:
                            registry.register_node_type(obj)
                    loaded_modules.append(module_name)
                else:
                    logger.warning(f"Could not create module spec for {filename}")
            except Exception as e:
                logger.error(f"Error loading plugin {filename}: {e}", exc_info=True)
    return loaded_modules


def reload_plugin_modules(module_names: list[str]) -> dict[str, type[Node]]:
    """
    Reloads a list of modules, un-registers old classes, and registers new ones.
    Returns a map of node type names to their new class definitions for hotswapping.
    """
    new_class_map = {}
    original_node_types = list(registry.node_types.keys())

    registry.node_types.clear()
    registry.node_categories.clear()

    for module_name in module_names:
        try:
            if module_name in sys.modules:
                module_obj = importlib.reload(sys.modules[module_name])
                for _, obj in inspect.getmembers(module_obj, inspect.isclass):
                    if issubclass(obj, Node) and obj is not Node:
                        registry.register_node_type(obj)
                        if obj.NODE_TYPE in original_node_types:
                            new_class_map[obj.NODE_TYPE] = obj
            else:
                logger.warning(f"Module '{module_name}' not found in sys.modules, cannot reload.")
        except Exception as e:
            logger.error(f"Failed to reload module {module_name}: {e}", exc_info=True)

    logger.info(f"Plugin reload complete. Found {len(new_class_map)} classes to hotswap.")
    return new_class_map
