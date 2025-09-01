import os
import importlib.util
import inspect
import logging
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


def load_plugins(plugins_dir: str = "plugins"):
    """Loads all valid Python modules from the specified plugins directory."""
    plugins_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), plugins_dir)

    if not os.path.isdir(plugins_path):
        logger.error(f"Plugins directory not found: {plugins_path}")
        return

    for filename in os.listdir(plugins_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_path = os.path.join(plugins_path, filename)
            module_name = f"{plugins_dir}.{filename[:-3]}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    for _, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, Node) and obj is not Node:
                            registry.register_node_type(obj)
                else:
                    logger.warning(f"Could not create module spec for {filename}")
            except Exception as e:
                logger.error(f"Error loading plugin {filename}: {e}", exc_info=True)
