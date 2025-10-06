import uuid
import logging
from collections import OrderedDict
import torch
import abc
import threading

logger = logging.getLogger(__name__)


# Define the interface for any object that can drive the processing clock.
class IClockProvider(abc.ABC):
    """
    An interface for nodes that can act as a processing clock source.
    Implementing this interface allows the Engine to discover and use the
    node to drive the processing loop. A node can have its own resources
    (like an audio stream) running even when it is not the active clock.
    """

    @abc.abstractmethod
    def start_clock(self, tick_callback: callable):
        """
        The Engine calls this to promote the node to the active clock provider.
        The implementation should start invoking the provided 'tick_callback'
        at its desired rate.
        """
        pass

    @abc.abstractmethod
    def stop_clock(self):
        """
        The Engine calls this to demote the node from its active clock role.
        The implementation should cease all calls to the tick_callback. It
        should NOT necessarily stop its own underlying resources (e.g., audio stream),
        as it may continue to function as a passive output.
        """
        pass


class Socket:
    """Represents a connection point on a node."""

    def __init__(self, name: str, node: "Node", is_input: bool, data_type: type = torch.Tensor):
        self.name = name
        self.node = node
        self.is_input = is_input
        self.data_type = data_type
        self.connections: list["Connection"] = []
        self._data: any = None  # Cached data for the current tick

    def __repr__(self):
        direction = "Input" if self.is_input else "Output"
        return f"<Socket {self.node.name}.{self.name} ({direction})>"


class Connection:
    """Represents a connection between two sockets."""

    def __init__(self, start_socket: Socket, end_socket: Socket):
        if not start_socket or start_socket.is_input:
            raise ValueError("Connection must start from a valid Output socket.")
        if not end_socket or not end_socket.is_input:
            raise ValueError("Connection must end at a valid Input socket.")

        self.start_socket = start_socket
        self.end_socket = end_socket
        self.id = str(uuid.uuid4())

    def to_dict(self):
        return {
            "start_node_id": self.start_socket.node.id,
            "start_socket_name": self.start_socket.name,
            "end_node_id": self.end_socket.node.id,
            "end_socket_name": self.end_socket.name,
            "id": self.id,
        }


class Node:
    """Base class for all processing nodes."""

    UI_CLASS = None  # The class used for the node's UI representation
    NODE_TYPE = None  # Nodes name, e.g. "Oscillator", "Filter"
    CATEGORY = None  # E.g. "Sources", "Effects", "Synthesis"
    DESCRIPTION = ""

    def __init__(self, name: str, node_id: str | None = None):
        self.name = name
        self.id = node_id or str(uuid.uuid4())
        self.inputs = OrderedDict()
        self.outputs = OrderedDict()
        self.pos = (0.0, 0.0)
        self.error_state: str | None = None  # Attribute to hold error messages
        self.ui_update_callback = lambda state_dict: None  # No-op callback by default
        self._lock = threading.Lock()

    def clear_error_state(self):
        """Resets the node's error state."""
        self.error_state = None

    def add_input(self, name: str, data_type: type = torch.Tensor) -> Socket:
        socket = Socket(name, self, True, data_type)
        self.inputs[name] = socket
        return socket

    def add_output(self, name: str, data_type: type = torch.Tensor) -> Socket:
        socket = Socket(name, self, False, data_type)
        self.outputs[name] = socket
        return socket

    def process(self, input_data: dict) -> dict:
        raise NotImplementedError

    def start(self):
        """
        Called when graph processing starts.
        Nodes should initialize or reset their processing state here.
        For nodes managing external resources (like audio streams),
        this method should start them in a passive state.
        """
        pass

    def stop(self):
        """
        Called when graph processing stops.
        Nodes should clean up any resources they manage.
        """
        pass

    def remove(self):
        self.stop()

    def to_dict(self):
        node_data = {
            "id": self.id,
            "name": self.name,
            "type": self.NODE_TYPE,
            "pos": list(self.pos),
        }
        extra_data = self.serialize_extra()
        if extra_data:
            node_data.update(extra_data)
        return node_data

    def serialize_extra(self) -> dict:
        return {}

    def deserialize_extra(self, data: dict):
        pass

    def __repr__(self):
        return f"<Node {self.name} ({self.NODE_TYPE} - {self.id[:4]})>"

    def _get_state_snapshot_locked(self) -> dict:
        """
        [Override in subclass]
        Return a dictionary of the node's state to be saved or sent to the UI.
        This method is called by the base class while the lock is held.
        """
        return {}

    def get_current_state_snapshot(self) -> dict:
        """
        Public, thread-safe method to get the current state.
        This base implementation handles the locking. Subclasses should not override this.
        """
        with self._lock:
            return self._get_state_snapshot_locked()


class NodeGraph:
    """
    A pure data container for the graph's state.
    """

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.connections: dict[str, Connection] = {}
        self.selected_clock_node_id: str | None = None

    def clear(self):
        self.nodes.clear()
        self.connections.clear()
        self.selected_clock_node_id = None
