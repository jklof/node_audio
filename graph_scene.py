import logging
import threading
from typing import Type

from PySide6.QtWidgets import QGraphicsScene, QMenu, QGraphicsItem, QGraphicsTextItem
from PySide6.QtCore import Signal, Slot, QPointF
from PySide6.QtGui import QTransform, QAction, Qt

from node_system import NodeGraph, Node
from ui_elements import NodeItem, ConnectionItem, SocketItem
from plugin_loader import registry

logger = logging.getLogger(__name__)


class NodeGraphScene(QGraphicsScene):
    """
    Manages the visual representation of the node graph. It requests logical
    changes from the controller but does not modify the graph logic directly.
    """

    nodeCreationRequested = Signal(type, str, tuple)  # class, name, (x, y)
    nodeRenameRequested = Signal(str, str)  # node_id, new_name
    itemDeletionRequested = Signal(str)  # For nodes, handled by view
    clockSourceSetRequested = Signal(str)  # node_id

    def __init__(self, graph_logic: NodeGraph, parent=None):
        super().__init__(parent)
        self.graph_logic = graph_logic
        self.node_items: dict[str, NodeItem] = {}
        self.connection_items: dict[str, ConnectionItem] = {}
        self.current_clock_id: str | None = None
        self._show_processing_load = False

    def clear_scene(self):
        """Removes all UI items from the scene, ensuring signals are disconnected first."""
        for conn_item in self.connection_items.values():
            conn_item.remove()

        for conn_item in list(self.connection_items.values()):
            self.removeItem(conn_item)
        for node_item in list(self.node_items.values()):
            self.removeItem(node_item)

        self.node_items.clear()
        self.connection_items.clear()

    @Slot(dict)
    def sync_from_graph(self, graph_snapshot: dict):
        """
        Performs a non-destructive sync of the UI from a NodeGraph model snapshot.
        It reconciles items in the scene with the logical graph state,
        adding, removing, or updating items as needed, which prevents flickering.
        This slot receives the snapshot from the engine's signal.
        """
        logger.info("Syncing UI from logical graph snapshot (non-destructive)...")

        nodes_snapshot = graph_snapshot.get("nodes", {})
        connections_snapshot = graph_snapshot.get("connections", {})
        self.current_clock_id = graph_snapshot.get("selected_clock_node_id")

        # --- RECONCILE NODES ---

        # 2. Identify and remove deleted nodes from the UI
        snapshot_node_ids = set(nodes_snapshot.keys())
        current_node_ids = set(self.node_items.keys())
        ids_to_remove = current_node_ids - snapshot_node_ids

        for node_id in ids_to_remove:
            node_item = self.node_items.pop(node_id)
            self.removeItem(node_item)
            logger.debug(f"UI Sync: Removed node item {node_id[:4]}")

        # 3. Add new nodes and update existing ones
        for node_id, node_logic in nodes_snapshot.items():
            if node_id not in self.node_items:
                # This is a new node, create its UI item
                NodeItemClass = registry.get_node_class(node_logic.NODE_TYPE).UI_CLASS or NodeItem
                node_item = NodeItemClass(node_logic)
                node_item.set_processing_bar_visible(self._show_processing_load)
                self.addItem(node_item)
                self.node_items[node_id] = node_item
                node_item.updateFromLogic()
                logger.debug(f"UI Sync: Added new node item {node_id[:4]}")
            else:
                # Node already exists, update its state
                node_item = self.node_items[node_id]

                # --- RECONCILE SOCKETS ---
                current_ui_sockets = set(node_item._socket_items.keys())
                desired_logic_sockets = set(node_logic.inputs.values()) | set(node_logic.outputs.values())

                sockets_to_remove = current_ui_sockets - desired_logic_sockets
                sockets_to_add = desired_logic_sockets - current_ui_sockets

                if sockets_to_remove or sockets_to_add:
                    logger.debug(f"UI Sync: Reconciling sockets for node {node_id[:4]}")
                    for logic_socket in sockets_to_remove:
                        socket_item_to_remove = node_item._socket_items.pop(logic_socket, None)
                        label_item_to_remove = node_item._socket_labels.pop(logic_socket, None)
                        if socket_item_to_remove and socket_item_to_remove.scene():
                            self.removeItem(socket_item_to_remove)
                        if label_item_to_remove and label_item_to_remove.scene():
                            self.removeItem(label_item_to_remove)

                    for logic_socket in sockets_to_add:
                        node_item._socket_items[logic_socket] = SocketItem(logic_socket, node_item)
                        label = QGraphicsTextItem(logic_socket.name, node_item)
                        label.setDefaultTextColor(Qt.GlobalColor.lightGray)
                        node_item._socket_labels[logic_socket] = label

                # Update position if it has changed
                if node_item.pos() != QPointF(*node_logic.pos):
                    node_item.setPos(QPointF(*node_logic.pos))

                # Directly tell the UI item to sync its state from its logic object.
                node_item.updateFromLogic()

        # --- RECONCILE CONNECTIONS ---

        # 4. Identify and remove deleted connections from the UI
        snapshot_conn_ids = set(connections_snapshot.keys())
        current_conn_ids = set(self.connection_items.keys())
        conn_ids_to_remove = current_conn_ids - snapshot_conn_ids

        for conn_id in conn_ids_to_remove:
            conn_item = self.connection_items.pop(conn_id)
            conn_item.remove()  # Disconnect signals
            self.removeItem(conn_item)
            logger.debug(f"UI Sync: Removed connection item {conn_id[:4]}")

        # 5. Add new connections
        for conn_id, conn_logic in connections_snapshot.items():
            if conn_id not in self.connection_items:
                # This is a new connection, create its UI item
                start_node_item = self.node_items.get(conn_logic.start_socket.node.id)
                end_node_item = self.node_items.get(conn_logic.end_socket.node.id)
                if start_node_item and end_node_item:
                    start_socket_item = start_node_item.get_socket_item(conn_logic.start_socket)
                    end_socket_item = end_node_item.get_socket_item(conn_logic.end_socket)
                    if start_socket_item and end_socket_item:
                        conn_item = ConnectionItem(conn_logic, start_socket_item, end_socket_item)
                        self.addItem(conn_item)
                        self.connection_items[conn_id] = conn_item
                        logger.debug(f"UI Sync: Added new connection item {conn_id[:4]}")

        logger.info("UI sync complete.")

    @Slot(dict)
    def on_node_stats_updated(self, stats: dict):
        """Receives processing stats from the engine and updates relevant node items."""
        # An empty dictionary signals that processing has stopped, so clear all bars.
        if not stats:
            for node_item in self.node_items.values():
                node_item.set_processing_percentage(0.0)
            return

        # Update each node with its new percentage
        for node_id, percentage in stats.items():
            if node_id in self.node_items:
                self.node_items[node_id].set_processing_percentage(percentage)

    @Slot(bool)
    def set_processing_load_visible(self, visible: bool):
        """Sets the visibility of the processing load bar on all nodes."""
        self._show_processing_load = visible
        for node_item in self.node_items.values():
            node_item.set_processing_bar_visible(visible)

    def contextMenuEvent(self, event):
        """Shows the node creation menu on right-click on the background."""
        item = self.itemAt(event.scenePos(), QTransform())
        if item is None:
            menu = QMenu()
            for category in registry.get_categories():
                category_menu = menu.addMenu(category)
                for node_type in registry.get_node_types_in_category(category):
                    action = category_menu.addAction(node_type)
                    action.setData(registry.get_node_class(node_type))

            selected_action = menu.exec(event.screenPos())
            if selected_action and selected_action.data():
                node_class = selected_action.data()
                pos = (event.scenePos().x(), event.scenePos().y())
                self.nodeCreationRequested.emit(node_class, node_class.NODE_TYPE, pos)
        else:
            # Let the item handle its own context menu
            super().contextMenuEvent(event)

    @Slot(str, str)
    def on_node_error(self, node_id: str, error_message: str):
        """Slot to handle a single node entering an error state."""
        logger.warning(f"UI received error for node {node_id[:4]}: {error_message}")
        if node_id in self.node_items:
            node_item = self.node_items[node_id]
            # Explicitly tell the UI item to update its state
            node_item.set_error_display_state(error_message)
