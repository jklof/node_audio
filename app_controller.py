import logging
import json
from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from engine import Engine, EngineState
from graph_view import NodeGraphWidget
from node_system import Socket

logger = logging.getLogger(__name__)


class AppController(QObject):
    """
    Mediator between the UI (View) and the Engine (Model/Processing).
    Connects UI actions to Engine commands and Engine signals to UI updates.
    """

    def __init__(self, engine: Engine, graph_widget: NodeGraphWidget, parent_window: QWidget):
        super().__init__(parent_window)
        self.engine = engine
        self.graph_widget = graph_widget
        self.parent_window = parent_window
        self.current_file_path = None

        # Connect Engine signals to UI update slots
        self.engine.signals.graphChanged.connect(self.graph_widget.graph_scene.sync_from_graph)
        self.engine.signals.processingError.connect(self._on_processing_error)
        self.engine.signals.nodeProcessingStatsUpdated.connect(self.graph_widget.graph_scene.on_node_stats_updated)

        # Connect View (UI) request signals to Controller slots
        self.graph_widget.graph_scene.nodeCreationRequested.connect(self.on_node_creation_requested)
        self.graph_widget.graph_scene.clockSourceSetRequested.connect(self.on_clock_source_set_requested)
        self.graph_widget.graph_scene.nodeRenameRequested.connect(self.on_node_rename_requested)
        self.graph_widget.connectionRequested.connect(self.on_connection_requested)
        self.graph_widget.nodeDeletionRequested.connect(self.on_node_deletion_requested)
        self.graph_widget.connectionDeletionRequested.connect(self.on_connection_deletion_requested)

    @Slot(list)
    def reload_all_plugins(self, module_names: list[str]):
        """
        Orchestrates the plugin reload process by delegating the entire
        transactional operation to the engine.
        """
        logger.info("Controller: Requesting engine to perform transactional plugin reload.")

        # Delegate the entire complex operation to the engine.
        # This call will block until the reload is complete.
        self.engine.reload_plugins_and_graph(module_names)


    @Slot(type, str, tuple)
    def on_node_creation_requested(self, node_class, name, pos):
        """Handles a request from the UI to create a new node."""
        logger.info(f"Controller: Received request to create node '{name}' of type {node_class.__name__}")
        self.engine.add_node(node_class, name, pos)

    @Slot(str, str)
    def on_node_rename_requested(self, node_id: str, new_name: str):
        """Handles a request from the UI to rename a node."""
        logger.info(f"Controller: Received request to rename node {node_id[:4]} to '{new_name}'")
        self.engine.rename_node(node_id, new_name)

    @Slot(str)
    def on_clock_source_set_requested(self, node_id: str):
        """Handles a request from the UI to change the clock source."""
        logger.info(f"Controller: Received request to set clock source to node {node_id[:4]}")
        self.engine.set_clock_source(node_id)

    @Slot(object, object)
    def on_connection_requested(self, start_socket: Socket, end_socket: Socket):
        """Handles a request from the UI to create a new connection."""
        logger.info(f"Controller: Received request to connect {start_socket} to {end_socket}")
        self.engine.add_connection(start_socket, end_socket)

    @Slot(str)
    def on_node_deletion_requested(self, node_id: str):
        """Handles a request from the UI to delete a node."""
        self.engine.remove_node(node_id)

    @Slot(str)
    def on_connection_deletion_requested(self, connection_id: str):
        """Handles a request from the UI to delete a connection."""
        self.engine.remove_connection(connection_id)

    @Slot(str)
    def _on_processing_error(self, error_msg: str):
        """Displays a critical error from the Engine and stops processing."""
        logger.error(f"Controller: Received processing error: {error_msg}")
        QMessageBox.critical(
            self.parent_window,
            "Processing Error",
            f"A critical error occurred:\n{error_msg}\nProcessing has been stopped.",
        )

    @Slot()
    def start_processing(self):
        self.engine.start_processing()

    @Slot()
    def stop_processing(self):
        self.engine.stop_processing()

    @Slot()
    def save_graph(self):
        """Opens a save file dialog and tells the engine to save the graph."""
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent_window, "Save Graph", self.current_file_path, "JSON Files (*.json)"
        )
        if file_path:
            try:
                self.engine.save_graph_to_file(file_path)
                self.current_file_path = file_path
                self.parent_window.statusBar().showMessage(f"Graph saved to {file_path}", 5000)
                logger.info(f"Graph saved to {file_path}")
            except Exception as e:
                logger.error(f"Failed to save graph: {e}", exc_info=True)
                QMessageBox.critical(self.parent_window, "Save Error", f"Could not save graph to file:\n{e}")

    def _load_graph_from_path(self, file_path: str, is_startup: bool = False):
        """Common logic for loading a graph from a file path."""
        try:
            self.engine.load_graph_from_file(file_path)
            self.current_file_path = file_path
            self.parent_window.statusBar().showMessage(f"Graph loaded from {file_path}", 5000)
            logger.info(f"Graph loaded from {file_path}")
        except Exception as e:
            title = "Startup Load Error" if is_startup else "Load Error"
            logger.error(f"Failed to load graph from {file_path}: {e}", exc_info=True)
            QMessageBox.critical(
                self.parent_window, title, f"Failed to load graph from file:\n{file_path}\n\nError: {e}"
            )
            self.current_file_path = None
            self.parent_window.statusBar().showMessage(f"Failed to load graph from {file_path}", 5000)

    @Slot()
    def load_graph(self):
        """Opens a load file dialog and tells the engine to load the graph."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent_window, "Load Graph", self.current_file_path, "JSON Files (*.json)"
        )
        if file_path:
            self._load_graph_from_path(file_path)

    @Slot(str)
    def load_graph_on_startup(self, file_path: str):
        """Loads a graph from a given path without a file dialog, for startup."""
        self._load_graph_from_path(file_path, is_startup=True)

    @Slot()
    def clear_graph(self):
        """Requests the engine to clear the entire graph."""
        logger.info("Controller: Received request to clear graph.")
        self.engine.clear_graph()
        self.current_file_path = None  # Also clear the current file path
        self.parent_window.statusBar().showMessage("Graph cleared.", 5000)

    def cleanup_on_exit(self):
        logger.info("Controller: Cleaning up on exit.")
        self.engine.shutdown()
