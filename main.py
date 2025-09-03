import sys
import logging
import os
import argparse
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QToolBar, QWidget, QSizePolicy, QLabel
from PySide6.QtGui import QAction
from PySide6.QtCore import Slot, QTimer, Qt, QSettings

from plugin_loader import load_plugins
from engine import Engine
from app_controller import AppController
from graph_view import NodeGraphWidget
from ui_icons import create_icon_from_svg, ICONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, clean_start: bool = False):
        super().__init__()
        self.setWindowTitle("Re Node Processor")

        if clean_start:
            logger.info("Clean startup requested. Clearing all saved settings.")
            QSettings("ReNode", "ReNodeProcessor").clear()

        load_plugins("plugins")
        #load_plugins("additional_plugins")

        self.engine = Engine()
        self.graph_widget = NodeGraphWidget(self.engine.graph, self)
        self.setCentralWidget(self.graph_widget)
        self.controller = AppController(self.engine, self.graph_widget, self)

        self._create_menus()
        self._create_toolbar()
        self.statusBar().showMessage("Ready")

        self.engine.signals.processingStateChanged.connect(self.update_ui_for_processing_state)
        self.engine.signals.graphChanged.connect(self.update_clock_display)

        self.update_ui_for_processing_state(False)
        self.update_clock_display(self.engine._create_graph_snapshot_locked())

        self._load_settings()

    def _load_settings(self):
        """
        Loads window geometry and validates it's on-screen.
        """
        settings = QSettings("ReNode", "ReNodeProcessor")

        # Restore window geometry from settings
        geometry_bytes = settings.value("geometry")
        if geometry_bytes:
            self.restoreGeometry(geometry_bytes)

            # Validate that the restored geometry is on-screen
            is_on_screen = False
            window_rect = self.geometry()
            screens = QApplication.screens()

            for screen in screens:
                if screen.geometry().intersects(window_rect):
                    is_on_screen = True
                    break

            if not is_on_screen:
                logger.warning("Restored window position is off-screen. Centering on primary display.")
                if QApplication.primaryScreen():
                    center_point = QApplication.primaryScreen().geometry().center()
                    self.move(center_point - self.rect().center())
        else:
            # Default geometry if none is saved, centered on the primary screen
            self.setGeometry(100, 100, 1200, 800)
            if QApplication.primaryScreen():
                center_point = QApplication.primaryScreen().geometry().center()
                self.move(center_point - self.rect().center())

    def _save_settings(self):
        """Saves window geometry and current graph file path."""
        settings = QSettings("ReNode", "ReNodeProcessor")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("lastGraphFile", self.controller.current_file_path)

    def _create_toolbar(self):
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        # File Operations
        new_action = QAction(
            create_icon_from_svg(ICONS["new_file"]), "New", self, triggered=self.controller.clear_graph
        )
        new_action.setShortcut("Ctrl+N")
        toolbar.addAction(new_action)

        load_action = QAction(
            create_icon_from_svg(ICONS["open_folder"]), "Open", self, triggered=self.controller.load_graph
        )
        load_action.setShortcut("Ctrl+O")
        toolbar.addAction(load_action)

        save_action = QAction(create_icon_from_svg(ICONS["save"]), "Save", self, triggered=self.controller.save_graph)
        save_action.setShortcut("Ctrl+S")
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        # Processing Control (reusing actions from menu)
        self.start_action.setIcon(create_icon_from_svg(ICONS["play"]))
        self.stop_action.setIcon(create_icon_from_svg(ICONS["stop"]))
        toolbar.addAction(self.start_action)
        toolbar.addAction(self.stop_action)

        toolbar.addSeparator()

        # View Control
        zoom_in_action = QAction(
            create_icon_from_svg(ICONS["zoom_in"]), "Zoom In", self, triggered=self.graph_widget.zoom_in
        )
        zoom_in_action.setShortcut("Ctrl+=")
        toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction(
            create_icon_from_svg(ICONS["zoom_out"]), "Zoom Out", self, triggered=self.graph_widget.zoom_out
        )
        zoom_out_action.setShortcut("Ctrl+-")
        toolbar.addAction(zoom_out_action)

        zoom_fit_action = QAction(
            create_icon_from_svg(ICONS["zoom_fit"]), "Fit View", self, triggered=self.graph_widget.zoom_to_fit
        )
        zoom_fit_action.setShortcut("Ctrl+0")
        toolbar.addAction(zoom_fit_action)

        # Spacer to push clock status to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        # Clock Source Display
        self.clock_status_label = QLabel("Clock: None ")
        toolbar.addWidget(self.clock_status_label)

    def _create_menus(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        process_menu = menu_bar.addMenu("&Process")
        view_menu = menu_bar.addMenu("&View")

        # File Menu Actions
        save_action = QAction("Save Graph", self, triggered=self.controller.save_graph)
        save_action.setShortcut("Ctrl+S")
        load_action = QAction("Load Graph", self, triggered=self.controller.load_graph)
        load_action.setShortcut("Ctrl+O")
        clear_action = QAction("Clear Graph", self, triggered=self.controller.clear_graph)
        clear_action.setShortcut("Ctrl+N")
        exit_action = QAction("Exit", self, triggered=self.close)
        exit_action.setShortcut("Ctrl+Q")

        file_menu.addAction(save_action)
        file_menu.addAction(load_action)
        file_menu.addAction(clear_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # Process Menu Actions
        self.start_action = QAction("Start", self, triggered=self.controller.start_processing)
        self.start_action.setShortcut("F5")
        self.stop_action = QAction("Stop", self, triggered=self.controller.stop_processing)
        self.stop_action.setShortcut("F6")

        process_menu.addAction(self.start_action)
        process_menu.addAction(self.stop_action)

        # View Menu Actions
        self.show_load_action = QAction("Show Processing Load", self, checkable=True)
        self.show_load_action.setChecked(False)
        self.show_load_action.toggled.connect(self.on_toggle_processing_load_view)
        view_menu.addAction(self.show_load_action)

    @Slot(bool)
    def update_ui_for_processing_state(self, is_processing: bool):
        self.start_action.setEnabled(not is_processing)
        self.stop_action.setEnabled(is_processing)
        self.statusBar().showMessage("Processing..." if is_processing else "Stopped.")

    @Slot(dict)
    def update_clock_display(self, graph_snapshot: dict):
        clock_id = graph_snapshot.get("selected_clock_node_id")
        nodes = graph_snapshot.get("nodes", {})

        if clock_id and clock_id in nodes:
            node_name = nodes[clock_id].name
            self.clock_status_label.setText(f"Clock: {node_name} ")
        else:
            self.clock_status_label.setText("Clock: None ")

    @Slot(bool)
    def on_toggle_processing_load_view(self, checked: bool):
        self.graph_widget.graph_scene.set_processing_load_visible(checked)

    def closeEvent(self, event):
        self._save_settings()
        self.controller.cleanup_on_exit()
        event.accept()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re Node Processor - A real-time audio node graph application.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Start with a clean configuration, ignoring saved window geometry and last opened file.",
    )
    parser.add_argument(
        "--load",
        type=str,
        metavar="PATH_TO_GRAPH.json",
        help="Load a specific graph file on startup, overriding the last saved session.",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)

    # Pass the clean flag to the MainWindow. This must happen before it reads any settings.
    window = MainWindow(clean_start=args.clean)

    # --- Determine which graph to load on startup ---
    file_to_load = None
    startup_error = None

    if args.load:
        # The --load argument takes highest priority
        if os.path.exists(args.load):
            file_to_load = args.load
            logger.info(f"Command-line request: Loading graph from '{file_to_load}'")
        else:
            startup_error = f"The specified graph file could not be found:\n{args.load}"
            logger.error(startup_error)
    elif not args.clean:
        # If not a clean start and no specific file is given, try to load the last one.
        settings = QSettings("ReNode", "ReNodeProcessor")
        last_file = settings.value("lastGraphFile")
        if last_file and os.path.exists(last_file):
            file_to_load = last_file
            logger.info(f"Restoring last session graph from: '{file_to_load}'")

    # --- Load the determined graph (if any) ---
    if file_to_load:
        # Use a QTimer to ensure this runs after the main window is set up
        QTimer.singleShot(0, lambda: window.controller.load_graph_on_startup(file_to_load))

    window.show()

    # --- Show a startup error (if any) after the main window is visible ---
    if startup_error:
        QMessageBox.critical(window, "Startup Error", startup_error)

    sys.exit(app.exec())
