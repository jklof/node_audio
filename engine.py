import threading
import traceback
import logging
import json
import time
import gc
from collections import deque
from typing import Type, Any
from enum import Enum, auto

from PySide6.QtCore import QObject, Signal

from node_system import NodeGraph, Node, Socket, Connection, IClockProvider
from plugin_loader import registry, scan_and_load_plugins
from constants import TICK_DURATION_NS

logger = logging.getLogger(__name__)

# Number of ticks to pre-process to fill the audio buffer before playback starts.
INITIAL_PRIME_TICKS = 2


class EngineState(Enum):
    """Defines the possible operational states of the Engine."""

    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()


class EngineSignals(QObject):
    graphChanged = Signal(dict)
    processingStateChanged = Signal(EngineState)
    processingError = Signal(str)
    nodeProcessingStatsUpdated = Signal(dict)
    nodeErrorOccurred = Signal(str, str)  # (node_id, error_message)


class Engine:
    def __init__(self):
        self.graph = NodeGraph()
        self.signals = EngineSignals()
        self._lock = threading.Lock()

        self._cached_plan: list[tuple[Node, dict[str, Socket], dict[str, Any]]] | None = None

        # -- a thread-safe command queue ---
        self._command_queue = deque()

        self._state = EngineState.STOPPED
        self._stop_event = threading.Event()
        self._tick_semaphore = threading.Semaphore(0)
        self._processing_thread: threading.Thread | None = None

        # Start the initial processing thread
        self._start_processing_thread()

    def _start_processing_thread(self):
        """Creates and starts a new processing thread. Safe to call only when thread is confirmed stopped."""
        if self._processing_thread is not None and self._processing_thread.is_alive():
            logger.warning("Engine: Attempted to start a thread when one was already running.")
            return
        self._stop_event.clear()
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        logger.info("Engine: Processing thread started.")

    def _synchronous_stop_processing_thread(self):
        """Signals the processing thread to stop and waits (joins) until it has fully terminated."""
        if self._processing_thread is None or not self._processing_thread.is_alive():
            return

        logger.info("Engine: Beginning synchronous stop of processing thread...")
        self._stop_event.set()
        # Release the semaphore one last time to unblock the thread if it's waiting on acquire()
        self._tick_semaphore.release()

        # Wait for the thread to finish its loop and exit. This is the critical synchronous step.
        self._processing_thread.join(timeout=2.0)
        if self._processing_thread.is_alive():
            logger.error("Engine: Processing thread failed to stop gracefully within the timeout!")

        self._processing_thread = None
        logger.info("Engine: Processing thread has been successfully stopped and joined.")

    def tick(self):
        """
        Called by the active IClockProvider from a high-priority thread (e.g., audio callback).
        This method must be lightweight and non-blocking.
        """
        # Only release the semaphore if the engine is actively running
        if self._state == EngineState.RUNNING:
            self._tick_semaphore.release()

    # --- Central command processor, called by the processing thread ---
    def _process_command_queue_locked(self):
        """
        Executes all pending commands from the UI thread. This is the ONLY
        place where the graph structure OR LIFECYCLE is mutated. Assumes lock is held.
        """
        if not self._command_queue:
            return

        # Invalidate plan cache if any command exists that might change the graph
        graph_mutating_commands = {
            "add_node",
            "remove_node",
            "add_connection",
            "remove_connection",
            "clear_graph",
            "load_graph",
        }
        if any(cmd[0] in graph_mutating_commands for cmd in self._command_queue):
            self._invalidate_plan_cache()

        while self._command_queue:
            command, payload = self._command_queue.popleft()

            # --- GRAPH MUTATION COMMANDS ---
            if command == "add_node":
                self._add_node_locked(payload["node_object"])
            elif command == "remove_node":
                self._remove_node_locked(payload["node_id"])
            elif command == "add_connection":
                self._add_connection_locked(payload["start_socket"], payload["end_socket"])
            elif command == "remove_connection":
                self._remove_connection_locked(payload["connection_id"])
            elif command == "rename_node":
                self._rename_node_locked(payload["node_id"], payload["new_name"])

            # --- LIFECYCLE COMMANDS ---
            elif command == "start_graph":
                self._start_graph_locked()
            elif command == "stop_graph":
                if self._state == EngineState.RUNNING or self._state == EngineState.STOPPING:
                    self._stop_graph_locked()
                    self.signals.processingStateChanged.emit(EngineState.STOPPED)
                    self.signals.nodeProcessingStatsUpdated.emit({})
                else:
                    logger.warning("Engine: Ignored 'stop_graph' command, not in a stoppable state.")
            elif command == "clear_graph":
                if self._state == EngineState.RUNNING:
                    self._stop_graph_locked()
                    self.signals.processingStateChanged.emit(EngineState.STOPPED)
                    self.signals.nodeProcessingStatsUpdated.emit({})
                self._clear_graph_locked_impl()
            elif command == "load_graph":
                self._load_graph_locked(payload["graph_data"])

            # --- New command to synchronize UI state after a file load ---
            elif command == "_sync_ui_after_load":
                logger.info("Engine: Executing post-load UI sync command.")
                for node in self.graph.nodes.values():
                    # This call is now safe because the graphChanged signal has been processed
                    # by the UI thread, meaning all NodeItems exist and callbacks are connected.
                    node.ui_update_callback(node.get_current_state_snapshot())

    def _processing_loop(self):
        next_update_time = 0.0
        processing_stats = {}
        while not self._stop_event.is_set():
            # --- Step 1: ALWAYS process command queue at the start of the cycle ---
            graph_was_changed = False
            with self._lock:
                if self._command_queue:
                    self._process_command_queue_locked()
                    graph_was_changed = True

            if graph_was_changed:
                self._emit_graph_changed()

            # --- Step 2: Check if the engine is in the RUNNING state ---
            with self._lock:
                is_running = self._state == EngineState.RUNNING

            if is_running:
                # --- State: RUNNING ---
                # This will block until tick() is called from the audio thread.
                acquired = self._tick_semaphore.acquire(timeout=0.1)
                if not acquired:
                    continue
                if self._stop_event.is_set():
                    break

                processing_plan = None
                error_to_emit = None
                with self._lock:
                    if self._state != EngineState.RUNNING:
                        continue
                    processing_plan = self._get_processing_plan()
                    if processing_plan is None:
                        error_to_emit = "Cycle detected in graph. Processing halted."
                        logger.error(error_to_emit)
                        self._stop_graph_locked()
                        self.signals.processingStateChanged.emit(EngineState.STOPPED)

                if error_to_emit:
                    self.signals.processingError.emit(error_to_emit)
                    continue

                try:
                    if processing_plan:
                        self._execute_unlocked_tick(processing_plan, processing_stats)
                        current_time = time.monotonic()
                        if current_time >= next_update_time:
                            next_update_time = current_time + 0.033
                            self.signals.nodeProcessingStatsUpdated.emit(dict(processing_stats))
                except Exception as e:
                    error_msg = f"Unhandled exception in unlocked processing tick: {e}"
                    logger.error(f"{error_msg}\n{traceback.format_exc()}")
                    with self._lock:
                        self._stop_graph_locked()
                    self.signals.processingError.emit(error_msg)
                    self.signals.processingStateChanged.emit(EngineState.STOPPED)
            else:
                # --- State: STOPPED, STARTING, or STOPPING ---
                # The engine is not running, so we don't process audio.
                # We sleep for a short duration to prevent this loop from
                # busy-waiting and consuming 100% CPU.
                time.sleep(0.01)  # 10ms sleep to yield the CPU

    def _execute_unlocked_tick(
        self,
        processing_plan: list[tuple[Node, dict[str, Socket], dict[str, Any]]],
        stats: dict[str, float],
    ):
        """
        Processes one tick of the graph using a pre-computed plan.
        Handles errors on a per-node basis.
        """
        stats.clear()
        for node, input_sources, input_data in processing_plan:
            if node.error_state is not None:
                stats[node.id] = 0.0  # Show zero processing load
                continue

            try:
                for name in node.inputs:
                    source_socket = input_sources.get(name)
                    input_data[name] = source_socket._data if source_socket else None
                start_time = time.perf_counter_ns()
                output_data = node.process(input_data)
                end_time = time.perf_counter_ns()
                processing_time_ns = end_time - start_time
                stats[node.id] = (processing_time_ns / TICK_DURATION_NS) * 100.0
                if output_data:
                    for name, data in output_data.items():
                        if name in node.outputs:
                            node.outputs[name]._data = data
            except Exception as e:
                error_msg = f"Error in '{node.name}': {e}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                node.error_state = error_msg
                for output_socket in node.outputs.values():
                    output_socket._data = None
                self.signals.nodeErrorOccurred.emit(node.id, error_msg)

    def _get_processing_plan(self) -> list[tuple[Node, dict[str, Socket], dict[str, Any]]] | None:
        """
        Gets the processing plan. If the cache is invalid, it computes a new one.
        Assumes the lock is held.
        """
        if self._cached_plan is None:
            self._cached_plan = self._compute_processing_plan()
        return self._cached_plan

    def _invalidate_plan_cache(self):
        """Invalidates the cached processing plan. Assumes lock is held."""
        self._cached_plan = None

    def _compute_processing_plan(self) -> list[tuple[Node, dict[str, Socket], dict[str, Any]]] | None:
        """
        Computes the topological sort and builds a detailed, thread-safe
        processing plan from it. Assumes lock is held.
        """
        in_degree = {node_id: 0 for node_id in self.graph.nodes}
        adj_list = {node_id: [] for node_id in self.graph.nodes}
        for conn in self.graph.connections.values():
            start_node_id = conn.start_socket.node.id
            end_node_id = conn.end_socket.node.id
            if start_node_id in adj_list and end_node_id not in adj_list[start_node_id]:
                adj_list[start_node_id].append(end_node_id)
                in_degree[end_node_id] += 1
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        sorted_node_ids = []
        while queue:
            node_id = queue.popleft()
            sorted_node_ids.append(node_id)
            for neighbor_id in adj_list.get(node_id, []):
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)
        if len(sorted_node_ids) != len(self.graph.nodes):
            return None  # Cycle detected
        plan = []
        for node_id in sorted_node_ids:
            node = self.graph.nodes[node_id]
            input_sources = {}
            for input_name, input_socket in node.inputs.items():
                if input_socket.connections:
                    input_sources[input_name] = input_socket.connections[0].start_socket
            node_input_dict = {name: None for name in node.inputs}
            plan.append((node, input_sources, node_input_dict))
        logger.info("Engine: Re-computed new processing plan.")
        return plan

    def start_processing(self):
        with self._lock:
            if self._state != EngineState.STOPPED:
                logger.warning(f"Engine: Cannot start. Current state is {self._state.name}.")
                return
            self._state = EngineState.STARTING
        self.signals.processingStateChanged.emit(EngineState.STARTING)
        self._queue_command("start_graph")
        logger.info("Engine: Queued command to start graph processing.")

    def _start_graph_locked(self):
        """Internal helper to start all nodes and the clock. Includes rollback on failure. Assumes lock is held."""
        logger.info("Engine: Internal start graph command executing...")
        self._invalidate_plan_cache()
        for node in self.graph.nodes.values():
            node.clear_error_state()
        started_nodes = []
        error_msg = None
        try:
            for node in self.graph.nodes.values():
                node.start()
                started_nodes.append(node)
            clock_node_id = self.graph.selected_clock_node_id
            clock_node = self.graph.nodes.get(clock_node_id)
            if clock_node and isinstance(clock_node, IClockProvider):
                clock_node.start_clock(self.tick)
                for _ in range(INITIAL_PRIME_TICKS):
                    self.tick()
            else:
                logger.warning("Engine: Starting processing without an active clock source.")
        except Exception as e:
            error_msg = f"Failed to start graph: {e}"
            logger.error(error_msg, exc_info=True)
            logger.info("Engine: Rolling back successfully started nodes...")
            for node_to_stop in reversed(started_nodes):
                try:
                    node_to_stop.stop()
                except Exception as stop_e:
                    logger.error(f"Error during rollback stopping node '{node_to_stop.name}': {stop_e}")
        if error_msg:
            self._state = EngineState.STOPPED
            self.signals.processingError.emit(error_msg)
            self.signals.processingStateChanged.emit(EngineState.STOPPED)
        else:
            self._state = EngineState.RUNNING
            self.signals.processingStateChanged.emit(EngineState.RUNNING)
            logger.info("Engine: Graph processing started successfully.")

    def _stop_graph_locked(self):
        """Internal helper to perform all actions to stop a running graph. Assumes lock is held."""
        logger.info("Engine: Internal stop graph command executing...")
        self._state = EngineState.STOPPING
        clock_node_id = self.graph.selected_clock_node_id
        if clock_node_id and clock_node_id in self.graph.nodes:
            clock_node = self.graph.nodes[clock_node_id]
            if isinstance(clock_node, IClockProvider):
                clock_node.stop_clock()
        for node in self.graph.nodes.values():
            try:
                node.stop()
            except Exception as e:
                logger.error(f"Error stopping node '{node.name}': {e}", exc_info=True)
        while self._tick_semaphore.acquire(blocking=False):
            pass
        self._state = EngineState.STOPPED
        logger.info("Engine: Graph processing stopped.")

    def stop_processing(self):
        with self._lock:
            if self._state != EngineState.RUNNING:
                logger.warning(f"Engine: Cannot stop. Current state is {self._state.name}.")
                return
            self._state = EngineState.STOPPING
        self.signals.processingStateChanged.emit(EngineState.STOPPING)
        self._queue_command("stop_graph")
        logger.info("Engine: Queued command to stop graph processing.")

    def _create_graph_snapshot_locked(self) -> dict:
        """Creates a copy of the graph state. Assumes lock is held."""
        return {
            "nodes": dict(self.graph.nodes),
            "connections": dict(self.graph.connections),
            "selected_clock_node_id": self.graph.selected_clock_node_id,
        }

    def _emit_graph_changed(self):
        """Creates a snapshot and emits the graphChanged signal. Thread-safe."""
        with self._lock:
            snapshot = self._create_graph_snapshot_locked()
        self.signals.graphChanged.emit(snapshot)

    def set_clock_source(self, node_id: str | None):
        was_running = self._state == EngineState.RUNNING
        if was_running:
            self.stop_processing()
        with self._lock:
            if self.graph.selected_clock_node_id == node_id:
                if was_running:
                    self.start_processing()
                return
            new_node = self.graph.nodes.get(node_id)
            if new_node and isinstance(new_node, IClockProvider):
                self.graph.selected_clock_node_id = node_id
                logger.info(f"Engine: Clock source set to '{new_node.name}'")
            else:
                if new_node:
                    logger.warning(f"Node '{new_node.name}' cannot be a clock source.")
                self.graph.selected_clock_node_id = None
                logger.info("Engine: Clock source cleared.")
        self._emit_graph_changed()
        if was_running:
            self.start_processing()

    def _queue_command(self, command_name: str, **payload):
        """Safely queues a command for the processing thread."""
        with self._lock:
            self._command_queue.append((command_name, payload))

    def add_node(self, node_class: Type[Node], name: str, pos: tuple[float, float]) -> Node:
        """Creates a node and queues it for addition to the graph."""
        new_node = node_class(name)
        new_node.pos = pos
        self._queue_command("add_node", node_object=new_node)
        return new_node

    def remove_node(self, node_id: str):
        """Queues a node for deferred deletion from the graph."""
        self._queue_command("remove_node", node_id=node_id)

    def rename_node(self, node_id: str, new_name: str):
        """Queues a node for a deferred rename operation."""
        if new_name:
            self._queue_command("rename_node", node_id=node_id, new_name=new_name)

    def add_connection(self, start_socket: Socket, end_socket: Socket):
        """Queues a connection for deferred addition to the graph."""
        start_type = start_socket.data_type if start_socket.data_type is not None else Any
        end_type = end_socket.data_type if end_socket.data_type is not None else Any
        is_compatible = start_type is Any or end_type is Any or start_type == end_type
        if not is_compatible:
            logger.error(f"Engine rejected incompatible connection: {start_socket} -> {end_socket}")
            return
        self._queue_command("add_connection", start_socket=start_socket, end_socket=end_socket)

    def remove_connection(self, connection_id: str):
        """Queues a connection for deferred deletion from the graph."""
        self._queue_command("remove_connection", connection_id=connection_id)

    def _add_node_locked(self, new_node: Node):
        """Adds a node to the graph. Assumes lock is held."""
        new_node.graph_invalidation_callback = self._emit_graph_changed
        self.graph.nodes[new_node.id] = new_node
        if isinstance(new_node, IClockProvider) and self.graph.selected_clock_node_id is None:
            self.graph.selected_clock_node_id = new_node.id
        if self._state == EngineState.RUNNING:
            new_node.start()

    def _remove_node_locked(self, node_id: str):
        """Removes a node from the graph. Assumes lock is held."""
        if node_id not in self.graph.nodes:
            return
        node_to_remove = self.graph.nodes[node_id]
        if self.graph.selected_clock_node_id == node_id:
            if self._state == EngineState.RUNNING and isinstance(node_to_remove, IClockProvider):
                node_to_remove.stop_clock()
            self.graph.selected_clock_node_id = None
        conns_to_remove = [
            conn
            for conn in self.graph.connections.values()
            if conn.start_socket.node == node_to_remove or conn.end_socket.node == node_to_remove
        ]
        for conn in conns_to_remove:
            self._remove_connection_locked(conn.id)
        node_to_remove.remove()
        del self.graph.nodes[node_id]

    def _rename_node_locked(self, node_id: str, new_name: str):
        """Renames a node in the graph. Assumes lock is held."""
        if node_id in self.graph.nodes and new_name:
            node = self.graph.nodes[node_id]
            logger.info(f"Engine: Renaming node '{node.name}' to '{new_name}'")
            node.name = new_name

    def _add_connection_locked(self, start_socket: Socket, end_socket: Socket):
        try:
            if end_socket.connections:
                self._remove_connection_locked(end_socket.connections[0].id)
            connection = Connection(start_socket, end_socket)
            self.graph.connections[connection.id] = connection
            start_socket.connections.append(connection)
            end_socket.connections.append(connection)
        except Exception as e:
            logger.error(f"Engine: Failed to create connection while holding lock: {e}", exc_info=True)

    def _remove_connection_locked(self, connection_id: str):
        """Removes a connection from the graph. Assumes lock is held."""
        if connection_id in self.graph.connections:
            conn = self.graph.connections[connection_id]
            if conn in conn.start_socket.connections:
                conn.start_socket.connections.remove(conn)
            if conn in conn.end_socket.connections:
                conn.end_socket.connections.remove(conn)
            del self.graph.connections[connection_id]

    def _clear_graph_locked_impl(self):
        """Actual implementation of clearing the graph."""
        for node in list(self.graph.nodes.values()):
            node.remove()
        self.graph.clear()

    def clear_graph(self):
        """Queues a command to clear the entire graph."""
        logger.info("Engine: Queued command to clear graph.")
        self._queue_command("clear_graph")

    def save_graph_to_file(self, file_path: str):
        with self._lock:
            graph_data = {
                "nodes": [node.to_dict() for node in self.graph.nodes.values()],
                "connections": [conn.to_dict() for conn in self.graph.connections.values()],
                "clock_source_id": self.graph.selected_clock_node_id,
            }
        with open(file_path, "w") as f:
            json.dump(graph_data, f, indent=4)

    def load_graph_from_file(self, file_path: str):
        """
        Reads a graph file, queues a load command, and then queues a UI sync command.
        """
        try:
            with open(file_path, "r") as f:
                graph_data = json.load(f)
            # Queue a single, atomic "load" command.
            self._queue_command("load_graph", graph_data=graph_data)
            # --- Queue a subsequent command to ensure UI is fully updated ---
            self._queue_command("_sync_ui_after_load")

        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read or parse graph file {file_path}: {e}")
            self.signals.processingError.emit(f"Could not load file: {e}")
            return

    def _load_graph_locked(self, graph_data: dict):
        """
        Internal helper to perform an atomic load operation. Assumes lock is held.
        This is called by the processing thread.
        """
        # 1. Ensure the graph is stopped before clearing and loading.
        if self._state == EngineState.RUNNING:
            self._stop_graph_locked()
            self.signals.processingStateChanged.emit(EngineState.STOPPED)
            self.signals.nodeProcessingStatsUpdated.emit({})

        # 2. Clear the current graph.
        self._clear_graph_locked_impl()

        # 3. Load the new graph from the data dictionary.
        try:
            for node_data in graph_data.get("nodes", []):
                node_type = node_data.get("type")
                node_class = registry.get_node_class(node_type)
                if not node_class:
                    logger.warning(f"Could not find node class for type '{node_type}'. Skipping.")
                    continue
                new_node = node_class(name=node_data["name"], node_id=node_data["id"])
                new_node.graph_invalidation_callback = self._emit_graph_changed
                new_node.pos = tuple(node_data.get("pos", (0, 0)))
                new_node.deserialize_extra(node_data)
                self.graph.nodes[new_node.id] = new_node
            for conn_data in graph_data.get("connections", []):
                start_node = self.graph.nodes.get(conn_data["start_node_id"])
                end_node = self.graph.nodes.get(conn_data["end_node_id"])
                if not start_node or not end_node:
                    logger.warning("Skipping connection due to missing node.")
                    continue
                start_socket = start_node.outputs.get(conn_data["start_socket_name"])
                end_socket = end_node.inputs.get(conn_data["end_socket_name"])
                if not start_socket or not end_socket:
                    logger.warning("Skipping connection due to missing socket.")
                    continue
                self._add_connection_locked(start_socket, end_socket)
            self.graph.selected_clock_node_id = graph_data.get("clock_source_id")
            logger.info("Engine: Graph loaded successfully from dictionary.")
        except Exception as e:
            error_msg = f"Failed during graph deserialization: {e}"
            logger.error(error_msg, exc_info=True)
            self.signals.processingError.emit(error_msg)
            # Ensure graph is clean after a failed load
            self._clear_graph_locked_impl()

    def reload_plugins_and_graph(self, plugin_dirs: list[str]):
        """Performs a full, transactional reload of plugins and the graph."""
        logger.info("Engine: Starting plugin reload transaction.")
        was_running = self._state == EngineState.RUNNING
        if was_running:
            self.stop_processing()
        self._synchronous_stop_processing_thread()
        with self._lock:
            graph_data = {
                "nodes": [node.to_dict() for node in self.graph.nodes.values()],
                "connections": [conn.to_dict() for conn in self.graph.connections.values()],
                "clock_source_id": self.graph.selected_clock_node_id,
            }
            self._clear_graph_locked_impl()
        self._emit_graph_changed()
        scan_and_load_plugins(plugin_dirs, clear_registry=True)
        try:
            # We use the internal load method here as it's part of a synchronous operation
            with self._lock:
                self._load_graph_locked(graph_data)
        except Exception as e:
            error_msg = f"Failed to reload graph from memory: {e}"
            logger.error(error_msg, exc_info=True)
            self.signals.processingError.emit(error_msg)
            self._start_processing_thread()
            return
        self._emit_graph_changed()
        self._start_processing_thread()
        if was_running:
            self.start_processing()
        logger.info("Engine: Plugin reload transaction complete.")

    def shutdown(self):
        """Performs a fully synchronous shutdown of the engine and all node resources."""
        logger.info("Engine: Shutting down...")
        if self._state == EngineState.RUNNING:
            self.stop_processing()
        self._synchronous_stop_processing_thread()
        logger.info("Engine: Processing thread stopped. Cleaning up nodes...")
        with self._lock:
            nodes_to_remove = list(self.graph.nodes.values())
        for node in nodes_to_remove:
            try:
                node.remove()
            except Exception as e:
                logger.error(f"Error during shutdown while removing node '{node.name}': {e}", exc_info=True)
        gc.collect()
        logger.info("Engine: Shutdown complete.")
