import threading
import traceback
import logging
import json
import time
from collections import deque
from typing import Type

from PySide6.QtCore import QObject, Signal

from node_system import NodeGraph, Node, Socket, Connection, IClockProvider
from plugin_loader import registry
from constants import TICK_DURATION_NS

logger = logging.getLogger(__name__)

# Number of ticks to pre-process to fill the audio buffer before playback starts.
INITIAL_PRIME_TICKS = 2

class EngineSignals(QObject):
    graphChanged = Signal(dict)
    processingStateChanged = Signal(bool)
    processingError = Signal(str)
    nodeProcessingStatsUpdated = Signal(dict)

class Engine:
    def __init__(self):
        self.graph = NodeGraph()
        self.signals = EngineSignals()
        self._lock = threading.Lock() # This is a non-reentrant lock

        self._cached_plan: list[tuple[Node, dict[str, Socket]]] | None = None

        self._tick_semaphore = threading.Semaphore(0)
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._stop_event = threading.Event()
        self._is_processing_flag = threading.Event()
        self._processing_thread.start()


    def tick(self):
        if self._is_processing_flag.is_set():
            self._tick_semaphore.release()

    def _processing_loop(self):
        next_update_time = 0.0
        while not self._stop_event.is_set():
            acquired = self._tick_semaphore.acquire(timeout=0.1)
            if not acquired:
                continue

            if self._stop_event.is_set():
                break
            
            processing_plan = None
            error_to_emit = None
            
            with self._lock:
                if not self._is_processing_flag.is_set():
                    continue
                
                processing_plan = self._get_processing_plan()
                
                if processing_plan is None:
                    error_to_emit = "Cycle detected in graph. Processing halted."
                    logger.error(error_to_emit)
                    self._stop_processing_locked()
            
            # emit the signals *after* the lock has been released.
            if error_to_emit:
                self.signals.processingError.emit(error_to_emit)
                self.signals.processingStateChanged.emit(False)
                continue

            try:
                # This check ensures we don't try to execute a None plan if an error occurred.
                if processing_plan:
                    stats = self._execute_unlocked_tick(processing_plan)

                    current_time = time.monotonic()
                    if current_time >= next_update_time:
                        next_update_time = current_time + 0.033 # ~30 FPS
                        self.signals.nodeProcessingStatsUpdated.emit(stats)

            except Exception as e:
                error_msg = f"Unhandled exception in unlocked processing tick: {e}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                with self._lock:
                    self._stop_processing_locked()
                self.signals.processingError.emit(error_msg)
                self.signals.processingStateChanged.emit(False)

    def _execute_unlocked_tick(self, processing_plan: list[tuple[Node, dict[str, Socket]]]):
        """Processes one tick of the graph using a pre-computed and cached plan."""
        stats = {}
        for node, input_sources in processing_plan:
            input_data = {}
            try:
                for name, socket in node.inputs.items():
                    source_socket = input_sources.get(name)
                    if source_socket:
                        input_data[name] = source_socket._data
                    else:
                        input_data[name] = None
                
                start_time = time.perf_counter_ns()
                output_data = node.process(input_data)
                end_time = time.perf_counter_ns()
                processing_time_ns = end_time - start_time
                stats[node.id] = ( processing_time_ns / TICK_DURATION_NS ) * 100.0

                for name, data in (output_data or {}).items():
                    if name in node.outputs:
                        node.outputs[name]._data = data

            except Exception as e:
                raise RuntimeError(f"Error processing node '{node.name}': {e}") from e

        return stats

    def _get_processing_plan(self) -> list[tuple[Node, dict[str, Socket]]] | None:
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

    def _compute_processing_plan(self) -> list[tuple[Node, dict[str, Socket]]] | None:
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
            return None # Cycle detected

        plan = []
        for node_id in sorted_node_ids:
            node = self.graph.nodes[node_id]
            input_sources = {}
            for input_name, input_socket in node.inputs.items():
                if input_socket.connections:
                    input_sources[input_name] = input_socket.connections[0].start_socket
            plan.append((node, input_sources))
            
        logger.info("Engine: Re-computed new processing plan.")
        return plan

    def start_processing(self):
        error_to_emit = None
        
        with self._lock:
            if self._is_processing_flag.is_set():
                return
            logger.info("Engine: Starting processing...")
            self._invalidate_plan_cache()
            
            # Call start() on ALL nodes first.
            for node in self.graph.nodes.values():
                try:
                    node.start()
                except Exception as e:
                    # Assign the error message to our variable and break the loop.
                    error_to_emit = f"Error starting node '{node.name}': {e}"
                    break # Do not proceed with other nodes
            
            # If an error occurred in the loop above, we skip the rest of the block.
            if not error_to_emit:
                clock_node_id = self.graph.selected_clock_node_id
                clock_node = self.graph.nodes.get(clock_node_id)
                
                if clock_node and isinstance(clock_node, IClockProvider):
                    try:
                        clock_node.start_clock(self.tick)
                        for _ in range(INITIAL_PRIME_TICKS):
                            self.tick()
                    except Exception as e:
                        # Assign the error message for the clock source.
                        error_to_emit = f"Failed to activate clock source '{clock_node.name}': {e}"
                else:
                    logger.warning("Engine: Starting processing without an active clock source.")
                
                # Only set the processing flag if there have been no errors.
                if not error_to_emit:
                    self._is_processing_flag.set()

        # After the lock is released, check if we need to emit an error.
        if error_to_emit:
            self.signals.processingError.emit(error_to_emit)
            return
    
        self.signals.processingStateChanged.emit(True)
        logger.info("Engine: Processing started.")

    def _stop_processing_locked(self):
        logger.info("Engine: Stopping processing...")
        clock_node_id = self.graph.selected_clock_node_id
        
        # Stop the active clock source
        if clock_node_id and clock_node_id in self.graph.nodes:
            clock_node = self.graph.nodes[clock_node_id]
            if isinstance(clock_node, IClockProvider):
                clock_node.stop_clock()
        
        # Now, call stop() on ALL nodes
        for node in self.graph.nodes.values():
            try:
                node.stop()
            except Exception as e:
                logger.error(f"Error stopping node '{node.name}': {e}", exc_info=True)
        
        self._is_processing_flag.clear()

    def stop_processing(self):
        with self._lock:
            if not self._is_processing_flag.is_set():
                return
            self._stop_processing_locked()
        
        # Drain any pending ticks from the semaphore
        while self._tick_semaphore.acquire(blocking=False):
            pass

        self.signals.nodeProcessingStatsUpdated.emit({}) # Clear UI bars
        self.signals.processingStateChanged.emit(False)
        logger.info("Engine: Processing stopped.")

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

    def rename_node(self, node_id: str, new_name: str):
        """Renames a node and signals the change."""
        with self._lock:
            if node_id in self.graph.nodes and new_name:
                node = self.graph.nodes[node_id]
                logger.info(f"Engine: Renaming node '{node.name}' to '{new_name}'")
                node.name = new_name
                # A name change doesn't affect the processing plan, so no need to invalidate the cache.
        self._emit_graph_changed()

    def set_clock_source(self, node_id: str | None):
        was_processing = self._is_processing_flag.is_set()

        if was_processing:
            self.stop_processing()

        with self._lock:
            # If the requested node is already the clock, do nothing, but restart processing if needed.
            if self.graph.selected_clock_node_id == node_id:
                if was_processing:
                    self.start_processing()
                return

            # Atomically update the clock source ID.
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

        if was_processing:
            self.start_processing()

    def add_node(self, node_class: Type[Node], name: str, pos: tuple[float, float]) -> Node:
        with self._lock:
            self._invalidate_plan_cache() 
            new_node = node_class(name)
            new_node.pos = pos
            self.graph.nodes[new_node.id] = new_node
            
            # Auto-select the first clock provider if none is selected.
            if isinstance(new_node, IClockProvider) and self.graph.selected_clock_node_id is None:
                self.graph.selected_clock_node_id = new_node.id

            # If processing, call the node's individual start() method.
            if self._is_processing_flag.is_set():
                new_node.start()

        self._emit_graph_changed()
        return new_node
    
    def remove_node(self, node_id: str):
        with self._lock:
            if node_id not in self.graph.nodes:
                return
            self._invalidate_plan_cache() 
            node_to_remove = self.graph.nodes[node_id]
            
            # If the node to remove is the active clock, demote it.
            if self.graph.selected_clock_node_id == node_id:
                if self._is_processing_flag.is_set() and isinstance(node_to_remove, IClockProvider):
                    node_to_remove.stop_clock()
                self.graph.selected_clock_node_id = None

            conns_to_remove = [conn for conn in self.graph.connections.values()
                               if conn.start_socket.node == node_to_remove or conn.end_socket.node == node_to_remove]
            
            for conn in conns_to_remove:
                self._remove_connection_internal(conn.id)

            # Node.remove() calls node.stop(), which will clean up its resources.
            node_to_remove.remove()
            del self.graph.nodes[node_id]

        self._emit_graph_changed()

    def add_connection(self, start_socket: Socket, end_socket: Socket):
        with self._lock:
            self._invalidate_plan_cache()
            self._add_connection_locked(start_socket, end_socket)
        self._emit_graph_changed()

    def _add_connection_locked(self, start_socket: Socket, end_socket: Socket):
        """Internal helper to add a connection. Assumes lock is held."""
        try:
            if end_socket.connections:
                self._remove_connection_internal(end_socket.connections[0].id)
            
            connection = Connection(start_socket, end_socket)
            self.graph.connections[connection.id] = connection
            start_socket.connections.append(connection)
            end_socket.connections.append(connection)
        except Exception as e:
            logger.error(f"Engine: Failed to create connection while holding lock: {e}", exc_info=True)

    def remove_connection(self, connection_id: str):
        with self._lock:
            self._invalidate_plan_cache()
            self._remove_connection_internal(connection_id)
        self._emit_graph_changed()

    def _remove_connection_internal(self, connection_id: str):
        """Internal helper. Assumes lock is held."""
        if connection_id in self.graph.connections:
            conn = self.graph.connections[connection_id]
            if conn in conn.start_socket.connections:
                conn.start_socket.connections.remove(conn)
            if conn in conn.end_socket.connections:
                conn.end_socket.connections.remove(conn)
            del self.graph.connections[connection_id]

    def _clear_graph_locked(self):
        """Internal helper to clear the graph. Assumes lock is held."""
        for node in list(self.graph.nodes.values()):
            node.remove()
        self.graph.clear()
        self._invalidate_plan_cache()

    def clear_graph(self):
        self.stop_processing()
        with self._lock:
            self._clear_graph_locked()
        self._emit_graph_changed()

    def save_graph_to_file(self, file_path: str):
        """Serializes the current graph state to a JSON file."""
        with self._lock:
            graph_data = {
                "nodes": [node.to_dict() for node in self.graph.nodes.values()],
                "connections": [conn.to_dict() for conn in self.graph.connections.values()],
                "clock_source_id": self.graph.selected_clock_node_id,
            }
        
        with open(file_path, 'w') as f:
            json.dump(graph_data, f, indent=4)

    def load_graph_from_file(self, file_path: str):
        """Clears the current graph and loads a new one from a JSON file."""
        # Call the public clear_graph method. This correctly stops processing,
        # clears the model, AND emits a signal with the empty graph state,
        # forcing the UI to fully clear itself before the new data arrives.
        self.clear_graph()
        
        try:
            with open(file_path, 'r') as f:
                graph_data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read or parse graph file {file_path}: {e}")
            self.signals.processingError.emit(f"Could not load file: {e}")
            return

        with self._lock:

            # 1. Recreate nodes
            for node_data in graph_data.get("nodes", []):
                node_type = node_data.get("type")
                node_class = registry.get_node_class(node_type)
                if not node_class:
                    logger.warning(f"Could not find node class for type '{node_type}'. Skipping.")
                    continue
                
                new_node = node_class(name=node_data['name'], node_id=node_data['id'])
                new_node.pos = tuple(node_data.get('pos', (0, 0)))
                new_node.deserialize_extra(node_data)
                self.graph.nodes[new_node.id] = new_node
            
            # 2. Recreate connections
            for conn_data in graph_data.get("connections", []):
                start_node = self.graph.nodes.get(conn_data['start_node_id'])
                end_node = self.graph.nodes.get(conn_data['end_node_id'])
                if not start_node or not end_node:
                    logger.warning("Skipping connection due to missing node.")
                    continue
                    
                start_socket = start_node.outputs.get(conn_data['start_socket_name'])
                end_socket = end_node.inputs.get(conn_data['end_socket_name'])
                if not start_socket or not end_socket:
                    logger.warning("Skipping connection due to missing socket.")
                    continue
                
                self._add_connection_locked(start_socket, end_socket)

            # 3. Restore clock source
            self.graph.selected_clock_node_id = graph_data.get("clock_source_id")

        # Emit the final signal to draw the newly loaded graph.
        self._emit_graph_changed()

    def shutdown(self):
        logger.info("Engine: Shutting down...")
        self.stop_processing()
        self._stop_event.set()
        self._tick_semaphore.release()
        self._processing_thread.join(timeout=2)
        if self._processing_thread.is_alive():
            logger.warning("Engine: Processing thread did not exit gracefully.")
        logger.info("Engine: Shutdown complete.")