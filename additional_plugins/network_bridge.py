import logging
import socket
import struct
import threading
import time
from collections import deque
from typing import Optional, Dict

import numpy as np

from node_system import Node, IClockProvider
from constants import (
    DEFAULT_SAMPLERATE,
    DEFAULT_BLOCKSIZE,
    DEFAULT_CHANNELS,
    DEFAULT_DTYPE,
)
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem
from PySide6.QtCore import Qt
from ui_elements import NodeItem, NODE_CONTENT_PADDING


logger = logging.getLogger(__name__)


# Global registry so the bridge server can find the active nodes
_BRIDGE_REGISTRY = {
    "sources": [],  # type: list["NetworkBridgeSourceNode"]
    "sinks": [],  # type: list["NetworkBridgeSinkNode"]
}


def _register_source(node: "NetworkBridgeSourceNode"):
    _BRIDGE_REGISTRY["sources"].append(node)


def _unregister_source(node: "NetworkBridgeSourceNode"):
    try:
        _BRIDGE_REGISTRY["sources"].remove(node)
    except ValueError:
        pass


def _register_sink(node: "NetworkBridgeSinkNode"):
    _BRIDGE_REGISTRY["sinks"].append(node)


def _unregister_sink(node: "NetworkBridgeSinkNode"):
    try:
        _BRIDGE_REGISTRY["sinks"].remove(node)
    except ValueError:
        pass


class NetworkBridgeSourceNode(Node):
    NODE_TYPE = "Network Bridge Input"
    CATEGORY = "Input / Output"
    DESCRIPTION = "Provides audio blocks coming from an external plugin/host over TCP."

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_output("out", data_type=np.ndarray)
        self._buffer = deque(maxlen=16)
        self._lock = threading.Lock()
        _register_source(self)

    def push_block(self, block: np.ndarray):
        try:
            with self._lock:
                self._buffer.append(block.astype(DEFAULT_DTYPE, copy=False))
        except Exception as e:
            logger.error(f"[{self.name}] Error pushing input block: {e}")

    def process(self, input_data: dict) -> dict:
        with self._lock:
            block = self._buffer.popleft() if self._buffer else None

        if block is None:
            # If nothing queued yet, output silence with default channel count
            block = np.zeros((DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS), dtype=DEFAULT_DTYPE)

        return {"out": block}

    def remove(self):
        _unregister_source(self)
        super().remove()


class NetworkBridgeSinkNode(Node, IClockProvider):
    NODE_TYPE = "Network Bridge Output"
    CATEGORY = "Input / Output"
    DESCRIPTION = (
        "Collects audio destined for an external plugin/host and can act as the graph clock."
    )
    UI_CLASS = None  # Will be set after UI class definition

    def __init__(self, name: str, node_id: Optional[str] = None):
        super().__init__(name, node_id)
        self.add_input("in", data_type=np.ndarray)

        self._tick_callback: Optional[callable] = None
        self._result_event = threading.Event()
        self._expected_seq: Optional[int] = None
        self._result_block: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        self._server: Optional["BridgeServer"] = None
        # Connected clients map: addr_str -> info dict
        self._connected_clients: Dict[str, Dict] = {}
        # Signal emitter for UI updates
        self.signal_emitter = BridgeSignalEmitter(parent_node_name=self.name)
        _register_sink(self)

    # IClockProvider
    def start_clock(self, tick_callback: callable):
        logger.info(f"[{self.name}] Network bridge promoted to ACTIVE clock source.")
        self._tick_callback = tick_callback
        if self._server is None:
            # Locate a source node if available
            source = _BRIDGE_REGISTRY["sources"][0] if _BRIDGE_REGISTRY["sources"] else None
            self._server = BridgeServer(self, source)
            self._server.start()

    def stop_clock(self):
        logger.info(f"[{self.name}] Network bridge demoted from active clock.")
        self._tick_callback = None

    def start(self):
        # No-op; server is started when the node becomes the clock
        pass

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None
        # Clear any connected clients on stop
        with self._lock:
            if self._connected_clients:
                self._connected_clients.clear()
                if self.signal_emitter:
                    self.signal_emitter.emit_client_list_changed()

    def remove(self):
        self.stop()
        _unregister_sink(self)
        super().remove()

    # Server coordination
    def set_expected_sequence(self, seq: int):
        with self._lock:
            self._expected_seq = seq
            self._result_block = None
            self._result_event.clear()

    def get_result_block(self, timeout_s: float) -> Optional[np.ndarray]:
        if self._result_event.wait(timeout_s):
            with self._lock:
                return self._result_block
        return None

    def _on_tick_request(self):
        if self._tick_callback:
            self._tick_callback()

    def process(self, input_data: dict) -> dict:
        block = input_data.get("in")
        if not isinstance(block, np.ndarray):
            block = np.zeros((DEFAULT_BLOCKSIZE, DEFAULT_CHANNELS), dtype=DEFAULT_DTYPE)

        with self._lock:
            # Publish the result for the currently expected sequence
            self._result_block = block.astype(DEFAULT_DTYPE, copy=False)
            self._result_event.set()

        return {}

    # ---- UI/Status helpers ----
    def _emit_status(self, message: str):
        if self.signal_emitter:
            self.signal_emitter.emit_server_status_update(message)

    def _notify_client_connected(self, addr_str: str, info: Dict):
        with self._lock:
            self._connected_clients[addr_str] = info
        if self.signal_emitter:
            self.signal_emitter.emit_client_list_changed()

    def _notify_client_disconnected(self, addr_str: str):
        removed = False
        with self._lock:
            if addr_str in self._connected_clients:
                del self._connected_clients[addr_str]
                removed = True
        if removed and self.signal_emitter:
            self.signal_emitter.emit_client_list_changed()

    def get_connected_clients_snapshot(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self._connected_clients)


class BridgeServer(threading.Thread):
    """
    Minimal TCP server to exchange audio blocks with a DAW plugin. Protocol:
      - Client connects to (host, port). Default: localhost:61000
      - Client sends handshake: b"NABR" + u32 sample_rate + u32 blocksize + u16 in_ch + u16 out_ch (little endian)
      - For each processing block:
          Client sends: opcode=1 (u8), seq=u32, payload: in_ch * blocksize float32 samples, interleaved by frame
          Server replies: opcode=2 (u8), seq=u32, payload: out_ch * blocksize float32, interleaved by frame
    """

    def __init__(self, sink: NetworkBridgeSinkNode, source: Optional[NetworkBridgeSourceNode], host: str = "127.0.0.1", port: int = 61000):
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._sink = sink
        self._source = source
        self._stop_event = threading.Event()
        self._server_sock: Optional[socket.socket] = None

        # Runtime negotiated values (from handshake)
        self.sample_rate = DEFAULT_SAMPLERATE
        self.blocksize = DEFAULT_BLOCKSIZE
        self.in_channels = DEFAULT_CHANNELS
        self.out_channels = DEFAULT_CHANNELS

    def shutdown(self):
        self._stop_event.set()
        try:
            if self._server_sock:
                try:
                    self._server_sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self._server_sock.close()
        except Exception:
            pass

    def run(self):
        logger.info(f"BridgeServer: starting on {self._host}:{self._port}")
        try:
            self._sink._emit_status(f"Listening on {self._host}:{self._port}")
        except Exception:
            pass
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self._host, self._port))
                s.listen(1)
                s.settimeout(0.5)
                self._server_sock = s

                while not self._stop_event.is_set():
                    try:
                        conn, addr = s.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    logger.info(f"BridgeServer: client connected from {addr}")
                    try:
                        self._serve_client(conn, addr)
                    except Exception as e:
                        logger.error(f"BridgeServer: client error: {e}", exc_info=True)
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        logger.info("BridgeServer: client disconnected")
        except Exception as e:
            logger.error(f"BridgeServer: fatal server error: {e}", exc_info=True)
        finally:
            logger.info("BridgeServer: stopped")
            try:
                self._sink._emit_status("Stopped")
            except Exception:
                pass

    def _recv_exact(self, conn: socket.socket, num: int) -> bytes:
        chunks = []
        remaining = num
        while remaining > 0:
            try:
                chunk = conn.recv(remaining)
                if not chunk:
                    raise ConnectionError("Client disconnected during receive")
                chunks.append(chunk)
                remaining -= len(chunk)
            except socket.timeout:
                raise ConnectionError("Receive timeout - client may have disconnected")
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
                raise ConnectionError(f"Client connection lost: {e}")
        return b"".join(chunks)

    def _serve_client(self, conn: socket.socket, addr):
        # Set a reasonable timeout for all socket operations
        conn.settimeout(2.0)
        addr_str = f"{addr[0]}:{addr[1]}"
        
        try:
            # Handshake
            hdr = self._recv_exact(conn, 4 + 4 + 4 + 2 + 2)
            magic, sr, bs, in_ch, out_ch = struct.unpack("<4sIIHH", hdr)
            if magic != b"NABR":
                raise RuntimeError("Bad handshake magic")

            self.sample_rate = int(sr)
            self.blocksize = int(bs)
            self.in_channels = int(in_ch)
            self.out_channels = int(out_ch)

            logger.info(
                f"BridgeServer: handshake OK (sr={self.sample_rate}, bs={self.blocksize}, in={self.in_channels}, out={self.out_channels})"
            )
            # Classify plugin type (best-effort)
            plugin_type = "Instrument" if self.in_channels == 0 and self.out_channels > 0 else "FX"
            
            # Notify UI about the connected client
            try:
                self._sink._notify_client_connected(
                    addr_str,
                    {
                        "type": plugin_type,
                        "sample_rate": self.sample_rate,
                        "blocksize": self.blocksize,
                        "in_channels": self.in_channels,
                        "out_channels": self.out_channels,
                    },
                )
                self._sink._emit_status(f"Client connected: {addr_str} ({plugin_type})")
            except Exception:
                pass

            if self.blocksize != DEFAULT_BLOCKSIZE:
                logger.warning(
                    f"BridgeServer: host blocksize {self.blocksize} != app DEFAULT_BLOCKSIZE {DEFAULT_BLOCKSIZE}. Expect additional latency/jitter."
                )

            # Processing loop
            frame_bytes_in = self.in_channels * self.blocksize * 4
            frame_bytes_out = self.out_channels * self.blocksize * 4

            while not self._stop_event.is_set():
                # Read request with timeout handling
                try:
                    header = self._recv_exact(conn, 1 + 4)
                except (ConnectionError, socket.timeout, socket.error) as e:
                    logger.info(f"BridgeServer: client {addr_str} disconnected during header read: {e}")
                    break
                except Exception as e:
                    logger.error(f"BridgeServer: unexpected error reading header from {addr_str}: {e}")
                    break
                    
                (opcode,) = struct.unpack("<B", header[:1])
                (seq,) = struct.unpack("<I", header[1:])
                if opcode != 1:
                    logger.warning(f"BridgeServer: unknown opcode {opcode} from {addr_str}, closing")
                    break

                in_block = None
                if self.in_channels > 0:
                    try:
                        payload = self._recv_exact(conn, frame_bytes_in)
                        in_f32 = np.frombuffer(payload, dtype=np.float32)
                        try:
                            in_block = in_f32.reshape(self.blocksize, self.in_channels).copy()
                        except Exception:
                            in_block = np.zeros((self.blocksize, self.in_channels), dtype=np.float32)
                    except (ConnectionError, socket.timeout, socket.error) as e:
                        logger.info(f"BridgeServer: client {addr_str} disconnected during payload read: {e}")
                        break
                    except Exception as e:
                        logger.error(f"BridgeServer: unexpected error reading payload from {addr_str}: {e}")
                        break

                # Deliver input block to source node (effect mode)
                if in_block is not None and self._source is not None:
                    self._source.push_block(in_block)

                # Trigger graph processing and collect result from sink node
                self._sink.set_expected_sequence(seq)
                self._sink._on_tick_request()
                out_block = self._sink.get_result_block(timeout_s=1.0)
                if out_block is None:
                    logger.warning("BridgeServer: timeout waiting for processed block; sending silence")
                    out_block = np.zeros((self.blocksize, self.out_channels), dtype=np.float32)

                # Ensure shape and dtype
                if not isinstance(out_block, np.ndarray) or out_block.ndim != 2 or out_block.shape[0] != self.blocksize:
                    out_block = np.zeros((self.blocksize, self.out_channels), dtype=np.float32)
                if out_block.shape[1] != self.out_channels:
                    ch_to_copy = min(out_block.shape[1], self.out_channels)
                    temp = np.zeros((self.blocksize, self.out_channels), dtype=np.float32)
                    if ch_to_copy > 0:
                        temp[:, :ch_to_copy] = out_block[:, :ch_to_copy]
                    out_block = temp
                out_block = out_block.astype(np.float32, copy=False)

                # Send response with proper error handling
                try:
                    conn.sendall(struct.pack("<B", 2) + struct.pack("<I", seq))
                    conn.sendall(out_block.astype(np.float32, copy=False).tobytes(order="C"))
                except (ConnectionError, socket.timeout, socket.error, BrokenPipeError) as e:
                    logger.info(f"BridgeServer: client {addr_str} disconnected during response send: {e}")
                    break
                except Exception as e:
                    logger.error(f"BridgeServer: unexpected error sending response to {addr_str}: {e}")
                    break

        except Exception as e:
            logger.error(f"BridgeServer: error in client handler for {addr_str}: {e}")
        finally:
            # Client disconnected; notify UI
            try:
                self._sink._notify_client_disconnected(addr_str)
                self._sink._emit_status(f"Client disconnected: {addr_str}")
            except Exception:
                pass


class BridgeSignalEmitter(QObject):
    """Qt signal emitter for bridge node UI updates."""

    server_status_update = Signal(str)
    client_list_changed = Signal()

    def __init__(self, parent_node_name: str = "NetworkBridge"):
        super().__init__()
        self._parent_node_name = parent_node_name

    def emit_server_status_update(self, status: str):
        try:
            self.server_status_update.emit(status)
        except RuntimeError:
            pass

    def emit_client_list_changed(self):
        try:
            self.client_list_changed.emit()
        except RuntimeError:
            pass


class NetworkBridgeSinkNodeItem(NodeItem):
    """UI for NetworkBridgeSinkNode listing connected plugins and status."""

    NODE_WIDTH = 260

    def __init__(self, node_logic: NetworkBridgeSinkNode):
        super().__init__(node_logic)

        self.container = QWidget()
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(4)

        self.status_label = QLabel("Status: Inactive")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.clients_list = QListWidget()
        self.clients_list.setMinimumHeight(60)
        layout.addWidget(self.clients_list)

        self.setContentWidget(self.container)

        emitter = getattr(self.node_logic, "signal_emitter", None)
        if emitter:
            emitter.server_status_update.connect(self._on_status_update)
            emitter.client_list_changed.connect(self._refresh_clients_list)

        # Initialize UI
        self._refresh_clients_list()

    def _on_status_update(self, status: str):
        self.status_label.setText(status)
        if "Error" in status:
            self.status_label.setStyleSheet("color: red;")
        elif "Listening" in status or "connected" in status or "Active" in status:
            self.status_label.setStyleSheet("color: lightgreen;")
        elif "Stopped" in status:
            self.status_label.setStyleSheet("color: lightgray;")
        else:
            self.status_label.setStyleSheet("")

    def _refresh_clients_list(self):
        self.clients_list.clear()
        if not self.node_logic:
            return
        snapshot = self.node_logic.get_connected_clients_snapshot()
        if not snapshot:
            item = QListWidgetItem("No plugins connected")
            item.setForeground(Qt.gray)
            self.clients_list.addItem(item)
            return
        for addr_str, info in snapshot.items():
            label = f"{addr_str} — {info.get('type','?')}  in:{info.get('in_channels')} out:{info.get('out_channels')}  sr:{info.get('sample_rate')} bs:{info.get('blocksize')}"
            self.clients_list.addItem(QListWidgetItem(label))

    def updateFromLogic(self):
        self._refresh_clients_list()
        super().updateFromLogic()


# Attach UI class to logic node
NetworkBridgeSinkNode.UI_CLASS = NetworkBridgeSinkNodeItem


