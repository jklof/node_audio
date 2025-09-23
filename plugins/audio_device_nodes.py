import numpy as np
import torch
import sounddevice as sd
from collections import deque
import threading
import time
import logging
from typing import List, Tuple, Dict, Any, Optional, Callable

from node_system import Node, IClockProvider
from ui_elements import NodeItem, NODE_CONTENT_PADDING

from PySide6.QtWidgets import QWidget, QComboBox, QLabel, QVBoxLayout, QSizePolicy, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt, Slot, QObject, Signal, QSignalBlocker, QTimer

from constants import (
    DEFAULT_SAMPLERATE,
    DEFAULT_BLOCKSIZE,
    DEFAULT_CHANNELS,
    DEFAULT_BUFFER_SIZE_BLOCKS,
    DEFAULT_DTYPE,
)

# Set up logging
logger = logging.getLogger(__name__)


# ==============================================================================
# Audio Device Manager
# ==============================================================================
class AudioDeviceManager:
    """Manages audio device discovery and compatibility checks."""

    TARGET_SAMPLERATE = DEFAULT_SAMPLERATE

    @staticmethod
    def rescan_devices():
        """Forces sounddevice to re-scan for available audio devices."""
        try:
            logger.info("AudioDeviceManager: Re-scanning for audio devices.")
            sd._terminate()
            sd._initialize()
        except Exception as e:
            logger.error(f"AudioDeviceManager: Error during device re-scan: {e}")

    @staticmethod
    def get_host_api_name(host_api_index: int) -> Optional[str]:
        try:
            apis = sd.query_hostapis()
            api_info = dict(apis[host_api_index]) if 0 <= host_api_index < len(apis) else None
            return api_info.get("name") if api_info else None
        except Exception as e:
            logger.debug(f"AudioDeviceManager: Error getting host API name: {e}")
            return None

    @staticmethod
    def is_device_compatible(index: int, is_input: bool) -> bool:
        try:
            info = AudioDeviceManager.get_device_info(index)
            if not info:
                return False
            device_max_channels = info.get("max_input_channels" if is_input else "max_output_channels", 0)
            if device_max_channels <= 0:
                return False
            check_channels = 1 if is_input else DEFAULT_CHANNELS
            check_channels = min(check_channels, device_max_channels)
            if is_input:
                sd.check_input_settings(
                    device=index, samplerate=AudioDeviceManager.TARGET_SAMPLERATE, channels=check_channels
                )
            else:
                sd.check_output_settings(
                    device=index, samplerate=AudioDeviceManager.TARGET_SAMPLERATE, channels=check_channels
                )
            return True
        except Exception:
            return False

    @staticmethod
    def get_compatible_devices(is_input: bool) -> List[Tuple[int, Dict]]:
        compatible_devices = []
        try:
            devices = sd.query_devices()
            for idx, dev_info_raw in enumerate(devices):
                info = dict(dev_info_raw)
                if AudioDeviceManager.is_device_compatible(idx, is_input):
                    compatible_devices.append((idx, info))
        except Exception as e:
            logger.error(f"AudioDeviceManager: Error querying devices: {e}")
        return compatible_devices

    @staticmethod
    def find_device_by_name(name: str, hostapi_name: Optional[str], is_input: bool) -> Optional[int]:
        try:
            devices = sd.query_devices()
            apis = sd.query_hostapis()
            target_hostapi_index = None
            if hostapi_name:
                for idx_api, api_info_raw in enumerate(apis):
                    if dict(api_info_raw).get("name") == hostapi_name:
                        target_hostapi_index = idx_api
                        break
            for idx_dev, dev_info_raw in enumerate(devices):
                info = dict(dev_info_raw)
                if info.get("name") != name:
                    continue
                if target_hostapi_index is not None and info.get("hostapi") != target_hostapi_index:
                    continue
                if AudioDeviceManager.is_device_compatible(idx_dev, is_input):
                    return idx_dev
        except Exception as e:
            logger.debug(f"AudioDeviceManager: Error finding device by name: {e}")
        return None

    @staticmethod
    def get_default_device_index(is_input: bool) -> Optional[int]:
        try:
            default_dev = sd.default.device
            default_idx = default_dev if isinstance(default_dev, int) else default_dev[0 if is_input else 1]
            if default_idx is None or default_idx < 0:
                return None
            if AudioDeviceManager.is_device_compatible(default_idx, is_input):
                return default_idx
        except Exception as e:
            logger.debug(f"AudioDeviceManager: Error getting default device index: {e}")
        return None

    @staticmethod
    def get_device_info(index: int) -> Optional[Dict]:
        try:
            return dict(sd.query_devices(index))
        except Exception:
            return None


# ==============================================================================
# Base UI Node Item for Audio Nodes
# ==============================================================================
class AudioDeviceNodeItem(NodeItem):

    NODE_WIDTH = 250

    def __init__(self, node_logic: "BaseAudioNode"):
        super().__init__(node_logic)

        self.container_widget = QWidget()
        layout = QVBoxLayout(self.container_widget)
        layout.setContentsMargins(
            NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING, NODE_CONTENT_PADDING
        )
        layout.setSpacing(3)

        device_row = QHBoxLayout()
        device_row.setSpacing(3)

        self.device_combobox = QComboBox()
        self.device_combobox.setToolTip("Select audio device")
        self.device_combobox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.device_combobox.setMinimumWidth(self.NODE_WIDTH - 40)  # Adjusted for button
        device_row.addWidget(self.device_combobox)

        self.refresh_button = QPushButton("🔄")
        self.refresh_button.setToolTip("Refresh available audio devices")
        self.refresh_button.setFixedSize(24, 24)
        self.refresh_button.clicked.connect(self._on_refresh_devices_clicked)
        device_row.addWidget(self.refresh_button)

        layout.addLayout(device_row)

        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.stream_channels_label = QLabel("Stream Ch: N/A")
        self.stream_channels_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.stream_channels_label)

        self.setContentWidget(self.container_widget)
        self.device_combobox.currentIndexChanged.connect(self._on_ui_device_selection_changed)

        self._populate_device_combobox()

    def _populate_device_combobox(self):
        is_input_node = self.node_logic._get_is_input_node()
        logger.debug(f"[{self.node_logic.name}] UI: Populating device combobox (is_input: {is_input_node}).")
        try:
            with QSignalBlocker(self.device_combobox):
                self.device_combobox.clear()
                default_idx = AudioDeviceManager.get_default_device_index(is_input_node)
                default_name = "Default"
                if default_idx is not None:
                    info = AudioDeviceManager.get_device_info(default_idx)
                    if info:
                        default_name = f"Default ({info.get('name', '')[:25]}...)"
                self.device_combobox.addItem(default_name, userData=None)
                devices = AudioDeviceManager.get_compatible_devices(is_input_node)
                for idx, info in devices:
                    name = f"{idx}: {info.get('name', 'Unknown Device')}"
                    self.device_combobox.addItem(name, userData=idx)
        except Exception as e:
            logger.error(f"[{self.node_logic.name}] UI: Error populating device combobox: {e}")
            if self.device_combobox.count() == 0:
                self.device_combobox.addItem("Error loading devices", userData=-1)

    def _set_combobox_to_device(self, device_identifier: Optional[int]):
        logger.debug(f"[{self.node_logic.name}] UI: Setting combobox to device ID: {device_identifier}.")
        found_index = self.device_combobox.findData(device_identifier)

        if found_index != -1:
            if self.device_combobox.currentIndex() != found_index:
                self.device_combobox.setCurrentIndex(found_index)
        else:
            if self.device_combobox.currentIndex() != 0:
                self.device_combobox.setCurrentIndex(0)  # Default to the 'Default' item
            if device_identifier is not None:
                logger.warning(
                    f"[{self.node_logic.name}] UI: Device ID {device_identifier} not in combobox. Selecting 'Default'."
                )

    @Slot(int)
    def _on_ui_device_selection_changed(self, combobox_idx: int):
        if combobox_idx < 0 or not self.node_logic:
            return
        selected_device_identifier = self.device_combobox.itemData(combobox_idx)
        logger.info(f"[{self.node_logic.name}] UI: User selected device id: {selected_device_identifier}")
        self.node_logic.set_user_selected_device(selected_device_identifier)

    @Slot(dict)
    def _on_state_updated_from_logic(self, state: dict):

        super()._on_state_updated_from_logic(state)

        with QSignalBlocker(self.device_combobox):
            user_selected_id = state.get("user_selected_device_identifier")
            self._set_combobox_to_device(user_selected_id)

        status_message = state.get("status")
        if status_message:
            self._update_status_label(status_message)

        channels = state.get("channels")
        if channels is not None:
            self.stream_channels_label.setText(f"Stream Ch: {channels}")

        if state.get("device_list_refreshed", False):
            logger.info(f"[{self.node_logic.name}] UI: Refreshing device list due to state update.")
            self._populate_device_combobox()
            with QSignalBlocker(self.device_combobox):
                self._set_combobox_to_device(user_selected_id)

    @Slot()
    def _on_refresh_devices_clicked(self):
        if not self.node_logic:
            return
        logger.info(f"[{self.node_logic.name}] UI: Refresh devices button clicked.")
        self.node_logic.refresh_device_list()

    def _update_status_label(self, message: str):
        self.status_label.setText(f"{message}")
        if "Error" in message or "Failed" in message:
            self.status_label.setStyleSheet("color: red;")
        elif "Active" in message:
            self.status_label.setStyleSheet("color: lightgreen;")
        else:
            self.status_label.setStyleSheet("color: lightgray;")


class AudioSourceNodeItem(AudioDeviceNodeItem):
    pass


class AudioSinkNodeItem(AudioDeviceNodeItem):
    pass


# ==============================================================================
# Base Audio Node Logic
# ==============================================================================
class BaseAudioNode(Node):

    NODE_TYPE = None

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)

        self.samplerate = AudioDeviceManager.TARGET_SAMPLERATE
        self.blocksize = DEFAULT_BLOCKSIZE
        self.dtype = DEFAULT_DTYPE
        self.sd_dtype = "float32"

        self.channels = DEFAULT_CHANNELS

        self._buffer = deque(maxlen=DEFAULT_BUFFER_SIZE_BLOCKS)
        self._stream: Optional[sd.Stream] = None
        self._stream_error_count = 0
        self._user_selected_device_identifier: Optional[int] = None
        self._user_selected_device_name: Optional[str] = "Default"
        self._user_selected_device_hostapi_name: Optional[str] = None
        self._active_device_info: Optional[Dict[str, Any]] = None
        self._current_status_message = "Status: Uninitialized"

        self._set_initial_default_device_selection()
        logger.debug(f"[{self.name}] BaseAudioNode initialized.")

    # --- Abstract methods ---
    def _get_is_input_node(self) -> bool:
        raise NotImplementedError

    def _get_stream_config_channels(self, device_info: Dict) -> int:
        raise NotImplementedError

    def _get_audio_callback(self) -> Callable:
        raise NotImplementedError

    def _get_sounddevice_stream_class(self) -> type:
        raise NotImplementedError

    def _update_channels_from_stream(self, stream_instance: sd.Stream) -> Optional[int]:
        return None

    def _reset_channels_on_failure(self) -> Optional[int]:
        return None

    def _post_stream_stop_actions(self):
        pass

    # --- Common methods ---
    def _set_initial_default_device_selection(self):
        self._user_selected_device_identifier = None
        self._user_selected_device_name = "Default"
        self._user_selected_device_hostapi_name = None
        is_input_node = self._get_is_input_node()
        default_idx = AudioDeviceManager.get_default_device_index(is_input_node)
        if default_idx is not None:
            info = AudioDeviceManager.get_device_info(default_idx)
            if info:
                self._user_selected_device_name = info.get("name", "Default")
                self._user_selected_device_hostapi_name = AudioDeviceManager.get_host_api_name(info.get("hostapi", -1))
        self._update_status_message("Awaiting configuration")

    def _update_status_message(self, message: str):
        self._current_status_message = message

    def _get_current_state_snapshot_locked(self) -> Dict:
        return {
            "status": self._current_status_message,
            "user_selected_device_identifier": self._user_selected_device_identifier,
            "channels": self.channels,
        }

    def get_current_state_snapshot(self) -> Dict:
        with self._lock:
            return self._get_current_state_snapshot_locked()

    def refresh_device_list(self):
        logger.info(f"[{self.name}] Logic: Refreshing device list.")

        stream_was_active = False
        if self._stream is not None:
            try:
                # This can fail if the device was disconnected, invalidating the pointer
                stream_was_active = self._stream.active
            except sd.PortAudioError as e:
                logger.warning(
                    f"[{self.name}] Could not check stream status (it may be invalid): {e}. Assuming inactive."
                )
                with self._lock:
                    self._stream = None  # Discard the invalid stream object
                stream_was_active = False

        if stream_was_active:
            self.stop()

        AudioDeviceManager.rescan_devices()
        is_input_node = self._get_is_input_node()

        with self._lock:
            current_id = self._user_selected_device_identifier
            current_name = self._user_selected_device_name
            current_hostapi = self._user_selected_device_hostapi_name

        if current_id is not None and not AudioDeviceManager.is_device_compatible(current_id, is_input_node):
            logger.warning(f"[{self.name}] Device '{current_name}' (ID {current_id}) no longer compatible.")
            found_by_name = AudioDeviceManager.find_device_by_name(current_name, current_hostapi, is_input_node)
            with self._lock:
                if found_by_name is not None:
                    logger.info(f"[{self.name}] Found same device with new ID: {found_by_name}.")
                    self._user_selected_device_identifier = found_by_name
                    info = AudioDeviceManager.get_device_info(found_by_name)
                    if info:
                        self._user_selected_device_name = info.get("name")
                        self._user_selected_device_hostapi_name = AudioDeviceManager.get_host_api_name(
                            info.get("hostapi", -1)
                        )
                else:
                    logger.warning(f"[{self.name}] Reverting to Default.")
                    self._set_initial_default_device_selection()

        state = self._get_current_state_snapshot_locked()
        state["device_list_refreshed"] = True
        self.ui_update_callback(state)

        if stream_was_active:
            self.start()

    @Slot(object)
    def set_user_selected_device(self, device_identifier: Optional[int]):
        should_restart_stream = False
        with self._lock:
            if self._user_selected_device_identifier == device_identifier:
                return

            self._user_selected_device_identifier = device_identifier
            if device_identifier is not None:
                info = AudioDeviceManager.get_device_info(device_identifier)
                if info:
                    self._user_selected_device_name = info.get("name")
                    self._user_selected_device_hostapi_name = AudioDeviceManager.get_host_api_name(
                        info.get("hostapi", -1)
                    )
                else:
                    self._user_selected_device_name = "Unknown"
                    self._user_selected_device_hostapi_name = None
            else:
                self._set_initial_default_device_selection()

            if self._stream is not None and self._stream.active:
                should_restart_stream = True

        if should_restart_stream:
            self.stop()
            self.start()
        else:
            self.ui_update_callback(self._get_current_state_snapshot_locked())

    def start(self):
        is_input_node = self._get_is_input_node()
        with self._lock:
            if self._stream and self._stream.active:
                return

            effective_device_index = self._user_selected_device_identifier
            if effective_device_index is None:
                effective_device_index = AudioDeviceManager.get_default_device_index(is_input_node)

            if effective_device_index is None:
                self._update_status_message(f"Error: No compatible device.")
            elif not AudioDeviceManager.is_device_compatible(effective_device_index, is_input_node):
                name_for_msg = self._user_selected_device_name or f"ID {effective_device_index}"
                self._update_status_message(f"Error: Device not compatible.")
            else:
                try:
                    device_info = AudioDeviceManager.get_device_info(effective_device_index)
                    if not device_info:
                        raise RuntimeError("Could not get device info.")

                    self._buffer.clear()
                    self._stream_error_count = 0
                    self._stream = self._get_sounddevice_stream_class()(
                        samplerate=self.samplerate,
                        blocksize=self.blocksize,
                        device=effective_device_index,
                        channels=self._get_stream_config_channels(device_info),
                        dtype=self.sd_dtype,
                        callback=self._get_audio_callback(),
                    )
                    self._stream.start()

                    actual_id = (
                        self._stream.device
                        if isinstance(self._stream.device, int)
                        else self._stream.device[0 if is_input_node else 1]
                    )
                    self._active_device_info = AudioDeviceManager.get_device_info(actual_id) or {}
                    self._update_status_message(f"Active: {self._active_device_info.get('name', 'N/A')}")
                    self._update_channels_from_stream(self._stream)
                except Exception as e:
                    self._update_status_message(f"Error: Stream start failed.")
                    logger.error(f"[{self.name}] Stream start failed: {e}", exc_info=True)
                    if self._stream:
                        self._stream.close()
                    self._stream = None
                    self._active_device_info = None
                    self._reset_channels_on_failure()

        self.ui_update_callback(self._get_current_state_snapshot_locked())

    def stop(self):
        with self._lock:
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.warning(f"[{self.name}] Error stopping stream: {e}")
                self._stream = None
            self._active_device_info = None
            self._update_status_message("Inactive")
            self._post_stream_stop_actions()
        self.ui_update_callback(self._get_current_state_snapshot_locked())

    def remove(self):
        self.stop()
        super().remove()

    def serialize_extra(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "samplerate": self.samplerate,
                "blocksize": self.blocksize,
                "user_selected_device_identifier": self._user_selected_device_identifier,
                "user_selected_device_name": self._user_selected_device_name,
                "user_selected_device_hostapi_name": self._user_selected_device_hostapi_name,
                "channels": self.channels,
            }

    def deserialize_extra(self, data: Dict[str, Any]):
        is_input_node = self._get_is_input_node()
        with self._lock:
            self.samplerate = data.get("samplerate", DEFAULT_SAMPLERATE)
            self.blocksize = data.get("blocksize", DEFAULT_BLOCKSIZE)
            self.channels = data.get("channels", DEFAULT_CHANNELS)
            loaded_id = data.get("user_selected_device_identifier")
            loaded_name = data.get("user_selected_device_name")
            loaded_hostapi = data.get("user_selected_device_hostapi_name")

            if loaded_id is not None and AudioDeviceManager.is_device_compatible(loaded_id, is_input_node):
                self._user_selected_device_identifier = loaded_id
                info = AudioDeviceManager.get_device_info(loaded_id)
                if info:
                    self._user_selected_device_name = info.get("name")
                    self._user_selected_device_hostapi_name = AudioDeviceManager.get_host_api_name(
                        info.get("hostapi", -1)
                    )
            elif loaded_name:
                found_by_name = AudioDeviceManager.find_device_by_name(loaded_name, loaded_hostapi, is_input_node)
                if found_by_name is not None:
                    self._user_selected_device_identifier = found_by_name
                    info = AudioDeviceManager.get_device_info(found_by_name)
                    if info:
                        self._user_selected_device_name = info.get("name")
                        self._user_selected_device_hostapi_name = AudioDeviceManager.get_host_api_name(
                            info.get("hostapi", -1)
                        )
                else:
                    self._set_initial_default_device_selection()
            else:
                self._set_initial_default_device_selection()
            self._buffer.clear()
            self._update_status_message("Deserialized - Inactive")

        # Emit after deserialization to sync UI
        self.ui_update_callback(self._get_current_state_snapshot_locked())


# ==============================================================================
# Concrete Audio Node Implementations
# ==============================================================================
class AudioSourceNode(BaseAudioNode):
    NODE_TYPE = "Audio Device Input"
    CATEGORY = "Input / Output"
    DESCRIPTION = "Reads audio from input device"
    UI_CLASS = AudioSourceNodeItem

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self._overflow_count = 0
        self._last_overflow_time = 0
        self.add_output("out", data_type=torch.Tensor)

    def _get_is_input_node(self) -> bool:
        return True

    def _get_audio_callback(self) -> Callable:
        return self._audio_callback_input

    def _get_sounddevice_stream_class(self) -> type:
        return sd.InputStream

    def _get_stream_config_channels(self, device_info: Dict) -> int:
        max_ch = device_info.get("max_input_channels", self.channels)
        return max_ch if max_ch > 0 else self.channels

    def _update_channels_from_stream(self, stream_instance: sd.InputStream) -> Optional[int]:
        if self.channels != stream_instance.channels:
            self.channels = stream_instance.channels
            return self.channels
        return None

    def _reset_channels_on_failure(self) -> Optional[int]:
        if self.channels != DEFAULT_CHANNELS:
            self.channels = DEFAULT_CHANNELS
            return self.channels
        return None

    def _audio_callback_input(self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        if status.input_overflow:
            self._overflow_count += 1
            now = time.monotonic()
            if now - self._last_overflow_time > 5.0:
                logger.warning(f"[{self.name}] Input overflow ({self._overflow_count} total).")
                self._last_overflow_time = now
        try:
            with self._lock:
                self._buffer.append(torch.from_numpy(indata.copy().T))
        except Exception as e:
            self._stream_error_count += 1
            if self._stream_error_count < 5:
                logger.error(f"[{self.name}] Error in input callback: {e}", exc_info=True)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            stream_ch = self.channels
            output_block = self._buffer.popleft() if self._buffer else None

        if output_block is None:
            output_block = torch.zeros((stream_ch, self.blocksize), dtype=self.dtype)
        elif output_block.shape[0] != stream_ch:
            temp = torch.zeros((stream_ch, self.blocksize), dtype=self.dtype)
            ch_to_copy = min(output_block.shape[0], stream_ch)
            if ch_to_copy > 0:
                temp[:ch_to_copy, :] = output_block[:ch_to_copy, :]
            output_block = temp
        return {"out": output_block}


class AudioSinkNode(BaseAudioNode, IClockProvider):
    NODE_TYPE = "Audio Device Output"
    CATEGORY = "Input / Output"
    DESCRIPTION = "Sends audio to output device"
    UI_CLASS = AudioSinkNodeItem

    def __init__(self, name, node_id=None):
        super().__init__(name, node_id)
        self._underflow_count = 0
        self._last_underflow_time = 0
        self._tick_callback: Optional[Callable] = None
        self._active_stream_channels = self.channels
        self.add_input("in", data_type=torch.Tensor)

    def _get_is_input_node(self) -> bool:
        return False

    def _get_audio_callback(self) -> Callable:
        return self._audio_callback_output

    def _get_sounddevice_stream_class(self) -> type:
        return sd.OutputStream

    def _get_stream_config_channels(self, device_info: Dict) -> int:
        return self.channels

    def _update_channels_from_stream(self, stream_instance: sd.OutputStream) -> Optional[int]:
        self._active_stream_channels = stream_instance.channels
        if self._active_stream_channels != self.channels:
            logger.info(f"[{self.name}] Stream using {self._active_stream_channels} ch (target: {self.channels}).")
        return None

    def _post_stream_stop_actions(self):
        super()._post_stream_stop_actions()
        self._active_stream_channels = self.channels

    def _audio_callback_output(self, outdata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        if self._tick_callback:
            try:
                self._tick_callback()
            except Exception as e:
                self._stream_error_count += 1
                if self._stream_error_count < 10:
                    logger.error(f"[{self.name}] Error in graph tick: {e}", exc_info=True)

        if status.output_underflow:
            self._underflow_count += 1
            now = time.monotonic()
            if now - self._last_underflow_time > 5.0:
                logger.warning(f"[{self.name}] Output underflow ({self._underflow_count} total).")
                self._last_underflow_time = now

        if not self._stream or not self._stream.active:
            outdata.fill(0)
            return

        with self._lock:
            if self._buffer:
                data_block: torch.Tensor = self._buffer.popleft()
                numpy_block = data_block.numpy().T
                if numpy_block.shape == outdata.shape:
                    outdata[:] = numpy_block
                else:
                    outdata.fill(0)
            else:
                outdata.fill(0)

    def start_clock(self, tick_callback: callable):
        logger.info(f"[{self.name}] Promoting to ACTIVE clock source.")
        self.start()
        self._tick_callback = tick_callback

    def stop_clock(self):
        logger.info(f"[{self.name}] Demoting from active clock source.")
        self._tick_callback = None

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        signal_block = input_data.get("in")
        with self._lock:
            target_chans = self._active_stream_channels

        processed_block = torch.zeros((target_chans, self.blocksize), dtype=self.dtype)
        if (
            isinstance(signal_block, torch.Tensor)
            and signal_block.shape[1] == self.blocksize
            and signal_block.ndim == 2
        ):
            input_chans = signal_block.shape[0]
            if input_chans == target_chans:
                processed_block = signal_block
            else:
                ch_to_copy = min(input_chans, target_chans)
                if ch_to_copy > 0:
                    processed_block[:ch_to_copy, :] = signal_block[:ch_to_copy, :]
                if target_chans > input_chans and input_chans > 0:
                    last_ch = signal_block[-1:, :]
                    processed_block[input_chans:, :] = last_ch.tile((target_chans - input_chans, 1))

        try:
            with self._lock:
                self._buffer.append(processed_block)
        except Exception as e:
            logger.error(f"[{self.name}] Error adding to buffer: {e}", exc_info=True)
        return {}
