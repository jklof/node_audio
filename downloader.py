import os
import time
import urllib.request
import hashlib
import logging
import threading
from typing import Callable, Optional, List, Dict

# Configure logging
logger = logging.getLogger(__name__)


class CancelledError(Exception):
    """Custom exception to signal a cancelled download."""

    pass


def verify_sha256(
    file_path: str,
    expected_hash: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Verifies the SHA256 hash of a file.

    Args:
        file_path: The path to the file to verify.
        expected_hash: The expected hexadecimal SHA256 hash string.
        status_callback: An optional function to receive status updates.

    Returns:
        True if the hash matches, False otherwise.
    """
    file_name = os.path.basename(file_path)
    if status_callback:
        status_callback(f"Verifying {file_name}...")

    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            chunk_size = 32 * 1024
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(chunk)
                time.sleep(0)  # Yield to other threads

        calculated_hash = sha256_hash.hexdigest()

        if calculated_hash == expected_hash:
            logger.info(f"SHA256 hash verified for {file_name}.")
            return True
        else:
            logger.critical(f"FATAL: SHA256 HASH MISMATCH for {file_name}!")
            logger.critical(f"  - Expected:   {expected_hash}")
            logger.critical(f"  - Calculated: {calculated_hash}")
            if status_callback:
                status_callback(f"Error: Hash mismatch for {file_name}")
            return False
    except Exception as e:
        logger.error(f"Could not verify hash for {file_path}: {e}", exc_info=True)
        if status_callback:
            status_callback(f"Error: File verification failed for {file_name}")
        return False


def _download_file_internal(
    url: str,
    dest_path: str,
    status_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Internal download logic with cancellation support."""
    file_name = os.path.basename(dest_path)
    _last_percent_reported = -1

    def reporthook(block_num: int, block_size: int, total_size: int):
        nonlocal _last_percent_reported
        # Check for cancellation at every progress update
        if cancel_event and cancel_event.is_set():
            raise CancelledError("Download cancelled by user.")

        if total_size > 0 and status_callback:
            downloaded = block_num * block_size
            percent = int((downloaded / total_size) * 100)
            if percent > _last_percent_reported:
                status_callback(f"Downloading {file_name} ({percent}%)")
                _last_percent_reported = percent

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)


def ensure_files(
    file_manifest: List[Dict[str, str]],
    status_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> bool:
    """
    Ensures a list of files exists and is valid.
    For each file, it checks for existence, then verifies the SHA256 hash.
    If the file is missing or invalid, it attempts to download it.

    Args:
        file_manifest: A list of dictionaries, where each dict contains:
            'path': The local destination path.
            'url': The download URL.
            'sha256': The expected SHA256 hash.
        status_callback: A function to receive string status updates.
        cancel_event: A threading.Event to signal cancellation.

    Returns:
        True if all files are present and valid by the end, False otherwise.
    """
    for file_info in file_manifest:
        dest_path = file_info["path"]
        url = file_info["url"]
        expected_hash = file_info["sha256"]
        file_name = os.path.basename(dest_path)

        if cancel_event and cancel_event.is_set():
            logger.info("File check cancelled before processing.")
            return False

        # 1. Check if file exists and is already valid
        if os.path.exists(dest_path) and verify_sha256(dest_path, expected_hash, status_callback):
            logger.info(f"File '{file_name}' already exists and is valid.")
            continue  # Move to the next file

        # 2. If not, attempt to download
        if status_callback:
            status_callback(f"Preparing to download {file_name}...")

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        try:
            _download_file_internal(url, dest_path, status_callback, cancel_event)
            if status_callback:
                status_callback(f"Download of {file_name} complete.")
        except CancelledError:
            logger.info(f"Download of {file_name} was cancelled.")
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
            return False
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}", exc_info=True)
            if status_callback:
                status_callback(f"Error: Failed to download {file_name}")
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
            return False

        # 3. Verify the newly downloaded file
        if not verify_sha256(dest_path, expected_hash, status_callback):
            logger.error(f"Verification failed for downloaded file: {file_name}")
            return False

    if status_callback:
        status_callback("All files are ready.")
    return True
