import os

os.environ["CL_SDK_VISUALISATION"] = "0"  # Disable WebSocket support for testing

import time
import subprocess
import sys
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import cl
import pytest

def generate_recording(directory: Path, duration_sec: float = 30) -> Path:
    """
    Generate a recording file in `directory` using the currently configured
    simulator data source and return the path to the resulting recording file.

    The caller is responsible for enabling accelerated time (e.g. via
    ``CL_SDK_ACCELERATED_TIME``) so generation completes quickly. The `Neurons`
    singleton is cleared afterwards so the next ``cl.open()`` starts fresh.
    """
    with cl.open() as neurons:
        recording = neurons.record(
            file_location      = str(directory),
            stop_after_seconds = duration_sec,
        )
        recording.wait_until_stopped()
        path = Path(recording.file["path"])
    cl.Neurons._clear_instance(force=True)
    return path


def cleanup():
    buffer_prefix = cl.Neurons._instance._buffer_name_prefix if cl.Neurons._instance is not None else None
    # Clear the Neurons singleton instance to ensure a fresh start for each test
    cl.Neurons._clear_instance(force=True)

    # Kill any orphaned producer processes
    # On Unix-like systems, use pkill to target the specific process by name
    if sys.platform != 'win32':
        try:
            subprocess.run(['pkill', '-9', '-f', 'cl-data-producer'],
                            capture_output=True, timeout=2)
        except Exception:
            pass
    # On Windows: the OS will clean up child processes when the parent (pytest) exits.

    # Give any lingering processes time to die
    time.sleep(0.2)

    # Clean up all shared memory segments matching the cl_sdk_ pattern
    # These now have dynamic names with prefixes: cl_sdk_{prefix}_{segment}

    if buffer_prefix:
        # We can target shared memory segments with the specific buffer prefix for more efficient cleanup
        shm_names = {f"cl_sdk_{buffer_prefix}_{suffix}" for suffix in {"frames", "ds_heap", "ds_index", "stims", "header", "spikes"}}

        for shm_name in shm_names:
            try:
                shm = SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
            except Exception:
                pass
    elif sys.platform != 'win32':
        # On some POSIX systems, shared memory segments are represented as files in /dev/shm
        try:
            shm_dir = Path('/dev/shm')
            if shm_dir.exists():
                for shm_file in shm_dir.glob('cl_sdk_*'):
                    try:
                        shm = SharedMemory(name=shm_file.name)
                        shm.close()
                        shm.unlink()
                    except FileNotFoundError:
                        pass
                    except Exception:
                        pass
        except Exception:
            pass

@pytest.fixture(autouse=True)
def cleanup_shared_memory():
    """Clean up any leaked shared memory before and after each test."""

    cleanup()
    try:
        yield
    finally:
        cleanup()
