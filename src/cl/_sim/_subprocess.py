"""Utilities for running SDK background workers via `subprocess.Popen`."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import pickle
import queue
import struct
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, BinaryIO

_MESSAGE_HEADER = struct.Struct("<I")

_logger = logging.getLogger("cl.subprocess")

class _FileDescriptorReader:
    """Minimal binary reader backed by `os.read`."""

    def __init__(self, fd: int):
        self._fd = fd

    def read(self, size: int) -> bytes:
        return os.read(self._fd, size)

def _child_env() -> dict[str, str]:
    """Build an environment for SDK child processes."""
    env = os.environ.copy()

    paths: list[str] = []
    existing = env.get("PYTHONPATH")
    if existing:
        paths.extend(p for p in existing.split(os.pathsep) if p)
    for path in sys.path:
        if path and path not in paths:
            paths.append(path)
    if paths:
        env["PYTHONPATH"] = os.pathsep.join(paths)

    return env

def _binary_stream(stream) -> BinaryIO:
    """Return the binary buffer for stdio streams."""
    if stream is sys.stdin:
        with contextlib.suppress(AttributeError, OSError):
            return _FileDescriptorReader(stream.fileno())  # type: ignore[return-value]
    return getattr(stream, "buffer", stream)

def _read_exact(stream: BinaryIO, size: int) -> bytes:
    """Read exactly *size* bytes or raise EOFError."""
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("IPC stream closed while reading a message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

def encode_ipc_message(message: dict[str, Any]) -> bytes:
    """Encode an IPC message as a length-prefixed pickle frame."""
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > 0xFFFFFFFF:
        raise ValueError("IPC message is too large")
    return _MESSAGE_HEADER.pack(len(payload)) + payload

def decode_ipc_message(payload: bytes) -> dict[str, Any]:
    """Decode a pickle IPC payload."""
    message = pickle.loads(payload)
    if not isinstance(message, dict):
        raise TypeError(f"Expected IPC dict message, got {type(message).__name__}")
    return message

def read_ipc_message(stream) -> dict[str, Any]:
    """Read and decode a single length-prefixed IPC message from *stream*."""
    binary_stream = _binary_stream(stream)
    header = _read_exact(binary_stream, _MESSAGE_HEADER.size)
    (payload_size,) = _MESSAGE_HEADER.unpack(header)
    return decode_ipc_message(_read_exact(binary_stream, payload_size))

def write_status(message: dict[str, Any]) -> None:
    """Write a subprocess status message to stdout."""
    sys.stdout.buffer.write(json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()

class StdoutStatusWriter:
    """Queue-like adapter that sends status dictionaries over stdout."""

    def put(self, message: dict[str, Any]) -> None:
        write_status(message)

def start_ipc_command_reader(
    command_queue: queue.Queue[Any],
    *,
    decode: Callable[[dict[str, Any]], Any] | None = None,
) -> threading.Thread:
    """Start a daemon thread that reads IPC commands from stdin."""

    def _reader() -> None:
        stream = _binary_stream(sys.stdin)
        while True:
            try:
                message = read_ipc_message(stream)
                command_queue.put(decode(message) if decode else message)
            except EOFError:
                break
            except Exception:
                _logger.exception("Failed to decode subprocess command")

    thread = threading.Thread(target=_reader, name="cl-ipc-command-reader", daemon=True)
    thread.start()
    return thread

@dataclass
class PopenProcess:
    """Small process-handle wrapper around `Popen`."""

    target      : str
    process_name: str
    config      : dict[str, Any]
    stdout      : int | None = None
    stderr      : int | None = None
    stdin       : int | None = subprocess.DEVNULL

    def __post_init__(self) -> None:
        self._popen: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._popen is not None:
            return
        command = [
            sys.executable,
            "-m",
            "cl._sim._subprocess_main",
            "--process-name",
            self.process_name,
            "--target",
            self.target,
            "--config-json",
            json.dumps(self.config, separators=(",", ":")),
        ]
        self._popen = subprocess.Popen(
            command,
            env               = _child_env(),
            stdin             = self.stdin,
            stdout            = self.stdout,
            stderr            = self.stderr,
            text              = True,
            bufsize           = 1,
            start_new_session = True,
        )
        _logger.debug("Started Popen subprocess %s with PID %s", self.process_name, self.pid)

    @property
    def pid(self) -> int | None:
        return self._popen.pid if self._popen is not None else None

    @property
    def exitcode(self) -> int | None:
        return self._popen.poll() if self._popen is not None else None

    def is_alive(self) -> bool:
        return self._popen is not None and self._popen.poll() is None

    def join(self, timeout: float | None = None) -> None:
        if self._popen is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._popen.wait(timeout=timeout)

    def terminate(self) -> None:
        if self.is_alive() and self._popen is not None:
            self._popen.terminate()

    def kill(self) -> None:
        if self.is_alive() and self._popen is not None:
            self._popen.kill()

class IpcProcess:
    """`Popen` wrapper with pickle stdin commands and JSON stdout statuses."""

    def __init__(
        self,
        target      : str,
        process_name: str,
        *,
        on_status   : Callable[[dict[str, Any]], None] | None = None,
    ):
        self._target       = target
        self._process_name = process_name
        self._on_status    = on_status

        self._popen        : subprocess.Popen[bytes] | None = None
        self._statuses     : queue.Queue[dict[str, Any]]    = queue.Queue()
        self._reader_thread: threading.Thread | None        = None

    def start(self) -> None:
        if self._popen is not None:
            return
        command = [
            sys.executable,
            "-m",
            "cl._sim._subprocess_main",
            "--process-name",
            self._process_name,
            "--target",
            self._target,
        ]
        self._popen = subprocess.Popen(
            command,
            env               = _child_env(),
            stdin             = subprocess.PIPE,
            stdout            = subprocess.PIPE,
            stderr            = None,
            bufsize           = 0,
            start_new_session = True,
        )
        _logger.debug("Started IPC Popen subprocess %s with PID %s", self._process_name, self.pid)
        self._reader_thread = threading.Thread(
            target = self._read_statuses,
            name   = f"{self._process_name}-status-reader",
            daemon = True,
        )
        self._reader_thread.start()

    @property
    def pid(self) -> int | None:
        return self._popen.pid if self._popen is not None else None

    @property
    def exitcode(self) -> int | None:
        return self._popen.poll() if self._popen is not None else None

    def is_alive(self) -> bool:
        return self._popen is not None and self._popen.poll() is None

    def send_message(self, message: dict[str, Any]) -> None:
        if self._popen is None or self._popen.stdin is None:
            raise RuntimeError("Subprocess has not been started")
        self._popen.stdin.write(encode_ipc_message(message))
        self._popen.stdin.flush()

    def get_status(self, timeout: float | None = None) -> dict[str, Any]:
        return self._statuses.get(timeout=timeout)

    def join(self, timeout: float | None = None) -> None:
        if self._popen is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._popen.wait(timeout=timeout)

    def terminate(self) -> None:
        if self.is_alive() and self._popen is not None:
            self._popen.terminate()

    def kill(self) -> None:
        if self.is_alive() and self._popen is not None:
            self._popen.kill()

    def _read_statuses(self) -> None:
        if self._popen is None or self._popen.stdout is None:
            return
        for line in self._popen.stdout:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise TypeError(f"Expected status dict, got {type(message).__name__}")
            except Exception:
                _logger.debug("Ignoring non-status subprocess stdout line: %r", line.rstrip())
                continue
            self._statuses.put(message)
            if self._on_status is not None:
                self._on_status(message)
