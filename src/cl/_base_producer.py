"""
Base classes for producer workers and their interfaces.

This module provides abstract base classes that capture common patterns
between DataProducer and PlaybackProducer:

- BaseProducerWorker: Common subprocess worker utilities (file I/O, buffer, GC)
- BaseProducer: Common main-process interface (start/stop, buffer management)

The base classes provide common utilities but allow subclasses to implement
their own run() loops since the timing requirements differ significantly
(e.g., accelerated mode in DataProducer vs seek in PlaybackProducer).
"""
from __future__ import annotations

import contextlib
import gc
import logging
import os
import time
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import tables

from ._data_buffer import SharedDataBuffer, SpikeRecord, StimRecord

_logger = logging.getLogger("cl.base_producer")


class BaseProducerWorker(ABC):
    """
    Abstract base class for producer subprocess workers.

    Provides common utilities:
    - Shared buffer attachment
    - Replay file opening (samples, spikes)
    - Spike timestamp loading for binary search
    - GC management for low-jitter operation
    - Cleanup on exit

    Subclasses implement their own run() loop and use the utilities.
    """

    def __init__(
        self,
        replay_file_path : str,
        channel_count    : int,
        frames_per_second: int,
        tick_rate_hz     : int,
        command_queue    : Queue,
        name_prefix      : str,
    ):
        self._replay_file_path  = replay_file_path
        self._channel_count     = channel_count
        self._frames_per_second = frames_per_second
        self._command_queue     = command_queue
        self._name_prefix       = name_prefix

        # Calculate frames per tick (subclasses may override)
        self._frames_per_tick  = max(1, frames_per_second // tick_rate_hz)
        self._tick_duration_ns = 1_000_000_000 // tick_rate_hz

        # State
        self._running = False

        # Shared buffer (attached via attach_buffer())
        self._buffer: SharedDataBuffer | None = None

        # Replay file handle and data references (opened via open_replay_file())
        self._replay_file   : tables.File | None = None
        self._replay_samples: tables.Array | None = None
        self._replay_spikes : tables.Table | None = None

        # Pre-loaded timestamp array for efficient binary search
        self._spike_timestamps: np.ndarray | None = None

    @property
    def buffer(self) -> SharedDataBuffer | None:
        """The shared data buffer."""
        return self._buffer

    @property
    def is_running(self) -> bool:
        """Whether the producer is running."""
        return self._running

    # --- Common utility methods ---

    @staticmethod
    def set_process_priority() -> None:
        """Set higher process priority to help ensure stable timing."""
        with contextlib.suppress(OSError, AttributeError):
            os.nice(-5)

    def attach_buffer(self) -> SharedDataBuffer:
        """
        Attach to the shared memory buffer as producer.

        Returns the attached buffer and stores it in self._buffer.
        """
        self._buffer = SharedDataBuffer.attach(
            as_producer=True,
            name_prefix=self._name_prefix
        )
        return self._buffer

    def open_replay_file(self) -> None:
        """
        Open the replay H5 file and set up samples/spikes references.

        Sets self._replay_file, self._replay_samples, self._replay_spikes.
        Subclasses can call this then add their own data loading (stims, datastreams).
        """
        h5file = tables.open_file(self._replay_file_path, mode='r')

        # Get samples (required)
        if not hasattr(h5file.root, 'samples'):
            raise ValueError("Recording file missing 'samples' dataset")
        self._replay_samples = h5file.root.samples

        # Get spikes (optional)
        self._replay_spikes = h5file.root.spikes if hasattr(h5file.root, 'spikes') else None

        # Store file handle
        self._replay_file = h5file

    def load_spike_timestamps(self) -> None:
        """Load spike timestamps into memory for efficient binary search."""
        if self._replay_spikes is None:
            self._spike_timestamps = np.array([], dtype=np.int64)
            return

        spike_count = len(self._replay_spikes)
        self._spike_timestamps = np.zeros(spike_count, dtype=np.int64)

        for i in range(spike_count):
            self._spike_timestamps[i] = int(self._replay_spikes[i]["timestamp"])

    @staticmethod
    def disable_gc() -> None:
        """Collect garbage then disable GC to avoid jitter."""
        gc.collect()
        gc.disable()

    @staticmethod
    def enable_gc() -> None:
        """Re-enable GC and collect garbage."""
        gc.enable()
        gc.collect()

    def read_spikes_in_range(
        self,
        from_timestamp  : int,
        to_timestamp    : int,
        timestamp_offset: int = 0,
    ) -> list[SpikeRecord]:
        """
        Read spikes from the replay file within a timestamp range.

        Uses binary search on pre-loaded timestamps for efficiency.

        Args:
            from_timestamp: Start timestamp (inclusive, in file coordinates)
            to_timestamp: End timestamp (exclusive, in file coordinates)
            timestamp_offset: Offset to add to spike timestamps in result

        Returns:
            List of SpikeRecord for spikes in the range
        """
        if (
            self._replay_spikes is None or
            self._spike_timestamps is None or
            len(self._spike_timestamps) == 0
        ):
            return []

        # Binary search for range
        left_idx  = np.searchsorted(self._spike_timestamps, from_timestamp, side="left")
        right_idx = np.searchsorted(self._spike_timestamps, to_timestamp, side="left")

        result = []
        for i in range(left_idx, right_idx):
            spike = self._replay_spikes[i]
            result.append(SpikeRecord(
                timestamp           = int(spike["timestamp"]) + timestamp_offset,
                channel             = int(spike["channel"]),
                channel_mean_sample = 0.0,  # Not typically stored in recording
                samples             = np.array(spike["samples"], dtype=np.float32)
            ))

        return result

    def get_next_command(self):
        """Get the next command from the queue, or None if empty."""
        try:
            return self._command_queue.get_nowait()
        except Exception:
            return None

    def sleep_until_next_tick(
        self,
        start_wall_ns: int,
        tick_count   : int,
    ) -> None:
        """Sleep until the next tick based on wall time."""
        target_wall_ns = start_wall_ns + (tick_count * self._tick_duration_ns)
        now_ns         = time.perf_counter_ns()
        sleep_ns       = target_wall_ns - now_ns

        if sleep_ns > 0:
            time.sleep(sleep_ns * 1e-9)

    def write_spikes_to_buffer(self, spikes: list[SpikeRecord]) -> None:
        """Write a list of spikes to the shared buffer."""
        if self._buffer is None:
            return

        for spike in spikes:
            self._buffer.write_spike(spike)

    def write_stims_to_buffer(self, stims: list[StimRecord]) -> None:
        """Write a list of stims to the shared buffer."""
        if self._buffer is None:
            return

        for stim in stims:
            self._buffer.write_stim(stim)

    def cleanup(self) -> None:
        """Clean up resources - call at end of run()."""
        if self._buffer:
            self._buffer.close()
            self._buffer = None

        if self._replay_file is not None:
            self._replay_file.close()
            self._replay_file = None

        BaseProducerWorker.enable_gc()
        _logger.info("Producer stopped")

    # --- Abstract method for subclasses ---

    @abstractmethod
    def run(self) -> None:
        """
        Main producer loop.

        Subclasses implement their own loop using the utility methods:
        - set_process_priority()
        - attach_buffer()
        - open_replay_file()
        - load_spike_timestamps()
        - disable_gc()
        - read_spikes_in_range()
        - cleanup()
        """


class BaseProducer(ABC):
    """
    Abstract base class for producer interfaces (main process side).

    Handles common functionality:
    - Shared buffer creation
    - Subprocess management (start/stop)
    - Buffer property access

    Subclasses implement:
    - _create_process(): Create the subprocess Process object
    - _send_shutdown(): Send shutdown command to subprocess
    """

    def __init__(
        self,
        replay_file_path : str | Path,
        channel_count    : int,
        frames_per_second: int,
    ):
        self._replay_file_path  = str(replay_file_path) if isinstance(replay_file_path, Path) else replay_file_path
        self._channel_count     = channel_count
        self._frames_per_second = frames_per_second

        # State
        self._started    = False
        self._process    : Process | None = None
        self._buffer     : SharedDataBuffer | None = None
        self._name_prefix: str | None = None

        # Command queue (main -> producer)
        self._command_queue: Queue = Queue()

    @property
    def buffer(self) -> SharedDataBuffer | None:
        """The shared data buffer."""
        return self._buffer

    @property
    def name_prefix(self) -> str | None:
        """The shared memory name prefix."""
        return self._name_prefix

    @property
    def is_started(self) -> bool:
        """Whether the producer subprocess has been started."""
        return self._started

    @property
    def current_timestamp(self) -> int:
        """Get the current timestamp from the buffer."""
        if self._buffer:
            return self._buffer.write_timestamp
        return 0

    @property
    def is_paused(self) -> bool:
        """Check if the producer is paused."""
        if self._buffer:
            return self._buffer.pause_flag
        return False

    def start(self, timeout: float = 15.0, start_timestamp: int = 0) -> None:
        """
        Start the producer subprocess.

        Args:
            timeout: Maximum time to wait for producer to signal ready
            start_timestamp: Starting timestamp for the buffer
        """
        if self._started:
            return

        # Create shared memory buffer
        self._buffer = SharedDataBuffer.create(
            channel_count     = self._channel_count,
            frames_per_second = self._frames_per_second,
            start_timestamp   = start_timestamp,
        )
        self._name_prefix = self._buffer.get_name_prefix()

        _logger.info("Created shared buffer with prefix: %s", self._name_prefix)

        # Create and start subprocess
        self._process = self._create_process()
        self._process.start()

        # Wait for producer to signal ready
        start_time = time.time()
        while not self._buffer.producer_ready:
            if time.time() - start_time > timeout:
                self.stop()
                raise TimeoutError("Producer did not start within timeout")
            time.sleep(0.01)

        self._started = True
        _logger.info("Producer started")

    def stop(self) -> None:
        """Stop the producer subprocess."""
        if not self._started:
            return

        # Send shutdown command
        self._send_shutdown()

        # Also set shutdown flag in buffer
        if self._buffer:
            self._buffer.shutdown_flag = True

        # Wait for process to exit
        if self._process:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                _logger.warning("Producer did not exit cleanly, terminating")
                self._process.terminate()
                self._process.join(timeout=1.0)

        # Close and unlink shared memory
        if self._buffer:
            self._buffer.close(unlink=True)
            self._buffer = None

        self._started = False
        _logger.info("Producer stopped")

    def set_paused(self, paused: bool) -> None:
        """Set the pause state of the producer."""
        if self._buffer:
            self._buffer.pause_flag = paused

    # --- Abstract methods for subclasses ---

    @abstractmethod
    def _create_process(self) -> Process:
        """
        Create the subprocess Process object.

        Override to create Process with appropriate target and args.
        """

    @abstractmethod
    def _send_shutdown(self) -> None:
        """
        Send shutdown command to the subprocess.

        Override to put shutdown command on queue.
        """
