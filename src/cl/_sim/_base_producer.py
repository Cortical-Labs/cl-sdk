"""
Base classes for producer workers and their interfaces.

This module provides abstract base classes that capture common patterns
between DataProducer and PlaybackProducer:

- BaseProducerWorker: Common subprocess worker utilities (buffer, GC, timing)
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
import sys
import time
import traceback
from abc import ABC, abstractmethod

from ._data_buffer import SharedDataBuffer, SpikeRecord, StimRecord
from ._subprocess import PopenProcess

_logger = logging.getLogger("cl.base_producer")


class BaseProducerWorker(ABC):
    """
    Abstract base class for producer subprocess workers.

    Provides common utilities:
    - Shared buffer attachment
    - GC management for low-jitter operation
    - Tick-rate sleep helper
    - Buffer spike/stim write helpers
    - Cleanup on exit

    Subclasses implement their own run() loop and own any source-specific
    state (replay files, live data sources, etc.).
    """

    def __init__(
        self,
        channel_count    : int,
        frames_per_second: int,
        tick_rate_hz     : int,
        name_prefix      : str,
        start_timestamp  : int = 0,
        duration_frames  : int = 0,
    ):
        self._channel_count     = channel_count
        self._frames_per_second = frames_per_second
        self._name_prefix       = name_prefix

        # Calculate frames per tick (subclasses may override)
        self._frames_per_tick  = max(1, frames_per_second // tick_rate_hz)
        self._tick_duration_ns = 1_000_000_000 // tick_rate_hz

        # Common playback state
        self._start_timestamp   = start_timestamp
        self._duration_frames   = duration_frames
        self._current_timestamp = start_timestamp

        # State
        self._running = False

        # Shared buffer (attached via attach_buffer())
        self._buffer: SharedDataBuffer | None = None

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
    def run_in_subprocess(worker: BaseProducerWorker, label: str = "Producer") -> None:
        """Run a producer worker with standard subprocess error handling."""
        try:
            worker.run()
        except KeyboardInterrupt:
            with contextlib.suppress(Exception):
                worker.cleanup()
        except Exception as e:
            print(f"{label} subprocess failed: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise

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
            as_producer                      = True,
            name_prefix                      = self._name_prefix,
            unregister_from_resource_tracker = True,
        )
        return self._buffer

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

    def _process_commands(self) -> None:
        """Process shared-memory control state. Subclasses may override."""
        if self._buffer is not None and self._buffer.shutdown_flag:
            self._running = False

    def _handle_command(self, cmd: object) -> None:
        """Handle a single non-shutdown command. Override in subclasses."""

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

        self._buffer.write_spikes(spikes)

    def write_stims_to_buffer(self, stims: list[StimRecord]) -> None:
        """Write a list of stims to the shared buffer."""
        if self._buffer is None:
            return

        self._buffer.write_stims(stims)

    def cleanup(self) -> None:
        """Clean up resources - call at end of run()."""
        if self._buffer:
            self._buffer.close()
            self._buffer = None

        BaseProducerWorker.enable_gc()
        _logger.info("Producer stopped")

    # --- Abstract method for subclasses ---

    @abstractmethod
    def run(self) -> None:
        """Main producer loop. Subclasses implement their own loop using the utility methods."""


class BaseProducer(ABC):
    """
    Abstract base class for producer interfaces (main process side).

    Handles common functionality:
    - Shared buffer creation
    - Subprocess management (start/stop)
    - Buffer property access

    Subclasses implement:
    - _create_process(): Create the subprocess handle
    """

    def __init__(
        self,
        channel_count    : int,
        frames_per_second: int,
        start_timestamp  : int = 0,
        duration_frames  : int = 0,
        tick_rate_hz     : int = 5000,
    ):
        self._channel_count     = channel_count
        self._frames_per_second = frames_per_second
        self._start_timestamp   = start_timestamp
        self._duration_frames   = duration_frames
        self._tick_rate_hz      = tick_rate_hz

        # State
        self._started    : bool                    = False
        self._process    : PopenProcess     | None = None
        self._buffer     : SharedDataBuffer | None = None
        self._name_prefix: str              | None = None

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

    @property
    def duration_frames(self) -> int:
        """Get total duration in frames."""
        return self._duration_frames

    @property
    def start_timestamp(self) -> int:
        """Get the recording start timestamp."""
        return self._start_timestamp

    @property
    def end_timestamp(self) -> int:
        """Get the recording end timestamp."""
        return self._start_timestamp + self._duration_frames

    def start(self, timeout: float = 15.0) -> None:
        """
        Start the producer subprocess.

        Args:
            timeout: Maximum time to wait for producer to signal ready
        """
        if self._started:
            return

        # Create shared memory buffer
        self._buffer = SharedDataBuffer.create(
            channel_count     = self._channel_count,
            frames_per_second = self._frames_per_second,
            start_timestamp   = self._start_timestamp,
        )
        self._name_prefix = self._buffer.get_name_prefix()

        _logger.info("Created shared buffer with prefix: %s", self._name_prefix)

        # Create and start subprocess
        self._process = self._create_process()
        self._process.start()

        # Wait for producer to signal ready
        start_time = time.time()
        while not self._buffer.producer_ready:
            if self._process is not None and not self._process.is_alive():
                exitcode = self._process.exitcode
                self._cleanup_after_failed_start()
                raise RuntimeError(f"Producer subprocess exited early with code {exitcode}")
            if time.time() - start_time > timeout:
                self._cleanup_after_failed_start()
                raise TimeoutError("Producer did not start within timeout")
            time.sleep(0.01)

        self._started = True
        _logger.info("Producer started")

    def stop(self) -> None:
        """Stop the producer subprocess."""
        if not self._started:
            return

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

    def _cleanup_after_failed_start(self) -> None:
        """Release resources allocated before producer startup completed."""
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)
        self._process = None

        if self._buffer:
            self._buffer.close(unlink=True)
            self._buffer = None

        self._started = False

    def set_paused(self, paused: bool) -> None:
        """Set the pause state of the producer."""
        if self._buffer:
            self._buffer.pause_flag = paused

    # --- Abstract methods for subclasses ---

    @abstractmethod
    def _create_process(self) -> PopenProcess:
        """
        Create the subprocess handle.

        Override to create PopenProcess with appropriate module and config.
        """
