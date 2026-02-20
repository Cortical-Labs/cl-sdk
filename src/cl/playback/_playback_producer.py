"""
Playback producer subprocess for recording replay.

This module provides a subprocess that reads from a recording file and produces
waveform, spike, stim, and datastream data at real-time rate.
The data is written to shared memory for consumption by the WebSocket server.

Unlike DataProducer, this reads ALL events (spikes, stims, datastreams) directly
from the recording file rather than from command queues.
"""
from __future__ import annotations

import logging
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING

import numpy as np

from .._base_producer import BaseProducer, BaseProducerWorker
from .._data_buffer import (
    DataStreamEventRecord,
    StimRecord,
)

if TYPE_CHECKING:
    from pathlib import Path

_logger = logging.getLogger("cl.playback.producer")

DEFAULT_TICK_RATE_HZ = 5000  # 5kHz producer rate (5 frames/tick at 25kHz)

@dataclass
class SeekCommand:
    """Command to seek to a specific timestamp."""
    target_timestamp: int

@dataclass
class SetPausedCommand:
    """Command to set the pause state."""
    paused: bool

@dataclass
class ShutdownCommand:
    """Command to shut down the producer."""

def _producer_main(
    replay_file_path : str,
    channel_count    : int,
    frames_per_second: int,
    duration_frames  : int,
    start_timestamp  : int,
    tick_rate_hz     : int,
    command_queue    : Queue,
    name_prefix      : str,
) -> None:
    """
    Main function for the playback producer subprocess.

    This runs in a separate process and:
    1. Attaches to the shared memory buffer (created by main process)
    2. Reads frames, spikes, stims from the recording file
    3. Writes data to shared memory at tick_rate_hz
    4. Handles seek/pause commands from the queue

    Args:
        replay_file_path: Path to the H5 recording file
        channel_count: Number of channels
        frames_per_second: Sample rate
        duration_frames: Total frames in recording file
        start_timestamp: Starting timestamp (from recording attributes)
        tick_rate_hz: Producer loop rate in Hz
        command_queue: Queue for receiving seek/pause/shutdown commands
        name_prefix: Unique prefix for shared memory (from parent process)
    """
    try:
        producer = PlaybackProducerWorker(
            replay_file_path  = replay_file_path,
            channel_count     = channel_count,
            frames_per_second = frames_per_second,
            duration_frames   = duration_frames,
            start_timestamp   = start_timestamp,
            tick_rate_hz      = tick_rate_hz,
            command_queue     = command_queue,
            name_prefix       = name_prefix,
        )
        producer.run()
    except Exception as e:
        print(f"Playback producer subprocess failed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise

class PlaybackProducerWorker(BaseProducerWorker):
    """
    Worker class that runs in the playback producer subprocess.

    Extends BaseProducerWorker with playback-specific functionality:
    - Duration/start_timestamp tracking
    - Pause state with start-paused behavior
    - Seek functionality
    - Stim and datastream reading from recording file
    """

    def __init__(
        self,
        replay_file_path : str,
        channel_count    : int,
        frames_per_second: int,
        duration_frames  : int,
        start_timestamp  : int,
        tick_rate_hz     : int,
        command_queue    : Queue,
        name_prefix      : str,
    ):
        super().__init__(
            replay_file_path  = replay_file_path,
            channel_count     = channel_count,
            frames_per_second = frames_per_second,
            tick_rate_hz      = tick_rate_hz,
            command_queue     = command_queue,
            name_prefix       = name_prefix,
        )

        self._duration_frames   = duration_frames
        self._start_timestamp   = start_timestamp

        # Playback state
        self._current_timestamp = start_timestamp
        self._paused            = True  # Start paused

        # Additional replay data (opened in run())
        self._replay_stims        = None
        self._replay_data_streams = None

        # Pre-loaded timestamp arrays for efficient binary search
        self._stim_timestamps: np.ndarray | None = None
        self._datastream_indices: dict[str, np.ndarray] = {}

    def run(self) -> None:
        """Main playback producer loop with pause/seek support."""
        # Use base class utilities for setup
        BaseProducerWorker.set_process_priority()
        self.attach_buffer()
        self.open_replay_file()
        self.load_spike_timestamps()

        # Load additional playback-specific data
        self._load_stims_and_datastreams()
        self._load_stim_timestamps()
        self._load_datastream_indices()

        # Disable GC
        BaseProducerWorker.disable_gc()

        self._running = True

        _logger.info(
            "Playback producer started: %d frames/tick, duration=%d frames, paused=%s",
            self._frames_per_tick, self._duration_frames, self._paused
        )

        # Signal readiness
        assert self._buffer is not None
        self._buffer.producer_ready = True
        self._buffer.pause_flag     = True  # Start paused
        start_wall_ns               = time.perf_counter_ns()
        tick_count                  = 0

        while self._running:
            # Process commands first
            self._process_commands()

            # Check for shutdown flag
            if self._buffer.shutdown_flag:
                _logger.info("Producer received shutdown signal")
                break

            # If paused, sleep and continue
            if self._paused:
                time.sleep(0.01)  # 10ms pause check interval
                # Reset timing when unpaused
                start_wall_ns = time.perf_counter_ns()
                tick_count    = 0
                continue

            # Calculate timestamp range for this tick
            from_ts = self._current_timestamp
            to_ts   = from_ts + self._frames_per_tick

            # Clamp to recording bounds
            end_timestamp = self._start_timestamp + self._duration_frames
            if to_ts > end_timestamp:
                to_ts = end_timestamp
                if from_ts >= end_timestamp:
                    # Recording complete - pause at end
                    self._paused = True
                    self._buffer.pause_flag = True
                    _logger.info("Reached end of recording, pausing")
                    continue

            # Read frames from replay file
            frame_count = to_ts - from_ts
            frames = self._read_frames(from_ts, frame_count)

            # Read spikes (using base class utility with timestamp offset)
            relative_from = from_ts - self._start_timestamp
            relative_to   = to_ts - self._start_timestamp
            spikes = self.read_spikes_in_range(
                from_timestamp   = relative_from,
                to_timestamp     = relative_to,
                timestamp_offset = self._start_timestamp,
            )

            # Read stims from replay file
            stims = self._read_stims(from_ts, to_ts)

            # Read datastream events from replay file
            datastream_events = self._read_datastream_events(from_ts, to_ts)

            # Write to shared buffer
            self._buffer.write_frames(frames, from_ts)
            self.write_spikes_to_buffer(spikes)
            self.write_stims_to_buffer(stims)
            for ds_event in datastream_events:
                self._buffer.write_datastream_event(ds_event)

            # Explicit cleanup with GC disabled
            del frames, spikes, stims, datastream_events

            # Sleep until next tick
            tick_count += 1
            self.sleep_until_next_tick(start_wall_ns, tick_count)

            # Advance timestamp
            self._current_timestamp = to_ts

        # Cleanup using base class utility
        self.cleanup()

    def _process_commands(self) -> None:
        """Process commands from the main process."""
        while True:
            cmd = self.get_next_command()
            if cmd is None:
                break

            match cmd:
                case SeekCommand():
                    self._seek_to(cmd.target_timestamp)
                case SetPausedCommand():
                    self._paused = cmd.paused
                    if self._buffer:
                        self._buffer.pause_flag = cmd.paused
                    _logger.info("Playback %s", "paused" if cmd.paused else "resumed")
                case ShutdownCommand():
                    self._running = False

    def _seek_to(self, target_timestamp: int) -> None:
        """Seek to the specified timestamp."""
        # Clamp to recording bounds
        end_timestamp = self._start_timestamp + self._duration_frames
        target_timestamp = max(self._start_timestamp, min(target_timestamp, end_timestamp))

        _logger.info("Seeking to timestamp %d (from %d)", target_timestamp, self._current_timestamp)
        self._current_timestamp = target_timestamp

        # Reset the buffer state for the new position
        # This clears all ring buffer indices so new data can be written from the seek position
        if self._buffer:
            self._buffer.reset_to_timestamp(target_timestamp)

    def _load_stims_and_datastreams(self) -> None:
        """Load stim and datastream references from the replay file."""
        # Base class already opened the file and set _replay_file, _replay_samples, _replay_spikes
        assert self._replay_file is not None

        # Get stims (optional)
        self._replay_stims = (
            self._replay_file.root.stims
            if hasattr(self._replay_file.root, 'stims')
            else None
        )

        # Get data streams (optional)
        self._replay_data_streams = (
            self._replay_file.root.data_stream
            if hasattr(self._replay_file.root, 'data_stream')
            else None
        )

    def _load_stim_timestamps(self) -> None:
        """Load stim timestamps into memory for efficient binary search."""
        if self._replay_stims is None:
            self._stim_timestamps = np.array([], dtype=np.int64)
            return

        stim_count = len(self._replay_stims)
        self._stim_timestamps = np.zeros(stim_count, dtype=np.int64)

        for i in range(stim_count):
            self._stim_timestamps[i] = int(self._replay_stims[i]["timestamp"])

    def _load_datastream_indices(self) -> None:
        """Load datastream timestamp indices for efficient searching."""
        self._datastream_indices: dict[str, np.ndarray] = {}

        if self._replay_data_streams is None:
            return

        # Access _v_children is the PyTables way to iterate child groups
        for ds_name in self._replay_data_streams._v_children:  # noqa: SLF001
            ds_group = self._replay_data_streams[ds_name]
            if hasattr(ds_group, 'index'):
                # Load timestamps for this data stream
                index_table = ds_group.index
                timestamps = np.zeros(len(index_table), dtype=np.int64)
                for i, row in enumerate(index_table):
                    timestamps[i] = int(row["timestamp"])
                self._datastream_indices[ds_name] = timestamps

    def _read_frames(self, from_timestamp: int, frame_count: int) -> np.ndarray:
        """Read frames from the replay file."""
        if self._replay_samples is None:
            return np.zeros((frame_count, self._channel_count), dtype=np.int16)

        # Convert timestamp to file index (timestamps are relative to start)
        relative_ts = from_timestamp - self._start_timestamp
        start_idx   = relative_ts
        end_idx     = start_idx + frame_count

        # Clamp to file bounds
        start_idx = max(0, min(start_idx, self._duration_frames))
        end_idx   = max(0, min(end_idx, self._duration_frames))

        if start_idx >= end_idx:
            return np.zeros((frame_count, self._channel_count), dtype=np.int16)

        return np.array(self._replay_samples[start_idx:end_idx], dtype=np.int16)

    def _read_stims(self, from_timestamp: int, to_timestamp: int) -> list[StimRecord]:
        """Read stims from the replay file for the given timestamp range."""
        if self._replay_stims is None or self._stim_timestamps is None or len(self._stim_timestamps) == 0:
            return []

        # Timestamps in file are relative to recording start
        relative_from = from_timestamp - self._start_timestamp
        relative_to   = to_timestamp - self._start_timestamp

        # Binary search for range
        left_idx  = np.searchsorted(self._stim_timestamps, relative_from, side="left")
        right_idx = np.searchsorted(self._stim_timestamps, relative_to, side="left")

        result = []
        for i in range(left_idx, right_idx):
            stim = self._replay_stims[i]
            # Convert relative timestamp back to absolute
            absolute_ts = int(stim["timestamp"]) + self._start_timestamp
            result.append(StimRecord(
                timestamp = absolute_ts,
                channel   = int(stim["channel"])
            ))

        return result

    def _read_datastream_events(
        self,
        from_timestamp: int,
        to_timestamp  : int
    ) -> list[DataStreamEventRecord]:
        """Read datastream events from the replay file for the given timestamp range."""
        if self._replay_data_streams is None:
            return []

        # Timestamps in file are relative to recording start
        relative_from = from_timestamp - self._start_timestamp
        relative_to   = to_timestamp - self._start_timestamp

        result = []

        for ds_name, timestamps in self._datastream_indices.items():
            if len(timestamps) == 0:
                continue

            # Binary search for range
            left_idx  = np.searchsorted(timestamps, relative_from, side="left")
            right_idx = np.searchsorted(timestamps, relative_to, side="left")

            if left_idx >= right_idx:
                continue

            ds_group = self._replay_data_streams[ds_name]
            index_table = ds_group.index
            data_array  = ds_group.data

            for i in range(left_idx, right_idx):
                row = index_table[i]
                # Convert relative timestamp back to absolute
                absolute_ts = int(row["timestamp"]) + self._start_timestamp

                # Read data from heap
                start_idx = int(row["start_index"])
                end_idx   = int(row["end_index"])
                data = bytes(data_array[start_idx:end_idx])

                result.append(DataStreamEventRecord(
                    timestamp   = absolute_ts,
                    stream_name = ds_name,
                    data        = data
                ))

        # Sort by timestamp (since we're merging from multiple streams)
        result.sort(key=lambda x: x.timestamp)

        return result


class PlaybackProducer(BaseProducer):
    """
    Interface to the playback producer subprocess.

    Extends BaseProducer with playback-specific functionality:
    - Seek/pause commands
    - Playback state properties

    Usage:
        producer = PlaybackProducer(recording_file, ...)
        producer.start()
        # ... control via commands ...
        producer.stop()
    """

    def __init__(
        self,
        replay_file_path : str | Path,
        channel_count    : int,
        frames_per_second: int,
        duration_frames  : int,
        start_timestamp  : int,
        tick_rate_hz     : int = DEFAULT_TICK_RATE_HZ,
    ):
        super().__init__(
            replay_file_path  = replay_file_path,
            channel_count     = channel_count,
            frames_per_second = frames_per_second,
        )

        self._duration_frames   = duration_frames
        self._start_timestamp   = start_timestamp
        self._tick_rate_hz      = tick_rate_hz

    def _create_process(self) -> Process:
        """Create the playback producer subprocess."""
        return Process(
            target = _producer_main,
            args   = (
                self._replay_file_path,
                self._channel_count,
                self._frames_per_second,
                self._duration_frames,
                self._start_timestamp,
                self._tick_rate_hz,
                self._command_queue,
                self._name_prefix,
            ),
            daemon = True,
        )

    def _send_shutdown(self) -> None:
        """Send shutdown command to the subprocess."""
        self._command_queue.put(ShutdownCommand())

    def start(self, timeout: float = 15.0, start_timestamp: int = 0) -> None:  # noqa: ARG002
        """Start the producer subprocess."""
        # Use base class start with our start_timestamp (ignore parameter)
        super().start(timeout=timeout, start_timestamp=self._start_timestamp)

    def set_paused(self, paused: bool) -> None:
        """Set the pause state (override to send command to subprocess)."""
        self._command_queue.put(SetPausedCommand(paused=paused))
        # Also use base class method to update buffer
        super().set_paused(paused)

    def seek_to(self, timestamp: int) -> None:
        """Seek to the specified timestamp."""
        self._command_queue.put(SeekCommand(target_timestamp=timestamp))

    def seek_relative(self, delta_frames: int) -> None:
        """Seek relative to current position by delta_frames."""
        # Use base class current_timestamp property
        current_ts = self.current_timestamp
        self.seek_to(current_ts + delta_frames)

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
