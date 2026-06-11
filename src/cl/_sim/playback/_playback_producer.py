"""
Playback producer subprocess for recording replay.

This module provides a subprocess that reads from a recording file and produces
waveform, spike, stim, and datastream data at real-time rate.
The data is written to shared memory for consumption by the WebSocket server.

Unlike DataProducer, this reads ALL events (spikes, stims, datastreams) directly
from the recording file rather than from command queues.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .._base_producer import BaseProducer, BaseProducerWorker
from .._data_buffer import (
    DataStreamEventRecord,
    SpikeRecord,
    StimRecord,
)
from .._subprocess import PopenProcess

if TYPE_CHECKING:
    import tables

_logger = logging.getLogger("cl.playback.producer")

DEFAULT_TICK_RATE_HZ = 5000  # 5kHz producer rate (5 frames/tick at 25kHz)

def _producer_main(
    replay_file_path : str,
    channel_count    : int,
    frames_per_second: int,
    duration_frames  : int,
    start_timestamp  : int,
    tick_rate_hz     : int,
    name_prefix      : str,
) -> None:
    producer = PlaybackProducerWorker(
        replay_file_path  = replay_file_path,
        channel_count     = channel_count,
        frames_per_second = frames_per_second,
        duration_frames   = duration_frames,
        start_timestamp   = start_timestamp,
        tick_rate_hz      = tick_rate_hz,
        name_prefix       = name_prefix,
    )
    BaseProducerWorker.run_in_subprocess(producer, "Playback producer")

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
        name_prefix      : str,
    ):
        super().__init__(
            channel_count     = channel_count,
            frames_per_second = frames_per_second,
            tick_rate_hz      = tick_rate_hz,
            name_prefix       = name_prefix,
            start_timestamp   = start_timestamp,
            duration_frames   = duration_frames,
        )
        self._replay_file_path = replay_file_path

        # Playback state
        self._paused             = True  # Start paused
        self._speed              = 1.0   # Playback speed multiplier
        self._last_seek_sequence = 0

        # Replay file handle and data references (opened in run())
        from cl.util import RecordingView
        self._recording_view     : RecordingView | None = None
        self._replay_file        : tables.File   | None = None
        self._replay_stims                              = None
        self._replay_data_streams                       = None

        # Pre-loaded timestamp arrays for efficient binary search
        self._spike_timestamps  : np.ndarray | None     = None
        self._stim_timestamps   : np.ndarray | None     = None
        self._datastream_indices: dict[str, np.ndarray] = {}

    def run(self) -> None:
        """Main playback producer loop with pause/seek support."""
        # Use base class utilities for setup
        BaseProducerWorker.set_process_priority()
        self.attach_buffer()
        self._open_replay_file()
        self._load_spike_timestamps()

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
        self._buffer.producer_ready  = True
        self._buffer.pause_flag      = True  # Start paused
        self._buffer.playback_speed  = self._speed
        start_wall_ns                = time.perf_counter_ns()
        tick_count                   = 0

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
            scaled_frames = max(1, int(self._frames_per_tick * self._speed))
            to_ts   = from_ts + scaled_frames

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

            # Read spikes (using helper with timestamp offset)
            relative_from = from_ts - self._start_timestamp
            relative_to   = to_ts - self._start_timestamp
            spikes        = self._read_spikes_in_range(
                from_timestamp   = relative_from,
                to_timestamp     = relative_to,
                timestamp_offset = self._start_timestamp,
            )

            # Read stims from replay file
            stims = self._read_stims(from_ts, to_ts)

            # Read datastream events from replay file
            datastream_events = self._read_datastream_events(from_ts, to_ts)

            # Write to shared buffer
            # Spikes, stims, and datastream events are written before frames so that
            # write_timestamp only advances after all associated data for this tick
            # is committed. This ensures wait_for_timestamp() returns only when all
            # data is visible.
            self.write_spikes_to_buffer(spikes)
            self.write_stims_to_buffer(stims)
            for ds_event in datastream_events:
                self._buffer.write_datastream_event(ds_event)
            self._buffer.stim_write_timestamp = to_ts
            self._buffer.write_frames(frames, from_ts)

            # Explicit cleanup with GC disabled
            del frames, spikes, stims, datastream_events

            # Sleep until next tick (then advance timestamp)
            self.sleep_until_next_tick(start_wall_ns, tick_count)
            self._current_timestamp = to_ts
            tick_count += 1

        # Cleanup using base class utility
        self.cleanup()

    def _process_commands(self) -> None:
        """Handle playback controls published through shared memory."""
        super()._process_commands()
        if self._buffer is None:
            return

        seek_sequence = self._buffer.playback_seek_sequence
        if seek_sequence != self._last_seek_sequence:
            self._last_seek_sequence = seek_sequence
            if seek_sequence != 0:
                self._seek_to(self._buffer.playback_seek_timestamp)

        paused = self._buffer.pause_flag
        if paused != self._paused:
            self._paused = paused
            _logger.info("Playback %s", "paused" if paused else "resumed")

        speed = self._buffer.playback_speed
        if speed != self._speed:
            self._speed = max(0.25, min(4.0, speed))
            _logger.info("Playback speed set to %.2fx", self._speed)

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

    def _open_replay_file(self) -> None:
        """Open the replay H5 file and set up samples/spikes references."""
        from cl.util import RecordingView
        self._recording_view = RecordingView(self._replay_file_path)
        self._replay_file    = self._recording_view.file

    def _load_spike_timestamps(self) -> None:
        """Load spike timestamps into memory for efficient binary search."""
        if self._recording_view is None or self._recording_view.spikes is None:
            self._spike_timestamps = np.array([], dtype=np.int64)
            return

        spikes = self._recording_view.spikes
        try:
            # Vectorized column read - much faster for PyTables Tables
            self._spike_timestamps = spikes.col('timestamp').astype(np.int64)
        except (AttributeError, KeyError):
            # Fallback for non-Table node types
            spike_count = len(spikes)
            self._spike_timestamps = np.zeros(spike_count, dtype=np.int64)
            for i in range(spike_count):
                self._spike_timestamps[i] = int(spikes[i]["timestamp"])

    def _read_spikes_in_range(
        self,
        from_timestamp  : int,
        to_timestamp    : int,
        timestamp_offset: int = 0,
    ) -> list[SpikeRecord]:
        """Read spikes from the replay file within a timestamp range."""
        if (
            self._recording_view is None or
            self._recording_view.spikes is None or
            self._spike_timestamps is None or
            len(self._spike_timestamps) == 0
        ):
            return []

        left_idx  = np.searchsorted(self._spike_timestamps, from_timestamp, side="left")
        right_idx = np.searchsorted(self._spike_timestamps, to_timestamp,   side="left")

        result = []
        for i in range(left_idx, right_idx):
            spike = self._recording_view.spikes[i]
            result.append(SpikeRecord(
                timestamp           = int(spike["timestamp"]) + timestamp_offset,
                channel             = int(spike["channel"]),
                channel_mean_sample = 0.0,  # Not typically stored in recording
                samples             = np.array(spike["samples"], dtype=np.float32)
            ))

        return result

    def cleanup(self) -> None:
        if self._recording_view is not None:
            self._recording_view.close()
            self._recording_view = None
        self._replay_file = None  # Already closed by recording_view.close() above
        super().cleanup()

    def _load_stims_and_datastreams(self) -> None:
        """Load stim and datastream references from the replay file."""
        # _open_replay_file already set _replay_file
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

        try:
            # Vectorized column read - much faster for PyTables Tables
            self._stim_timestamps = self._replay_stims.col('timestamp').astype(np.int64)
        except (AttributeError, KeyError):
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
        for ds_name in self._replay_data_streams._v_children:
            ds_group = self._replay_data_streams[ds_name]
            if hasattr(ds_group, 'index'):
                index_table = ds_group.index
                try:
                    # Vectorized column read - much faster for PyTables Tables
                    timestamps = index_table.col('timestamp').astype(np.int64)
                except (AttributeError, KeyError):
                    timestamps = np.zeros(len(index_table), dtype=np.int64)
                    for i, row in enumerate(index_table):
                        timestamps[i] = int(row["timestamp"])
                self._datastream_indices[ds_name] = timestamps

    def _read_frames(self, from_timestamp: int, frame_count: int) -> np.ndarray:
        """Read frames from the replay file."""
        if self._recording_view is None or self._recording_view.samples is None:
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

        return np.array(self._recording_view.samples[start_idx:end_idx], dtype=np.int16)

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
                timestamp          = absolute_ts,
                intended_timestamp = absolute_ts,          # For playback, intended timestamp is always the same as actual
                channel            = int(stim["channel"])
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
            channel_count     = channel_count,
            frames_per_second = frames_per_second,
            start_timestamp   = start_timestamp,
            duration_frames   = duration_frames,
            tick_rate_hz      = tick_rate_hz,
        )
        self._replay_file_path = str(replay_file_path) if isinstance(replay_file_path, Path) else replay_file_path

    def _create_process(self) -> PopenProcess:
        """Create the playback producer subprocess."""
        assert self._name_prefix is not None
        return PopenProcess(
            target       = "cl._sim.playback._playback_producer:run_from_config",
            process_name = "cl-playback-producer",
            config       = {
                "replay_file_path" : self._replay_file_path,
                "channel_count"    : self._channel_count,
                "frames_per_second": self._frames_per_second,
                "duration_frames"  : self._duration_frames,
                "start_timestamp"  : self._start_timestamp,
                "tick_rate_hz"     : self._tick_rate_hz,
                "name_prefix"      : self._name_prefix,
            },
        )

    def set_paused(self, paused: bool) -> None:
        """Set the pause state (override to send command to subprocess)."""
        super().set_paused(paused)

    def seek_to(self, timestamp: int) -> None:
        """Seek to the specified timestamp."""
        if self._buffer:
            self._buffer.request_playback_seek(timestamp)

    def seek_relative(self, delta_frames: int) -> None:
        """Seek relative to current position by delta_frames."""
        current_ts = self.current_timestamp
        self.seek_to(current_ts + delta_frames)

    def set_speed(self, speed: float) -> None:
        """Set the playback speed (0.25 to 4.0)."""
        if self._buffer:
            self._buffer.playback_speed = speed

    @property
    def playback_speed(self) -> float:
        """Get the current playback speed from the buffer."""
        if self._buffer:
            return self._buffer.playback_speed
        return 1.0

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CL SDK playback producer subprocess")
    parser.add_argument("--process-name", default="cl-playback-producer")
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()
    config = json.loads(args.config_json)
    run_from_config(config)

def run_from_config(config: dict) -> None:
    _producer_main(**config)

if __name__ == "__main__":
    main()
