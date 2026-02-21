"""
Data producer subprocess for mock API.

This module provides a subprocess that reads from a replay file and produces
waveform, spike, and stim data at a configurable rate (default 5kHz).
The data is written to shared memory for consumption by neurons.read(),
the closed loop, and the WebSocket server.

The producer respects debugger pause flags and can run in accelerated time mode.
"""
from __future__ import annotations

import itertools
import logging
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING, override

import numpy as np

from ._base_producer import BaseProducer, BaseProducerWorker
from ._data_buffer import SPIKE_SAMPLES_TOTAL, SpikeRecord, StimRecord
from ._stim_queue import ChannelStimQueue

if TYPE_CHECKING:
    from pathlib import Path

_logger = logging.getLogger("cl.producer")

DEFAULT_TICK_RATE_HZ = 5000         # 5kHz producer rate (2ms ticks, 5 frames/tick at 25kHz)
STALE_THRESHOLD_NS   = 200_000_000  # 200ms

@dataclass(order=True)
class StimCommand:
    """Command to queue a stim."""
    timestamp    : int
    channel      : int = field(compare=False)
    end_timestamp: int = field(compare=False)  # When the channel becomes available again

@dataclass
class InterruptCommand:
    """Command to interrupt stims on channels."""
    timestamp: int
    channels : list[int]

@dataclass
class SyncCommand:
    """Command to sync channels."""
    timestamp           : int
    channels            : list[int]
    wait_for_frame_start: bool

@dataclass
class ShutdownCommand:
    """Command to shut down the producer."""

@dataclass
class SetPauseFlagCommand:
    """Command to set the pause flag."""
    paused: bool

def _producer_main(
    replay_file_path   : str,
    start_timestamp    : int,
    replay_start_offset: int,
    channel_count      : int,
    frames_per_second  : int,
    duration_frames    : int,
    tick_rate_hz       : int,
    accelerated_time   : bool,
    command_queue      : Queue,
    name_prefix        : str,
) -> None:
    """
    Main function for the producer subprocess.

    This runs in a separate process and:
    1. Attaches to the shared memory buffer (created by main process)
    2. Reads frames from the replay file
    3. Generates spikes from replay data
    4. Processes stim commands from the queue
    5. Writes data to shared memory at tick_rate_hz

    Args:
        replay_file_path: Path to the H5 replay file
        start_timestamp: Initial timestamp value
        replay_start_offset: Offset into the replay file
        channel_count: Number of channels
        frames_per_second: Sample rate
        duration_frames: Total frames in replay file
        tick_rate_hz: Producer loop rate in Hz
        accelerated_time: If True, run as fast as possible
        command_queue: Queue for receiving stim/control commands
        name_prefix: Unique prefix for shared memory (from parent process)
    """
    try:
        producer = DataProducerWorker(
            replay_file_path    = replay_file_path,
            start_timestamp     = start_timestamp,
            replay_start_offset = replay_start_offset,
            channel_count       = channel_count,
            frames_per_second   = frames_per_second,
            duration_frames     = duration_frames,
            tick_rate_hz        = tick_rate_hz,
            accelerated_time    = accelerated_time,
            command_queue       = command_queue,
            name_prefix         = name_prefix,
        )
        producer.run()
    except Exception as e:
        # Print directly to stderr in case logging isn't working
        import traceback
        print(f"Producer subprocess failed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise

class DataProducerWorker(BaseProducerWorker):
    """
    Worker class that runs in the producer subprocess.

    Extends BaseProducerWorker with data producer specific functionality:
    - Stim queue management
    - Accelerated time mode
    - Replay offset and wrapping
    """

    def __init__(
        self,
        replay_file_path   : str,
        start_timestamp    : int,
        replay_start_offset: int,
        channel_count      : int,
        frames_per_second  : int,
        duration_frames    : int,
        tick_rate_hz       : int,
        accelerated_time   : bool,
        command_queue      : Queue,
        name_prefix        : str,
    ):
        super().__init__(
            replay_file_path  = replay_file_path,
            channel_count     = channel_count,
            frames_per_second = frames_per_second,
            tick_rate_hz      = tick_rate_hz,
            command_queue     = command_queue,
            name_prefix       = name_prefix,
        )

        # Data producer specific state
        self._start_timestamp     = start_timestamp
        self._replay_start_offset = replay_start_offset
        self._duration_frames     = duration_frames
        self._accelerated_time    = accelerated_time

        # Override frames per tick for accelerated mode
        if accelerated_time:
            # Produce in batches of ~1ms worth of frames for efficiency
            self._frames_per_tick = max(25, frames_per_second // tick_rate_hz)

        # Data producer state
        self._current_timestamp = start_timestamp

        # Stim queue: channel-indexed for efficient interrupt handling
        self._stim_queue: ChannelStimQueue[StimCommand] = ChannelStimQueue()

        # Track when each channel becomes available
        self._channel_available_from = np.full(channel_count, fill_value=start_timestamp, dtype=np.int64)

    def run(self) -> None:
        """Main producer loop."""

        # Use base class utilities for setup
        BaseProducerWorker.set_process_priority()
        self.attach_buffer()
        self.open_replay_file()
        self.load_spike_timestamps()
        BaseProducerWorker.disable_gc()

        self._running = True

        _logger.info("Producer started: %d frames/tick, accelerated=%s", self._frames_per_tick, self._accelerated_time)

        assert self._buffer is not None, "Data buffer not initialized"

        # Signal readiness and start timing together - consumer can start waiting
        self._buffer.producer_ready = True
        start_wall_ns               = time.perf_counter_ns()
        tick_count                  = 0

        while self._running:
            # Check for pause flag
            if self._buffer.pause_flag:
                time.sleep(0.01)  # 10ms pause check interval
                # Reset timing when unpaused
                start_wall_ns = time.perf_counter_ns()
                tick_count = 0
                continue

            # Check for shutdown flag
            if self._buffer.shutdown_flag:
                _logger.info("Producer received shutdown signal")
                break

            self._process_commands()

            # In accelerated mode, wait for consumer to request more data
            if self._accelerated_time:
                requested_ts = self._buffer.requested_timestamp
                # Producer should wait while it has caught up to the requested timestamp
                # This ensures the loop controls time advancement
                while self._current_timestamp >= requested_ts:
                    # Check for debugger pause via heartbeat
                    if self._check_heartbeat_stale():
                        time.sleep(0.01)  # Pause while debugger is active
                        continue

                    time.sleep(0.0001)  # 100μs sleep to avoid busy waiting
                    requested_ts = self._buffer.requested_timestamp
                    # Check for shutdown while waiting
                    if self._buffer.shutdown_flag:
                        _logger.info("Producer received shutdown signal while waiting")
                        self._running = False
                        break

                # If shutdown was triggered, exit immediately
                if not self._running:
                    break

                # Process commands again after waking up to catch any last-minute stims
                self._process_commands()

            elif tick_count % 10 == 0:  # Check every 10 ticks instead of every tick
                # Check for debugger pause via heartbeat (less frequently in real-time)
                while self._check_heartbeat_stale():
                    time.sleep(0.01)  # Pause while debugger is active

            # Calculate timestamp range for this tick
            from_ts = self._current_timestamp
            to_ts   = from_ts + self._frames_per_tick

            # Read frames from replay file
            frames = self._read_replay_frames(from_ts, self._frames_per_tick)

            # Read spikes from replay file
            spikes = self._read_replay_spikes(from_ts, self._frames_per_tick)

            # Process stims that should fire in this range
            stims = self._process_stims(from_ts, to_ts)

            # Write to shared buffer
            self._buffer.write_frames(frames, from_ts)
            self.write_spikes_to_buffer(spikes)
            self.write_stims_to_buffer(stims)

            # Explicit cleanup with GC disabled
            del frames, spikes, stims

            # Sleep until next tick (unless in accelerated mode)
            if not self._accelerated_time:
                self.sleep_until_next_tick(start_wall_ns, tick_count)

            # Advance timestamp
            self._current_timestamp = to_ts
            tick_count += 1

        # Cleanup using base class utility
        self.cleanup()

    def _check_heartbeat_stale(self) -> bool:
        """
        Check if the main process heartbeat has gone stale (indicating debugger pause).

        Returns True if the heartbeat hasn't been updated in over 200ms, suggesting
        the main process is paused at a breakpoint.
        """
        if self._buffer is None:
            return False

        heartbeat_ns = self._buffer.main_process_heartbeat_ns
        if heartbeat_ns == 0:
            return False  # Not yet initialized

        current_ns = time.perf_counter_ns()
        elapsed_ns = current_ns - heartbeat_ns

        # If heartbeat hasn't updated in 200ms, consider it stale
        return elapsed_ns > STALE_THRESHOLD_NS

    def _read_replay_frames(self, from_timestamp: int, frame_count: int) -> np.ndarray:
        """
        Read frames from the replay file, handling wrap-around.

        Returns array of shape (frame_count, channel_count) with int16 values.
        """
        # Convert to replay file indices
        elapsed_frames = from_timestamp - self._start_timestamp + self._replay_start_offset
        start_idx      = elapsed_frames % self._duration_frames
        end_idx        = start_idx + frame_count

        assert self._replay_samples is not None

        if end_idx <= self._duration_frames:
            # No wrap
            return np.array(self._replay_samples[start_idx:end_idx], dtype=np.int16)
        else:
            # Wrap around
            first_part          = self._duration_frames - start_idx
            result              = np.empty((frame_count, self._channel_count), dtype=np.int16)
            result[:first_part] = self._replay_samples[start_idx:]
            result[first_part:] = self._replay_samples[:end_idx - self._duration_frames]
            return result

    def _read_replay_spikes(
        self,
        from_timestamp: int,
        frame_count   : int,
    ) -> list[SpikeRecord]:
        """
        Read spikes from the replay file for the given timestamp range.

        Also extracts spike waveform samples from the frames array.
        """
        if self._replay_spikes is None or self._spike_timestamps is None:
            return []

        result = []

        offset_from_ts     = from_timestamp + self._replay_start_offset
        loop_count, mod_ts = divmod(offset_from_ts, self._duration_frames)

        left_ts  = mod_ts
        right_ts = (offset_from_ts + frame_count) % self._duration_frames

        # Binary search for range of spikes
        # Spike timestamps are in file-relative coordinates (same as left_ts before wrapping)
        # We need to search in the modular arithmetic space [left_ts, right_ts) with wrapping

        left_idx  = np.searchsorted(self._spike_timestamps, left_ts, side="left")
        right_idx = np.searchsorted(self._spike_timestamps, right_ts, side="left")

        if right_ts > left_ts:
            # No wrap-around: simple case
            indices = range(left_idx, right_idx)
        else:
            # Wrap-around case: [left_ts, duration_frames) + [0, right_ts)
            indices = itertools.chain(range(left_idx, len(self._spike_timestamps)), range(right_idx))

        # Fetch spike records only for those in the range
        for i in indices:
            replay_spike = self._replay_spikes[i]
            spike_ts     = int(replay_spike["timestamp"]) + (loop_count * self._duration_frames) - self._replay_start_offset

            # Get spike samples from the frames if available
            spike_samples = (
                replay_spike["samples"]
                if replay_spike.dtype.names and "samples" in replay_spike.dtype.names
                else np.zeros(SPIKE_SAMPLES_TOTAL, dtype=np.float32)
            )

            result.append(
                SpikeRecord(
                    timestamp           = spike_ts,
                    channel             = int(replay_spike["channel"]),
                    channel_mean_sample = float(np.mean(spike_samples)),
                    samples             = np.asarray(spike_samples, dtype=np.float32),
                )
            )

        return result

    def _process_commands(self) -> None:
        """Process any pending commands from the queue."""
        while True:
            cmd = self.get_next_command()
            if cmd is None:
                break

            if isinstance(cmd, StimCommand):
                # Add to stim queue using channel-indexed structure
                self._stim_queue.put(
                    timestamp = cmd.timestamp,
                    channel   = cmd.channel,
                    payload   = cmd,
                )
                # Update channel availability
                self._channel_available_from[cmd.channel] = cmd.end_timestamp

            elif isinstance(cmd, InterruptCommand):
                # Before removing stims, get the last kept stim's end timestamp for each channel
                # This preserves channel availability info needed for sync()
                for ch in cmd.channels:
                    last_kept = self._stim_queue.get_last_entry_before(ch, cmd.timestamp)
                    if last_kept is not None:
                        _, stim_cmd = last_kept
                        # Use the end timestamp of the last kept stim
                        self._channel_available_from[ch] = stim_cmd.end_timestamp
                    else:
                        # No stims kept, channel is available at interrupt time
                        self._channel_available_from[ch] = cmd.timestamp

                # Use efficient channel-indexed removal: O(c * log k) instead of O(n) drain-rebuild
                self._stim_queue.interrupt_channels(cmd.channels, cmd.timestamp)

            elif isinstance(cmd, ShutdownCommand):
                self._running = False

            elif isinstance(cmd, SetPauseFlagCommand):
                if self._buffer:
                    self._buffer.pause_flag = cmd.paused

    def _process_stims(self, from_timestamp: int, to_timestamp: int) -> list[StimRecord]:
        """Process stims that should fire in the given timestamp range.

        Stims scheduled slightly in the past (within tolerance) are still delivered
        at from_timestamp to account for IPC latency between main process and producer.

        In accelerated mode, this also processes stims from earlier timestamps that
        may have been queued late (after the producer already advanced past them).
        """
        result = []

        # Tolerance for late stims:
        # - In real-time mode: accounts for IPC latency (~1ms producer tick interval)
        # - In accelerated mode: producer runs ahead by up to jitter tolerance,
        #   so we need larger tolerance to accept stims scheduled during loop ticks
        late_tolerance_frames = max(self._frames_per_tick, 200) if self._accelerated_time else 50

        # Pop all stims up to to_timestamp using efficient batch operation
        popped_stims = self._stim_queue.pop_until(to_timestamp)

        for stim_ts, stim_channel, _stim_cmd in popped_stims:
            # In accelerated mode, accept all stims (they may have been queued late)
            if self._accelerated_time:
                actual_ts = stim_ts  # Keep exact timestamp
            else:
                # Real-time mode: check if stim is too late
                if stim_ts < from_timestamp - late_tolerance_frames:
                    _logger.warning("Skipped stim at %d (too late by %d frames)", stim_ts, from_timestamp - stim_ts)
                    continue
                # Adjust late stims to current time
                actual_ts = max(stim_ts, from_timestamp)
                if stim_ts < from_timestamp:
                    _logger.debug("Stim at %d delivered late at %d on channel %d", stim_ts, actual_ts, stim_channel)
                else:
                    _logger.debug("Stim at %d on channel %d", stim_ts, stim_channel)

            result.append(
                StimRecord(
                    timestamp = actual_ts,
                    channel   = stim_channel,
                )
            )

        return result

class DataProducer(BaseProducer):
    """
    Interface to the data producer subprocess.

    Extends BaseProducer with data producer specific functionality:
    - Stim command queuing
    - Channel availability tracking
    - Accelerated time mode
    - Debugger pause flag

    Usage:
        producer = DataProducer(replay_file, ...)
        producer.start()
        # ... use neurons.read(), etc ...
        producer.stop()
    """

    def __init__(
        self,
        replay_file_path   : str | Path,
        start_timestamp    : int,
        replay_start_offset: int,
        channel_count      : int,
        frames_per_second  : int,
        duration_frames    : int,
        tick_rate_hz       : int  = DEFAULT_TICK_RATE_HZ,
        accelerated_time   : bool = False,
    ):
        """
        Initialize the data producer.

        Args:
            replay_file_path   : Path to the H5 replay file
            start_timestamp    : Initial timestamp value
            replay_start_offset: Offset into the replay file
            channel_count      : Number of channels
            frames_per_second  : Sample rate
            duration_frames    : Total frames in replay file
            tick_rate_hz       : Producer loop rate in Hz (default 5000)
            accelerated_time   : If True, run as fast as possible
        """
        super().__init__(
            replay_file_path  = replay_file_path,
            channel_count     = channel_count,
            frames_per_second = frames_per_second,
        )

        # Data producer specific state
        self._start_timestamp     = start_timestamp
        self._replay_start_offset = replay_start_offset
        self._duration_frames     = duration_frames
        self._tick_rate_hz        = tick_rate_hz
        self._accelerated_time    = accelerated_time

        # Track channel availability (mirror of producer state)
        self._channel_available_from = np.full(channel_count, fill_value=start_timestamp, dtype=np.int64)

        # Stim frequency bin duration for burst calculations
        self._stim_frequency_bin_duration_us = 20
        self._frame_duration_us = int(1_000_000 / frames_per_second)

    def _create_process(self) -> Process:
        """Create the data producer subprocess."""
        return Process(
            target = _producer_main,
            args   = (
                self._replay_file_path,
                self._start_timestamp,
                self._replay_start_offset,
                self._channel_count,
                self._frames_per_second,
                self._duration_frames,
                self._tick_rate_hz,
                self._accelerated_time,
                self._command_queue,
                self._name_prefix,
            ),
            daemon = True,
            name   = "cl-data-producer",
        )

    def _send_shutdown(self) -> None:
        """Send shutdown command to the subprocess."""
        self._command_queue.put(ShutdownCommand())

    @property
    def is_running(self) -> bool:
        """Check if the producer is running."""
        return self.is_started and self._process is not None and self._process.is_alive()

    @override
    def start(self, timeout: float = 15.0, start_timestamp: int = 0) -> None:
        """Start the producer subprocess with data producer specific start_timestamp."""
        # Use base class start with our start_timestamp (ignore parameter)
        super().start(timeout=timeout, start_timestamp=self._start_timestamp)

    def queue_stim(
        self,
        timestamp    : int,
        channel      : int,
        end_timestamp: int,
    ) -> None:
        """
        Queue a stim to be delivered at the specified timestamp.

        Args:
            timestamp: When to deliver the stim
            channel: Channel to stim
            end_timestamp: When the channel becomes available again
        """
        if self._command_queue is None:
            raise RuntimeError("Producer not running")

        self._command_queue.put(StimCommand(
            timestamp     = timestamp,
            channel       = channel,
            end_timestamp = end_timestamp,
        ))

        # Update local tracking
        self._channel_available_from[channel] = end_timestamp

    def interrupt_channels(self, timestamp: int, channels: list[int]) -> None:
        """
        Interrupt stims on the specified channels.

        Args:
            timestamp: Current timestamp
            channels: Channels to interrupt
        """
        if self._command_queue is None:
            raise RuntimeError("Producer not running")

        self._command_queue.put(
            InterruptCommand(
                timestamp = timestamp,
                channels  = channels,
            )
        )

        # Update local tracking
        for ch in channels:
            self._channel_available_from[ch] = timestamp

    def get_channel_available_from(self, channel: int) -> int:
        """Get the timestamp when a channel becomes available."""
        return int(self._channel_available_from[channel])

    def set_channel_available_from(self, channel: int, timestamp: int) -> None:
        """Set when a channel becomes available (for local tracking)."""
        self._channel_available_from[channel] = timestamp
