"""
Data producer subprocess for the mock API.

The producer reads from a configured simulator data source and writes waveform
and spike data to shared memory for neurons.read(), the closed loop, recording,
and the WebSocket server.

Stim data is written directly to the shared buffer by the main process
(neurons.py) for exact timing -- no IPC delay.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._base_producer import BaseProducer, BaseProducerWorker
from ._data_buffer import SPIKE_SAMPLES_TOTAL, BufferHeader, SpikeRecord
from ..sim._data_source import (
    DataSourceBatch,
    DataSourceSpike,
    DataSourceStim,
    SimulatorDataSource,
    create_data_source_from_config,
    validate_metadata,
)
from ._subprocess import PopenProcess

_logger = logging.getLogger("cl.producer")

DEFAULT_TICK_RATE_HZ                     = 5000         # 5kHz producer rate (2ms ticks, 5 frames/tick at 25kHz)
DEFAULT_ACCELERATED_MAX_FRAMES_PER_BATCH = 1024
STALE_THRESHOLD_NS                       = 200_000_000  # 200ms
ACCELERATED_WAIT_SPIN_POLLS              = 64

def _producer_main(
    data_source_config: dict[str, Any],
    start_timestamp   : int,
    channel_count     : int,
    frames_per_second : int,
    duration_frames   : int,
    tick_rate_hz      : int,
    accelerated_time  : bool,
    name_prefix       : str,
) -> None:
    producer = DataProducerWorker(
        data_source_config = data_source_config,
        start_timestamp    = start_timestamp,
        channel_count      = channel_count,
        frames_per_second  = frames_per_second,
        duration_frames    = duration_frames,
        tick_rate_hz       = tick_rate_hz,
        accelerated_time   = accelerated_time,
        name_prefix        = name_prefix,
    )
    BaseProducerWorker.run_in_subprocess(producer, "Data producer")

class DataProducerWorker(BaseProducerWorker):
    """
    Worker class that runs in the producer subprocess.

    Extends BaseProducerWorker with:
    - Configurable simulator data sources
    - Accelerated time mode
    - Debugger pause detection
    """

    def __init__(
        self,
        data_source_config: Mapping[str, Any],
        start_timestamp   : int,
        channel_count     : int,
        frames_per_second : int,
        duration_frames   : int,
        tick_rate_hz      : int,
        accelerated_time  : bool,
        name_prefix       : str,
    ):
        super().__init__(
            channel_count     = channel_count,
            frames_per_second = frames_per_second,
            tick_rate_hz      = tick_rate_hz,
            name_prefix       = name_prefix,
            start_timestamp   = start_timestamp,
            duration_frames   = duration_frames,
        )

        self._data_source: SimulatorDataSource | None = None
        self._data_source_config                      = dict(data_source_config)
        self._accelerated_time                        = accelerated_time
        self._last_stim_read_ts                       = int(start_timestamp)
        self._accelerated_max_frames_per_batch        = DEFAULT_ACCELERATED_MAX_FRAMES_PER_BATCH

        if accelerated_time:
            self._accelerated_max_frames_per_batch = max(1, self._accelerated_max_frames_per_batch)

    def run(self) -> None:
        """Main producer loop."""

        BaseProducerWorker.set_process_priority()
        self.attach_buffer()
        self.open_data_source()
        BaseProducerWorker.disable_gc()

        self._running = True

        _logger.info("Producer started: %d frames/tick, accelerated=%s", self._frames_per_tick, self._accelerated_time)

        assert self._buffer is not None, "Data buffer not initialized"

        self._buffer.producer_ready = True
        start_wall_ns               = time.perf_counter_ns()
        tick_count                  = 0

        # Bind hot header accessors once. The accelerated spin loop reads these
        # tens of times per tick; going through the Python @property + numpy
        # scalar machinery on every read is the dominant per-tick overhead in
        # accelerated mode. The flag bytes are read directly off the memoryview
        # and requested_timestamp off its pre-built numpy view.
        buf            = self._buffer
        header_buf     = buf._shm_header.buf
        assert header_buf is not None, "Shared memory header buffer not initialized"
        requested_view = buf._hdr_requested_ts
        pause_off      = BufferHeader.PAUSE_FLAG_OFFSET
        shutdown_off   = BufferHeader.SHUTDOWN_FLAG_OFFSET

        while self._running:
            if header_buf[pause_off]:
                time.sleep(0.01)
                start_wall_ns = time.perf_counter_ns()
                tick_count = 0
                continue

            if header_buf[shutdown_off]:
                _logger.info("Producer received shutdown signal")
                break

            self._process_commands()

            if self._accelerated_time:
                # In accelerated mode the main process drives requested_timestamp
                # directly. Spin briefly on active requests to reduce scheduler
                # handoff cost, then yield so idle accel mode does not burn a core.
                requested_ts = int(requested_view[0])
                spin_count   = 0
                while self._current_timestamp >= requested_ts:
                    if requested_ts > 0 and spin_count < ACCELERATED_WAIT_SPIN_POLLS:
                        spin_count += 1
                    else:
                        spin_count = 0
                        time.sleep(0)
                    requested_ts = int(requested_view[0])
                    if header_buf[shutdown_off]:
                        _logger.info("Producer received shutdown signal while waiting")
                        self._running = False
                        break

                if not self._running:
                    break

                self._process_commands()

            elif tick_count % 10 == 0:
                while self._check_heartbeat_stale():
                    time.sleep(0.01)

            from_ts     = self._current_timestamp
            frame_count = self._frames_per_tick
            if self._accelerated_time:
                frame_count = min(
                    self._accelerated_max_frames_per_batch,
                    requested_ts - from_ts,
                )
                if frame_count <= 0:
                    continue
            to_ts = from_ts + frame_count

            self._deliver_stims_to_source(to_ts)
            batch  = self._read_source_batch(from_ts, frame_count)
            frames = self._coerce_frames(batch.frames, frame_count)
            spikes = self._coerce_spikes(batch.spikes, from_ts, to_ts)

            self.write_spikes_to_buffer(spikes)
            self._buffer.write_frames_validated(frames, from_ts)

            if not self._accelerated_time:
                self.sleep_until_next_tick(start_wall_ns, tick_count)

            self._current_timestamp = to_ts
            tick_count += 1

        self.cleanup()

    def open_data_source(self) -> None:
        """Instantiate and open the configured data source."""
        self._data_source = create_data_source_from_config(self._data_source_config)
        metadata = validate_metadata(self._data_source.metadata)
        if "metadata" in self._data_source_config:
            expected_metadata = validate_metadata(self._data_source_config["metadata"])
            if metadata != expected_metadata:
                raise ValueError(
                    "Data source metadata changed between parent and producer process: "
                    f"expected {expected_metadata}, got {metadata}"
                )
        if metadata.channel_count != self._channel_count:
            raise ValueError(
                f"Data source channel_count {metadata.channel_count} does not match "
                f"producer channel_count {self._channel_count}"
            )
        if metadata.frames_per_second != self._frames_per_second:
            raise ValueError(
                f"Data source frames_per_second {metadata.frames_per_second} does not match "
                f"producer frames_per_second {self._frames_per_second}"
            )
        if self._accelerated_time and not metadata.supports_accelerated:
            raise ValueError("Configured simulator data source does not support accelerated time")

        self._data_source.open()

    def _deliver_stims_to_source(self, up_to_timestamp: int) -> None:
        """Forward newly committed stim events to the data source."""
        if self._buffer is None or self._data_source is None:
            return

        stim_to_ts = min(int(up_to_timestamp), int(self._buffer.stim_write_timestamp))
        if stim_to_ts <= self._last_stim_read_ts:
            return

        stim_records            = self._buffer.read_stims(self._last_stim_read_ts, stim_to_ts)
        self._last_stim_read_ts = stim_to_ts
        if not stim_records:
            return

        self._data_source.on_stims([
            self._stim_record_to_data_source_stim(stim)
            for stim in stim_records
        ])

    @staticmethod
    def _stim_record_to_data_source_stim(stim) -> DataSourceStim:
        phase_count = max(0, min(int(stim.phase_count), 3))
        return DataSourceStim(
            timestamp          = int(stim.timestamp),
            intended_timestamp = int(stim.intended_timestamp),
            channel            = int(stim.channel),
            phase_durations_us = tuple(int(x) for x in stim.phase_durations_us[:phase_count]),
            phase_currents_uA  = tuple(float(x) for x in stim.phase_currents_uA[:phase_count]),
        )

    def _read_source_batch(self, from_timestamp: int, frame_count: int) -> DataSourceBatch:
        assert self._data_source is not None
        batch = self._data_source.read(from_timestamp, frame_count)
        if not isinstance(batch, DataSourceBatch):
            raise TypeError(f"Data source read() must return DataSourceBatch, got {type(batch).__name__}")
        return batch

    def _coerce_frames(self, frames: np.ndarray | None, frame_count: int) -> np.ndarray:
        if frames is None:
            return np.zeros((frame_count, self._channel_count), dtype=np.int16)

        # Fast path: data source returned exactly what we want — skip the
        # asarray/shape/contiguity validation done on every tick.
        if (
            type(frames) is np.ndarray
            and frames.dtype == np.int16
            and frames.shape == (frame_count, self._channel_count)
            and frames.flags.c_contiguous
        ):
            return frames

        frames = np.asarray(frames, dtype=np.int16)
        expected_shape = (frame_count, self._channel_count)
        if frames.shape != expected_shape:
            raise ValueError(f"Data source frames must have shape {expected_shape}, got {frames.shape}")
        if not frames.flags.c_contiguous:
            frames = np.ascontiguousarray(frames)
        return frames

    def _coerce_spikes(
        self,
        spikes        : Sequence[DataSourceSpike],
        from_timestamp: int,
        to_timestamp  : int,
    ) -> list[SpikeRecord]:
        # Fast path: most ticks yield no spikes from typical simulator sources.
        if not spikes:
            return []
        result: list[SpikeRecord] = []
        channel_count = self._channel_count
        for spike in spikes:
            spike_type = type(spike)
            record: SpikeRecord
            if spike_type is SpikeRecord:
                record = spike  # type: ignore[assignment]
            elif spike_type is DataSourceSpike:
                record = self._spike_to_record(spike)  # type: ignore[arg-type]
            elif isinstance(spike, Mapping):
                record = self._spike_to_record(DataSourceSpike(**spike))
            elif isinstance(spike, SpikeRecord):
                record = spike
            elif isinstance(spike, DataSourceSpike):
                record = self._spike_to_record(spike)
            else:
                raise TypeError(
                    f"Data source spikes must be DataSourceSpike or SpikeRecord, got {type(spike).__name__}"
                )

            ts = record.timestamp
            if not (from_timestamp <= ts < to_timestamp):
                raise ValueError(
                    f"Spike timestamp {ts} outside requested range "
                    f"[{from_timestamp}, {to_timestamp})"
                )
            ch = record.channel
            if ch < 0 or ch >= channel_count:
                raise ValueError(f"Spike channel {ch} outside channel range 0-{channel_count - 1}")
            result.append(record)
        return result

    @staticmethod
    def _spike_to_record(spike: DataSourceSpike) -> SpikeRecord:
        if spike.samples is None:
            samples = np.zeros(SPIKE_SAMPLES_TOTAL, dtype=np.float32)
        else:
            samples = np.asarray(spike.samples, dtype=np.float32)
            if samples.shape != (SPIKE_SAMPLES_TOTAL,):
                raise ValueError(
                    f"Spike samples must have shape ({SPIKE_SAMPLES_TOTAL},), got {samples.shape}"
                )
            if not samples.flags.c_contiguous:
                samples = np.ascontiguousarray(samples)

        channel_mean_sample = (
            float(spike.channel_mean_sample)
            if spike.channel_mean_sample is not None
            else float(np.mean(samples))
        )

        return SpikeRecord(
            timestamp           = int(spike.timestamp),
            channel             = int(spike.channel),
            channel_mean_sample = channel_mean_sample,
            samples             = samples,
        )

    def _check_heartbeat_stale(self) -> bool:
        """
        Check if the main process heartbeat has gone stale.

        Returns True if the heartbeat has not updated in over 200ms, which
        suggests the main process is paused at a breakpoint.
        """
        if self._buffer is None:
            return False

        heartbeat_ns = self._buffer.main_process_heartbeat_ns
        if heartbeat_ns == 0:
            return False

        current_ns = time.perf_counter_ns()
        elapsed_ns = current_ns - heartbeat_ns
        return elapsed_ns > STALE_THRESHOLD_NS

    def cleanup(self) -> None:
        if self._data_source is not None:
            with contextlib.suppress(Exception):
                self._data_source.close()
            self._data_source = None
        super().cleanup()

class DataProducer(BaseProducer):
    """
    Interface to the data producer subprocess.

    Extends BaseProducer with:
    - Configurable simulator data source
    - Accelerated time mode
    - Debugger pause flag
    """

    def __init__(
        self,
        data_source_config: Mapping[str, Any],
        start_timestamp   : int,
        channel_count     : int,
        frames_per_second : int,
        duration_frames   : int,
        tick_rate_hz      : int  = DEFAULT_TICK_RATE_HZ,
        accelerated_time  : bool = False,
    ):
        """
        Initialize the data producer.

        Args:
            data_source_config: Process config for a simulator data source
            start_timestamp   : Initial timestamp value (25kHz buffer rate)
            channel_count     : Number of channels
            frames_per_second : Buffer sample rate (25kHz)
            duration_frames   : Total frames at buffer rate (25kHz)
            tick_rate_hz      : Producer loop rate in Hz (default 5000)
            accelerated_time  : If True, run as fast as possible
        """
        super().__init__(
            channel_count     = channel_count,
            frames_per_second = frames_per_second,
            start_timestamp   = start_timestamp,
            duration_frames   = duration_frames,
            tick_rate_hz      = tick_rate_hz,
        )

        self._data_source_config = dict(data_source_config)
        self._accelerated_time   = accelerated_time

    def _create_process(self) -> PopenProcess:
        """Create the data producer subprocess."""
        assert self._name_prefix is not None
        return PopenProcess(
            target       = "cl._sim._data_producer:run_from_config",
            process_name = "cl-data-producer",
            config       = {
                "data_source_config": self._data_source_config,
                "start_timestamp"   : self._start_timestamp,
                "channel_count"     : self._channel_count,
                "frames_per_second" : self._frames_per_second,
                "duration_frames"   : self._duration_frames,
                "tick_rate_hz"      : self._tick_rate_hz,
                "accelerated_time"  : self._accelerated_time,
                "name_prefix"       : self._name_prefix,
            },
        )

    @property
    def is_running(self) -> bool:
        """Check if the producer is running."""
        return self.is_started and self._process is not None and self._process.is_alive()

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CL SDK data producer subprocess")
    parser.add_argument("--process-name", default="cl-data-producer")
    parser.add_argument("--config-json", required=True)
    args   = parser.parse_args()
    config = json.loads(args.config_json)
    run_from_config(config)

def run_from_config(config: dict) -> None:
    _producer_main(**config)

if __name__ == "__main__":
    main()
