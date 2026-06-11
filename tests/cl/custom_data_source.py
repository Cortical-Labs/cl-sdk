from __future__ import annotations

import threading

import numpy as np

from cl.sim import (
    DataSourceBatch,
    DataSourceSpike,
    DataSourceStim,
    LiveDataSink,
    LiveSimulatorDataSource,
    SimulatorDataSource,
    SimulatorDataSourceMetadata,
)
from cl._sim._data_buffer import SPIKE_SAMPLES_TOTAL

class DeterministicSource(SimulatorDataSource):
    def __init__(self, spike_channel: int = 3, sample_offset: int = 0):
        self._spike_channel = spike_channel
        self._sample_offset = sample_offset
        self._metadata      = SimulatorDataSourceMetadata(
            channel_count      = 64,
            frames_per_second  = 25_000,
            uV_per_sample_unit = 0.195,
            start_timestamp    = 0,
        )

    @property
    def metadata(self) -> SimulatorDataSourceMetadata:
        return self._metadata

    def read(self, from_timestamp: int, frame_count: int) -> DataSourceBatch:
        frames = np.zeros((frame_count, self._metadata.channel_count), dtype=np.int16)
        frames[:, 0] = np.arange(from_timestamp, from_timestamp + frame_count, dtype=np.int64) % 32_000
        frames[:, 1] = self._sample_offset

        spikes = []
        spike_ts = 125
        if from_timestamp <= spike_ts < from_timestamp + frame_count:
            spikes.append(DataSourceSpike(
                timestamp = spike_ts,
                channel   = self._spike_channel,
                samples   = np.arange(SPIKE_SAMPLES_TOTAL, dtype=np.float32),
            ))

        return DataSourceBatch(frames=frames, spikes=spikes)

def make_source(spike_channel: int = 3, sample_offset: int = 0) -> DeterministicSource:
    return DeterministicSource(spike_channel=spike_channel, sample_offset=sample_offset)

class StimAwareSource(SimulatorDataSource):
    def __init__(self):
        self._metadata = SimulatorDataSourceMetadata(
            channel_count      = 64,
            frames_per_second  = 25_000,
            uV_per_sample_unit = 0.195,
            start_timestamp    = 0,
        )
        self._stims: list[DataSourceStim] = []

    @property
    def metadata(self) -> SimulatorDataSourceMetadata:
        return self._metadata

    def on_stim(self, stim: DataSourceStim) -> None:
        self._stims.append(stim)

    def read(self, from_timestamp: int, frame_count: int) -> DataSourceBatch:
        frames = np.zeros((frame_count, self._metadata.channel_count), dtype=np.int16)
        to_timestamp = from_timestamp + frame_count
        for stim_index, stim in enumerate(self._stims, start=1):
            if stim.timestamp >= to_timestamp:
                break
            frame_index = max(0, stim.timestamp - from_timestamp)
            if frame_index >= frame_count:
                continue
            first_current = stim.phase_currents_uA[0] if stim.phase_currents_uA else 0.0
            last_current  = stim.phase_currents_uA[-1] if stim.phase_currents_uA else 0.0
            frames[frame_index:, 3] = stim_index
            frames[frame_index:, 4] = stim.channel
            frames[frame_index:, 5] = len(stim.phase_durations_us)
            frames[frame_index:, 6] = stim.duration_us
            frames[frame_index:, 7] = int(round(first_current * 1000))
            frames[frame_index:, 8] = int(round(last_current * 1000))
        return DataSourceBatch(frames=frames)

def make_stim_aware_source() -> StimAwareSource:
    return StimAwareSource()

class ThreadedLiveSource(LiveSimulatorDataSource):
    def __init__(self, sample_offset: int = 0, batch_frames: int = 5):
        self._sample_offset = sample_offset
        self._batch_frames  = batch_frames
        self._stop_event    = threading.Event()
        self._thread: threading.Thread | None = None
        self._sink  : LiveDataSink | None     = None
        super().__init__(
            metadata             = SimulatorDataSourceMetadata(
                channel_count      = 64,
                frames_per_second  = 25_000,
                uV_per_sample_unit = 0.195,
                start_timestamp    = 0,
            ),
            max_buffer_frames    = 128,
            read_timeout_seconds = 5.0,
        )

    def start(self, sink: LiveDataSink) -> None:
        self._sink = sink
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, args=(sink,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sink is not None:
            self._sink.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self, sink: LiveDataSink) -> None:
        try:
            while not self._stop_event.is_set():
                start_ts = sink.next_timestamp
                stop_ts  = start_ts + self._batch_frames
                frames   = np.zeros((self._batch_frames, self.metadata.channel_count), dtype=np.int16)
                frames[:, 0] = np.arange(start_ts, stop_ts, dtype=np.int64) % 32_000
                frames[:, 2] = self._sample_offset
                sink.emit_batch(DataSourceBatch(
                    frames = frames,
                    spikes = [DataSourceSpike(timestamp=start_ts + 1, channel=6)],
                ))
        except RuntimeError:
            pass

def make_live_source(sample_offset: int = 0, batch_frames: int = 5) -> ThreadedLiveSource:
    return ThreadedLiveSource(sample_offset=sample_offset, batch_frames=batch_frames)

class ReactiveLiveSource(LiveSimulatorDataSource):
    def __init__(self, batch_frames: int = 5, max_buffer_frames: int = 64):
        self._batch_frames = batch_frames
        self._stop_event   = threading.Event()
        self._thread: threading.Thread | None = None
        self._sink  : LiveDataSink | None     = None
        super().__init__(
            metadata=SimulatorDataSourceMetadata(
                channel_count      = 64,
                frames_per_second  = 25_000,
                uV_per_sample_unit = 0.195,
                start_timestamp    = 0,
            ),
            max_buffer_frames    = max_buffer_frames,
            read_timeout_seconds = 5.0,
        )

    def start(self, sink: LiveDataSink) -> None:
        self._sink = sink
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, args=(sink,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sink is not None:
            self._sink.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def on_stim(self, stim: DataSourceStim) -> None:
        if self._sink is None:
            return
        spike_ts = max(stim.timestamp + 1, self._sink.read_timestamp)
        self._sink.emit_spikes(
            [
                DataSourceSpike(
                    timestamp = spike_ts,
                    channel   = stim.channel,
                    samples   = np.zeros(SPIKE_SAMPLES_TOTAL, dtype=np.float32),
                )
            ]
        )

    def _run(self, sink: LiveDataSink) -> None:
        try:
            while not self._stop_event.is_set():
                frames = np.zeros(
                    (self._batch_frames, self.metadata.channel_count), dtype=np.int16
                )
                sink.emit_frames(frames)
        except RuntimeError:
            pass

def make_reactive_live_source(
    batch_frames: int = 5, max_buffer_frames: int = 64
) -> ReactiveLiveSource:
    return ReactiveLiveSource(batch_frames=batch_frames, max_buffer_frames=max_buffer_frames)
