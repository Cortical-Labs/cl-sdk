"""Simulator data-source interfaces and built-ins."""
from __future__ import annotations

import bisect
import contextlib
import importlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from threading import Condition, Lock
from typing import Any

import numpy as np
from numpy import ndarray

from .._sim._data_buffer import DEFAULT_CHANNEL_COUNT, DEFAULT_FRAMES_PER_SECOND, SPIKE_SAMPLES_BEFORE, SPIKE_SAMPLES_TOTAL

_logger = logging.getLogger("cl.sim.data_source")

DATA_SOURCE_ENV          = "CL_SDK_DATA_SOURCE"
DATA_SOURCE_CONFIG_ENV   = "CL_SDK_DATA_SOURCE_CONFIG"
DATA_SOURCE_METADATA_ENV = "CL_SDK_DATA_SOURCE_METADATA"

DEFAULT_UV_PER_SAMPLE_UNIT = 0.195
DEFAULT_DURATION_FRAMES    = 60 * DEFAULT_FRAMES_PER_SECOND

_simulator_data_source_spec: _SimulatorDataSourceSpec | None = None

@dataclass(frozen=True)
class SimulatorDataSourceMetadata:
    """
    Static description of a simulator data source.

    Every `SimulatorDataSource` must publish a `metadata` instance describing
    its shape and capabilities. The simulator subprocess reads metadata once
    immediately after construction (before `open()`) to configure the shared
    buffer, timestamp accounting, and accelerated-time support.

    Attributes:
        channel_count: Number of channels per frame. Must be positive. Frames
            returned by `read()` must have shape `(frame_count, channel_count)`.
        frames_per_second: Sample rate in Hz. Must match the SDK's expected
            rate (currently 25,000); other values are rejected by
            `validate_metadata`.
        uV_per_sample_unit: Scaling factor used to convert raw `int16` sample
            units into microvolts. Used by downstream analysis but not by the
            simulator itself.
        start_timestamp: Frame timestamp at which this source's data begins.
            Reads use timestamps relative to this value.
        duration_frames: Total number of frames the source can produce, or
            `None` for unbounded sources. Bounded sources loop or stop at this
            length depending on the producer's policy.
        seekable: `True` if `read()` can be called with arbitrary
            `from_timestamp` values within the source's range. `False` for
            sources that must be consumed sequentially (e.g. live streams).
        realtime_only: `True` if the source can only produce data at real
            wall-clock speed. Set automatically for `LiveSimulatorDataSource`.
        supports_accelerated: `True` if the source can produce frames faster
            than real time (used by `CL_SDK_ACCELERATED_TIME`). File-backed
            sources are accelerated; live sources usually are not.
    """

    channel_count       : int        = DEFAULT_CHANNEL_COUNT
    frames_per_second   : int        = DEFAULT_FRAMES_PER_SECOND
    uV_per_sample_unit  : float      = DEFAULT_UV_PER_SAMPLE_UNIT
    start_timestamp     : int        = 0
    duration_frames     : int | None = None
    seekable            : bool       = True
    realtime_only       : bool       = False
    supports_accelerated: bool       = True

@dataclass(frozen=True)
class DataSourceSpike:
    """
    A spike detection event produced by a simulator data source.

    Sources may emit spikes alongside frames (as part of a `DataSourceBatch`)
    when they have ground-truth knowledge of spike events (e.g. when replaying
    a recording with annotated spikes, or generating synthetic data). Sources
    that don't know about spikes simply leave the `spikes` field of
    `DataSourceBatch` empty and let downstream consumers do the detection.

    Attributes:
        timestamp: Frame timestamp at which the spike occurred. Must fall
            within the time range of the batch the spike is emitted in.
        channel: Zero-based channel index on which the spike was detected.
            Must be `< metadata.channel_count`.
        samples: Optional waveform snippet centred on the spike, in
            microvolts (`float32`). Length is fixed by
            `cl._sim._data_buffer.SPIKE_SAMPLES_TOTAL`. `None` if no waveform
            is available.
        channel_mean_sample: Optional baseline (mean) sample value for the
            channel at the time of the spike, used by some analyses for
            normalisation. `None` if not computed.
    """

    timestamp          : int
    channel            : int
    samples            : ndarray | None = None
    channel_mean_sample: float   | None = None

@dataclass(frozen=True)
class DataSourceStim:
    """
    A stimulation event observed by a simulator data source.

    These timestamps use the same 25 kHz frame clock as `read()`. Bursts are
    delivered as individual stim events. Phase fields mirror the `StimDesign`
    used for that individual pulse.
    """

    timestamp         : int
    channel           : int
    intended_timestamp: int | None        = None
    phase_durations_us: tuple[int, ...]   = ()
    phase_currents_uA : tuple[float, ...] = ()

    @cached_property
    def duration_us(self) -> int:
        return int(sum(self.phase_durations_us))

    @cached_property
    def stim_design(self):
        """Reconstruct a `cl.StimDesign`, or `None` if phase data is absent."""
        if not self.phase_durations_us:
            return None
        from cl import StimDesign
        args: list[int | float] = []
        for duration_us, current_uA in zip(self.phase_durations_us, self.phase_currents_uA, strict=True):
            args.extend([int(duration_us), float(current_uA)])
        return StimDesign(*args)  # type: ignore[call-arg]

@dataclass(frozen=True)
class DataSourceBatch:
    """
    A frames + spikes payload returned by `SimulatorDataSource.read()`.

    A batch describes the data for a contiguous range of frame timestamps
    `[from_timestamp, from_timestamp + frame_count)`. Either field may be
    omitted: a source may return only frames (the common case), only spikes
    (e.g. when push-emitting out-of-band spike events on a live source), or
    both.

    Attributes:
        frames: `int16` array shaped `(frame_count, channel_count)` in raw
            sample units. `None` indicates no new frames for this batch.
            When present, `frame_count` must equal the value requested by the
            caller of `read()`.
        spikes: Sequence of `DataSourceSpike` events whose timestamps fall
            within the batch's time range. Should be ordered by timestamp.
            Defaults to empty.
    """

    frames: ndarray | None = None
    spikes: Sequence[DataSourceSpike] = ()

class SimulatorDataSource(ABC):
    """
    Abstract base class for pull-style simulator data sources.

    A data source is the component that produces neural sample frames (and
    optionally spikes) on behalf of the SDK's in-process simulator. Subclass
    this directly when your source can answer `read(from_timestamp,
    frame_count)` on demand. For example, file-backed replay sources,
    deterministic generators, or any source that is randomly seekable. For
    push-style sources driven by their own clock (hardware capture, network
    streams, etc.) subclass `LiveSimulatorDataSource` instead.

    ## Lifecycle

    Sources are constructed in the parent process via
    `cl.sim.set_simulator_data_source("module:factory", config={...})`, then
    re-constructed inside the simulator subprocess from the same factory.
    The subprocess then calls, in order:

    1. `metadata`: read once to size the shared buffer and configure timing.
    2. `open()`: acquire any per-process resources (file handles, sockets,
       thread pools, etc.). Do not do this work in `__init__`; the constructor
       runs in both the parent and child processes.
    3. `read(from_timestamp, frame_count)`: called repeatedly to drive the
       simulator. Must return a `DataSourceBatch` whose `frames` (if any)
       have the exact shape `(frame_count, metadata.channel_count)`.
    4. `close()`: release the resources acquired in `open()`.

    ## Requirements for implementations

    - The class and its factory must be importable from the simulator
      subprocess (`from module import factory`). Lambdas, local classes, and
      `__main__`-defined symbols are rejected by `set_simulator_data_source`.
    - Any constructor `config` passed to `set_simulator_data_source` must be
      JSON-serialisable; it is forwarded as `**kwargs` to the factory in
      the subprocess.
    - `metadata.frames_per_second` must match the SDK's expected sample rate
      (currently 25,000).
    - If `metadata.seekable` is `True`, `read()` must accept any
      `from_timestamp` in `[start_timestamp, start_timestamp +
      duration_frames)`. If `False`, the source may require sequential reads.

    See the module doc (`cl.sim`) for a full worked example.
    """

    @property
    @abstractmethod
    def metadata(self) -> SimulatorDataSourceMetadata:
        """
        Return the static metadata describing this source.

        Called once by the simulator subprocess immediately after
        construction, before `open()`. Must return the same value on every
        call; the simulator does not poll for metadata changes.
        """

    def open(self) -> None:
        """
        Prepare the source for `read()` calls in the simulator subprocess.

        Use this hook (rather than `__init__`) to acquire process-local
        resources such as file handles, sockets, threads, or GPU contexts.
        The constructor runs in both the parent and child processes, while
        `open()` runs only in the simulator subprocess.

        Default implementation does nothing.
        """

    def close(self) -> None:
        """
        Release any resources acquired in `open()`.

        Always paired with a successful `open()` call. Implementations should
        be idempotent and tolerant of being called even if `open()` raised.

        Default implementation does nothing.
        """

    def on_stim(self, stim: DataSourceStim) -> None:
        """
        Handle one committed stim event from the SDK simulator.

        Called in the data producer subprocess after `Neurons` publishes stim
        events to the shared stim ring. Override this to let a simulated model
        react to stimulation. Default implementation does nothing.
        """

    def on_stims(self, stims: Sequence[DataSourceStim]) -> None:
        """
        Handle a batch of committed stim events from the SDK simulator.

        Override this for batch handling; otherwise each event is forwarded to `on_stim()`.
        """
        for stim in stims:
            self.on_stim(stim)

    @abstractmethod
    def read(self, from_timestamp: int, frame_count: int) -> DataSourceBatch:
        """
        Produce a batch of frames (and optionally spikes) for the simulator.

        Returns a `DataSourceBatch` covering the half-open timestamp range
        `[from_timestamp, from_timestamp + frame_count)`. If `batch.frames`
        is not `None`, it must have shape
        `(frame_count, metadata.channel_count)` and dtype `int16`. Any
        spikes in `batch.spikes` must have timestamps within the requested
        range.

        Args:
            from_timestamp: First frame timestamp to produce, relative to
                `metadata.start_timestamp`.
            frame_count: Number of frames the simulator expects. Always > 0.

        May block for live/streamed sources, but should return promptly for
        seekable sources to avoid stalling the simulator.
        """

class LiveDataSink:
    """
    Thread-safe handle for pushing data into a `LiveSimulatorDataSource`.

    A `LiveDataSink` is passed to `LiveSimulatorDataSource.start()` and is
    the only way a live source should emit data. All methods are safe to call
    from background threads owned by the source implementation.

    Backpressure: `emit_batch` / `emit_frames` will block if the source's
    internal buffer is full, until the simulator drains it (or the source is
    closed). This naturally throttles producers that run faster than the
    simulator can consume.
    """

    def __init__(self, source: LiveSimulatorDataSource):
        self._source = source

    @property
    def next_timestamp(self) -> int:
        """
        Timestamp that will be assigned to the next emitted frame.

        Useful when constructing spikes whose `timestamp` must fall inside
        the batch they are emitted with.
        """
        return self._source.next_write_timestamp

    @property
    def read_timestamp(self) -> int:
        """
        Timestamp of the next frame the simulator will read (the read head).

        This is the lowest timestamp an out-of-band spike (`emit_spikes`) can
        still be delivered with; earlier spikes have already been passed and are
        dropped. Clamp reactive spikes to `max(desired_ts, read_timestamp)` to
        minimise stim->spike latency without risking a silently dropped spike.
        Always `<= next_timestamp` (the write head runs ahead by the buffered
        frame depth).
        """
        return self._source.current_read_timestamp

    def emit_batch(self, batch: DataSourceBatch) -> None:
        """
        Emit a frames + spikes batch atomically.

        Any spikes in `batch.spikes` must have timestamps within
        `[next_timestamp, next_timestamp + batch.frames.shape[0])`. Blocks
        if the internal buffer is full.

        Raises:
            ValueError: if frames have the wrong shape/dtype or if a spike's
                timestamp falls outside the batch range.
            RuntimeError: if the source has already been closed.
        """
        self._source._emit_live_batch(batch)

    def emit_frames(self, frames: ndarray) -> None:
        """
        Emit frames with no associated spikes.

        Convenience wrapper around `emit_batch(DataSourceBatch(frames=frames))`.
        """
        self.emit_batch(DataSourceBatch(frames=frames))

    def emit_spikes(self, spikes: Sequence[DataSourceSpike]) -> None:
        """
        Emit spikes out-of-band, without advancing the frame timeline.

        Use this when spike detection runs asynchronously to frame production
        and needs to be reported as soon as it's available. Spike timestamps
        may be any past or future value; spikes are sorted and merged into
        the pending queue.
        """
        self._source._emit_live_spikes(spikes)

    def close(self) -> None:
        """
        Signal that no more frames or spikes will arrive.

        After this call, pending `read()`s on the simulator side will drain
        the buffer and then raise once data runs out. Any subsequent
        `emit_*` call will raise.
        """
        self._source._mark_live_closed()

class LiveSimulatorDataSource(SimulatorDataSource):
    """
    Abstract base class for push-style (live) simulator data sources.

    Use this base when data arrives on its own clock, e.g., from hardware
    capture, a network stream, or a child process, rather than being
    fetched on demand. Subclasses implement `start(sink)` (and optionally
    `stop()`) and push frames/spikes into the supplied `LiveDataSink` from a
    background thread or event loop. The base class buffers what's pushed and
    serves the simulator's `read()` calls sequentially.

    Live sources are inherently non-seekable and real-time only: the base
    class overrides `metadata.seekable` to `False` and `metadata.realtime_only`
    to `True` regardless of what the supplied metadata says.

    ## Lifecycle

    1. `__init__(metadata, ...)`: subclass calls `super().__init__()` with
       optional metadata and buffering parameters.
    2. `open()`: called by the simulator subprocess; resets internal state
       and invokes `start(sink)`.
    3. The subclass's `start(sink)` implementation begins producing data,
       typically by spawning a thread that calls `sink.emit_*` in a loop.
    4. The simulator drains the buffer via `read()`; emit calls block when
       the buffer is full.
    5. `close()`: calls `stop()`, marks the sink closed, and unblocks any
       waiting reader.

    ## Backpressure & timeouts

    - `max_buffer_frames` caps the number of pending frames; emit calls block
      when full, throttling fast producers.
    - `read_timeout_seconds` bounds how long the simulator will wait for
      enough frames to satisfy a read before raising `TimeoutError`.

    See the module doc (`cl.sim`) for a full worked example.
    """

    def __init__(
        self,
        metadata                : SimulatorDataSourceMetadata | None = None,
        *,
        max_buffer_frames       : int = DEFAULT_FRAMES_PER_SECOND,
        read_timeout_seconds    : float = 30.0,
        supports_accelerated    : bool = False,
    ):
        """
        Args:
            metadata: Static metadata for the source. The `seekable` and
                `realtime_only` fields are ignored and forced to `False` and
                `True` respectively. Defaults to `SimulatorDataSourceMetadata()`.
            max_buffer_frames: Maximum number of pending frames to buffer
                before `emit_*` calls block. Defaults to one second's worth
                of frames at the SDK's sample rate.
            read_timeout_seconds: Maximum time the simulator will wait for a
                `read()` to be satisfied. Raises `TimeoutError` on expiry.
            supports_accelerated: Whether the source can keep up with
                accelerated time. Defaults to `False` (live sources are
                normally locked to wall-clock).
        """
        metadata = metadata or SimulatorDataSourceMetadata()
        self._metadata = replace(
            metadata,
            seekable             = False,
            realtime_only        = True,
            supports_accelerated = supports_accelerated,
        )
        self._max_buffer_frames    = int(max_buffer_frames)
        self._read_timeout_seconds = float(read_timeout_seconds)
        self._condition            = Condition()
        self._frame_chunks         : deque[np.ndarray] = deque()
        self._spikes               : deque[DataSourceSpike] = deque()
        self._available_frames     = 0
        self._read_timestamp       = self._metadata.start_timestamp
        self._write_timestamp      = self._metadata.start_timestamp
        self._live_closed          = False
        self._started              = False
        if self._max_buffer_frames <= 0:
            raise ValueError("max_buffer_frames must be positive")
        if self._read_timeout_seconds <= 0:
            raise ValueError("read_timeout_seconds must be positive")

    @property
    def metadata(self) -> SimulatorDataSourceMetadata:
        return self._metadata

    @property
    def next_write_timestamp(self) -> int:
        with self._condition:
            return self._write_timestamp

    @property
    def current_read_timestamp(self) -> int:
        """
        Timestamp of the next frame the simulator will read (the read head).

        This is the earliest timestamp an out-of-band spike emitted via
        `LiveDataSink.emit_spikes()` can still be delivered with: spikes whose
        timestamp falls *before* the read head have already been passed and are
        discarded. Unlike `next_write_timestamp` (the source's write head, which
        runs ahead of the read head by the buffered frame depth), clamping a
        reactive spike to `max(desired_ts, current_read_timestamp)` places it as
        close as possible to the stimulus that triggered it — minimising the
        stim->spike latency while guaranteeing the spike is never dropped.
        """
        with self._condition:
            return self._read_timestamp

    def open(self) -> None:
        with self._condition:
            self._frame_chunks.clear()
            self._spikes.clear()
            self._available_frames = 0
            self._read_timestamp = self._metadata.start_timestamp
            self._write_timestamp = self._metadata.start_timestamp
            self._live_closed = False
        self.start(LiveDataSink(self))
        self._started = True

    def close(self) -> None:
        self._mark_live_closed()
        if self._started:
            with contextlib.suppress(Exception):
                self.stop()
        self._started = False

    @abstractmethod
    def start(self, sink: LiveDataSink) -> None:
        """
        Begin producing live data into `sink`.

        Called once per `open()`. Implementations typically spawn a daemon
        thread (or start an async task / external process) that calls
        `sink.emit_batch`, `sink.emit_frames`, or `sink.emit_spikes` as data
        becomes available. Must return promptly, do not block here.
        """

    def stop(self) -> None:
        """
        Stop producing live data. Called once per `close()`.

        Implementations should signal their background workers to exit and
        wait for them to do so. Exceptions raised here are swallowed by the
        base class to ensure cleanup proceeds.

        Default implementation does nothing.
        """

    def read(self, from_timestamp: int, frame_count: int) -> DataSourceBatch:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")

        with self._condition:
            if from_timestamp != self._read_timestamp:
                raise ValueError(
                    "LiveSimulatorDataSource is sequential; "
                    f"expected timestamp {self._read_timestamp}, got {from_timestamp}"
                )

            deadline = time.monotonic() + self._read_timeout_seconds
            while self._available_frames < frame_count and not self._live_closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for {frame_count} live frames "
                        f"at timestamp {from_timestamp}"
                    )
                self._condition.wait(timeout=remaining)

            if self._available_frames < frame_count:
                raise RuntimeError("Live data source closed before enough frames were available")

            frames = self._pop_live_frames(frame_count)
            to_timestamp = from_timestamp + frame_count
            spikes = self._pop_live_spikes(from_timestamp, to_timestamp)
            self._read_timestamp = to_timestamp
            self._condition.notify_all()
            return DataSourceBatch(frames=frames, spikes=spikes)

    def _emit_live_batch(self, batch: DataSourceBatch) -> None:
        if not isinstance(batch, DataSourceBatch):
            raise TypeError(f"emit_batch() expects DataSourceBatch, got {type(batch).__name__}")

        frames = self._coerce_live_frames(batch.frames)
        spikes = self._coerce_live_spikes(batch.spikes)
        frame_count = 0 if frames is None else frames.shape[0]

        with self._condition:
            if self._live_closed:
                raise RuntimeError("Cannot emit after live data source is closed")
            if frames is not None and frame_count > 0:
                while (
                    self._available_frames + frame_count > self._max_buffer_frames and
                    not self._live_closed
                ):
                    self._condition.wait(timeout=0.1)
                if self._live_closed:
                    raise RuntimeError("Cannot emit frames after live data source is closed")

                batch_start = self._write_timestamp
                batch_end   = batch_start + frame_count
                for spike in spikes:
                    if not (batch_start <= spike.timestamp < batch_end):
                        raise ValueError(
                            f"Live batch spike timestamp {spike.timestamp} outside "
                            f"batch range [{batch_start}, {batch_end})"
                        )
                self._frame_chunks.append(frames)
                self._available_frames += frame_count
                self._write_timestamp   = batch_end

            self._append_live_spikes(spikes)
            self._condition.notify_all()

    def _emit_live_spikes(self, spikes: Sequence[DataSourceSpike]) -> None:
        spikes = self._coerce_live_spikes(spikes)
        with self._condition:
            if self._live_closed:
                raise RuntimeError("Cannot emit spikes after live data source is closed")
            self._append_live_spikes(spikes)
            self._condition.notify_all()

    def _mark_live_closed(self) -> None:
        with self._condition:
            self._live_closed = True
            self._condition.notify_all()

    def _coerce_live_frames(self, frames: ndarray | None) -> np.ndarray | None:
        if frames is None:
            return None
        frames = np.asarray(frames, dtype=np.int16)
        if frames.ndim != 2 or frames.shape[1] != self._metadata.channel_count:
            raise ValueError(
                f"frames must have shape (frame_count, {self._metadata.channel_count}), "
                f"got {frames.shape}"
            )
        if not frames.flags.c_contiguous:
            frames = np.ascontiguousarray(frames)
        return frames

    def _coerce_live_spikes(self, spikes: Sequence[DataSourceSpike]) -> list[DataSourceSpike]:
        result: list[DataSourceSpike] = []
        for spike in spikes:
            if isinstance(spike, Mapping):
                spike = DataSourceSpike(**spike)
            if not isinstance(spike, DataSourceSpike):
                raise TypeError(f"spikes must contain DataSourceSpike objects, got {type(spike).__name__}")
            if spike.channel < 0 or spike.channel >= self._metadata.channel_count:
                raise ValueError(f"Spike channel {spike.channel} outside channel range")
            result.append(spike)
        result.sort(key=lambda spike: spike.timestamp)
        return result

    def _append_live_spikes(self, spikes: Sequence[DataSourceSpike]) -> None:
        if not spikes:
            return
        if self._spikes and spikes[0].timestamp < self._spikes[-1].timestamp:
            combined = list(self._spikes)
            combined.extend(spikes)
            combined.sort(key=lambda spike: spike.timestamp)
            self._spikes = deque(combined)
            return
        self._spikes.extend(spikes)

    def _pop_live_frames(self, frame_count: int) -> np.ndarray:
        frames = np.empty((frame_count, self._metadata.channel_count), dtype=np.int16)
        copied = 0
        while copied < frame_count:
            chunk = self._frame_chunks[0]
            take = min(frame_count - copied, chunk.shape[0])
            frames[copied:copied + take] = chunk[:take]
            copied += take
            if take == chunk.shape[0]:
                self._frame_chunks.popleft()
            else:
                self._frame_chunks[0] = chunk[take:]
        self._available_frames -= frame_count
        return frames

    def _pop_live_spikes(self, from_timestamp: int, to_timestamp: int) -> list[DataSourceSpike]:
        spikes: list[DataSourceSpike] = []
        while self._spikes and self._spikes[0].timestamp < to_timestamp:
            spike = self._spikes.popleft()
            if spike.timestamp >= from_timestamp:
                spikes.append(spike)
        return spikes

@dataclass(frozen=True)
class _SimulatorDataSourceSpec:
    """Importable source factory plus JSON-serialisable configuration."""

    factory : str
    config  : dict[str, Any]
    metadata: SimulatorDataSourceMetadata | None = None

    def to_process_config(self) -> dict[str, Any]:
        process_config: dict[str, Any] = {
            "kind"   : "factory",
            "factory": self.factory,
            "config" : self.config,
        }
        if self.metadata is not None:
            process_config["metadata"] = asdict(self.metadata)
        return process_config

class FileRecordingDataSource(SimulatorDataSource):
    """Built-in source that loops an HDF5 recording file."""

    def __init__(
        self,
        replay_file_path   : str | Path,
        replay_start_offset: int = 0,
    ):
        from cl.util import RecordingView

        self._replay_file_path    = str(replay_file_path)
        self._replay_start_offset = int(replay_start_offset)
        self._recording_view    : RecordingView | None = None
        self._spike_timestamps  : np.ndarray | None    = None
        self._frame_cache       : np.ndarray | None    = None
        self._frame_cache_start : int                  = 0
        self._frame_cache_end   : int                  = 0
        self._warned_no_samples : bool                 = False

        with RecordingView(self._replay_file_path) as recording:
            attrs: dict[str, Any] = dict(recording.attributes)

        self._metadata = SimulatorDataSourceMetadata(
            channel_count      = int(attrs["channel_count"]),
            frames_per_second  = int(attrs["frames_per_second"]),
            uV_per_sample_unit = float(attrs["uV_per_sample_unit"]),
            start_timestamp    = int(attrs["start_timestamp"]),
            duration_frames    = int(attrs["duration_frames"]),
        )

        if self._metadata.duration_frames is None or self._metadata.duration_frames <= 0:
            raise ValueError("Replay recording duration_frames must be positive")

        self._frame_cache_frames = max(
            1,
            min(
                self._metadata.duration_frames,
                int(self._metadata.frames_per_second * 0.25),
            ),
        )

    @property
    def metadata(self) -> SimulatorDataSourceMetadata:
        return self._metadata

    def open(self) -> None:
        from cl.util import RecordingView

        self._recording_view = RecordingView(self._replay_file_path)
        self._load_spike_timestamps()

    def close(self) -> None:
        if self._recording_view is not None:
            self._recording_view.close()
            self._recording_view = None
        self._spike_timestamps = None
        self._frame_cache = None

    def read(self, from_timestamp: int, frame_count: int) -> DataSourceBatch:
        return DataSourceBatch(
            frames = self._read_frames(from_timestamp, frame_count),
            spikes = self._read_spikes(from_timestamp, frame_count),
        )

    def _load_spike_timestamps(self) -> None:
        if self._recording_view is None or self._recording_view.spikes is None:
            self._spike_timestamps = np.array([], dtype=np.int64)
            return

        spikes = self._recording_view.spikes
        try:
            self._spike_timestamps = spikes.col("timestamp").astype(np.int64)
        except (AttributeError, KeyError):
            spike_count = len(spikes)
            self._spike_timestamps = np.zeros(spike_count, dtype=np.int64)
            for i in range(spike_count):
                self._spike_timestamps[i] = int(spikes[i]["timestamp"])

    def _read_frames(self, from_timestamp: int, frame_count: int) -> np.ndarray:
        recording_has_samples = self._recording_view is not None and self._recording_view.samples is not None
        if not recording_has_samples:
            if not self._warned_no_samples:
                self._warned_no_samples = True
                _logger.warning(
                    "Replay recording %s has no sample data; producing zero-valued frames",
                    self._replay_file_path,
                )
            return np.zeros((frame_count, self._metadata.channel_count), dtype=np.int16)

        file_duration = self._metadata.duration_frames
        assert file_duration is not None

        elapsed   = from_timestamp - self._metadata.start_timestamp
        file_pos  = elapsed + self._replay_start_offset
        start_idx = file_pos % file_duration

        frames = np.empty((frame_count, self._metadata.channel_count), dtype=np.int16)
        filled = 0
        idx    = start_idx
        while filled < frame_count:
            take = min(frame_count - filled, file_duration - idx)
            frames[filled:filled + take] = self._read_frame_slice(idx, take)
            filled += take
            idx = 0
        return frames

    def _read_frame_slice(self, start_idx: int, frame_count: int) -> np.ndarray:
        if frame_count <= 0:
            return np.empty((0, self._metadata.channel_count), dtype=np.int16)

        end_idx = start_idx + frame_count
        if (
            self._frame_cache is None or
            start_idx < self._frame_cache_start or
            end_idx > self._frame_cache_end
        ):
            assert self._recording_view is not None and self._recording_view.samples is not None
            file_duration = self._metadata.duration_frames
            assert file_duration is not None
            cache_end = min(
                file_duration,
                start_idx + max(self._frame_cache_frames, frame_count),
            )
            self._frame_cache = np.asarray(
                self._recording_view.samples[start_idx:cache_end],
                dtype=np.int16,
            )
            self._frame_cache_start = start_idx
            self._frame_cache_end   = cache_end

        cache_start = start_idx - self._frame_cache_start
        return self._frame_cache[cache_start:cache_start + frame_count]

    def _read_spikes(
        self,
        from_timestamp: int,
        frame_count   : int,
    ) -> list[DataSourceSpike]:
        if (
            self._recording_view is None or
            self._recording_view.spikes is None or
            self._spike_timestamps is None or
            len(self._spike_timestamps) == 0
        ):
            return []

        file_duration = self._metadata.duration_frames
        assert file_duration is not None

        result: list[DataSourceSpike] = []

        elapsed            = from_timestamp - self._metadata.start_timestamp
        offset_from_ts_25k = elapsed + self._replay_start_offset
        cursor             = offset_from_ts_25k
        end_cursor         = offset_from_ts_25k + frame_count

        while cursor < end_cursor:
            loop_count, left_ts = divmod(cursor, file_duration)
            take     = min(end_cursor - cursor, file_duration - left_ts)
            right_ts = left_ts + take

            left_idx  = np.searchsorted(self._spike_timestamps, left_ts, side="left")
            right_idx = np.searchsorted(self._spike_timestamps, right_ts, side="left")

            for i in range(left_idx, right_idx):
                replay_spike = self._recording_view.spikes[i]
                spike_file_ts = int(replay_spike["timestamp"])

                spike_elapsed = (
                    spike_file_ts
                    + loop_count * file_duration
                    - self._replay_start_offset
                )
                spike_ts = spike_elapsed + self._metadata.start_timestamp

                spike_samples = (
                    replay_spike["samples"]
                    if replay_spike.dtype.names and "samples" in replay_spike.dtype.names
                    else np.zeros(SPIKE_SAMPLES_TOTAL, dtype=np.float32)
                )

                result.append(
                    DataSourceSpike(
                        timestamp           = spike_ts,
                        channel             = int(replay_spike["channel"]),
                        channel_mean_sample = float(np.mean(spike_samples)),
                        samples             = np.ascontiguousarray(spike_samples, dtype=np.float32),
                    )
                )

            cursor += take

        return result

class RandomDataSource(SimulatorDataSource):
    """
    Generate deterministic synthetic neural data on the fly.

    Produces Poisson-distributed sample frames (and ground-truth spikes)
    directly in response to `read()`, instead of pre-generating a recording
    file and replaying it. Frames are computed deterministically from
    `random_seed` in fixed-size blocks, so the source is randomly seekable.
    The stream is unbounded: time always advances and fresh data is generated
    for every new block, never wrapping around to repeat earlier frames.
    """

    _BLOCK_FRAMES         = DEFAULT_FRAMES_PER_SECOND
    _CALIBRATION_FRAMES   = 10 * DEFAULT_FRAMES_PER_SECOND
    _MAX_CACHED_BLOCKS    = 16
    _GROUNDED_CHANNELS    = (0, 7, 56, 63)
    _REFERENCE_CHANNELS   = (4,)
    _NON_SPIKING_CHANNELS = _GROUNDED_CHANNELS + _REFERENCE_CHANNELS

    # Poisson block generation dominates this source's cost and dwarfs every
    # other per-read operation. numpy's Generator.poisson releases the GIL
    # while sampling, so upcoming blocks are generated on a small worker pool
    # while the read loop consumes the current block. This overlaps generation
    # across cores (crucial in accelerated mode) without changing any output
    # bytes: each block is still computed deterministically from
    # SeedSequence([random_seed, block_index]).
    _DEFAULT_PREFETCH_WORKERS = min(4, max(1, (os.cpu_count() or 1) - 1))

    def __init__(
        self,
        replay_start_offset: int   = 0,
        sample_mean        : int   | None = None,
        spike_percentile   : float | None = None,
        random_seed        : int   | None = None,
        amplify_spikes     : bool  | None = None,
    ):
        self._replay_start_offset = int(replay_start_offset)
        self._sample_mean         = int(os.getenv("CL_SDK_SAMPLE_MEAN", "170") if sample_mean is None else sample_mean)
        self._spike_percentile    = float(os.getenv("CL_SDK_SPIKE_PERCENTILE", "99.995") if spike_percentile is None else spike_percentile)
        self._random_seed         = int(os.getenv("CL_SDK_RANDOM_SEED", self._generate_random_seed()) if random_seed is None else random_seed)
        self._amplify_spikes      = (os.getenv("CL_SDK_SPIKE_VISIBILITY", "0") == "1") if amplify_spikes is None else bool(amplify_spikes)

        self._channel_count      = DEFAULT_CHANNEL_COUNT
        self._frames_per_second  = DEFAULT_FRAMES_PER_SECOND
        self._uV_per_sample_unit = DEFAULT_UV_PER_SAMPLE_UNIT

        self._metadata = SimulatorDataSourceMetadata(
            channel_count      = self._channel_count,
            frames_per_second  = self._frames_per_second,
            uV_per_sample_unit = self._uV_per_sample_unit,
            start_timestamp    = 0,
            duration_frames    = None,  # unbounded: data is generated forever
        )

        self._spike_channel_mask = np.ones(self._channel_count, dtype=bool)
        self._spike_channel_mask[list(self._NON_SPIKING_CHANNELS)] = False

        self._block_cache    : OrderedDict[int, np.ndarray] = OrderedDict()
        self._spike_cache    : OrderedDict[int, tuple[list[int], list[tuple[int, int, np.ndarray]]]] = OrderedDict()
        self._spike_threshold: float | None = None

        self._block_lock       = Lock()
        self._prefetch_workers = int(os.getenv("CL_SDK_RANDOM_PREFETCH_WORKERS", str(self._DEFAULT_PREFETCH_WORKERS)))
        self._prefetch_ahead   = max(1, self._prefetch_workers)
        self._block_futures: dict[int, Future[np.ndarray]] = {}
        self._executor:      ThreadPoolExecutor | None     = None

        # LRU window over the most recently used blocks. Reads advance forward,
        # so we only need to retain the current block plus the in-flight
        # prefetches; the window keeps slack for both to avoid self-eviction.
        self._max_cached_blocks = max(self._MAX_CACHED_BLOCKS, self._prefetch_ahead + 2)

    @property
    def metadata(self) -> SimulatorDataSourceMetadata:
        return self._metadata

    def open(self) -> None:
        self._ensure_executor()
        self._spike_threshold = self._compute_spike_threshold()
        # Pre-compute spikes for the blocks already generated during threshold
        # calibration so the producer hot read loop sees uniform, cheap reads.
        with self._block_lock:
            calibrated_blocks = list(self._block_cache.keys())
        for block_index in calibrated_blocks:
            self._block_spikes(block_index)

        start_block = self._replay_start_offset // self._BLOCK_FRAMES
        self._block_spikes(start_block)

        # Print a message informing the user what seed is being used
        print(f"Random data source is using seed: {self._random_seed}")

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        with self._block_lock:
            self._block_futures.clear()
            self._block_cache.clear()
        self._spike_cache.clear()
        self._spike_threshold = None

    def read(self, from_timestamp: int, frame_count: int) -> DataSourceBatch:
        return DataSourceBatch(
            frames = self._read_frames(from_timestamp, frame_count),
            spikes = self._read_spikes(from_timestamp, frame_count),
        )

    def _generate_random_seed(self) -> int:
        """Generate a random seed."""
        seed_sequence = np.random.SeedSequence(None)
        return seed_sequence.generate_state(1)[0]

    def _read_frames(self, from_timestamp: int, frame_count: int) -> np.ndarray:
        elapsed   = from_timestamp - self._metadata.start_timestamp
        start_idx = elapsed + self._replay_start_offset
        return self._read_samples(start_idx, frame_count)

    def _read_spikes(self, from_timestamp: int, frame_count: int) -> list[DataSourceSpike]:
        elapsed  = from_timestamp - self._metadata.start_timestamp
        offset   = elapsed + self._replay_start_offset
        from_idx = offset
        to_idx   = offset + frame_count

        result: list[DataSourceSpike] = []
        for frame, channel, samples in self._spikes_in_range(from_idx, to_idx):
            spike_ts = frame - self._replay_start_offset + self._metadata.start_timestamp
            result.append(DataSourceSpike(
                timestamp           = spike_ts,
                channel             = channel,
                channel_mean_sample = float(self._sample_mean),
                samples             = samples,
            ))
        return result

    def _spikes_in_range(self, from_idx: int, to_idx: int) -> list[tuple[int, int, np.ndarray]]:
        if to_idx <= from_idx:
            return []

        first_block = from_idx // self._BLOCK_FRAMES
        last_block  = (to_idx - 1) // self._BLOCK_FRAMES
        result: list[tuple[int, int, np.ndarray]] = []
        for block_index in range(first_block, last_block + 1):
            cached = self._spike_cache.get(block_index)
            if cached is not None:
                self._spike_cache.move_to_end(block_index)
                frames, spikes = cached
                if not frames:
                    continue
                lo = bisect.bisect_left(frames, from_idx)
                hi = bisect.bisect_left(frames, to_idx)
                if lo < hi:
                    result.extend(spikes[lo:hi])
                continue

            block_start = block_index * self._BLOCK_FRAMES
            base        = self._base_block(block_index)
            rel_from    = max(0, from_idx - block_start)
            rel_to      = min(base.shape[0], to_idx - block_start)
            if rel_to <= rel_from:
                continue

            # Small accelerated reads should not force full-block spike detection.
            # Scan only the requested slice; full blocks still get cached below.
            if rel_from == 0 and rel_to == base.shape[0]:
                frames, spikes = self._block_spikes(block_index)
                if spikes:
                    result.extend(spikes)
                continue

            candidate_frames, candidate_channels = np.where(
                np.abs(base[rel_from:rel_to]) > self._spike_threshold_value()
            )
            for slice_frame, channel in zip(candidate_frames, candidate_channels, strict=True):
                spike = self._spike_from_candidate(
                    base,
                    block_start,
                    rel_from + int(slice_frame),
                    int(channel),
                )
                if spike is not None:
                    result.append(spike)
        return result

    def _block_spikes(self, block_index: int) -> tuple[list[int], list[tuple[int, int, np.ndarray]]]:
        cached = self._spike_cache.get(block_index)
        if cached is not None:
            self._spike_cache.move_to_end(block_index)
            return cached

        threshold   = self._spike_threshold_value()
        block_start = block_index * self._BLOCK_FRAMES
        base        = self._base_block(block_index)
        abs_base    = np.abs(base)

        candidate_frames, candidate_channels = np.where(abs_base > threshold)
        frames: list[int] = []
        spikes: list[tuple[int, int, np.ndarray]] = []
        for relative_frame, channel in zip(candidate_frames, candidate_channels, strict=True):
            spike = self._spike_from_candidate(
                base,
                block_start,
                int(relative_frame),
                int(channel),
            )
            if spike is None:
                continue
            frames.append(spike[0])
            spikes.append(spike)

        value = (frames, spikes)
        self._spike_cache[block_index] = value
        if len(self._spike_cache) > self._max_cached_blocks:
            self._spike_cache.popitem(last=False)
        return value

    def _spike_from_candidate(
        self,
        base          : np.ndarray,
        block_start   : int,
        relative_frame: int,
        channel       : int,
    ) -> tuple[int, int, np.ndarray] | None:
        frame = block_start + relative_frame
        if (
            frame < SPIKE_SAMPLES_BEFORE or
            not self._spike_channel_mask[channel]
        ):
            return None

        window_start = relative_frame - SPIKE_SAMPLES_BEFORE
        window_end   = window_start + SPIKE_SAMPLES_TOTAL
        if not self._amplify_spikes and window_start >= 0 and window_end <= base.shape[0]:
            window = base[window_start:window_end, channel]
        else:
            window = self._read_samples(frame - SPIKE_SAMPLES_BEFORE, SPIKE_SAMPLES_TOTAL)[:, channel]

        return (
            frame,
            channel,
            np.ascontiguousarray(window * self._uV_per_sample_unit, dtype=np.float32),
        )

    def _spike_threshold_value(self) -> float:
        if self._spike_threshold is None:
            self._spike_threshold = self._compute_spike_threshold()
        return self._spike_threshold

    def _compute_spike_threshold(self) -> float:
        calibration_frames = self._CALIBRATION_FRAMES
        sample_count       = calibration_frames * self._channel_count
        abs_samples        = np.empty(sample_count, dtype=np.int16)

        filled = 0
        idx    = 0
        while idx < calibration_frames:
            block_index = idx // self._BLOCK_FRAMES
            block       = self._base_block(block_index)
            within      = idx - block_index * self._BLOCK_FRAMES
            take        = min(calibration_frames - idx, block.shape[0] - within)
            out_view    = abs_samples[
                filled:filled + take * self._channel_count
            ].reshape(take, self._channel_count)
            np.absolute(block[within:within + take], out=out_view)
            filled += take * self._channel_count
            idx    += take

        return float(np.percentile(abs_samples, self._spike_percentile))

    def _read_samples(self, start_idx: int, count: int) -> np.ndarray:
        samples = self._base_samples(start_idx, count)
        if self._amplify_spikes and count > 0:
            samples                            = samples.copy()
            threshold                          = self._spike_threshold_value()
            mask                               = np.abs(samples) > threshold
            mask[:, self._GROUNDED_CHANNELS]   = False
            mask[:, self._REFERENCE_CHANNELS]  = False
            frame_indices                      = start_idx + np.arange(count)
            edge                               = frame_indices < SPIKE_SAMPLES_BEFORE
            mask[edge, :]                      = False
            samples[mask]                     *= 3
        return samples

    def _base_samples(self, start_idx: int, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0, self._channel_count), dtype=np.int16)

        block_index = start_idx // self._BLOCK_FRAMES
        block       = self._base_block(block_index)
        within      = start_idx - block_index * self._BLOCK_FRAMES
        if within + count <= block.shape[0]:
            return block[within:within + count]

        out    = np.empty((count, self._channel_count), dtype=np.int16)
        filled = 0
        idx    = start_idx
        while filled < count:
            block_index = idx // self._BLOCK_FRAMES
            block       = self._base_block(block_index)
            within      = idx - block_index * self._BLOCK_FRAMES
            take        = min(count - filled, block.shape[0] - within)
            out[filled:filled + take] = block[within:within + take]
            filled += take
            idx    += take
        return out

    def _base_block(self, block_index: int) -> np.ndarray:
        with self._block_lock:
            cached = self._block_cache.get(block_index)
            if cached is not None:
                self._block_cache.move_to_end(block_index)
                self._schedule_prefetch_locked(block_index + 1)
                return cached

            future = self._block_futures.get(block_index)
            if future is None and self._executor is not None:
                future = self._submit_block_locked(block_index)

        block = future.result() if future is not None else self._compute_block(block_index)

        with self._block_lock:
            self._store_block_locked(block_index, block)
            self._schedule_prefetch_locked(block_index + 1)
        return block

    def _compute_block(self, block_index: int) -> np.ndarray:
        rng     = np.random.default_rng(np.random.SeedSequence([self._random_seed, block_index]))
        samples = rng.poisson(
            self._sample_mean,
            size=(self._BLOCK_FRAMES, self._channel_count),
        ).astype(np.int16) - self._sample_mean

        samples[:, self._GROUNDED_CHANNELS]    = 0
        samples[:, self._REFERENCE_CHANNELS] //= 2  # The reference channel is less noisy than the others (mimic hardware behaviour)
        return samples

    def _ensure_executor(self) -> None:
        if self._prefetch_workers <= 0 or self._executor is not None:
            return
        self._executor = ThreadPoolExecutor(
            max_workers        = self._prefetch_workers,
            thread_name_prefix = "cl-random-source",
        )

    def _submit_block_locked(self, block_index: int) -> Future[np.ndarray]:
        assert self._executor is not None
        future = self._executor.submit(self._compute_block, block_index)
        self._block_futures[block_index] = future
        return future

    def _store_block_locked(self, block_index: int, block: np.ndarray) -> None:
        self._block_futures.pop(block_index, None)
        self._block_cache[block_index] = block
        self._block_cache.move_to_end(block_index)
        while len(self._block_cache) > self._max_cached_blocks:
            self._block_cache.popitem(last=False)

    def _schedule_prefetch_locked(self, start_index: int) -> None:
        if self._executor is None:
            return
        for offset in range(self._prefetch_ahead):
            index = start_index + offset
            if index in self._block_cache or index in self._block_futures:
                continue
            self._submit_block_locked(index)

def set_simulator_data_source(
    factory : str | Callable[..., SimulatorDataSource],
    config  : Mapping[str, Any] | None = None,
    metadata: SimulatorDataSourceMetadata | Mapping[str, Any] | None = None,
) -> None:
    """
    Register a custom simulator data source for subsequent simulator opens.

    The next time the SDK opens the simulator (e.g. via `cl.neurons()`), the
    producer subprocess will import `factory` and call it with `**config` to
    construct its `SimulatorDataSource`. Sources registered here take
    precedence over the `CL_SDK_DATA_SOURCE` / `CL_SDK_DATA_SOURCE_CONFIG`
    environment variables.

    Args:
        factory: Either a `"module:attribute"` / `"module.attribute"` import
            path string, or a callable defined at module level in an
            importable package. Lambdas, local functions, and factories
            defined in `__main__`/REPL/notebooks are rejected because the
            simulator subprocess cannot re-import them by name; move such
            factories into a `.py` file inside an importable package.
        config: Keyword arguments forwarded to `factory()` in the subprocess.
            Must be JSON-serialisable. Defaults to no arguments.
        metadata: Optional static metadata for the source. Supplying this avoids
            constructing the source in the parent process before opening.

    Raises:
        ValueError: if `factory` is not importable by name.
        TypeError: if `config` is not JSON-serialisable.

    Calling this also invalidates any cached `Neurons` instance so the next
    open picks up the new source. Pair with `clear_simulator_data_source()`
    to revert to the default (random/file replay) source.
    """
    global _simulator_data_source_spec

    factory_path = _factory_to_import_path(factory)
    config_dict  = dict(config or {})
    _validate_json_serialisable(config_dict)
    metadata_obj                = validate_metadata(metadata) if metadata is not None else None
    _simulator_data_source_spec = _SimulatorDataSourceSpec(factory_path, config_dict, metadata_obj)

    with contextlib.suppress(ImportError):
        from cl.neurons import Neurons
        Neurons._clear_instance()

def clear_simulator_data_source() -> None:
    """
    Unregister any custom simulator data source previously set.

    After this call, subsequent simulator opens fall back to the
    `CL_SDK_DATA_SOURCE` env-var configuration if set, otherwise to the
    default behaviour (replay `CL_SDK_REPLAY_PATH` if set, else use the built-in
    random on-the-fly simulator source). Also invalidates any cached `Neurons`
    instance so the change takes effect on next open.

    No-op if no custom source was registered.
    """
    global _simulator_data_source_spec
    _simulator_data_source_spec = None

    with contextlib.suppress(ImportError):
        from cl.neurons import Neurons
        Neurons._clear_instance()

def get_configured_data_source_config() -> dict[str, Any] | None:
    """Return custom source config from setter/env, or ``None``."""
    if _simulator_data_source_spec is not None:
        return _simulator_data_source_spec.to_process_config()

    factory = os.getenv(DATA_SOURCE_ENV)
    if not factory:
        return None

    raw_config = os.getenv(DATA_SOURCE_CONFIG_ENV, "{}")
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as e:
        raise ValueError(f"{DATA_SOURCE_CONFIG_ENV} must be valid JSON") from e
    if not isinstance(config, dict):
        raise ValueError(f"{DATA_SOURCE_CONFIG_ENV} must decode to a JSON object")

    _validate_json_serialisable(config)
    result = {
        "kind"   : "factory",
        "factory": factory,
        "config" : config,
    }
    raw_metadata = os.getenv(DATA_SOURCE_METADATA_ENV)
    if raw_metadata:
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError as e:
            raise ValueError(f"{DATA_SOURCE_METADATA_ENV} must be valid JSON") from e
        result["metadata"] = asdict(validate_metadata(metadata))
    return result

def default_data_source_config(replay_start_offset: int = 0) -> dict[str, Any]:
    """Build process config for the default simulator source."""
    replay_path = os.getenv("CL_SDK_REPLAY_PATH")
    if replay_path:
        return file_recording_data_source_config(
            replay_path,
            replay_start_offset=replay_start_offset,
        )
    return random_data_source_config(replay_start_offset=replay_start_offset)

def file_recording_data_source_config(
    replay_file_path   : str | Path,
    replay_start_offset: int = 0,
) -> dict[str, Any]:
    """Build process config for the built-in recording-file replay source."""
    return {
        "kind"               : "file_recording",
        "replay_file_path"   : str(replay_file_path),
        "replay_start_offset": int(replay_start_offset),
    }

def random_data_source_config(
    replay_start_offset: int = 0,
    sample_mean        : int | None = None,
    spike_percentile   : float | None = None,
    random_seed        : int | None = None,
    amplify_spikes     : bool | None = None,
) -> dict[str, Any]:
    """Build process config for the built-in on-the-fly random source."""
    config: dict[str, Any] = {
        "kind"               : "random",
        "replay_start_offset": int(replay_start_offset),
    }
    optional_values = {
        "sample_mean"     : sample_mean,
        "spike_percentile": spike_percentile,
        "random_seed"     : random_seed,
        "amplify_spikes"  : amplify_spikes,
    }
    config.update({name: value for name, value in optional_values.items() if value is not None})
    return config

def create_data_source_from_config(config: Mapping[str, Any]) -> SimulatorDataSource:
    """Instantiate a simulator data source from process config."""
    kind = config.get("kind")
    if kind == "file_recording":
        return FileRecordingDataSource(
            replay_file_path    = str(config["replay_file_path"]),
            replay_start_offset = int(config.get("replay_start_offset", 0)),
        )

    if kind == "random":
        return RandomDataSource(
            replay_start_offset = int(config.get("replay_start_offset", 0)),
            sample_mean         = config.get("sample_mean"),
            spike_percentile    = config.get("spike_percentile"),
            random_seed         = config.get("random_seed"),
            amplify_spikes      = config.get("amplify_spikes"),
        )

    if kind == "factory":
        factory = _import_attr(str(config["factory"]))
        kwargs  = dict(config.get("config") or {})
        source  = factory(**kwargs)
        if not isinstance(source, SimulatorDataSource):
            raise TypeError(
                f"Simulator data source factory {config['factory']!r} returned "
                f"{type(source).__name__}, expected SimulatorDataSource"
            )
        return source

    raise ValueError(f"Unknown simulator data source kind: {kind!r}")

def load_data_source_metadata(config: Mapping[str, Any]) -> SimulatorDataSourceMetadata:
    """Return source metadata, constructing the source only when needed."""
    if "metadata" in config:
        return validate_metadata(config["metadata"])

    source = create_data_source_from_config(config)
    try:
        return validate_metadata(source.metadata)
    finally:
        with contextlib.suppress(Exception):
            source.close()

def validate_metadata(metadata: SimulatorDataSourceMetadata | Mapping[str, Any]) -> SimulatorDataSourceMetadata:
    """Coerce and validate source metadata."""
    if isinstance(metadata, Mapping):
        metadata = SimulatorDataSourceMetadata(**metadata)
    if not isinstance(metadata, SimulatorDataSourceMetadata):
        raise TypeError(
            f"metadata must be SimulatorDataSourceMetadata, got {type(metadata).__name__}"
        )
    if metadata.channel_count <= 0:
        raise ValueError("channel_count must be positive")
    if metadata.frames_per_second != DEFAULT_FRAMES_PER_SECOND:
        raise ValueError(
            f"frames_per_second must be {DEFAULT_FRAMES_PER_SECOND}; "
            f"got {metadata.frames_per_second}"
        )
    if metadata.start_timestamp < 0:
        raise ValueError("start_timestamp must be non-negative")
    if metadata.duration_frames is not None and metadata.duration_frames <= 0:
        raise ValueError("duration_frames must be positive when provided")
    return metadata

def _factory_to_import_path(factory: str | Callable[..., SimulatorDataSource]) -> str:
    if isinstance(factory, str):
        _import_attr(factory)
        return factory

    module     = getattr(factory, "__module__", "") or ""
    qualname   = getattr(factory, "__qualname__", "") or ""
    importable = module and qualname and module != "__main__" and "<locals>" not in qualname
    if importable:
        try:
            if _import_attr(f"{module}:{qualname}") is factory:
                return f"{module}:{qualname}"
        except (ImportError, AttributeError, ValueError):
            pass

    where = f"{module}:{qualname}" if module or qualname else repr(factory)
    raise ValueError(
        f"Simulator data source factory {where} is not importable by the "
        "simulator subprocess. Define the factory at module level in an "
        "installed/importable package (not in __main__, a REPL, a notebook, "
        "or inside another function) and pass it, or its 'module:factory' "
        "import path, to set_simulator_data_source()."
    )

def _import_attr(import_path: str) -> Any:
    if ":" in import_path:
        module_name, attr_path = import_path.split(":", 1)
    else:
        module_name, _, attr_path = import_path.rpartition(".")
    if not module_name or not attr_path:
        raise ValueError(
            "Simulator data source factory must use 'module:factory' or 'module.factory'"
        )

    module = importlib.import_module(module_name)
    attr   = module
    for part in attr_path.split("."):
        attr = getattr(attr, part)
    return attr

def _validate_json_serialisable(config: Mapping[str, Any]) -> None:
    try:
        json.dumps(config)
    except TypeError as e:
        raise TypeError("Simulator data source config must be JSON-serialisable") from e
