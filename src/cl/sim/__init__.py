"""
# Cortical Labs SDK: Simulator-Only Components

## Overview

This **simulator-only** module defines the pluggable data-source layer that feeds
the SDK's in-process simulator. A data source produces sample frames
and may also produce spikes. It may react to committed stimulation
events, so closed-loop tests can model stim-to-spike responses without running
the real data provider.

The SDK ships two built-in sources:

- **File recording replay**: replays an HDF5 recording selected with
  `CL_SDK_REPLAY_PATH`.
- **Random source**: generates deterministic synthetic frames and spikes on
  demand. This is the default when no replay file or custom source is set.

Custom sources can be backed by hardware, a network stream, a model, a test
fixture, or a precomputed dataset. As long as the source exposes stable
metadata and returns frames at the SDK sample rate, `cl.neurons`, recording,
analysis, and WebSocket visualisation use it like any other simulator source.

## Concepts

- **Frames** are the atomic neural data unit: an `int16` array shaped
  `(frame_count, channel_count)` in raw sample units. Use
  `SimulatorDataSourceMetadata.uV_per_sample_unit` to convert to microvolts.
- **Spikes** are `DataSourceSpike` records. A source can return spikes inline
  with frames or emit them asynchronously from a live source.
- **Stims** are committed `Neurons.stim()` events delivered to
  `on_stim(stim)` / `on_stims(stims)` in the producer subprocess.
- **Batches** (`DataSourceBatch`) are returned by `read()`: frames plus any
  spikes whose timestamps fall in that read window.
- **Metadata** (`SimulatorDataSourceMetadata`) describes channel count, sample
  rate, start timestamp, duration, seekability, real-time constraints, and
  accelerated-time support. The sample rate must currently be 25 kHz.

## Source Types

There are two source styles:

1. **Pull sources** subclass `SimulatorDataSource` and implement
   `read(from_timestamp, frame_count)`. The simulator asks them for each block.
   Use these for seekable files, deterministic fixtures, generated data, or
   models that can produce arbitrary timestamp ranges.
2. **Push/live sources** subclass `LiveSimulatorDataSource` and implement
   `start(sink)`. They push frames and spikes into a `LiveDataSink` from their
   own thread, event loop, or external process. The base class buffers data and
   serves sequential `read()` calls. Live sources are forced non-seekable,
   real-time only, and not accelerated unless `supports_accelerated=True`.

## Registering a Custom Source

Register custom sources by import path so the simulator subprocess can build
them independently:

```python
import cl

cl.sim.set_simulator_data_source(
    "my_package.my_module:my_source_factory",
    config={"param1": 42, "param2": "hello"},
    metadata=cl.sim.SimulatorDataSourceMetadata(channel_count=64),
)
```

The first argument is a `"module:attribute"` string, or an importable callable
that the SDK can resolve to that form. The optional `config` dict is forwarded
as keyword arguments to the factory in the simulator subprocess and must be
JSON-serialisable.

Pass `metadata=` when it is cheap and static. This avoids constructing the
source in the parent process just to inspect metadata. If metadata is omitted,
the SDK constructs the source once in the parent, reads `source.metadata`, then
passes that validated metadata to the producer process.

The producer always constructs the real source and validates its runtime
metadata against the parent metadata before opening. If the two differ, opening
fails. This catches sources whose metadata depends on process-local state,
environment, random choices, or mutable config.

To revert to the default source, call:

```python
cl.sim.clear_simulator_data_source()
```

The same configuration can be applied by environment variables:

- `CL_SDK_DATA_SOURCE`: import path of the factory.
- `CL_SDK_DATA_SOURCE_CONFIG`: JSON object of factory keyword arguments.
- `CL_SDK_DATA_SOURCE_METADATA`: JSON object of
  `SimulatorDataSourceMetadata` fields. Optional, but avoids parent-process
  construction for metadata discovery.

## Implementing a Pull Source

Pull sources are constructed in the parent process for metadata discovery
unless metadata is supplied at registration time, and are always constructed in
the simulator subprocess for real reads. Keep factories importable and avoid
depending on non-importable parent objects.

```python
import numpy as np

from cl.sim import (
    DataSourceBatch,
    DataSourceStim,
    SimulatorDataSource,
    SimulatorDataSourceMetadata,
)

class MyDataSource(SimulatorDataSource):
    def __init__(self, scale: float = 1.0):
        self._scale = scale

    @property
    def metadata(self) -> SimulatorDataSourceMetadata:
        return SimulatorDataSourceMetadata(
            channel_count   = 64,
            duration_frames = None,  # unbounded
            seekable        = True,
        )

    def open(self) -> None:
        # Acquire subprocess-local resources here.
        pass

    def close(self) -> None:
        pass

    def on_stim(self, stim: DataSourceStim) -> None:
        # Called when a stim has been committed to the simulator timeline.
        pass

    def read(self, from_timestamp: int, frame_count: int) -> DataSourceBatch:
        frames = (np.random.randn(frame_count, 64) * self._scale).astype(np.int16)
        return DataSourceBatch(frames=frames)

def my_source_factory(scale: float = 1.0) -> SimulatorDataSource:
    return MyDataSource(scale=scale)
```

## Implementing a Push/Live Source

Live sources are useful when frames arrive on their own clock, e.g. hardware,
a websocket, or another process. Implement `start(sink)`, emit frames with
`sink.emit_frames()` / `sink.emit_batch()`, and optionally emit reactive spikes
with `sink.emit_spikes()`.

Queued stims wake the simulator's stim publisher immediately. Outside
`neurons.loop()` and accelerated time, publishing no longer has to wait for the
next 10 ms heartbeat tick; the heartbeat is now the fallback clock while stim
publishing waits adaptively near the next due stim. A reactive live source can
therefore achieve sub-5 ms simulator timestamp latency when it keeps a shallow
buffer and places spikes at the earliest deliverable timestamp.

```python
import threading

import numpy as np

from cl.sim import (
    DataSourceSpike,
    DataSourceStim,
    LiveDataSink,
    LiveSimulatorDataSource,
    SimulatorDataSourceMetadata,
)

class MyLiveSource(LiveSimulatorDataSource):
    def __init__(self):
        super().__init__(
            metadata=SimulatorDataSourceMetadata(channel_count=64),
            max_buffer_frames=64,
            read_timeout_seconds=5.0,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sink: LiveDataSink | None = None

    def start(self, sink: LiveDataSink) -> None:
        self._sink = sink
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(sink,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def on_stim(self, stim: DataSourceStim) -> None:
        if self._sink is None:
            return

        # Spikes before the live read head have already been passed and are
        # dropped. Clamping to read_timestamp keeps the response deliverable
        # while avoiding extra latency from the live write-head buffer depth.
        spike_ts = max(stim.timestamp + 1, self._sink.read_timestamp)
        self._sink.emit_spikes(
            [
                DataSourceSpike(
                    timestamp=spike_ts,
                    channel=stim.channel,
                    samples=np.zeros(75, dtype=np.float32),
                )
            ]
        )

    def _run(self, sink: LiveDataSink) -> None:
        try:
            while not self._stop.is_set():
                frames = np.zeros((5, self.metadata.channel_count), dtype=np.int16)
                sink.emit_frames(frames)
        except RuntimeError:
            # The sink raises after the simulator closes the source.
            pass

def my_source_factory() -> LiveSimulatorDataSource:
    return MyLiveSource()
```
"""

from ._data_source import (
    DataSourceBatch,
    DataSourceSpike,
    DataSourceStim,
    LiveDataSink,
    LiveSimulatorDataSource,
    SimulatorDataSource,
    SimulatorDataSourceMetadata,
    clear_simulator_data_source,
    set_simulator_data_source,
)

__all__ = [
    "DataSourceBatch",
    "DataSourceSpike",
    "DataSourceStim",
    "LiveDataSink",
    "LiveSimulatorDataSource",
    "SimulatorDataSource",
    "SimulatorDataSourceMetadata",
    "clear_simulator_data_source",
    "set_simulator_data_source",
]
