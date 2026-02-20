"""
Threaded HDF5 recording writer for non-blocking data persistence.

This module provides a background writer that receives data via a queue
and writes it incrementally to an HDF5 file, avoiding blocking the main thread.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

import numpy as np
import tables
from numpy import ndarray

if TYPE_CHECKING:
    from pathlib import Path

    from . import Spike, Stim

# Queue batch processing settings
QUEUE_TIMEOUT_SECS = 0.2  # How long to wait for items when queue is empty
BATCH_SIZE         = 100  # Max items to process per batch for efficiency

# Flush data stream tables every N rows to prevent buffer bloat
DATA_STREAM_FLUSH_ROWS = 100

class SpikeRow(tables.IsDescription):
    """Descriptor for a row of spike data within recording_file.root.spikes table."""

    timestamp = tables.Int64Col(pos=0)
    """ Timestamp of the spike. """

    channel = tables.UInt8Col(pos=1)
    """ Channel that spiked. """

    samples = tables.Float32Col(shape=(75,))
    """
    25 samples before + 50 samples from the time of the spike,
    shifted by the channel mean and converted to µV.
    """

class StimRow(tables.IsDescription):
    """Descriptor for a row of stim data within recording_file.root.stims table."""

    timestamp = tables.Int64Col(pos=0)
    """ Timestamp of the stim. """

    channel = tables.UInt8Col(pos=1)
    """ Channel that stim was conducted. """

class DataStreamIndexRow(tables.IsDescription):
    timestamp = tables.Int64Col(pos=0)
    """ Timestamp of the datastream row. """

    start_index = tables.UInt64Col(pos=1)
    """ Index of first byte in data array. """

    end_index = tables.UInt64Col(pos=2)
    """ Index + 1 of last byte in data array. """

@dataclass
class SamplesBatch:
    """A batch of raw sample frames to write."""
    samples: ndarray

@dataclass
class SpikeBatch:
    """A batch of spikes to write."""
    spikes: list[Spike]

@dataclass
class StimBatch:
    """A batch of stims to write."""
    stims: list[Stim]

@dataclass
class DataStreamEvent:
    """A data stream event to write."""
    stream_name: str
    timestamp  : int
    data       : bytes

@dataclass
class DataStreamInit:
    """Initialize a data stream group in the H5 file."""
    stream_name: str
    attributes : dict[str, Any]

class DataStreamState:
    """Mutable state for a data stream (avoids tuple allocation on every event)."""
    __slots__ = ('data_array', 'index_table', 'next_data_index', 'rows_since_flush')

    def __init__(self, index_table: tables.Table, data_array: tables.EArray):
        self.index_table     : tables.Table  = index_table
        self.data_array      : tables.EArray = data_array
        self.next_data_index : int           = 0
        self.rows_since_flush: int           = 0

# Sentinel value to signal writer shutdown
class Shutdown:
    pass

# Type for queue items
type QueueItem = SamplesBatch | SpikeBatch | StimBatch | DataStreamEvent | DataStreamInit | Shutdown

class RecordingWriter:
    """
    Background thread writer for HDF5 recording files.

    Opens the H5 file immediately and writes data as it arrives via a queue.
    This allows non-blocking recording from the main application thread.
    """

    def __init__(
        self,
        file_path           : Path,
        channel_count       : int,
        start_timestamp     : int,
        include_spikes      : bool = True,
        include_stims       : bool = True,
        include_raw_samples : bool = True,
        include_data_streams: bool = True,
        exclude_data_streams: list[str] | set[str] | None = None,
        initial_attributes  : dict[str, Any] | None = None,
    ):
        """
        Initialize the recording writer.

        Args:
            file_path           : Path to the H5 file to create.
            channel_count       : Number of channels for sample data.
            start_timestamp     : Starting timestamp for relative time calculations.
            include_spikes      : Whether to record spikes.
            include_stims       : Whether to record stims.
            include_raw_samples : Whether to record raw sample data.
            include_data_streams: Whether to record data streams.
            exclude_data_streams: List of data stream names to exclude.
            initial_attributes  : Initial attributes to write to the file root.
        """
        self._file_path            = file_path
        self._channel_count        = channel_count
        self._start_timestamp      = start_timestamp
        self._include_spikes       = include_spikes
        self._include_stims        = include_stims
        self._include_raw_samples  = include_raw_samples
        self._include_data_streams = include_data_streams
        # Convert to set for O(1) lookup (important with GC disabled)
        self._exclude_data_streams: set[str] = set(exclude_data_streams) if exclude_data_streams else set()
        self._initial_attributes = initial_attributes or {}

        # Queue for incoming data
        self._queue: Queue[QueueItem] = Queue()

        # Thread management
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False

        # H5 file handle (only accessed from writer thread)
        self._h5_file   : tables.File | None   = None
        self._h5_spikes : tables.Table | None  = None
        self._h5_stims  : tables.Table | None  = None
        self._h5_samples: tables.EArray | None = None

        # Data stream tracking (stream_name -> mutable state object)
        self._data_streams: dict[str, DataStreamState] = {}

        self._logger = logging.getLogger(f"{file_path.name}.RecordingWriter")

    def start(self) -> None:
        """Start the background writer thread."""
        if self._started:
            return

        self._started = True
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        self._logger.debug("RecordingWriter started for %s", self._file_path)

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the writer thread and finalize the file.

        Args:
            timeout: Maximum seconds to wait for the writer thread to finish.
        """
        if self._stopped:
            return

        self._stopped = True
        self._queue.put(Shutdown())

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self._logger.warning("RecordingWriter thread did not stop within timeout")

    def update_attributes(self, attributes: dict[str, Any]) -> None:
        """
        Update file root attributes. These will be written when the file is finalized.

        Note: This queues an attribute update but actual writing happens in finalize.
        For simplicity, we store updates and apply them at close time.
        """
        # Thread-safe update of initial attributes dict
        self._initial_attributes.update(attributes)

    def write_samples(self, samples: ndarray) -> None:
        """Queue a batch of sample frames to write."""
        if not self._include_raw_samples or self._stopped:
            return
        self._queue.put(SamplesBatch(samples=samples))

    def write_spikes(self, spikes: list[Spike]) -> None:
        """Queue a batch of spikes to write."""
        if not self._include_spikes or self._stopped or not spikes:
            return
        self._queue.put(SpikeBatch(spikes=spikes))

    def write_stims(self, stims: list[Stim]) -> None:
        """Queue a batch of stims to write."""
        if not self._include_stims or self._stopped or not stims:
            return
        self._queue.put(StimBatch(stims=stims))

    def init_data_stream(self, stream_name: str, attributes: dict[str, Any]) -> None:
        """Initialize a data stream in the H5 file."""
        if not self._include_data_streams or self._stopped:
            return
        if stream_name in self._exclude_data_streams:
            return
        self._queue.put(DataStreamInit(stream_name=stream_name, attributes=attributes))

    def write_data_stream_event(
        self,
        stream_name: str,
        timestamp  : int,
        data       : bytes
    ) -> None:
        """Queue a data stream event to write."""
        if not self._include_data_streams or self._stopped:
            return
        if stream_name in self._exclude_data_streams:
            return
        self._queue.put(DataStreamEvent(stream_name=stream_name, timestamp=timestamp, data=data))

    def _writer_loop(self) -> None:
        """Main loop for the writer thread with batching."""
        try:
            self._open_file()

            while True:
                # Blocking get - no timeout, no exception overhead when idle
                item = self._queue.get()

                if isinstance(item, Shutdown):
                    # Drain remaining items before shutdown
                    self._drain_queue()
                    break

                # Collect a batch of items to process together
                batch: list[QueueItem] = [item]

                # Try to get up to BATCH_SIZE-1 more items without blocking
                for _ in range(BATCH_SIZE - 1):
                    try:
                        item = self._queue.get_nowait()
                        if isinstance(item, Shutdown):
                            # Process what we have, then exit
                            self._process_batch(batch)
                            self._drain_queue()
                            return
                        batch.append(item)
                    except Empty:
                        break

                # Process the entire batch
                self._process_batch(batch)

        except Exception as e:
            self._logger.error("RecordingWriter error: %s", e)
        finally:
            self._close_file()

    def _drain_queue(self) -> None:
        """Process all remaining items in the queue."""
        batch = []

        while True:
            try:
                item = self._queue.get_nowait()
                if isinstance(item, Shutdown):
                    continue  # Skip additional shutdown signals
                batch.append(item)

                # Process in chunks to avoid building huge batch
                if len(batch) >= BATCH_SIZE:
                    self._process_batch(batch)
                    batch = []
            except Empty:
                break

        # Process any remaining items
        if batch:
            self._process_batch(batch)

    def _process_batch(self, batch: list[QueueItem]) -> None:
        """Process a batch of queue items efficiently by grouping by type."""
        # Group items by type for efficient batch processing
        samples_batches             : list[SamplesBatch]               = []
        spike_batches               : list[SpikeBatch]                 = []
        stim_batches                : list[StimBatch]                  = []
        data_stream_inits           : list[DataStreamInit]             = []
        data_stream_events_by_stream: dict[str, list[DataStreamEvent]] = {}

        for item in batch:
            match item:
                case SamplesBatch():
                    samples_batches.append(item)
                case SpikeBatch():
                    spike_batches.append(item)
                case StimBatch():
                    stim_batches.append(item)
                case DataStreamInit():
                    data_stream_inits.append(item)
                case DataStreamEvent():
                    if item.stream_name not in data_stream_events_by_stream:
                        data_stream_events_by_stream[item.stream_name] = []
                    data_stream_events_by_stream[item.stream_name].append(item)

        # Process samples batches
        for samples_batch in samples_batches:
            self._write_samples_batch(samples_batch)

        # Process all spikes together
        if spike_batches:
            self._write_spikes_batches(spike_batches)

        # Process all stims together
        if stim_batches:
            self._write_stims_batches(stim_batches)

        # Initialize data streams
        for init in data_stream_inits:
            self._init_data_stream(init)

        # Process data stream events in batches per stream
        for stream_name, events in data_stream_events_by_stream.items():
            self._write_data_stream_events_batch(stream_name, events)

    def _write_spikes_batches(self, batches: list[SpikeBatch]) -> None:
        """Write multiple spike batches efficiently."""
        if self._h5_spikes is None:
            return

        row_count = 0
        for batch in batches:
            for spike in batch.spikes:
                row = self._h5_spikes.row
                row["timestamp"] = spike.timestamp - self._start_timestamp
                row["channel"] = spike.channel
                row["samples"] = spike.samples
                row.append()
                row_count += 1

        # Flush once for the entire batch group
        if row_count > 0:
            self._h5_spikes.flush()

    def _write_stims_batches(self, batches: list[StimBatch]) -> None:
        """Write multiple stim batches efficiently."""
        if self._h5_stims is None:
            return

        row_count = 0
        for batch in batches:
            for stim in batch.stims:
                row = self._h5_stims.row
                row["timestamp"] = stim.timestamp - self._start_timestamp
                row["channel"] = stim.channel
                row.append()
                row_count += 1

        # Flush once for the entire batch group
        if row_count > 0:
            self._h5_stims.flush()

    def _write_data_stream_events_batch(self, stream_name: str, events: list[DataStreamEvent]) -> None:
        """Write multiple data stream events for the same stream efficiently."""
        if stream_name not in self._data_streams:
            # Auto-initialize stream if not already done
            self._init_data_stream(DataStreamInit(stream_name=stream_name, attributes={}))

        state = self._data_streams.get(stream_name)
        if state is None:
            return  # Initialization failed

        # Batch append all index rows
        for event in events:
            start_index = state.next_data_index
            data_len = len(event.data)
            state.next_data_index += data_len
            end_index = state.next_data_index

            row = state.index_table.row
            row["timestamp"] = event.timestamp - self._start_timestamp
            row["start_index"] = start_index
            row["end_index"] = end_index
            row.append()

        # Concatenate all data bytes and append once
        if events:
            all_data = b"".join(event.data for event in events)
            state.data_array.append(np.frombuffer(all_data, dtype=np.uint8))

        # Update flush counter and flush if needed
        state.rows_since_flush += len(events)
        if state.rows_since_flush >= DATA_STREAM_FLUSH_ROWS:
            state.index_table.flush()
            state.data_array.flush()
            state.rows_since_flush = 0

    def _open_file(self) -> None:
        """Open the H5 file and create initial structure."""
        # Ensure parent directory exists
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        self._h5_file = tables.open_file(str(self._file_path), mode="w")

        # Write initial attributes
        for key, value in self._initial_attributes.items():
            self._h5_file.root._v_attrs[key] = value

        # Create spikes table
        if self._include_spikes:
            self._h5_spikes = self._h5_file.create_table(
                where        = "/",
                name         = "spikes",
                description  = SpikeRow,
                expectedrows = 10_000_000,
                filters      = None,
            )
            # Index is created at close time to avoid O(log n) overhead per append

        # Create stims table
        if self._include_stims:
            self._h5_stims = self._h5_file.create_table(
                where        = "/",
                name         = "stims",
                description  = StimRow,
                expectedrows = 10_000_000,
                filters      = None,
            )
            # Index is created at close time to avoid O(log n) overhead per append

        # Create samples array
        if self._include_raw_samples:
            self._h5_samples = self._h5_file.create_earray(
                where      = "/",
                name       = "samples",
                atom       = tables.Int16Atom(),
                shape      = (0, self._channel_count),
                chunkshape = (256, self._channel_count),
                filters    = None,
            )

        # Create data_stream group if needed
        if self._include_data_streams:
            self._h5_file.create_group("/", "data_stream")

        self._logger.debug("H5 file opened: %s", self._file_path)

    def _close_file(self) -> None:
        """Finalize and close the H5 file."""
        if self._h5_file is None:
            return

        try:
            # Create indexes now that all data has been written
            # (avoids O(log n) overhead per append during recording)
            if self._h5_spikes is not None:
                self._h5_spikes.cols.timestamp.create_index()
            if self._h5_stims is not None:
                self._h5_stims.cols.timestamp.create_index()
            for state in self._data_streams.values():
                state.index_table.cols.timestamp.create_index()

            # Close any open data stream tables
            for state in self._data_streams.values():
                state.index_table.close()
                state.data_array.close()

            # Update final attributes
            for key, value in self._initial_attributes.items():
                self._h5_file.root._v_attrs[key] = value

            # Flush and close file tables
            if self._h5_spikes is not None:
                self._h5_spikes.close()
            if self._h5_stims is not None:
                self._h5_stims.close()
            if self._h5_samples is not None:
                self._h5_samples.close()

            self._h5_file.close()
            self._logger.debug("H5 file closed: %s", self._file_path)

        except Exception as e:
            self._logger.error("Error closing H5 file: %s", e)

    def _write_samples_batch(self, batch: SamplesBatch) -> None:
        """Write a batch of sample frames."""
        if self._h5_samples is not None:
            self._h5_samples.append(batch.samples)

    def _write_spikes_batch(self, batch: SpikeBatch) -> None:
        """Write a batch of spikes."""
        if self._h5_spikes is None:
            return

        for spike in batch.spikes:
            row = self._h5_spikes.row
            row["timestamp"] = spike.timestamp - self._start_timestamp
            row["channel"]   = spike.channel
            row["samples"]   = spike.samples
            row.append()

        self._h5_spikes.flush()

    def _write_stims_batch(self, batch: StimBatch) -> None:
        """Write a batch of stims."""
        if self._h5_stims is None:
            return

        for stim in batch.stims:
            row = self._h5_stims.row
            row["timestamp"] = stim.timestamp - self._start_timestamp
            row["channel"]   = stim.channel
            row.append()

        self._h5_stims.flush()

    def _init_data_stream(self, init: DataStreamInit) -> None:
        """Initialize a data stream in the H5 file."""
        if self._h5_file is None:
            return
        if init.stream_name in self._data_streams:
            return  # Already initialized

        # Create group for this data stream
        group = self._h5_file.create_group("/data_stream", init.stream_name)
        group._v_attrs["name"]        = init.stream_name
        group._v_attrs["application"] = init.attributes

        # Create index table
        # Note: Index is created at close time to avoid O(log n) overhead per append
        index_table = self._h5_file.create_table(
            where       = group,
            name        = "index",
            description = DataStreamIndexRow,
        )

        # Create data array
        data_array = self._h5_file.create_earray(
            where      = group,
            name       = "data",
            atom       = tables.UInt8Atom(),
            shape      = (0,),
            chunkshape = (2**15,),
        )

        self._data_streams[init.stream_name] = DataStreamState(index_table, data_array)
