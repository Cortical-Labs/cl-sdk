"""
Shared memory ring buffer for mock API data distribution.

This module provides a lock-free ring buffer implementation using Python's
multiprocessing.shared_memory for sharing waveform, spike, and stim data
between the data producer subprocess and consumers (neurons.read(),
closed loop, WebSocket server).

The buffer holds ~5.24 seconds of data at 25kHz internal sample rate (131,072 frames).
"""
from __future__ import annotations

import contextlib
import logging
import struct
import time
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from threading import Lock
from typing import ClassVar, Final, Self
from uuid import uuid4

import numpy as np
from numpy import ndarray

# Buffer sizing
DEFAULT_BUFFER_SIZE_FRAMES      = 131_072  # ~5.24 seconds at 25kHz buffer rate, power of 2
DEFAULT_FRAMES_PER_SECOND       = 25_000   # 25kHz buffer rate (external rate)
DEFAULT_BUFFER_DURATION_SECONDS = DEFAULT_BUFFER_SIZE_FRAMES / DEFAULT_FRAMES_PER_SECOND  # ~5.24 seconds
DEFAULT_CHANNEL_COUNT           = 64

# Spike buffer sizing (max spikes per second × buffer duration)
MAX_SPIKES_PER_SECOND = 10_000  # Conservative upper bound
DEFAULT_MAX_SPIKES    = int(MAX_SPIKES_PER_SECOND * DEFAULT_BUFFER_DURATION_SECONDS)

# Stim buffer sizing
MAX_STIMS_PER_SECOND = 12_000  # 60 channels @ 200 Hz each
DEFAULT_MAX_STIMS    = int(MAX_STIMS_PER_SECOND * DEFAULT_BUFFER_DURATION_SECONDS)

# Data stream event buffer sizing
MAX_DATASTREAM_EVENTS_PER_SECOND   = 500
DEFAULT_MAX_DATASTREAM_EVENTS      = int(MAX_DATASTREAM_EVENTS_PER_SECOND * DEFAULT_BUFFER_DURATION_SECONDS)
DEFAULT_DATASTREAM_HEAP_SIZE_BYTES = 64 * 1024 * 1024  # 64 MB heap for variable-length data

# Spike sample window
SPIKE_SAMPLES_BEFORE = 25
SPIKE_SAMPLES_AFTER  = 49
SPIKE_SAMPLES_TOTAL  = SPIKE_SAMPLES_BEFORE + 1 + SPIKE_SAMPLES_AFTER  # 75

# Shared memory name template - actual names are instance-specific
SHM_NAME_TEMPLATE           = "cl_sdk_{prefix}_{{segment}}"
UINT32_MODULUS              = 2**32
ACCELERATED_WAIT_SPIN_POLLS = 64
SEQLOCK_SPIN_POLLS          = 64
SEQLOCK_YIELD_POLLS         = 512
SEQLOCK_SLEEP_SECONDS       = 1 / DEFAULT_FRAMES_PER_SECOND  # 40us, one 25kHz frame
SEQLOCK_MAX_SLEEP_SECONDS   = 0.001

_logger = logging.getLogger("cl.data_buffer")

def _seqlock_backoff(attempt: int) -> None:
    """Back off while retrying a seqlock-protected shared-memory read."""
    if attempt < SEQLOCK_SPIN_POLLS:
        return
    if attempt < SEQLOCK_YIELD_POLLS:
        time.sleep(0)
        return

    exponent = min((attempt - SEQLOCK_YIELD_POLLS) // 128, 5)
    time.sleep(min(SEQLOCK_MAX_SLEEP_SECONDS, SEQLOCK_SLEEP_SECONDS * (2**exponent)))

class _BinaryLayout:
    """Small wrapper around struct layouts with computed offsets."""

    def __init__(self, *fields: tuple[str | None, str], endian: str = "<"):
        self._fields = fields
        self._struct = struct.Struct(endian + "".join(fmt for _, fmt in fields))
        self.format  = self._struct.format
        self.size    = self._struct.size

        self._names       : list[str]      = []
        self._offsets     : dict[str, int] = {}
        self._sizes       : dict[str, int] = {}
        self._value_counts: dict[str, int] = {}

        offset = 0
        for name, fmt in fields:
            field_struct = struct.Struct(endian + fmt)
            field_size = field_struct.size
            if name is not None:
                if name in self._offsets:
                    raise ValueError(f"Duplicate field name in binary layout: {name}")
                self._names.append(name)
                self._offsets[name]      = offset
                self._sizes[name]        = field_size
                self._value_counts[name] = len(field_struct.unpack(bytes(field_size)))
            offset += field_size

        if offset != self.size:
            raise ValueError("Binary layout field sizes do not match struct size")

    def offset(self, name: str) -> int:
        return self._offsets[name]

    def end_offset(self, name: str) -> int:
        return self._offsets[name] + self._sizes[name]

    def pack(self, *values) -> bytes:
        return self._struct.pack(*values)

    def pack_into(self, buffer, offset: int, *values) -> None:
        self._struct.pack_into(buffer, offset, *values)

    def pack_from_object(self, obj) -> bytes:
        values = []
        for name in self._names:
            value = getattr(obj, name)
            if self._value_counts[name] == 1:
                values.append(value)
            else:
                values.extend(value)
        return self._struct.pack(*values)

    def unpack_from(self, buffer, offset: int = 0) -> tuple:
        return self._struct.unpack_from(buffer, offset)

    def unpack_dict_from(self, buffer, offset: int = 0) -> dict[str, object]:
        unpacked  = self._struct.unpack_from(buffer, offset)
        values    = {}
        value_idx = 0
        for name in self._names:
            value_count = self._value_counts[name]
            if value_count == 1:
                values[name] = unpacked[value_idx]
            else:
                values[name] = unpacked[value_idx:value_idx + value_count]
            value_idx += value_count
        return values

class _BinaryStructMixin:
    """Mixin for dataclasses whose binary layout fields match constructor fields."""

    _LAYOUT: ClassVar[_BinaryLayout]

    def pack(self) -> bytes:
        """Pack this dataclass using its binary layout."""
        return self._LAYOUT.pack_from_object(self)

    @classmethod
    def unpack_from(cls, buffer, offset: int = 0) -> Self:
        """Unpack this dataclass from a buffer using its binary layout."""
        return cls(**cls._LAYOUT.unpack_dict_from(buffer, offset))

@dataclass
class BufferHeader(_BinaryStructMixin):
    """
    Shared memory header containing buffer metadata and synchronization state.

    Layout (all little-endian, all multi-byte fields naturally aligned):
        Offset  Size  Field

        ABI prefix
        0       8     magic (0x434C53444B415049 = "CLSDKAPI")
        8       4     version (6)
        12      4     sequence_number (increments on each header update, for detecting torn reads)
        16      4     channel_count
        20      4     frames_per_second
        24      4     buffer_duration_frames
        28      1     pause_flag (1 = paused for debugger)
        29      1     shutdown_flag (1 = producer should exit)
        30      1     producer_ready (1 = producer is ready)
        31      1     playback_speed_q (speed * 4, e.g. 4 = 1x, 8 = 2x; 0 treated as 1x)

        Timestamp/control block
        32      8     start_timestamp (timestamp of first frame in buffer)
        40      8     write_timestamp (timestamp of last written frame + 1)
        48      8     requested_timestamp (consumer sets this to request data up to this point)
        56      8     main_process_heartbeat_ns (nanosecond timestamp, updated periodically by main process)
        64      8     stim_write_timestamp (main process sets this after writing stims)
        72      8     playback_seek_timestamp

        Ring/config block
        80      4     write_index (circular index into frame buffer)
        84      4     spike_write_index (circular index into spike buffer)
        88      4     spike_count (total spikes written, for overflow detection)
        92      4     stim_write_index (circular index into stim buffer)
        96      4     stim_count (total stims written)
        100     4     max_spikes (size of spike buffer)
        104     4     max_stims (size of stim buffer)
        108     4     datastream_write_index (circular index into datastream event index buffer)
        112     4     datastream_count (total datastream events written)
        116     4     max_datastream_events (size of datastream event index buffer)
        120     4     datastream_heap_size (size of datastream heap in bytes)
        124     4     datastream_heap_write_offset (current write position in heap)
        128     4     datastream_heap_generation (increments each time heap wraps)
        132     4     playback_seek_sequence (increments on each seek request)
        136     4     stim_update_sequence (odd while stim ring is being written)
        140     4     datastream_update_sequence (odd while datastream index/heap is being written)
        144     (end)
    """
    MAGIC  : ClassVar[Final] = 0x434C53444B415049  # "CLSDKAPI" in ASCII
    VERSION: ClassVar[Final] = 6

    _LAYOUT: ClassVar = _BinaryLayout(
        ("magic",                        "Q"),
        ("version",                      "I"),
        ("sequence_number",              "I"),
        ("channel_count",                "I"),
        ("frames_per_second",            "I"),
        ("buffer_duration_frames",       "I"),
        ("pause_flag",                   "B"),
        ("shutdown_flag",                "B"),
        ("producer_ready",               "B"),
        ("playback_speed_q",             "B"),
        ("start_timestamp",              "q"),
        ("write_timestamp",              "q"),
        ("requested_timestamp",          "q"),
        ("main_process_heartbeat_ns",    "q"),
        ("stim_write_timestamp",         "q"),
        ("playback_seek_timestamp",      "q"),
        ("write_index",                  "I"),
        ("spike_write_index",            "I"),
        ("spike_count",                  "I"),
        ("stim_write_index",             "I"),
        ("stim_count",                   "I"),
        ("max_spikes",                   "I"),
        ("max_stims",                    "I"),
        ("datastream_write_index",       "I"),
        ("datastream_count",             "I"),
        ("max_datastream_events",        "I"),
        ("datastream_heap_size",         "I"),
        ("datastream_heap_write_offset", "I"),
        ("datastream_heap_generation",   "I"),
        ("playback_seek_sequence",       "I"),
        ("stim_update_sequence",         "I"),
        ("datastream_update_sequence",   "I"),
    )
    FORMAT: ClassVar[Final] = _LAYOUT.format
    SIZE  : ClassVar[Final] = _LAYOUT.size

    MAGIC_OFFSET                     : ClassVar[Final] = _LAYOUT.offset("magic")
    VERSION_OFFSET                   : ClassVar[Final] = _LAYOUT.offset("version")
    SEQUENCE_NUMBER_OFFSET           : ClassVar[Final] = _LAYOUT.offset("sequence_number")
    CHANNEL_COUNT_OFFSET             : ClassVar[Final] = _LAYOUT.offset("channel_count")
    FRAMES_PER_SECOND_OFFSET         : ClassVar[Final] = _LAYOUT.offset("frames_per_second")
    BUFFER_DURATION_FRAMES_OFFSET    : ClassVar[Final] = _LAYOUT.offset("buffer_duration_frames")
    PAUSE_FLAG_OFFSET                : ClassVar[Final] = _LAYOUT.offset("pause_flag")
    SHUTDOWN_FLAG_OFFSET             : ClassVar[Final] = _LAYOUT.offset("shutdown_flag")
    PRODUCER_READY_OFFSET            : ClassVar[Final] = _LAYOUT.offset("producer_ready")
    PLAYBACK_SPEED_Q_OFFSET          : ClassVar[Final] = _LAYOUT.offset("playback_speed_q")
    START_TIMESTAMP_OFFSET           : ClassVar[Final] = _LAYOUT.offset("start_timestamp")
    WRITE_TIMESTAMP_OFFSET           : ClassVar[Final] = _LAYOUT.offset("write_timestamp")
    REQUESTED_TIMESTAMP_OFFSET       : ClassVar[Final] = _LAYOUT.offset("requested_timestamp")
    MAIN_PROCESS_HEARTBEAT_NS_OFFSET : ClassVar[Final] = _LAYOUT.offset("main_process_heartbeat_ns")
    STIM_WRITE_TIMESTAMP_OFFSET      : ClassVar[Final] = _LAYOUT.offset("stim_write_timestamp")
    PLAYBACK_SEEK_TIMESTAMP_OFFSET   : ClassVar[Final] = _LAYOUT.offset("playback_seek_timestamp")
    WRITE_INDEX_OFFSET               : ClassVar[Final] = _LAYOUT.offset("write_index")
    SPIKE_WRITE_INDEX_OFFSET         : ClassVar[Final] = _LAYOUT.offset("spike_write_index")
    SPIKE_COUNT_OFFSET               : ClassVar[Final] = _LAYOUT.offset("spike_count")
    STIM_WRITE_INDEX_OFFSET          : ClassVar[Final] = _LAYOUT.offset("stim_write_index")
    STIM_COUNT_OFFSET                : ClassVar[Final] = _LAYOUT.offset("stim_count")
    MAX_SPIKES_OFFSET                : ClassVar[Final] = _LAYOUT.offset("max_spikes")
    MAX_STIMS_OFFSET                 : ClassVar[Final] = _LAYOUT.offset("max_stims")
    DATASTREAM_WRITE_INDEX_OFFSET    : ClassVar[Final] = _LAYOUT.offset("datastream_write_index")
    DATASTREAM_COUNT_OFFSET          : ClassVar[Final] = _LAYOUT.offset("datastream_count")
    MAX_DATASTREAM_EVENTS_OFFSET     : ClassVar[Final] = _LAYOUT.offset("max_datastream_events")
    DATASTREAM_HEAP_SIZE_OFFSET      : ClassVar[Final] = _LAYOUT.offset("datastream_heap_size")
    DATASTREAM_HEAP_WRITE_OFFSET     : ClassVar[Final] = _LAYOUT.offset("datastream_heap_write_offset")
    DATASTREAM_HEAP_GENERATION_OFFSET: ClassVar[Final] = _LAYOUT.offset("datastream_heap_generation")
    PLAYBACK_SEEK_SEQUENCE_OFFSET    : ClassVar[Final] = _LAYOUT.offset("playback_seek_sequence")
    STIM_UPDATE_SEQUENCE_OFFSET      : ClassVar[Final] = _LAYOUT.offset("stim_update_sequence")
    DATASTREAM_UPDATE_SEQUENCE_OFFSET: ClassVar[Final] = _LAYOUT.offset("datastream_update_sequence")

    magic                       : int = MAGIC
    version                     : int = VERSION
    sequence_number             : int = 0  # Increments on each update for detecting torn reads
    channel_count               : int = DEFAULT_CHANNEL_COUNT
    frames_per_second           : int = DEFAULT_FRAMES_PER_SECOND
    buffer_duration_frames      : int = DEFAULT_BUFFER_SIZE_FRAMES
    start_timestamp             : int = 0
    write_timestamp             : int = 0
    write_index                 : int = 0
    pause_flag                  : int = 0
    shutdown_flag               : int = 0
    producer_ready              : int = 0  # Producer sets this when ready
    playback_speed_q            : int = 0  # speed * 4 (0 treated as 1x for backwards compat)
    spike_write_index           : int = 0
    spike_count                 : int = 0
    stim_write_index            : int = 0
    stim_count                  : int = 0
    max_spikes                  : int = DEFAULT_MAX_SPIKES
    max_stims                   : int = DEFAULT_MAX_STIMS
    requested_timestamp         : int = 0  # Consumer sets this to request data in accelerated mode
    datastream_write_index      : int = 0
    datastream_count            : int = 0
    max_datastream_events       : int = DEFAULT_MAX_DATASTREAM_EVENTS
    main_process_heartbeat_ns   : int = 0  # Updated periodically by main process for debugger detection
    datastream_heap_size        : int = DEFAULT_DATASTREAM_HEAP_SIZE_BYTES
    datastream_heap_write_offset: int = 0  # Current write position in heap (circular)
    datastream_heap_generation  : int = 0  # Increments each time heap wraps
    stim_write_timestamp        : int = 0  # Main process sets this after writing stims to ring
    playback_seek_sequence      : int = 0  # Main process increments to request a playback seek
    playback_seek_timestamp     : int = 0  # Target timestamp for the latest playback seek request
    stim_update_sequence        : int = 0  # Odd while stim ring write is in progress
    datastream_update_sequence  : int = 0  # Odd while datastream heap/index write is in progress

@dataclass
class SpikeRecord:
    """
    Binary layout for a spike in shared memory.

    Layout:
        Offset  Size  Field
        0       8     timestamp
        8       4     channel
        12      4     channel_mean_sample (float32)
        16      300   samples (75 × float32)
        316     4     padding
        320     (end)
    """
    _LAYOUT: ClassVar = _BinaryLayout(
        ("timestamp",           "q"),
        ("channel",             "I"),
        ("channel_mean_sample", "f"),
        ("samples",             f"{SPIKE_SAMPLES_TOTAL}f"),
        (None,                  "4x"),
    )
    _FIXED_LAYOUT: ClassVar[Final] = _BinaryLayout(
        ("timestamp",           "q"),
        ("channel",             "I"),
        ("channel_mean_sample", "f"),
    )

    TIMESTAMP_OFFSET          : ClassVar[Final] = _LAYOUT.offset("timestamp")
    CHANNEL_OFFSET            : ClassVar[Final] = _LAYOUT.offset("channel")
    CHANNEL_MEAN_SAMPLE_OFFSET: ClassVar[Final] = _LAYOUT.offset("channel_mean_sample")
    SAMPLES_OFFSET            : ClassVar[Final] = _LAYOUT.offset("samples")
    PACKED_SIZE               : ClassVar[Final] = _LAYOUT.end_offset("samples")
    SIZE                      : ClassVar[Final] = _LAYOUT.size
    FORMAT                    : ClassVar[Final] = _LAYOUT.format

    timestamp          : int
    channel            : int
    channel_mean_sample: float
    samples            : ndarray

    def pack_into(self, buffer, offset: int) -> None:
        """Pack spike directly into an existing writable buffer."""
        samples = np.asarray(self.samples, dtype=np.float32)

        if samples.shape != (SPIKE_SAMPLES_TOTAL,):
            raise ValueError(
                f"Expected {SPIKE_SAMPLES_TOTAL} spike samples, got shape {samples.shape}"
            )

        if not samples.flags.c_contiguous:
            samples = np.ascontiguousarray(samples)

        self._FIXED_LAYOUT.pack_into(
            buffer,
            offset,
            self.timestamp,
            self.channel,
            self.channel_mean_sample,
        )

        buffer[offset + self.SAMPLES_OFFSET:offset + self.PACKED_SIZE] = samples.tobytes()
        buffer[offset + self.PACKED_SIZE:offset + self.SIZE] = b"\x00" * (self.SIZE - self.PACKED_SIZE)

    @classmethod
    def unpack_from(cls, buffer, offset: int) -> SpikeRecord:
        """Unpack spike directly from a buffer without creating intermediate bytes."""
        # Unpack fixed fields first
        timestamp, channel, channel_mean = cls._FIXED_LAYOUT.unpack_from(buffer, offset)
        # Read samples directly from the buffer with a single allocation
        # (np.frombuffer view + np.array copy, avoiding the intermediate bytes slice).
        samples = np.array(np.frombuffer(
            buffer,
            dtype  = np.float32,
            count  = SPIKE_SAMPLES_TOTAL,
            offset = offset + cls.SAMPLES_OFFSET,
        ))
        return cls(
            timestamp           = timestamp,
            channel             = channel,
            channel_mean_sample = channel_mean,
            samples             = samples
        )

@dataclass
class StimRecord(_BinaryStructMixin):
    """
    Binary layout for a stim in shared memory.

    Phase data stores up to three StimDesign phases. Empty phase slots are
    zero-filled for recordings/playback that only know timestamp/channel.

    Layout:
        Offset  Size  Field
        0       8     timestamp
        8       8     intended_timestamp
        16      4     channel
        20      4     phase_count
        24      12    phase_durations_us (3 x uint32)
        36      12    phase_currents_uA (3 x float32)
        48      (end)
    """
    MAX_PHASES: ClassVar[Final] = 3
    _LAYOUT: ClassVar = _BinaryLayout(
        ("timestamp",          "q"),
        ("intended_timestamp", "q"),
        ("channel",            "I"),
        ("phase_count",        "I"),
        ("phase_durations_us", "III"),
        ("phase_currents_uA",  "fff"),
    )

    SIZE  : ClassVar[Final] = _LAYOUT.size
    FORMAT: ClassVar[Final] = _LAYOUT.format

    timestamp         : int
    intended_timestamp: int
    channel           : int
    phase_count       : int = 0
    phase_durations_us: tuple[int, int, int] = (0, 0, 0)
    phase_currents_uA : tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class DataStreamEventIndexRecord:
    """
    Fixed-size index record for data stream events in shared memory.

    This index record points to variable-length data stored in a separate heap.

    Layout:
        Offset  Size  Field
        0       8     timestamp
        8       2     stream_name_length
        10      6     padding
        16      64    stream_name (utf-8, padded to 64 bytes)
        80      4     heap_offset (offset into datastream heap where data starts)
        84      4     data_length (length of data in bytes)
        88      4     generation (heap generation when this data was written)
        92      4     padding
        96      (end)
    """
    MAX_STREAM_NAME_LENGTH: ClassVar[Final] = 64

    _LAYOUT: ClassVar = _BinaryLayout(
        ("timestamp",          "q"),
        ("stream_name_length", "H"),
        (None,                 "6x"),
        ("stream_name_bytes",  f"{MAX_STREAM_NAME_LENGTH}s"),
        ("heap_offset",        "I"),
        ("data_length",        "I"),
        ("generation",         "I"),
        (None,                 "4x"),
    )

    TIMESTAMP_OFFSET      : ClassVar[Final] = _LAYOUT.offset("timestamp")
    STREAM_NAME_LEN_OFFSET: ClassVar[Final] = _LAYOUT.offset("stream_name_length")
    STREAM_NAME_OFFSET    : ClassVar[Final] = _LAYOUT.offset("stream_name_bytes")
    HEAP_OFFSET_OFFSET    : ClassVar[Final] = _LAYOUT.offset("heap_offset")
    DATA_LENGTH_OFFSET    : ClassVar[Final] = _LAYOUT.offset("data_length")
    GENERATION_OFFSET     : ClassVar[Final] = _LAYOUT.offset("generation")

    FORMAT: ClassVar[Final] = _LAYOUT.format
    SIZE  : ClassVar[Final] = _LAYOUT.size

    timestamp  : int
    stream_name: str
    heap_offset: int  # Offset into the datastream heap
    data_length: int  # Length of data in bytes
    generation : int  # Heap generation when this data was written

    def pack(self) -> bytes:
        """Pack index record to bytes."""
        stream_name_bytes = self.stream_name.encode('utf-8')
        if len(stream_name_bytes) > self.MAX_STREAM_NAME_LENGTH:
            raise ValueError(
                f"stream_name encoded length must be <= {self.MAX_STREAM_NAME_LENGTH} bytes, "
                f"got {len(stream_name_bytes)}"
            )

        return self._LAYOUT.pack(
            self.timestamp,
            len(stream_name_bytes),
            stream_name_bytes.ljust(self.MAX_STREAM_NAME_LENGTH, b'\x00'),
            self.heap_offset,
            self.data_length,
            self.generation,
        )

    @classmethod
    def unpack_from(cls, buffer, offset: int) -> DataStreamEventIndexRecord:
        """Unpack index record directly from buffer."""
        timestamp, name_len, stream_name_bytes, heap_offset, data_length, generation = cls._LAYOUT.unpack_from(
            buffer,
            offset,
        )
        if name_len > cls.MAX_STREAM_NAME_LENGTH:
            raise ValueError(
                f"Invalid data stream name length {name_len}; "
                f"maximum is {cls.MAX_STREAM_NAME_LENGTH}"
            )
        stream_name = stream_name_bytes[:name_len].decode('utf-8')
        return cls(
            timestamp   = timestamp,
            stream_name = stream_name,
            heap_offset = heap_offset,
            data_length = data_length,
            generation  = generation
        )

@dataclass
class DataStreamEventRecord:
    """
    A data stream event with arbitrary-length payload.

    This represents a fully-reconstructed data stream event with the actual data.
    The underlying storage uses a separate index and heap for efficiency.
    """

    timestamp  : int
    stream_name: str
    data       : bytes  # msgpack-encoded data (arbitrary length)

assert struct.calcsize(BufferHeader.FORMAT)               == BufferHeader.SIZE
assert struct.calcsize(SpikeRecord.FORMAT)                == SpikeRecord.SIZE
assert struct.calcsize(StimRecord.FORMAT)                 == StimRecord.SIZE
assert struct.calcsize(DataStreamEventIndexRecord.FORMAT) == DataStreamEventIndexRecord.SIZE

class SharedDataBuffer:
    """
    Shared memory ring buffer for distributing mock API data.

    This class can be used as either a producer (creates shared memory)
    or consumer (attaches to existing shared memory).

    CONCURRENCY MODEL:
    This implementation provides best-effort concurrent access for single-producer,
    multiple-consumer scenarios. It uses a sequence counter to detect torn reads
    (when a read spans a producer's write), allowing consumers to retry.

    LIMITATIONS:
    - NOT truly lock-free due to Python's memory model (no memory barriers)
    - Producer writes may become visible to consumers in unexpected order
    - Consumers may occasionally read inconsistent state (detected via sequence number)
    - Designed for single producer only; multiple producers will cause corruption

    USAGE:
    The producer writes frames sequentially, incrementing the sequence number
    before and after each write. Consumers use _read_header_consistent() to
    retry reads until they get a consistent view (matching sequence numbers).

    Usage (producer):
        buffer = SharedDataBuffer.create(channel_count=64, frames_per_second=25000)
        buffer.write_frames(frames, timestamp)
        buffer.close(unlink=True)

    Usage (consumer):
        buffer = SharedDataBuffer.attach()
        frames = buffer.read_frames(from_timestamp, frame_count)
        spikes = buffer.read_spikes(from_timestamp, to_timestamp)
        buffer.close()
    """

    def __init__(
        self,
        shm_header          : SharedMemory,
        shm_frames          : SharedMemory,
        shm_spikes          : SharedMemory,
        shm_stims           : SharedMemory,
        shm_datastream_index: SharedMemory,
        shm_datastream_heap : SharedMemory,
        is_producer         : bool       = False,
        name_prefix         : str | None = None,
    ):
        self._shm_header           = shm_header
        self._shm_frames           = shm_frames
        self._shm_spikes           = shm_spikes
        self._shm_stims            = shm_stims
        self._shm_datastream_index = shm_datastream_index
        self._shm_datastream_heap  = shm_datastream_heap
        self._is_producer          = is_producer
        self._name_prefix          = name_prefix or self._extract_prefix_from_shm_name(shm_header.name)

        # Read header to get buffer configuration
        header = self._read_header()

        # Validate header magic and version
        if header.magic != BufferHeader.MAGIC:
            raise ValueError(
                f"Invalid shared memory header magic: expected {BufferHeader.MAGIC:#x}, "
                f"got {header.magic:#x}. Buffer may be corrupted or from incompatible version."
            )
        if header.version != BufferHeader.VERSION:
            raise ValueError(
                f"Incompatible shared memory version: expected {BufferHeader.VERSION}, "
                f"got {header.version}. Producer and consumer must use same version."
            )

        self._channel_count          = header.channel_count
        self._frames_per_second      = header.frames_per_second
        self._buffer_duration_frames = header.buffer_duration_frames
        self._max_spikes             = header.max_spikes
        self._max_stims              = header.max_stims
        self._max_datastream_events  = header.max_datastream_events
        self._datastream_heap_size   = header.datastream_heap_size
        self._stim_write_lock        = Lock()
        self._datastream_write_lock  = Lock()

        # Create numpy views into shared memory
        self._frames_view = np.ndarray(
            (self._buffer_duration_frames, self._channel_count),
            dtype  = np.int16,
            buffer = shm_frames.buf
        )

        # Zero-copy numpy views over hot header fields. On x86-64 aligned 8-byte
        # loads/stores are atomic, so these can replace struct.pack/unpack on the
        # tight closed-loop and producer paths.
        header_buf = shm_header.buf
        self._hdr_write_ts        = np.ndarray((1,), dtype=np.int64,  buffer=header_buf, offset=BufferHeader.WRITE_TIMESTAMP_OFFSET)
        self._hdr_start_ts        = np.ndarray((1,), dtype=np.int64,  buffer=header_buf, offset=BufferHeader.START_TIMESTAMP_OFFSET)
        self._hdr_write_idx       = np.ndarray((1,), dtype=np.uint32, buffer=header_buf, offset=BufferHeader.WRITE_INDEX_OFFSET)
        self._hdr_spike_write_idx = np.ndarray((1,), dtype=np.uint32, buffer=header_buf, offset=BufferHeader.SPIKE_WRITE_INDEX_OFFSET)
        self._hdr_spike_count     = np.ndarray((1,), dtype=np.uint32, buffer=header_buf, offset=BufferHeader.SPIKE_COUNT_OFFSET)
        self._hdr_stim_count      = np.ndarray((1,), dtype=np.uint32, buffer=header_buf, offset=BufferHeader.STIM_COUNT_OFFSET)
        self._hdr_requested_ts    = np.ndarray((1,), dtype=np.int64,  buffer=header_buf, offset=BufferHeader.REQUESTED_TIMESTAMP_OFFSET)
        self._hdr_stim_write_ts   = np.ndarray((1,), dtype=np.int64,  buffer=header_buf, offset=BufferHeader.STIM_WRITE_TIMESTAMP_OFFSET)
        self._hdr_heartbeat_ns    = np.ndarray((1,), dtype=np.int64,  buffer=header_buf, offset=BufferHeader.MAIN_PROCESS_HEARTBEAT_NS_OFFSET)
        self._hdr_seq             = np.ndarray((1,), dtype=np.uint32, buffer=header_buf, offset=BufferHeader.SEQUENCE_NUMBER_OFFSET)
        self._hdr_stim_seq        = np.ndarray((1,), dtype=np.uint32, buffer=header_buf, offset=BufferHeader.STIM_UPDATE_SEQUENCE_OFFSET)
        self._hdr_datastream_seq  = np.ndarray((1,), dtype=np.uint32, buffer=header_buf, offset=BufferHeader.DATASTREAM_UPDATE_SEQUENCE_OFFSET)

        # Strided views of the timestamp column in the spike/stim rings, so we can
        # read timestamps for the in-range filter without per-record struct.unpack_from.
        self._spike_ts_view = np.ndarray(
            (self._max_spikes,),
            dtype   = np.int64,
            buffer  = shm_spikes.buf,
            offset  = SpikeRecord.TIMESTAMP_OFFSET,
            strides = (SpikeRecord.SIZE,),
        )
        self._stim_ts_view = np.ndarray(
            (self._max_stims,),
            dtype   = np.int64,
            buffer  = shm_stims.buf,
            offset  = StimRecord._LAYOUT.offset("timestamp"),
            strides = (StimRecord.SIZE,),
        )

        # Pre-allocated scratch buffer for batch packing in write_stims.
        # Avoids per-call list comprehension + b"".join allocations on the hot path.
        self._stim_pack_scratch = bytearray(StimRecord.SIZE * 256)

        # Cached local sequence counters. Producer is the sole writer of the global
        # frame sequence; the main process is the sole writer of the stim ring; the
        # producer is the sole writer of the datastream ring. Caching avoids a
        # read-modify-write on the shared header on every increment.
        self._cached_seq            = int(self._hdr_seq[0])
        self._cached_datastream_seq = int(self._hdr_datastream_seq[0])

        # For spikes and stims, we access them as raw bytes
        # since they have variable-length records

    @staticmethod
    def _extract_prefix_from_shm_name(shm_name: str) -> str:
        """Extract the unique prefix from a shared memory name."""
        # Name format: "cl_sdk_{prefix}_{segment}"
        # Extract the prefix between "cl_sdk_" and the last "_"
        if shm_name.startswith("cl_sdk_"):
            parts = shm_name[7:].rsplit("_", 1)  # Remove "cl_sdk_" and split from right
            if parts:
                return parts[0]
        raise ValueError(f"Invalid shared memory name format: {shm_name}")

    @staticmethod
    def _make_shm_names(name_prefix: str) -> dict[str, str]:
        """Generate all shared memory names for a given prefix."""
        return {
            "header"  : f"cl_sdk_{name_prefix}_header",
            "frames"  : f"cl_sdk_{name_prefix}_frames",
            "spikes"  : f"cl_sdk_{name_prefix}_spikes",
            "stims"   : f"cl_sdk_{name_prefix}_stims",
            "ds_index": f"cl_sdk_{name_prefix}_ds_index",
            "ds_heap" : f"cl_sdk_{name_prefix}_ds_heap",
        }

    @staticmethod
    def _unregister_attached_segments(*segments: SharedMemory) -> None:
        """Prevent independent Popen child resource trackers from unlinking parent-owned shm."""
        with contextlib.suppress(Exception):
            from multiprocessing import resource_tracker
            for shm in segments:
                resource_tracker.unregister(getattr(shm, "_name", shm.name), "shared_memory")

    @staticmethod
    def _cleanup_created_segments(segments: list[SharedMemory]) -> None:
        """Close and unlink shared-memory segments after a partial create failure."""
        for shm in reversed(segments):
            with contextlib.suppress(Exception):
                shm.close()
            with contextlib.suppress(Exception):
                shm.unlink()

    def get_name_prefix(self) -> str:
        """Get the unique name prefix for this shared memory instance.

        This prefix can be passed to subprocesses to allow them to attach
        to the same shared memory using SharedDataBuffer.attach(name_prefix=...).
        """
        return self._name_prefix

    @classmethod
    def create(
        cls,
        channel_count          : int   = DEFAULT_CHANNEL_COUNT,
        frames_per_second      : int   = DEFAULT_FRAMES_PER_SECOND,
        buffer_duration_seconds: float = DEFAULT_BUFFER_DURATION_SECONDS,
        max_spikes             : int   = DEFAULT_MAX_SPIKES,
        max_stims              : int   = DEFAULT_MAX_STIMS,
        max_datastream_events  : int   = DEFAULT_MAX_DATASTREAM_EVENTS,
        datastream_heap_size   : int   = DEFAULT_DATASTREAM_HEAP_SIZE_BYTES,
        start_timestamp        : int   = 0,
        name_prefix            : str | None = None,
    ) -> SharedDataBuffer:
        """
        Create new shared memory buffers (producer mode).

        Args:
            channel_count: Number of channels (default 64)
            frames_per_second: Sample rate (default 50000)
            buffer_duration_seconds: Ring buffer duration (default 10)
            max_spikes: Maximum spikes in spike buffer
            max_stims: Maximum stims in stim buffer
            max_datastream_events: Maximum datastream event index entries
            datastream_heap_size: Size of heap for variable-length data (default 16MB)
            start_timestamp: Initial timestamp value
            name_prefix: Optional unique prefix for shared memory names.
                        If not provided, a random prefix is generated.
                        Use get_name_prefix() to retrieve it for passing to subprocesses.

        Returns:
            SharedDataBuffer instance in producer mode
        """
        # Generate unique prefix for this instance
        if name_prefix is None:
            name_prefix = uuid4().hex[:8]

        shm_names = cls._make_shm_names(name_prefix)

        buffer_duration_frames = int(frames_per_second * buffer_duration_seconds)
        if channel_count <= 0:
            raise ValueError(f"channel_count must be positive, got {channel_count}")
        if frames_per_second <= 0:
            raise ValueError(f"frames_per_second must be positive, got {frames_per_second}")
        if buffer_duration_frames <= 0:
            raise ValueError(
                f"buffer_duration_seconds produces no frames: {buffer_duration_seconds}"
            )
        if max_spikes <= 0:
            raise ValueError(f"max_spikes must be positive, got {max_spikes}")
        if max_stims <= 0:
            raise ValueError(f"max_stims must be positive, got {max_stims}")
        if max_datastream_events <= 0:
            raise ValueError(
                f"max_datastream_events must be positive, got {max_datastream_events}"
            )
        if datastream_heap_size <= 0:
            raise ValueError(
                f"datastream_heap_size must be positive, got {datastream_heap_size}"
            )

        # Calculate buffer sizes
        header_size            = BufferHeader.SIZE
        frames_size            = buffer_duration_frames * channel_count * 2  # int16
        spikes_size            = max_spikes * SpikeRecord.SIZE
        stims_size             = max_stims * StimRecord.SIZE
        datastream_index_size  = max_datastream_events * DataStreamEventIndexRecord.SIZE
        datastream_heap_size_  = datastream_heap_size  # Use exact requested size

        # Clean up any existing shared memory with same names
        # Try unlink first (works even if open by another process on some systems)
        cleanup_errors = []
        for name in shm_names.values():
            try:
                existing = SharedMemory(name=name)
                existing.close()
                existing.unlink()
            except FileNotFoundError:
                # Doesn't exist, good
                pass
            except Exception as e:
                # May fail if another process has it open
                # Store error but continue - we'll try to create anyway
                cleanup_errors.append((name, e))

        # Small delay to allow OS to clean up
        time.sleep(0.01)

        # Create shared memory segments
        # If creation fails, it may be because cleanup didn't complete - report helpful error
        created_segments: list[SharedMemory] = []
        try:
            shm_header           = SharedMemory(name=shm_names["header"], create=True, size=header_size)
            created_segments.append(shm_header)
            shm_frames           = SharedMemory(name=shm_names["frames"], create=True, size=frames_size)
            created_segments.append(shm_frames)
            shm_spikes           = SharedMemory(name=shm_names["spikes"], create=True, size=spikes_size)
            created_segments.append(shm_spikes)
            shm_stims            = SharedMemory(name=shm_names["stims"], create=True, size=stims_size)
            created_segments.append(shm_stims)
            shm_datastream_index = SharedMemory(name=shm_names["ds_index"], create=True, size=datastream_index_size)
            created_segments.append(shm_datastream_index)
            shm_datastream_heap  = SharedMemory(name=shm_names["ds_heap"], create=True, size=datastream_heap_size_)
            created_segments.append(shm_datastream_heap)
        except FileExistsError as e:
            cls._cleanup_created_segments(created_segments)
            error_msg = (
                f"Failed to create shared memory with prefix '{name_prefix}': {e}. "
                f"Previous cleanup may have failed (errors: {cleanup_errors}). "
                f"Another producer may be running, or system resources may not be released yet."
            )
            raise RuntimeError(error_msg) from e
        except Exception:
            cls._cleanup_created_segments(created_segments)
            raise

        # Initialize header
        header = BufferHeader(
            channel_count          = channel_count,
            frames_per_second      = frames_per_second,
            buffer_duration_frames = buffer_duration_frames,
            start_timestamp        = start_timestamp,
            write_timestamp        = start_timestamp,
            write_index            = 0,
            max_spikes             = max_spikes,
            max_stims              = max_stims,
            max_datastream_events  = max_datastream_events,
            datastream_heap_size   = datastream_heap_size,
            stim_write_timestamp   = start_timestamp,
        )
        assert shm_header.buf is not None
        shm_header.buf[:header_size] = header.pack()

        return cls(
            shm_header,
            shm_frames,
            shm_spikes,
            shm_stims,
            shm_datastream_index,
            shm_datastream_heap,
            is_producer = True,
            name_prefix = name_prefix,
        )

    @classmethod
    def attach(
        cls,
        as_producer                     : bool       = False,
        name_prefix                     : str | None = None,
        max_retries                     : int        = 50,
        retry_delay                     : float      = 0.1,
        unregister_from_resource_tracker: bool       = False,
    ) -> SharedDataBuffer:
        """
        Attach to existing shared memory buffers.

        Args:
            as_producer: If True, attach as producer (can write data).
                         If False, attach as consumer (read-only).
            name_prefix: Unique prefix for the shared memory to attach to.
                        Required when multiple SharedDataBuffer instances exist.
                        Get this from the producer via get_name_prefix().
            max_retries: Maximum number of retries if shared memory not found.
            retry_delay: Delay between retries in seconds.
            unregister_from_resource_tracker: If True, unregister attached segments
                        from this process' resource tracker. Use this for independent
                        Popen children that do not own the shared memory.

        Returns:
            SharedDataBuffer instance

        Raises:
            FileNotFoundError: If shared memory doesn't exist after retries
            ValueError: If name_prefix is required but not provided
        """

        # For backwards compatibility, try default prefix if none provided
        # This will only work if there's exactly one instance
        if name_prefix is None:
            # On systems where we can list /dev/shm, try to find it
            with contextlib.suppress(Exception):
                shm_dir = Path('/dev/shm')
                if shm_dir.exists():
                    candidates = [f.name for f in shm_dir.iterdir() if f.name.startswith('cl_sdk_')]
                    if candidates:
                        # Extract prefix from first candidate
                        first_name = candidates[0]
                        name_prefix = cls._extract_prefix_from_shm_name(first_name)

        if name_prefix is None:
            raise ValueError(
                "name_prefix is required to attach to shared memory. "
                "Get it from the producer using buffer.get_name_prefix() and pass it to subprocesses."
            )

        shm_names = cls._make_shm_names(name_prefix)

        last_error = None
        for _ in range(max_retries):
            try:
                shm_header           = SharedMemory(name=shm_names["header"])
                shm_frames           = SharedMemory(name=shm_names["frames"])
                shm_spikes           = SharedMemory(name=shm_names["spikes"])
                shm_stims            = SharedMemory(name=shm_names["stims"])
                shm_datastream_index = SharedMemory(name=shm_names["ds_index"])
                shm_datastream_heap  = SharedMemory(name=shm_names["ds_heap"])

                if unregister_from_resource_tracker:
                    cls._unregister_attached_segments(
                        shm_header,
                        shm_frames,
                        shm_spikes,
                        shm_stims,
                        shm_datastream_index,
                        shm_datastream_heap,
                    )

                buffer = cls(
                    shm_header,
                    shm_frames,
                    shm_spikes,
                    shm_stims,
                    shm_datastream_index,
                    shm_datastream_heap,
                    is_producer=as_producer,
                    name_prefix=name_prefix,
                )

                # If attaching as producer, reset control flags in case they're stale
                if as_producer:
                    buffer.shutdown_flag = False
                    buffer.pause_flag    = False

                return buffer
            except FileNotFoundError as e:
                last_error = e
                time.sleep(retry_delay)

        # All retries exhausted
        raise last_error or FileNotFoundError(
            f"Shared memory with prefix '{name_prefix}' not found after {max_retries} retries. "
            f"Ensure the producer has created the shared memory and the correct prefix is being used."
        )

    def _read_header(self) -> BufferHeader:
        """Read current header from shared memory (may be inconsistent)."""
        assert self._shm_header.buf is not None
        return BufferHeader.unpack_from(self._shm_header.buf)

    def _read_header_consistent(self, max_retries: int = 10_000) -> BufferHeader:
        """
        Read header with consistency checking using sequence number.

        Retries until we get two consecutive reads with matching sequence numbers,
        indicating the header wasn't updated mid-read.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            Consistent BufferHeader snapshot

        Raises:
            RuntimeError: If unable to get consistent read after max_retries
        """
        assert self._shm_header.buf is not None
        buf      = self._shm_header.buf
        seq_view = self._hdr_seq

        for attempt in range(max_retries):
            # Read sequence number before reading header.
            seq_before = int(seq_view[0])

            # Writers increment the sequence once before a multi-field/header
            # update and once again after it is complete.  Odd values therefore
            # mean a writer is in progress, even if the value stays stable for a
            # few reads; accepting those snapshots exposes partially updated
            # write indices/counts to consumers.
            if seq_before & 1:
                _seqlock_backoff(attempt)
                continue

            # Read full header directly from buffer
            header = BufferHeader.unpack_from(buf)

            # Read sequence number again
            seq_after = int(seq_view[0])

            # If sequence numbers match, header is consistent
            if seq_before == seq_after and header.sequence_number == seq_before:
                return header

            # Brief pause before retry to avoid spinning too hard
            _seqlock_backoff(attempt)

        raise RuntimeError(
            f"Failed to read consistent header after {max_retries} attempts. "
            f"Producer may be writing too frequently or system is overloaded."
        )

    def _read_frame_state_consistent(self, max_retries: int = 10_000) -> tuple[int, int, int]:
        """Fast-path seqlock read of (start_timestamp, write_timestamp, write_index).

        Returns just the three fields needed by read_frames, avoiding the cost
        of unpacking the full 30-field BufferHeader on every read.
        """
        seq_view       = self._hdr_seq
        start_ts_view  = self._hdr_start_ts
        write_ts_view  = self._hdr_write_ts
        write_idx_view = self._hdr_write_idx

        for attempt in range(max_retries):
            seq_before = int(seq_view[0])
            if seq_before & 1:
                _seqlock_backoff(attempt)
                continue
            start_ts  = int(start_ts_view[0])
            write_ts  = int(write_ts_view[0])
            write_idx = int(write_idx_view[0])
            if int(seq_view[0]) == seq_before:
                return start_ts, write_ts, write_idx
            _seqlock_backoff(attempt)
        raise RuntimeError("Failed to read consistent frame state")

    def _read_spike_state_consistent(self, max_retries: int = 10_000) -> tuple[int, int]:
        """Fast-path seqlock read of (spike_write_index, spike_count)."""
        seq_view       = self._hdr_seq
        spike_idx_view = self._hdr_spike_write_idx
        spike_cnt_view = self._hdr_spike_count

        for attempt in range(max_retries):
            seq_before = int(seq_view[0])
            if seq_before & 1:
                _seqlock_backoff(attempt)
                continue
            spike_idx = int(spike_idx_view[0])
            spike_cnt = int(spike_cnt_view[0])
            if int(seq_view[0]) == seq_before:
                return spike_idx, spike_cnt
            _seqlock_backoff(attempt)
        raise RuntimeError("Failed to read consistent spike state")

    def _write_header_field(self, offset: int, fmt: str, value) -> None:
        """Write a single field to the header."""
        assert self._shm_header.buf is not None
        data = struct.pack(fmt, value)
        self._shm_header.buf[offset:offset + len(data)] = data

    def _increment_sequence(self) -> None:
        """Increment the sequence counter to signal a header update."""
        # Producer is the sole writer of this counter — cache locally and just
        # publish the new value, avoiding the per-write shared-memory read.
        self._cached_seq = (self._cached_seq + 1) & 0xFFFFFFFF
        self._hdr_seq[0] = self._cached_seq

    def _read_update_sequence(self, offset: int) -> int:
        """Read a per-ring update sequence from the shared header."""
        if offset == BufferHeader.STIM_UPDATE_SEQUENCE_OFFSET:
            return int(self._hdr_stim_seq[0])
        if offset == BufferHeader.DATASTREAM_UPDATE_SEQUENCE_OFFSET:
            return int(self._hdr_datastream_seq[0])
        assert self._shm_header.buf is not None
        return struct.unpack_from("<I", self._shm_header.buf, offset)[0]

    def _increment_update_sequence(self, offset: int) -> None:
        """Increment a per-ring update sequence to bracket non-frame writes."""
        if offset == BufferHeader.STIM_UPDATE_SEQUENCE_OFFSET:
            # Stim ring is written from both the main process (cl.stim queue flush)
            # and the producer subprocess (data sources that emit stims), so we cannot
            # cache locally — must read-modify-write the shared header field.
            self._hdr_stim_seq[0] += 1
            return
        if offset == BufferHeader.DATASTREAM_UPDATE_SEQUENCE_OFFSET:
            # Producer is the sole writer of the datastream ring.
            self._cached_datastream_seq = (self._cached_datastream_seq + 1) & 0xFFFFFFFF
            self._hdr_datastream_seq[0] = self._cached_datastream_seq
            return
        assert self._shm_header.buf is not None
        current_seq = struct.unpack_from("<I", self._shm_header.buf, offset)[0]
        struct.pack_into("<I", self._shm_header.buf, offset, (current_seq + 1) % UINT32_MODULUS)

    @staticmethod
    def _write_record_bytes_to_ring(
        ring_buffer,
        start_idx   : int,
        max_records : int,
        record_size : int,
        record_bytes,
    ) -> None:
        """Write packed fixed-size records into a circular byte ring."""
        record_count = len(record_bytes) // record_size
        end_idx = start_idx + record_count
        if end_idx <= max_records:
            start_offset = start_idx * record_size
            ring_buffer[start_offset:end_idx * record_size] = record_bytes
            return

        first_record_count = max_records - start_idx
        first_byte_count = first_record_count * record_size
        ring_buffer[start_idx * record_size:max_records * record_size] = record_bytes[:first_byte_count]
        ring_buffer[:(record_count - first_record_count) * record_size] = record_bytes[first_byte_count:]

    @property
    def channel_count(self) -> int:
        return self._channel_count

    @property
    def frames_per_second(self) -> int:
        return self._frames_per_second

    @property
    def frame_duration_us(self) -> int:
        return int(1_000_000 / self._frames_per_second)

    @property
    def buffer_duration_frames(self) -> int:
        return self._buffer_duration_frames

    @property
    def write_timestamp(self) -> int:
        """Current write position timestamp (next frame to be written)."""
        return int(self._hdr_write_ts[0])

    @property
    def stim_write_timestamp(self) -> int:
        """Timestamp up to which the main process has written stims to the ring.

        The recording process should read stims up to
        min(write_timestamp, stim_write_timestamp) to avoid reading a range
        before the main process has had a chance to write stims for it.
        """
        return int(self._hdr_stim_write_ts[0])

    @stim_write_timestamp.setter
    def stim_write_timestamp(self, value: int) -> None:
        self._hdr_stim_write_ts[0] = value

    @property
    def start_timestamp(self) -> int:
        """Timestamp of first frame currently in buffer."""
        assert self._shm_header.buf is not None
        return struct.unpack_from("<q", self._shm_header.buf, BufferHeader.START_TIMESTAMP_OFFSET)[0]

    @property
    def pause_flag(self) -> bool:
        """Whether the producer should pause (debugger attached)."""
        assert self._shm_header.buf is not None
        return bool(self._shm_header.buf[BufferHeader.PAUSE_FLAG_OFFSET])

    @pause_flag.setter
    def pause_flag(self, value: bool) -> None:
        """Set the pause flag."""
        assert self._shm_header.buf is not None
        self._shm_header.buf[BufferHeader.PAUSE_FLAG_OFFSET] = 1 if value else 0

    @property
    def shutdown_flag(self) -> bool:
        """Whether the producer should shut down."""
        assert self._shm_header.buf is not None
        return bool(self._shm_header.buf[BufferHeader.SHUTDOWN_FLAG_OFFSET])

    @shutdown_flag.setter
    def shutdown_flag(self, value: bool) -> None:
        """Set the shutdown flag."""
        assert self._shm_header.buf is not None
        self._shm_header.buf[BufferHeader.SHUTDOWN_FLAG_OFFSET] = 1 if value else 0

    @property
    def producer_ready(self) -> bool:
        """Whether the producer is ready and has started its main loop."""
        assert self._shm_header.buf is not None
        return bool(self._shm_header.buf[BufferHeader.PRODUCER_READY_OFFSET])

    @producer_ready.setter
    def producer_ready(self, value: bool) -> None:
        """Set the producer ready flag."""
        assert self._shm_header.buf is not None
        self._shm_header.buf[BufferHeader.PRODUCER_READY_OFFSET] = 1 if value else 0

    @property
    def playback_speed(self) -> float:
        """Playback speed multiplier (0.25 to 4.0, default 1.0)."""
        assert self._shm_header.buf is not None
        q = self._shm_header.buf[BufferHeader.PLAYBACK_SPEED_Q_OFFSET]
        return q / 4.0 if q > 0 else 1.0

    @playback_speed.setter
    def playback_speed(self, value: float) -> None:
        """Set the playback speed multiplier (stored as speed * 4)."""
        assert self._shm_header.buf is not None
        q = max(1, min(16, round(value * 4)))
        self._shm_header.buf[BufferHeader.PLAYBACK_SPEED_Q_OFFSET] = q

    @property
    def requested_timestamp(self) -> int:
        """Consumer-requested timestamp for accelerated mode."""
        return int(self._hdr_requested_ts[0])

    @requested_timestamp.setter
    def requested_timestamp(self, value: int) -> None:
        """Set the requested timestamp (consumer tells producer how far to advance)."""
        self._hdr_requested_ts[0] = value

    @property
    def main_process_heartbeat_ns(self) -> int:
        """Main process heartbeat timestamp in nanoseconds (for debugger detection)."""
        return int(self._hdr_heartbeat_ns[0])

    @main_process_heartbeat_ns.setter
    def main_process_heartbeat_ns(self, value: int) -> None:
        """Update the main process heartbeat timestamp."""
        self._hdr_heartbeat_ns[0] = value

    @property
    def playback_seek_sequence(self) -> int:
        """Monotonic sequence number for playback seek requests."""
        assert self._shm_header.buf is not None
        return struct.unpack_from("<I", self._shm_header.buf, BufferHeader.PLAYBACK_SEEK_SEQUENCE_OFFSET)[0]

    @property
    def playback_seek_timestamp(self) -> int:
        """Target timestamp for the latest playback seek request."""
        assert self._shm_header.buf is not None
        return struct.unpack_from("<q", self._shm_header.buf, BufferHeader.PLAYBACK_SEEK_TIMESTAMP_OFFSET)[0]

    @property
    def datastream_count(self) -> int:
        """Total number of data stream events written to the shared ring."""
        assert self._shm_header.buf is not None
        return struct.unpack_from("<I", self._shm_header.buf, BufferHeader.DATASTREAM_COUNT_OFFSET)[0]

    def request_playback_seek(self, timestamp: int) -> None:
        """Request that the playback producer seek to *timestamp*."""
        assert self._shm_header.buf is not None
        current_seq = self.playback_seek_sequence
        next_seq = (current_seq + 1) % UINT32_MODULUS
        if next_seq == 0:
            next_seq = 1
        struct.pack_into("<q", self._shm_header.buf, BufferHeader.PLAYBACK_SEEK_TIMESTAMP_OFFSET, timestamp)
        struct.pack_into("<I", self._shm_header.buf, BufferHeader.PLAYBACK_SEEK_SEQUENCE_OFFSET, next_seq)

    def write_frames(
        self,
        frames   : ndarray,
        timestamp: int,
    ) -> None:
        """
        Write frames to the ring buffer with sequence-based consistency.

        Args:
            frames: Array of shape (frame_count, channel_count) with int16 values
            timestamp: Timestamp of first frame
        """
        if not self._is_producer:
            raise RuntimeError("write_frames() can only be called by producer")

        frames = np.asarray(frames, dtype=np.int16)
        if frames.ndim != 2 or frames.shape[1] != self._channel_count:
            raise ValueError(
                f"frames must have shape (frame_count, {self._channel_count}), "
                f"got {frames.shape}"
            )
        if not frames.flags.c_contiguous:
            frames = np.ascontiguousarray(frames)

        frame_count = frames.shape[0]
        if frame_count == 0:
            return
        self._write_frames_validated(frames, timestamp, frame_count)

    def write_frames_validated(
        self,
        frames   : ndarray,
        timestamp: int,
    ) -> None:
        """Write frames that have already passed dtype/shape/contiguity checks."""
        if not self._is_producer:
            raise RuntimeError("write_frames_validated() can only be called by producer")

        frame_count = frames.shape[0]
        if frame_count == 0:
            return
        self._write_frames_validated(frames, timestamp, frame_count)

    def _write_frames_validated(
        self,
        frames: ndarray,
        timestamp: int,
        frame_count: int,
    ) -> None:
        assert self._shm_header.buf is not None

        if frame_count > self._buffer_duration_frames:
            raise ValueError(
                f"Cannot write {frame_count} frames in one call; "
                f"buffer capacity is {self._buffer_duration_frames} frames"
            )

        start_ts  = int(self._hdr_start_ts[0])
        write_idx = int(self._hdr_write_idx[0])

        # Increment sequence to signal start of update
        self._increment_sequence()

        # Handle wrap-around
        end_idx = write_idx + frame_count
        if end_idx <= self._buffer_duration_frames:
            # No wrap
            self._frames_view[write_idx:end_idx] = frames
        else:
            # Wrap around
            first_part = self._buffer_duration_frames - write_idx
            self._frames_view[write_idx:] = frames[:first_part]
            self._frames_view[:end_idx - self._buffer_duration_frames] = frames[first_part:]

        # Calculate new header values
        new_write_idx = end_idx % self._buffer_duration_frames
        new_write_ts  = timestamp + frame_count
        new_start_ts  = max(start_ts, new_write_ts - self._buffer_duration_frames)

        self._hdr_write_ts[0]  = new_write_ts
        self._hdr_start_ts[0]  = new_start_ts
        self._hdr_write_idx[0] = new_write_idx

        self._increment_sequence()

    def write_spike(self, spike: SpikeRecord) -> None:
        """Write a spike to the spike ring buffer."""
        self.write_spikes([spike])

    def write_spikes(self, spikes: list[SpikeRecord]) -> None:
        """Write spikes to the spike ring buffer with a single header update."""
        if not spikes:
            return
        if not self._is_producer:
            raise RuntimeError("write_spikes() can only be called by producer")

        assert self._shm_header.buf is not None
        assert self._shm_spikes.buf is not None

        total_count = len(spikes)
        records_to_write = spikes[-self._max_spikes:]
        write_count = len(records_to_write)

        # Increment sequence to signal start of update
        self._increment_sequence()

        # Read current indices from header
        spike_write_idx = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.SPIKE_WRITE_INDEX_OFFSET)[0]
        spike_count     = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.SPIKE_COUNT_OFFSET)[0]

        start_idx = (spike_write_idx + total_count - write_count) % self._max_spikes
        spike_buf = self._shm_spikes.buf

        for n, spike in enumerate(records_to_write):
            idx = (start_idx + n) % self._max_spikes
            offset = idx * SpikeRecord.SIZE
            spike.pack_into(spike_buf, offset)

        # Update indices in header
        new_idx = (spike_write_idx + total_count) % self._max_spikes
        self._write_header_field(BufferHeader.SPIKE_WRITE_INDEX_OFFSET, "<I", new_idx)
        self._write_header_field(BufferHeader.SPIKE_COUNT_OFFSET, "<I", spike_count + total_count)

        # Increment sequence to signal completion
        self._increment_sequence()

    def write_stim(self, stim: StimRecord) -> None:
        """Write a stim to the stim ring buffer.

        Can be called from any process with access to the shared memory.
        The stim ring uses its own write index, independent of the frame buffer.
        """
        self.write_stims([stim])

    def write_stims(self, stims: list[StimRecord]) -> None:
        """Write stims to the stim ring buffer with a single header update."""
        if not stims:
            return

        assert self._shm_header.buf is not None
        assert self._shm_stims.buf is not None

        with self._stim_write_lock:
            total_count      = len(stims)
            records_to_write = stims[-self._max_stims:]
            write_count      = len(records_to_write)

            # The stim ring can be written independently from frames/spikes, so
            # it uses its own sequence counter rather than the global frame
            # seqlock.  Readers retry if the sequence changes under them.
            stim_count = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.STIM_COUNT_OFFSET)[0]

            start_idx    = (stim_count + total_count - write_count) % self._max_stims

            # Pack records directly into a reusable scratch bytearray, avoiding
            # a per-record .pack() + b"".join allocation pair.
            needed_size = write_count * StimRecord.SIZE
            scratch = self._stim_pack_scratch
            if len(scratch) < needed_size:
                scratch = bytearray(needed_size)
                self._stim_pack_scratch = scratch
            stim_pack_into = StimRecord._LAYOUT.pack_into
            for n, stim in enumerate(records_to_write):
                d = stim.phase_durations_us
                c = stim.phase_currents_uA
                stim_pack_into(
                    scratch, n * StimRecord.SIZE,
                    stim.timestamp, stim.intended_timestamp, stim.channel,
                    stim.phase_count,
                    d[0], d[1], d[2],
                    c[0], c[1], c[2],
                )
            record_bytes = memoryview(scratch)[:needed_size]

            self._increment_update_sequence(BufferHeader.STIM_UPDATE_SEQUENCE_OFFSET)
            try:
                self._write_record_bytes_to_ring(
                    self._shm_stims.buf,
                    start_idx,
                    self._max_stims,
                    StimRecord.SIZE,
                    record_bytes,
                )

                # Update indices in header
                new_idx   = (stim_count + total_count) % self._max_stims
                new_count = stim_count + total_count
                self._write_header_field(BufferHeader.STIM_WRITE_INDEX_OFFSET, "<I", new_idx)
                self._write_header_field(BufferHeader.STIM_COUNT_OFFSET, "<I", new_count)
            finally:
                self._increment_update_sequence(BufferHeader.STIM_UPDATE_SEQUENCE_OFFSET)

    def write_datastream_event(self, event: DataStreamEventRecord) -> None:
        """
        Write a data stream event to the ring buffer with variable-length data support.

        The event data is stored in a separate heap, with the index entry pointing to it.
        When the heap fills up and wraps around, the generation counter is incremented.
        Index entries store their generation, so readers can determine if data is still valid.
        """
        if not self._is_producer:
            raise RuntimeError("write_datastream_event() can only be called by producer")

        assert self._shm_header.buf is not None
        assert self._shm_datastream_index.buf is not None
        assert self._shm_datastream_heap.buf is not None

        data_bytes = bytes(event.data)
        data_len = len(data_bytes)
        heap_size = self._datastream_heap_size
        if data_len > heap_size:
            raise ValueError(
                f"Data stream event payload is {data_len} bytes, "
                f"but heap capacity is {heap_size} bytes"
            )

        with self._datastream_write_lock:
            # Data stream events can be appended independently from frames/spikes,
            # so they use their own sequence counter rather than the global frame
            # seqlock.  Readers retry if the sequence changes under them.
            datastream_count  = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.DATASTREAM_COUNT_OFFSET)[0]
            heap_write_offset = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.DATASTREAM_HEAP_WRITE_OFFSET)[0]
            heap_generation   = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.DATASTREAM_HEAP_GENERATION_OFFSET)[0]

            record_heap_offset = heap_write_offset
            record_generation  = heap_generation

            # Keep each payload contiguous in the heap. Splitting one event across
            # the end of the heap makes generation-based validity ambiguous: part
            # of the event belongs to the old generation and part to the new one.
            if data_len > 0 and record_heap_offset + data_len > heap_size:
                record_heap_offset = 0
                record_generation  = (heap_generation + 1) % UINT32_MODULUS

            new_heap_write_offset = record_heap_offset
            new_heap_generation   = record_generation
            if data_len > 0:
                end_offset = record_heap_offset + data_len
                if end_offset == heap_size:
                    new_heap_write_offset = 0
                    new_heap_generation   = (record_generation + 1) % UINT32_MODULUS
                else:
                    new_heap_write_offset = end_offset

            # Pack the index record before publishing the update sequence so
            # validation errors cannot leave readers waiting on an odd sequence.
            index_record = DataStreamEventIndexRecord(
                timestamp   = event.timestamp,
                stream_name = event.stream_name,
                heap_offset = record_heap_offset,
                data_length = data_len,
                generation  = record_generation,
            )
            index_record_bytes = index_record.pack()

            self._increment_update_sequence(BufferHeader.DATASTREAM_UPDATE_SEQUENCE_OFFSET)
            try:
                # Write index record
                index_offset = (datastream_count % self._max_datastream_events) * DataStreamEventIndexRecord.SIZE
                if data_len > 0:
                    self._shm_datastream_heap.buf[record_heap_offset:record_heap_offset + data_len] = data_bytes
                self._shm_datastream_index.buf[index_offset:index_offset + DataStreamEventIndexRecord.SIZE] = index_record_bytes

                # Update header indices.  datastream_count is written last so readers
                # only observe the new index record after its payload and metadata are
                # available.
                new_idx = (datastream_count + 1) % self._max_datastream_events
                self._write_header_field(BufferHeader.DATASTREAM_WRITE_INDEX_OFFSET, "<I", new_idx)
                self._write_header_field(BufferHeader.DATASTREAM_HEAP_WRITE_OFFSET, "<I", new_heap_write_offset)
                self._write_header_field(BufferHeader.DATASTREAM_HEAP_GENERATION_OFFSET, "<I", new_heap_generation)
                self._write_header_field(BufferHeader.DATASTREAM_COUNT_OFFSET, "<I", datastream_count + 1)
            finally:
                self._increment_update_sequence(BufferHeader.DATASTREAM_UPDATE_SEQUENCE_OFFSET)

    def reset_to_timestamp(self, timestamp: int) -> None:
        """
        Reset the buffer state for seeking to a new timestamp.

        This resets all ring buffer indices and counts, preparing the buffer
        to be filled from a new position. Used when seeking in playback mode.

        Args:
            timestamp: The new starting timestamp
        """
        if not self._is_producer:
            raise RuntimeError("reset_to_timestamp() can only be called by producer")

        assert self._shm_header.buf is not None

        # Increment sequence to signal start of update
        self._increment_sequence()
        self._increment_update_sequence(BufferHeader.STIM_UPDATE_SEQUENCE_OFFSET)
        self._increment_update_sequence(BufferHeader.DATASTREAM_UPDATE_SEQUENCE_OFFSET)

        try:
            # Reset frame buffer pointers
            self._write_header_field(BufferHeader.START_TIMESTAMP_OFFSET, "<q", timestamp)
            self._write_header_field(BufferHeader.WRITE_TIMESTAMP_OFFSET, "<q", timestamp)
            self._write_header_field(BufferHeader.WRITE_INDEX_OFFSET, "<I", 0)

            # Reset spike buffer
            self._write_header_field(BufferHeader.SPIKE_WRITE_INDEX_OFFSET, "<I", 0)
            self._write_header_field(BufferHeader.SPIKE_COUNT_OFFSET, "<I", 0)

            # Reset stim buffer
            self._write_header_field(BufferHeader.STIM_WRITE_INDEX_OFFSET, "<I", 0)
            self._write_header_field(BufferHeader.STIM_COUNT_OFFSET, "<I", 0)

            # Reset stim_write_timestamp
            self._write_header_field(BufferHeader.STIM_WRITE_TIMESTAMP_OFFSET, "<q", timestamp)

            # Reset datastream buffer and heap
            self._write_header_field(BufferHeader.DATASTREAM_WRITE_INDEX_OFFSET, "<I", 0)
            self._write_header_field(BufferHeader.DATASTREAM_COUNT_OFFSET, "<I", 0)
            self._write_header_field(BufferHeader.DATASTREAM_HEAP_WRITE_OFFSET, "<I", 0)
            # Increment generation to invalidate any cached heap data
            current_gen = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.DATASTREAM_HEAP_GENERATION_OFFSET)[0]
            self._write_header_field(BufferHeader.DATASTREAM_HEAP_GENERATION_OFFSET, "<I", (current_gen + 1) % UINT32_MODULUS)

        finally:
            self._increment_update_sequence(BufferHeader.DATASTREAM_UPDATE_SEQUENCE_OFFSET)
            self._increment_update_sequence(BufferHeader.STIM_UPDATE_SEQUENCE_OFFSET)
            # Increment sequence to signal completion
            self._increment_sequence()

    def read_frames(
        self,
        from_timestamp: int,
        frame_count   : int,
    ) -> ndarray:
        """
        Read frames from the ring buffer with consistency checking.

        Args:
            from_timestamp: Starting timestamp
            frame_count: Number of frames to read

        Returns:
            Array of shape (frame_count, channel_count) with int16 values

        Raises:
            ValueError: If requested data is not available in buffer or if parameters invalid
        """
        # Bounds checking
        if frame_count <= 0:
            raise ValueError(f"frame_count must be positive, got {frame_count}")
        if frame_count > self._buffer_duration_frames:
            raise ValueError(
                f"Requested {frame_count} frames exceeds buffer capacity "
                f"of {self._buffer_duration_frames} frames"
            )

        start_timestamp, write_timestamp, write_index = self._read_frame_state_consistent()
        to_timestamp = from_timestamp + frame_count

        # Handle timestamps before buffer start by returning zeros for the missing part
        # This allows reading "virtual" past data similar to old mock behaviour
        if from_timestamp < start_timestamp:
            zeros_count = start_timestamp - from_timestamp
            if zeros_count >= frame_count:
                # All requested frames are before buffer start - return zeros
                return np.zeros((frame_count, self._channel_count), dtype=np.int16)
            else:
                # Partial: some zeros, some from buffer
                result               = np.zeros((frame_count, self._channel_count), dtype=np.int16)
                buffer_from          = start_timestamp
                buffer_count         = frame_count - zeros_count
                buffer_data          = self._read_frames_internal_fast(write_timestamp, write_index, buffer_from, buffer_count)
                result[zeros_count:] = buffer_data
                return result

        if to_timestamp > write_timestamp:
            raise ValueError(
                f"Requested end timestamp {to_timestamp} is beyond current "
                f"write position {write_timestamp} (data not yet available)"
            )

        return self._read_frames_internal_fast(write_timestamp, write_index, from_timestamp, frame_count)

    def _read_frames_internal_fast(
        self,
        write_timestamp: int,
        write_index    : int,
        from_timestamp : int,
        frame_count    : int,
    ) -> ndarray:
        """Internal method to read frames from within valid buffer range."""
        frames_from_write = write_timestamp - from_timestamp
        start_idx         = (write_index - frames_from_write) % self._buffer_duration_frames
        end_idx           = start_idx + frame_count

        # Handle wrap-around
        if end_idx <= self._buffer_duration_frames:
            # No wrap
            return self._frames_view[start_idx:end_idx].copy()
        else:
            # Wrap around
            first_part = self._buffer_duration_frames - start_idx
            result = np.empty((frame_count, self._channel_count), dtype=np.int16)
            result[:first_part] = self._frames_view[start_idx:]
            result[first_part:] = self._frames_view[:end_idx - self._buffer_duration_frames]
            return result

    def read_spikes(
        self,
        from_timestamp: int,
        to_timestamp  : int,
    ) -> list[SpikeRecord]:
        """
        Read spikes from the spike ring buffer within a timestamp range.

        Args:
            from_timestamp: Start of range (inclusive)
            to_timestamp: End of range (exclusive)

        Returns:
            List of SpikeRecord objects
        """

        assert self._shm_header.buf is not None
        assert self._shm_spikes.buf is not None

        # Read spike state consistently using fast-path (avoids full header unpack)
        spike_write_idx, spike_count = self._read_spike_state_consistent()

        if spike_count == 0:
            return []

        # Determine how far back we can read
        readable_count = min(spike_count, self._max_spikes)

        result    = []
        spike_buf = self._shm_spikes.buf
        ts_view   = self._spike_ts_view

        for i in range(readable_count):
            # Scan backwards from most recent
            idx = (spike_write_idx - 1 - i) % self._max_spikes

            # Read timestamp via numpy view (zero-copy 8B load) instead of struct.unpack_from
            record_ts = int(ts_view[idx])

            if record_ts >= to_timestamp:
                # Too new, keep scanning backwards
                continue
            elif record_ts < from_timestamp:
                # Too old - scanning is always in reverse chronological order
                # (indices wrap but timestamps don't), so we can stop here
                break
            else:
                # In range - now unpack the full record
                offset = idx * SpikeRecord.SIZE
                record = SpikeRecord.unpack_from(spike_buf, offset)
                result.append(record)

        # Reverse to maintain chronological order
        result.reverse()
        return result

    def read_stims(
        self,
        from_timestamp: int,
        to_timestamp  : int,
    ) -> list[StimRecord]:
        """
        Read stims from the stim ring buffer within a timestamp range.

        Args:
            from_timestamp: Start of range (inclusive)
            to_timestamp: End of range (exclusive)

        Returns:
            List of StimRecord objects
        """

        assert self._shm_header.buf is not None
        assert self._shm_stims.buf is not None

        stim_buf = self._shm_stims.buf
        for attempt in range(10_000):
            seq_before = self._read_update_sequence(BufferHeader.STIM_UPDATE_SEQUENCE_OFFSET)
            if seq_before & 1:
                _seqlock_backoff(attempt)
                continue

            # Stim records are committed by publishing stim_count after the records
            # have been copied into the ring.  Derive the write index from the
            # committed count so a reader cannot observe a new index paired with an
            # old count while the main process is writing stims.
            stim_count     = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.STIM_COUNT_OFFSET)[0]
            stim_write_idx = stim_count % self._max_stims

            result: list[StimRecord] = []
            if stim_count > 0:
                # Determine how far back we can read
                readable_count = min(stim_count, self._max_stims)

                for i in range(readable_count):
                    # Scan backwards from most recent
                    idx       = (stim_write_idx - 1 - i) % self._max_stims

                    # Read timestamp via numpy view (zero-copy 8B load)
                    record_ts = int(self._stim_ts_view[idx])

                    if record_ts >= to_timestamp:
                        # Too new, keep scanning backwards
                        continue
                    elif record_ts < from_timestamp:
                        # Too old - scanning is always in reverse chronological order
                        # (indices wrap but timestamps don't), so we can stop here
                        break
                    else:
                        # In range - now unpack the full record
                        offset = idx * StimRecord.SIZE
                        record = StimRecord.unpack_from(stim_buf, offset)
                        result.append(record)

                # Reverse to maintain chronological order
                result.reverse()

            seq_after = self._read_update_sequence(BufferHeader.STIM_UPDATE_SEQUENCE_OFFSET)
            if seq_before == seq_after and not (seq_after & 1):
                return result
            _seqlock_backoff(attempt)

        raise RuntimeError("Failed to read a consistent stim ring snapshot")

    def read_datastream_events(
        self,
        from_timestamp: int,
        to_timestamp  : int,
    ) -> list[DataStreamEventRecord]:
        """
        Read data stream events from the ring buffer within a timestamp range.

        Args:
            from_timestamp: Start of range (inclusive)
            to_timestamp: End of range (exclusive)

        Returns:
            List of DataStreamEventRecord objects with reconstructed data
        """

        assert self._shm_header.buf is not None
        assert self._shm_datastream_index.buf is not None
        assert self._shm_datastream_heap.buf is not None

        index_buf = self._shm_datastream_index.buf
        heap_buf  = self._shm_datastream_heap.buf
        heap_size = self._datastream_heap_size
        for attempt in range(10_000):
            seq_before = self._read_update_sequence(BufferHeader.DATASTREAM_UPDATE_SEQUENCE_OFFSET)
            if seq_before & 1:
                _seqlock_backoff(attempt)
                continue

            datastream_count   = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.DATASTREAM_COUNT_OFFSET)[0]
            current_generation = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.DATASTREAM_HEAP_GENERATION_OFFSET)[0]
            heap_write_offset  = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.DATASTREAM_HEAP_WRITE_OFFSET)[0]

            result: list[DataStreamEventRecord] = []
            if datastream_count > 0:
                # Determine how far back we can read
                readable_count = min(datastream_count, self._max_datastream_events)

                for i in range(readable_count):
                    # Scan backwards from most recent
                    idx          = (datastream_count - 1 - i) % self._max_datastream_events
                    index_offset = idx * DataStreamEventIndexRecord.SIZE

                    # Read just the timestamp first (8 bytes) to check range
                    record_ts = struct.unpack_from("<q", index_buf, index_offset)[0]

                    if record_ts >= to_timestamp:
                        # Too new, keep scanning backwards
                        continue
                    elif record_ts < from_timestamp:
                        # Too old - scanning is always in reverse chronological order
                        break
                    else:
                        # In range - unpack the index record
                        index_record = DataStreamEventIndexRecord.unpack_from(index_buf, index_offset)

                        # Read data from heap if still valid
                        data = self._read_heap_data(
                            index_record.heap_offset,
                            index_record.data_length,
                            index_record.generation,
                            current_generation,
                            heap_write_offset,
                            heap_buf,
                            heap_size
                        )

                        # Create full event record
                        event = DataStreamEventRecord(
                            timestamp   = index_record.timestamp,
                            stream_name = index_record.stream_name,
                            data        = data
                        )
                        result.append(event)

                # Reverse to maintain chronological order
                result.reverse()

            seq_after = self._read_update_sequence(BufferHeader.DATASTREAM_UPDATE_SEQUENCE_OFFSET)
            if seq_before == seq_after and not (seq_after & 1):
                return result
            _seqlock_backoff(attempt)

        raise RuntimeError("Failed to read a consistent data stream snapshot")

    def read_datastream_events_since_count(
        self,
        from_count: int,
        to_count  : int | None = None,
    ) -> tuple[list[DataStreamEventRecord], int]:
        """
        Read data stream events written after a monotonic event count.

        Unlike timestamp-window reads, this catches late-appended events whose
        timestamps are older than a consumer's frame cursor. Returns the events
        in append order and the count the caller should store for the next read.

        This is done in two stages to avoid a fast writer (e.g. in accelerated mode)
        exhausting the seqlock retry budget:
        1. Seqlock stage: reads three header fields under a seqlock. The window is
           O(1) regardless of how many events are pending, so a fast writer cannot
           prevent it from succeeding.
        2. Read stage: iterates events without holding the seqlock. Safe because
           DATASTREAM_COUNT_OFFSET is written last in write_datastream_event, so all
           events below the snapshot count have fully committed index and heap data.
           Raises RuntimeError if the writer laps into the read range during this stage.
        """

        assert self._shm_header.buf is not None
        assert self._shm_datastream_index.buf is not None
        assert self._shm_datastream_heap.buf is not None

        index_buf = self._shm_datastream_index.buf
        heap_buf  = self._shm_datastream_heap.buf
        heap_size = self._datastream_heap_size

        # Stage 1: seqlock on header metadata only (3 uint32 reads — nanoseconds).
        # The event-reading loop is intentionally outside this window; see Stage 2.
        for attempt in range(10_000):
            seq_before = self._read_update_sequence(BufferHeader.DATASTREAM_UPDATE_SEQUENCE_OFFSET)
            if seq_before & 1:
                _seqlock_backoff(attempt)
                continue

            current_count     : int = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.DATASTREAM_COUNT_OFFSET)[0]
            current_generation: int = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.DATASTREAM_HEAP_GENERATION_OFFSET)[0]
            heap_write_offset : int = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.DATASTREAM_HEAP_WRITE_OFFSET)[0]

            seq_after = self._read_update_sequence(BufferHeader.DATASTREAM_UPDATE_SEQUENCE_OFFSET)
            if seq_before == seq_after and not (seq_after & 1):
                break
            _seqlock_backoff(attempt)
        else:
            raise RuntimeError("Failed to read a consistent data stream metadata snapshot")

        # Stage 2: read committed event data — no seqlock needed.
        #
        # DATASTREAM_COUNT_OFFSET is written last in write_datastream_event, so it acts as
        # a commitment point: every event below current_count has its index record and heap
        # payload fully written. The writer always advances forward and never modifies
        # already-committed events, so these reads are safe without holding the seqlock.
        read_to_count = current_count if to_count is None or to_count > current_count else to_count

        if from_count >= read_to_count:
            return [], from_count

        readable_count = min(current_count, self._max_datastream_events)
        oldest_count   = current_count - readable_count
        start_count    = max(from_count, oldest_count)

        if from_count < oldest_count:
            _logger.warning(
                "DataStream ring overflow: %d events lost (from_count=%d, oldest_count=%d)",
                oldest_count - from_count,
                from_count,
                oldest_count,
            )

        # Track previous-generation payloads that were valid at Stage 1 snapshot time.
        # These must be re-validated against the post-Stage-2 heap write offset: even
        # without a generation change, the write pointer advancing within the same
        # generation can overwrite their data regions.
        prev_gen_payloads: list[tuple[int, int, int]] = []  # (heap_offset, data_length, record_generation)

        result: list[DataStreamEventRecord] = []
        for event_count in range(start_count, read_to_count):
            idx             = event_count % self._max_datastream_events
            index_offset    = idx * DataStreamEventIndexRecord.SIZE
            index_record    = DataStreamEventIndexRecord.unpack_from(index_buf, index_offset)
            generation_diff = (current_generation - index_record.generation) % UINT32_MODULUS

            data = self._read_heap_data(
                index_record.heap_offset,
                index_record.data_length,
                index_record.generation,
                current_generation,
                heap_write_offset,
                heap_buf,
                heap_size,
            )
            if generation_diff == 1 and data:
                prev_gen_payloads.append(
                    (index_record.heap_offset, index_record.data_length, index_record.generation)
                )
            result.append(DataStreamEventRecord(
                timestamp   = index_record.timestamp,
                stream_name = index_record.stream_name,
                data        = data,
            ))

        # Post-Stage-2 seqlock snapshot.  The writer mutates index slots and heap data
        # before committing DATASTREAM_COUNT_OFFSET, so an unprotected read of post_count
        # could observe a stale value while a slot already written to is not yet counted,
        # causing the index guard to under-count concurrent advances.
        assert self._shm_header.buf is not None
        for attempt in range(10_000):
            post_seq = self._read_update_sequence(BufferHeader.DATASTREAM_UPDATE_SEQUENCE_OFFSET)
            if post_seq & 1:
                _seqlock_backoff(attempt)
                continue
            post_count             : int = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.DATASTREAM_COUNT_OFFSET)[0]
            post_heap_generation   : int = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.DATASTREAM_HEAP_GENERATION_OFFSET)[0]
            post_heap_write_offset : int = struct.unpack_from("<I", self._shm_header.buf, BufferHeader.DATASTREAM_HEAP_WRITE_OFFSET)[0]
            if self._read_update_sequence(BufferHeader.DATASTREAM_UPDATE_SEQUENCE_OFFSET) == post_seq and not (post_seq & 1):
                break
            _seqlock_backoff(attempt)
        else:
            raise RuntimeError("Failed to read a consistent post-Stage-2 guard snapshot")

        # Index ring guard: the index ring has max_datastream_events slots. The writer
        # is (max_datastream_events - batch_size) slots ahead of the start of our read
        # range. If it advanced that many slots during Stage 2, it has lapped into the
        # range and some index slots now contain newer events than what we read.
        advance       : int = (post_count - read_to_count) % UINT32_MODULUS
        batch_size    : int = read_to_count - start_count
        safety_margin : int = self._max_datastream_events - batch_size
        if advance > safety_margin:
            raise RuntimeError(
                f"DataStream index ring wrapped during Stage 2 read: writer advanced {advance} events "
                f"while {batch_size} events were being read; "
                f"returned events from the first {advance - safety_margin} slots are stale."
            )

        # Heap guard (generation): if the heap wrapped at least once during Stage 2,
        # _is_heap_data_valid was called with a stale generation and may have returned
        # True for regions the writer has since overwritten with newer payloads.
        heap_generation_advance: int = (post_heap_generation - current_generation) % UINT32_MODULUS
        if heap_generation_advance > 0:
            raise RuntimeError(
                f"DataStream heap wrapped during Stage 2 read: heap generation advanced "
                f"{heap_generation_advance} time(s); payload reads used a stale heap "
                f"snapshot and may have returned newer bytes in place of the expected payload."
            )

        # Heap guard (write offset): even without a generation change, the write pointer
        # advancing within the same generation can overwrite previous-generation payloads
        # that were valid at Stage 1 snapshot time. Re-validate each such payload against
        # the post-Stage-2 write offset; raise if any are now invalid.
        for prev_heap_offset, prev_data_length, prev_record_generation in prev_gen_payloads:
            if not self._is_heap_data_valid(
                prev_heap_offset,
                prev_data_length,
                prev_record_generation,
                post_heap_generation,
                post_heap_write_offset,
                heap_size,
            ):
                raise RuntimeError(
                    f"DataStream heap payload overwritten during Stage 2 read: "
                    f"previous-generation payload at heap offset {prev_heap_offset} "
                    f"(length {prev_data_length}) was valid at Stage 1 snapshot "
                    f"(heap_write_offset={heap_write_offset}) but is no longer valid "
                    f"at post-Stage-2 snapshot (heap_write_offset={post_heap_write_offset})."
                )

        return result, read_to_count

    def _read_heap_data(
        self,
        heap_offset       : int,
        data_length       : int,
        record_generation : int,
        current_generation: int,
        heap_write_offset : int,
        heap_buf          : memoryview[int],
        heap_size         : int
    ) -> bytes:
        """
        Read data from the heap, handling wrap-around.

        Returns empty bytes if the data has been overwritten.
        """
        if data_length == 0:
            return b''

        # Check if data is still valid using generation counter
        if not self._is_heap_data_valid(
            heap_offset, data_length, record_generation,
            current_generation, heap_write_offset, heap_size
        ):
            return b''

        # Read data, handling wrap-around
        if heap_offset + data_length <= heap_size:
            # No wrap
            return bytes(heap_buf[heap_offset:heap_offset + data_length])
        else:
            # Wrap around
            first_chunk_size = heap_size - heap_offset
            first_chunk      = bytes(heap_buf[heap_offset:heap_size])
            second_chunk     = bytes(heap_buf[:data_length - first_chunk_size])
            return first_chunk + second_chunk

    @staticmethod
    def _is_heap_data_valid(
        data_start        : int,
        data_length       : int,
        record_generation : int,
        current_generation: int,
        heap_write_offset : int,
        heap_size         : int,
    ) -> bool:
        """
        Check if the data is still valid using generation counters.

        Data is valid if:
        - Same generation: data hasn't been wrapped over yet
        - Previous generation: data is behind current write offset (hasn't been overwritten yet)
        - Older than previous generation: definitely overwritten
        """
        if data_length == 0:
            return True

        generation_diff = (current_generation - record_generation) % UINT32_MODULUS

        if generation_diff == 0:
            # Same generation - data is definitely valid
            return True
        elif generation_diff == 1:
            # Previous generation - data is valid if it's behind the write pointer
            # (i.e., the write pointer hasn't reached it yet after wrapping)
            data_end = data_start + data_length
            if data_end <= heap_size:
                # Data doesn't wrap - valid if entirely after current write offset
                return data_start >= heap_write_offset
            else:
                # Data wraps - the portion in [0, data_end - heap_size) must not overlap
                # with [0, heap_write_offset)
                # This is only valid if the wrapped portion starts at or after write_offset,
                # but that's impossible since wrapped portion is in [0, ...)
                # So wrapped data from previous generation is invalid once we've wrapped
                return False
        else:
            # Older than previous generation - definitely overwritten
            return False

    def wait_for_timestamp(
        self,
        target_timestamp: int,
        timeout_seconds : float | None = None,
    ) -> bool:
        """
        Wait until the buffer has data up to target_timestamp.

        Uses adaptive polling that works for both accelerated and real-time modes:
        - Detects accelerated mode via requested_timestamp and polls aggressively
        - In real-time mode, uses hybrid sleep/spin for efficiency

        Args:
            target_timestamp: Timestamp to wait for
            timeout_seconds: Maximum time to wait (None = forever)

        Returns:
            True if data is available, False if timeout
        """

        # Quick check if data is already available (common case)
        write_ts_view = self._hdr_write_ts
        if int(write_ts_view[0]) >= target_timestamp:
            return True

        requested_ts_view = self._hdr_requested_ts
        start_time_ns     = time.perf_counter_ns()
        timeout_ns        = None if timeout_seconds is None else int(timeout_seconds * 1_000_000_000)

        # Calculate frame timing for sleep/spin approach
        frame_duration_ns = 1_000_000_000 // self._frames_per_second
        # Spin-wait threshold: for waits shorter than this, just spin (more accurate)
        # macOS has ~1ms sleep granularity, so spin for the final 2ms
        spin_threshold_ns = 2_000_000  # 2ms
        accelerated_spins = 0

        while True:
            current_ts = int(write_ts_view[0])
            if current_ts >= target_timestamp:
                return True

            now_ns = time.perf_counter_ns()
            if timeout_ns is not None and now_ns - start_time_ns > timeout_ns:
                return False

            # Detect accelerated mode: if requested_timestamp >= target, producer
            # should be advancing quickly - poll without sleeping
            if int(requested_ts_view[0]) >= target_timestamp:
                if accelerated_spins < ACCELERATED_WAIT_SPIN_POLLS:
                    accelerated_spins += 1
                else:
                    # Accelerated mode: yield while polling to avoid monopolising the CPU.
                    accelerated_spins = 0
                    time.sleep(0)
                continue

            # Real-time mode: hybrid sleep/spin approach
            accelerated_spins = 0
            frames_remaining  = target_timestamp - current_ts
            expected_wait_ns  = frames_remaining * frame_duration_ns

            if expected_wait_ns > spin_threshold_ns:
                # Sleep for most of the wait, wake up early to spin
                sleep_ns = expected_wait_ns - spin_threshold_ns
                time.sleep(sleep_ns * 1e-9)
            # else: spin-wait (just loop back immediately)

    def close(self, unlink: bool = False) -> None:
        """
        Close shared memory handles.

        Args:
            unlink: If True, also unlink (delete) the shared memory.
                    Only the producer should do this.
        """
        if hasattr(self, "_frames_view"):
            del self._frames_view  # Release memoryview before closing shared memory

        segments = [
            self._shm_header,
            self._shm_frames,
            self._shm_spikes,
            self._shm_stims,
            self._shm_datastream_index,
            self._shm_datastream_heap,
        ]

        for shm in segments:
            try:
                shm.close()
            except BufferError:
                _logger.exception("Failed to close shared memory segment %s", shm.name)
            except Exception:
                _logger.exception("Unexpected error closing shared memory segment %s", shm.name)

            if unlink:
                with contextlib.suppress(FileNotFoundError):
                    shm.unlink()
