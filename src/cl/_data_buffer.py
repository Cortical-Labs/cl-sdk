"""
Shared memory ring buffer for mock API data distribution.

This module provides a lock-free ring buffer implementation using Python's
multiprocessing.shared_memory for sharing waveform, spike, and stim data
between the data producer subprocess and consumers (neurons.read(),
closed loop, WebSocket server).

The buffer holds 5 seconds of data at 25kHz sample rate (125,000 frames).
"""
from __future__ import annotations

import contextlib
import struct
import time
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import ClassVar, Final
from uuid import uuid4

import numpy as np
from numpy import ndarray

# Buffer sizing
DEFAULT_BUFFER_DURATION_SECONDS = 5
DEFAULT_FRAMES_PER_SECOND       = 25_000
DEFAULT_CHANNEL_COUNT           = 64

# Spike buffer sizing (max spikes per second × buffer duration)
MAX_SPIKES_PER_SECOND = 10_000  # Conservative upper bound
DEFAULT_MAX_SPIKES    = MAX_SPIKES_PER_SECOND * DEFAULT_BUFFER_DURATION_SECONDS

# Stim buffer sizing
MAX_STIMS_PER_SECOND = 1_000
DEFAULT_MAX_STIMS    = MAX_STIMS_PER_SECOND * DEFAULT_BUFFER_DURATION_SECONDS

# Data stream event buffer sizing
MAX_DATASTREAM_EVENTS_PER_SECOND   = 500
DEFAULT_MAX_DATASTREAM_EVENTS      = MAX_DATASTREAM_EVENTS_PER_SECOND * DEFAULT_BUFFER_DURATION_SECONDS
DEFAULT_DATASTREAM_HEAP_SIZE_BYTES = 64 * 1024 * 1024  # 64 MB heap for variable-length data

# Scan limits for ring buffer reads to prevent O(n) performance degradation
# These are multipliers applied to the expected count based on time window
# Higher values give more safety margin but slower worst-case performance
SCAN_LIMIT_MULTIPLIER = 4     # Allow 4x expected entries to account for bursts
MIN_SCAN_LIMIT        = 1000  # Minimum entries to scan regardless of time window

# Spike sample window
SPIKE_SAMPLES_BEFORE = 25
SPIKE_SAMPLES_AFTER  = 49
SPIKE_SAMPLES_TOTAL  = SPIKE_SAMPLES_BEFORE + 1 + SPIKE_SAMPLES_AFTER  # 75

# Shared memory name template - actual names are instance-specific
SHM_NAME_TEMPLATE = "cl_sdk_{prefix}_{{segment}}"

@dataclass
class BufferHeader:
    """
    Shared memory header containing buffer metadata and synchronization state.

    Layout (all little-endian):
        Offset  Size  Field
        0       8     magic (0x434C53444B415049 = "CLSDKAPI")
        8       4     version (2)
        12      4     sequence_number (increments on each header update, for detecting torn reads)
        16      4     channel_count
        20      4     frames_per_second
        24      4     buffer_duration_frames
        28      8     start_timestamp (timestamp of first frame in buffer)
        36      8     write_timestamp (timestamp of last written frame + 1)
        44      4     write_index (circular index into frame buffer)
        48      1     pause_flag (1 = paused for debugger)
        49      1     shutdown_flag (1 = producer should exit)
        50      1     producer_ready (1 = producer is ready)
        51      1     reserved
        52      4     spike_write_index (circular index into spike buffer)
        56      4     spike_count (total spikes written, for overflow detection)
        60      4     stim_write_index (circular index into stim buffer)
        64      4     stim_count (total stims written)
        68      4     max_spikes (size of spike buffer)
        72      4     max_stims (size of stim buffer)
        76      8     requested_timestamp (consumer sets this to request data up to this point)
        84      4     datastream_write_index (circular index into datastream event index buffer)
        88      4     datastream_count (total datastream events written)
        92      4     max_datastream_events (size of datastream event index buffer)
        96      8     main_process_heartbeat_ns (nanosecond timestamp, updated periodically by main process)
        104     4     datastream_heap_size (size of datastream heap in bytes)
        108     4     datastream_heap_write_offset (current write position in heap)
        112     4     datastream_heap_generation (increments each time heap wraps)
        116     (end)
    """
    MAGIC  : Final = 0x434C53444B415049  # "CLSDKAPI" in ASCII
    VERSION: Final = 2
    SIZE   : Final = 116

    # Struct format for header
    FORMAT: Final = "<QIIIIIqqIBBBBIIIIIIqIIIqIII"

    magic                       : int = MAGIC
    version                     : int = VERSION
    sequence_number             : int = 0  # Increments on each update for detecting torn reads
    channel_count               : int = DEFAULT_CHANNEL_COUNT
    frames_per_second           : int = DEFAULT_FRAMES_PER_SECOND
    buffer_duration_frames      : int = DEFAULT_FRAMES_PER_SECOND * DEFAULT_BUFFER_DURATION_SECONDS
    start_timestamp             : int = 0
    write_timestamp             : int = 0
    write_index                 : int = 0
    pause_flag                  : int = 0
    shutdown_flag               : int = 0
    producer_ready              : int = 0  # Producer sets this when ready
    reserved1                   : int = 0
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

    def pack(self) -> bytes:
        """Pack header to bytes."""
        return struct.pack(
            self.FORMAT,
            self.magic,
            self.version,
            self.sequence_number,
            self.channel_count,
            self.frames_per_second,
            self.buffer_duration_frames,
            self.start_timestamp,
            self.write_timestamp,
            self.write_index,
            self.pause_flag,
            self.shutdown_flag,
            self.producer_ready,
            self.reserved1,
            self.spike_write_index,
            self.spike_count,
            self.stim_write_index,
            self.stim_count,
            self.max_spikes,
            self.max_stims,
            self.requested_timestamp,
            self.datastream_write_index,
            self.datastream_count,
            self.max_datastream_events,
            self.main_process_heartbeat_ns,
            self.datastream_heap_size,
            self.datastream_heap_write_offset,
            self.datastream_heap_generation,
        )

    @classmethod
    def unpack_from(cls, buffer, offset: int = 0) -> BufferHeader:
        """Unpack header directly from a buffer without creating intermediate bytes."""
        values = struct.unpack_from(cls.FORMAT, buffer, offset)
        return cls._from_values(values)

    @classmethod
    def _from_values(cls, values: tuple) -> BufferHeader:
        """Create a BufferHeader from unpacked values."""
        return cls(
            magic                        = values[ 0],
            version                      = values[ 1],
            sequence_number              = values[ 2],
            channel_count                = values[ 3],
            frames_per_second            = values[ 4],
            buffer_duration_frames       = values[ 5],
            start_timestamp              = values[ 6],
            write_timestamp              = values[ 7],
            write_index                  = values[ 8],
            pause_flag                   = values[ 9],
            shutdown_flag                = values[10],
            producer_ready               = values[11],
            reserved1                    = values[12],
            spike_write_index            = values[13],
            spike_count                  = values[14],
            stim_write_index             = values[15],
            stim_count                   = values[16],
            max_spikes                   = values[17],
            max_stims                    = values[18],
            requested_timestamp          = values[19],
            datastream_write_index       = values[20],
            datastream_count             = values[21],
            max_datastream_events        = values[22],
            main_process_heartbeat_ns    = values[23],
            datastream_heap_size         = values[24],
            datastream_heap_write_offset = values[25],
            datastream_heap_generation   = values[26],
        )

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
        316     (end)
    """
    SIZE  : ClassVar[Final] = 8 + 4 + 4 + (SPIKE_SAMPLES_TOTAL * 4)  # 316 bytes
    FORMAT: ClassVar[Final] = f"<qIf{SPIKE_SAMPLES_TOTAL}f"

    timestamp          : int
    channel            : int
    channel_mean_sample: float
    samples            : ndarray

    def pack(self) -> bytes:
        """Pack spike to bytes."""
        return struct.pack(
            self.FORMAT,
            self.timestamp,
            self.channel,
            self.channel_mean_sample,
            *self.samples.astype(np.float32).tolist()
        )

    @classmethod
    def unpack_from(cls, buffer, offset: int) -> SpikeRecord:
        """Unpack spike directly from a buffer without creating intermediate bytes."""
        # Unpack fixed fields first
        timestamp, channel, channel_mean = struct.unpack_from("<qIf", buffer, offset)
        # Read samples directly from buffer as numpy array
        samples_offset = offset + 16  # 8 + 4 + 4
        samples = np.frombuffer(
            buffer[samples_offset:samples_offset + SPIKE_SAMPLES_TOTAL * 4],
            dtype=np.float32
        ).copy()
        return cls(
            timestamp           = timestamp,
            channel             = channel,
            channel_mean_sample = channel_mean,
            samples             = samples
        )

@dataclass
class StimRecord:
    """
    Binary layout for a stim in shared memory.

    Layout:
        Offset  Size  Field
        0       8     timestamp
        8       4     channel
        12      4     padding
        16      (end)
    """
    SIZE  : ClassVar[Final] = 16
    FORMAT: ClassVar[Final] = "<qI4x"

    timestamp: int
    channel: int

    def pack(self) -> bytes:
        """Pack stim to bytes."""
        return struct.pack(self.FORMAT, self.timestamp, self.channel)

    @classmethod
    def unpack_from(cls, buffer, offset: int) -> StimRecord:
        """Unpack stim directly from a buffer without creating intermediate bytes."""
        timestamp, channel = struct.unpack_from("<qI", buffer, offset)
        return cls(timestamp=timestamp, channel=channel)

@dataclass
class DataStreamEventIndexRecord:
    """
    Fixed-size index record for data stream events in shared memory.

    This index record points to variable-length data stored in a separate heap.

    Layout:
        Offset  Size  Field
        0       8     timestamp
        8       2     stream_name_length
        10      64    stream_name (utf-8, padded to 64 bytes)
        74      4     heap_offset (offset into datastream heap where data starts)
        78      4     data_length (length of data in bytes)
        82      4     generation (heap generation when this data was written)
        86      (end)
    """
    MAX_STREAM_NAME_LENGTH: ClassVar[Final] = 64
    SIZE                  : ClassVar[Final] = 8 + 2 + MAX_STREAM_NAME_LENGTH + 4 + 4 + 4  # 86 bytes

    timestamp  : int
    stream_name: str
    heap_offset: int  # Offset into the datastream heap
    data_length: int  # Length of data in bytes
    generation : int  # Heap generation when this data was written

    def pack(self) -> bytes:
        """Pack index record to bytes."""
        stream_name_bytes = self.stream_name.encode('utf-8')[:self.MAX_STREAM_NAME_LENGTH]

        result  = struct.pack("<qH", self.timestamp, len(stream_name_bytes))
        result += stream_name_bytes.ljust(self.MAX_STREAM_NAME_LENGTH, b'\x00')
        result += struct.pack("<III", self.heap_offset, self.data_length, self.generation)
        return result

    @classmethod
    def unpack_from(cls, buffer, offset: int) -> DataStreamEventIndexRecord:
        """Unpack index record directly from buffer."""
        timestamp, name_len = struct.unpack_from("<qH", buffer, offset)
        stream_name = bytes(buffer[offset + 10:offset + 10 + name_len]).decode('utf-8')
        heap_offset, data_length, generation = struct.unpack_from("<III", buffer, offset + 74)
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

        # Create numpy views into shared memory
        self._frames_view = np.ndarray(
            (self._buffer_duration_frames, self._channel_count),
            dtype  = np.int16,
            buffer = shm_frames.buf
        )

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
            frames_per_second: Sample rate (default 25000)
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
        try:
            shm_header           = SharedMemory(name=shm_names["header"], create=True, size=header_size)
            shm_frames           = SharedMemory(name=shm_names["frames"], create=True, size=frames_size)
            shm_spikes           = SharedMemory(name=shm_names["spikes"], create=True, size=spikes_size)
            shm_stims            = SharedMemory(name=shm_names["stims"], create=True, size=stims_size)
            shm_datastream_index = SharedMemory(name=shm_names["ds_index"], create=True, size=datastream_index_size)
            shm_datastream_heap  = SharedMemory(name=shm_names["ds_heap"], create=True, size=datastream_heap_size_)
        except FileExistsError as e:
            error_msg = (
                f"Failed to create shared memory with prefix '{name_prefix}': {e}. "
                f"Previous cleanup may have failed (errors: {cleanup_errors}). "
                f"Another producer may be running, or system resources may not be released yet."
            )
            raise RuntimeError(error_msg) from e

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
        as_producer: bool = False,
        name_prefix: str | None = None,
        max_retries: int = 50,
        retry_delay: float = 0.1,
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

    def _read_header_consistent(self, max_retries: int = 100) -> BufferHeader:
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
        buf = self._shm_header.buf

        for attempt in range(max_retries):
            # Read sequence number before reading header (offset 12, uint32)
            seq_before = struct.unpack_from("<I", buf, 12)[0]

            # Read full header directly from buffer
            header = BufferHeader.unpack_from(buf)

            # Read sequence number again
            seq_after = struct.unpack_from("<I", buf, 12)[0]

            # If sequence numbers match, header is consistent
            if seq_before == seq_after and header.sequence_number == seq_before:
                return header

            # Brief pause before retry to avoid spinning too hard
            if attempt % 10 == 9:
                time.sleep(0.0001)  # 100 microseconds every 10 attempts

        raise RuntimeError(
            f"Failed to read consistent header after {max_retries} attempts. "
            f"Producer may be writing too frequently or system is overloaded."
        )

    def _write_header_field(self, offset: int, fmt: str, value) -> None:
        """Write a single field to the header."""
        assert self._shm_header.buf is not None
        data = struct.pack(fmt, value)
        self._shm_header.buf[offset:offset + len(data)] = data

    def _increment_sequence(self) -> None:
        """Increment the sequence counter to signal a header update."""
        assert self._shm_header.buf is not None
        current_seq = struct.unpack_from("<I", self._shm_header.buf, 12)[0]
        new_seq = (current_seq + 1) % (2**32)  # Wrap at 32-bit max
        self._shm_header.buf[12:16] = struct.pack("<I", new_seq)

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
        assert self._shm_header.buf is not None
        return struct.unpack_from("<q", self._shm_header.buf, 36)[0]

    @property
    def start_timestamp(self) -> int:
        """Timestamp of first frame currently in buffer."""
        assert self._shm_header.buf is not None
        return struct.unpack_from("<q", self._shm_header.buf, 28)[0]

    @property
    def pause_flag(self) -> bool:
        """Whether the producer should pause (debugger attached)."""
        assert self._shm_header.buf is not None
        return bool(self._shm_header.buf[48])

    @pause_flag.setter
    def pause_flag(self, value: bool) -> None:
        """Set the pause flag."""
        assert self._shm_header.buf is not None
        self._shm_header.buf[48] = 1 if value else 0

    @property
    def shutdown_flag(self) -> bool:
        """Whether the producer should shut down."""
        assert self._shm_header.buf is not None
        return bool(self._shm_header.buf[49])

    @shutdown_flag.setter
    def shutdown_flag(self, value: bool) -> None:
        """Set the shutdown flag."""
        assert self._shm_header.buf is not None
        self._shm_header.buf[49] = 1 if value else 0

    @property
    def producer_ready(self) -> bool:
        """Whether the producer is ready and has started its main loop."""
        assert self._shm_header.buf is not None
        return bool(self._shm_header.buf[50])

    @producer_ready.setter
    def producer_ready(self, value: bool) -> None:
        """Set the producer ready flag."""
        assert self._shm_header.buf is not None
        self._shm_header.buf[50] = 1 if value else 0

    @property
    def requested_timestamp(self) -> int:
        """Consumer-requested timestamp for accelerated mode."""
        assert self._shm_header.buf is not None
        return struct.unpack_from("<q", self._shm_header.buf, 76)[0]

    @requested_timestamp.setter
    def requested_timestamp(self, value: int) -> None:
        """Set the requested timestamp (consumer tells producer how far to advance)."""
        assert self._shm_header.buf is not None
        data = struct.pack("<q", value)
        self._shm_header.buf[76:84] = data

    @property
    def main_process_heartbeat_ns(self) -> int:
        """Main process heartbeat timestamp in nanoseconds (for debugger detection)."""
        assert self._shm_header.buf is not None
        return struct.unpack_from("<q", self._shm_header.buf, 96)[0]

    @main_process_heartbeat_ns.setter
    def main_process_heartbeat_ns(self, value: int) -> None:
        """Update the main process heartbeat timestamp."""
        assert self._shm_header.buf is not None
        data = struct.pack("<q", value)
        self._shm_header.buf[96:104] = data

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

        frame_count = len(frames)
        header      = self._read_header()
        write_idx   = header.write_index

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
        new_start_ts  = max(header.start_timestamp, new_write_ts - self._buffer_duration_frames)

        # Update header fields (consumers will detect inconsistency via sequence number)
        self._write_header_field(28, "<q", new_start_ts)   # start_timestamp
        self._write_header_field(44, "<I", new_write_idx)  # write_index
        self._write_header_field(36, "<q", new_write_ts)   # write_timestamp

        # Increment sequence again to signal completion
        self._increment_sequence()

    def write_spike(self, spike: SpikeRecord) -> None:
        """Write a spike to the spike ring buffer."""
        if not self._is_producer:
            raise RuntimeError("write_spike() can only be called by producer")

        assert self._shm_header.buf is not None
        assert self._shm_spikes.buf is not None

        # Increment sequence to signal start of update
        self._increment_sequence()

        # Read current indices from header
        spike_write_idx = struct.unpack_from("<I", self._shm_header.buf, 52)[0]
        spike_count     = struct.unpack_from("<I", self._shm_header.buf, 56)[0]

        # Write spike record
        offset = (spike_write_idx % self._max_spikes) * SpikeRecord.SIZE
        self._shm_spikes.buf[offset:offset + SpikeRecord.SIZE] = spike.pack()

        # Update indices in header
        new_idx = (spike_write_idx + 1) % self._max_spikes
        self._write_header_field(52, "<I", new_idx)
        self._write_header_field(56, "<I", spike_count + 1)

        # Increment sequence to signal completion
        self._increment_sequence()

    def write_stim(self, stim: StimRecord) -> None:
        """Write a stim to the stim ring buffer."""
        if not self._is_producer:
            raise RuntimeError("write_stim() can only be called by producer")

        assert self._shm_header.buf is not None
        assert self._shm_stims.buf is not None

        # Increment sequence to signal start of update
        self._increment_sequence()

        # Read current index and count
        stim_write_idx = struct.unpack_from("<I", self._shm_header.buf, 60)[0]
        stim_count     = struct.unpack_from("<I", self._shm_header.buf, 64)[0]

        # Write stim record
        offset = (stim_write_idx % self._max_stims) * StimRecord.SIZE
        self._shm_stims.buf[offset:offset + StimRecord.SIZE] = stim.pack()

        # Update indices in header
        new_idx   = (stim_write_idx + 1) % self._max_stims
        new_count = stim_count + 1
        self._write_header_field(60, "<I", new_idx)
        self._write_header_field(64, "<I", new_count)

        # Increment sequence to signal completion
        self._increment_sequence()

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

        data_bytes = event.data
        data_len = len(data_bytes)

        # Increment sequence to signal start of update
        self._increment_sequence()

        # Read current indices from header
        datastream_write_idx = struct.unpack_from("<I", self._shm_header.buf, 84)[0]
        datastream_count     = struct.unpack_from("<I", self._shm_header.buf, 88)[0]
        heap_write_offset    = struct.unpack_from("<I", self._shm_header.buf, 108)[0]
        heap_generation      = struct.unpack_from("<I", self._shm_header.buf, 112)[0]

        # Write data to heap (may wrap around)
        heap_size  = self._datastream_heap_size
        end_offset = heap_write_offset + data_len

        if end_offset <= heap_size:
            # No wrap needed
            self._shm_datastream_heap.buf[heap_write_offset:end_offset] = data_bytes
            new_heap_write_offset = end_offset % heap_size if end_offset == heap_size else end_offset
            new_heap_generation = heap_generation + 1 if end_offset == heap_size else heap_generation
        else:
            # Wrap around - increment generation
            first_chunk_size = heap_size - heap_write_offset
            self._shm_datastream_heap.buf[heap_write_offset:heap_size] = data_bytes[:first_chunk_size]
            self._shm_datastream_heap.buf[:end_offset - heap_size] = data_bytes[first_chunk_size:]
            new_heap_write_offset = end_offset - heap_size
            new_heap_generation = heap_generation + 1

        # Create index record with current generation
        index_record = DataStreamEventIndexRecord(
            timestamp   = event.timestamp,
            stream_name = event.stream_name,
            heap_offset = heap_write_offset,
            data_length = data_len,
            generation  = heap_generation  # Generation when this data was written
        )

        # Write index record
        index_offset = (datastream_write_idx % self._max_datastream_events) * DataStreamEventIndexRecord.SIZE
        self._shm_datastream_index.buf[index_offset:index_offset + DataStreamEventIndexRecord.SIZE] = index_record.pack()

        # Update header indices
        new_idx = (datastream_write_idx + 1) % self._max_datastream_events
        self._write_header_field(84, "<I", new_idx)                   # datastream_write_index
        self._write_header_field(88, "<I", datastream_count + 1)      # datastream_count
        self._write_header_field(108, "<I", new_heap_write_offset)    # datastream_heap_write_offset
        self._write_header_field(112, "<I", new_heap_generation)      # datastream_heap_generation

        # Increment sequence to signal completion
        self._increment_sequence()

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

        # Reset frame buffer pointers
        self._write_header_field(28, "<q", timestamp)  # start_timestamp
        self._write_header_field(36, "<q", timestamp)  # write_timestamp
        self._write_header_field(44, "<I", 0)          # write_index

        # Reset spike buffer
        self._write_header_field(52, "<I", 0)          # spike_write_index
        self._write_header_field(56, "<I", 0)          # spike_count

        # Reset stim buffer
        self._write_header_field(60, "<I", 0)          # stim_write_index
        self._write_header_field(64, "<I", 0)          # stim_count

        # Reset datastream buffer and heap
        self._write_header_field(84, "<I", 0)          # datastream_write_index
        self._write_header_field(88, "<I", 0)          # datastream_count
        self._write_header_field(108, "<I", 0)         # datastream_heap_write_offset
        # Increment generation to invalidate any cached heap data
        current_gen = struct.unpack_from("<I", self._shm_header.buf, 112)[0]
        self._write_header_field(112, "<I", current_gen + 1)  # datastream_heap_generation

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

        header       = self._read_header_consistent()
        to_timestamp = from_timestamp + frame_count

        # Handle timestamps before buffer start by returning zeros for the missing part
        # This allows reading "virtual" past data similar to old mock behavior
        if from_timestamp < header.start_timestamp:
            zeros_count = header.start_timestamp - from_timestamp
            if zeros_count >= frame_count:
                # All requested frames are before buffer start - return zeros
                return np.zeros((frame_count, self._channel_count), dtype=np.int16)
            else:
                # Partial: some zeros, some from buffer
                result               = np.zeros((frame_count, self._channel_count), dtype=np.int16)
                buffer_from          = header.start_timestamp
                buffer_count         = frame_count - zeros_count
                buffer_data          = self._read_frames_internal(header, buffer_from, buffer_count)
                result[zeros_count:] = buffer_data
                return result

        if to_timestamp > header.write_timestamp:
            raise ValueError(
                f"Requested end timestamp {to_timestamp} is beyond current "
                f"write position {header.write_timestamp} (data not yet available)"
            )

        return self._read_frames_internal(header, from_timestamp, frame_count)

    def _read_frames_internal(
        self,
        header        : BufferHeader,
        from_timestamp: int,
        frame_count   : int,
    ) -> ndarray:
        """Internal method to read frames from within valid buffer range."""
        # Calculate buffer indices
        # The write_index corresponds to write_timestamp
        # So we need to work backwards
        frames_from_write = header.write_timestamp - from_timestamp
        start_idx         = (header.write_index - frames_from_write) % self._buffer_duration_frames
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

        # Read header consistently to get current spike state
        header = self._read_header_consistent()
        spike_write_idx = header.spike_write_index
        spike_count     = header.spike_count

        if spike_count == 0:
            return []

        # Determine how far back we can read
        readable_count = min(spike_count, self._max_spikes)

        # Limit scan depth based on expected data rate and time window.
        # This prevents O(n) scans when the buffer is full (n = 50,000 spikes).
        # The scan limit is calculated as:
        #   (time_window_seconds) * MAX_SPIKES_PER_SECOND * SCAN_LIMIT_MULTIPLIER
        # with a minimum of MIN_SCAN_LIMIT to handle edge cases.
        time_window_frames  = to_timestamp - from_timestamp
        time_window_seconds = max(time_window_frames / self._frames_per_second, 0.1)  # At least 100ms
        expected_spikes     = int(time_window_seconds * MAX_SPIKES_PER_SECOND * SCAN_LIMIT_MULTIPLIER)
        scan_limit          = max(MIN_SCAN_LIMIT, expected_spikes)
        readable_count      = min(readable_count, scan_limit)

        result = []
        spike_buf = self._shm_spikes.buf

        for i in range(readable_count):
            # Scan backwards from most recent
            idx = (spike_write_idx - 1 - i) % self._max_spikes
            offset = idx * SpikeRecord.SIZE

            # Read just the timestamp first (8 bytes) to avoid unpacking full record (316 bytes)
            record_ts = struct.unpack_from("<q", spike_buf, offset)[0]

            if record_ts >= to_timestamp:
                # Too new, keep scanning backwards
                continue
            elif record_ts < from_timestamp:
                # Too old - scanning is always in reverse chronological order
                # (indices wrap but timestamps don't), so we can stop here
                break
            else:
                # In range - now unpack the full record
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

        # Read header consistently to get current stim state
        header = self._read_header_consistent()
        stim_write_idx = header.stim_write_index
        stim_count     = header.stim_count

        if stim_count == 0:
            return []

        # Determine how far back we can read
        readable_count = min(stim_count, self._max_stims)

        # Limit scan depth based on expected data rate and time window.
        time_window_frames  = to_timestamp - from_timestamp
        time_window_seconds = max(time_window_frames / self._frames_per_second, 0.1)
        expected_stims      = int(time_window_seconds * MAX_STIMS_PER_SECOND * SCAN_LIMIT_MULTIPLIER)
        scan_limit          = max(MIN_SCAN_LIMIT, expected_stims)
        readable_count      = min(readable_count, scan_limit)

        result = []
        stim_buf = self._shm_stims.buf

        for i in range(readable_count):
            # Scan backwards from most recent
            idx    = (stim_write_idx - 1 - i) % self._max_stims
            offset = idx * StimRecord.SIZE

            # Read just the timestamp first (8 bytes) to avoid unpacking full record
            record_ts = struct.unpack_from("<q", stim_buf, offset)[0]

            if record_ts >= to_timestamp:
                # Too new, keep scanning backwards
                continue
            elif record_ts < from_timestamp:
                # Too old - scanning is always in reverse chronological order
                # (indices wrap but timestamps don't), so we can stop here
                break
            else:
                # In range - now unpack the full record
                record = StimRecord.unpack_from(stim_buf, offset)
                result.append(record)

        # Reverse to maintain chronological order
        result.reverse()
        return result

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

        # Read header consistently to get current state
        header = self._read_header_consistent()
        datastream_write_idx = header.datastream_write_index
        datastream_count     = header.datastream_count
        current_generation   = header.datastream_heap_generation
        heap_write_offset    = header.datastream_heap_write_offset

        if datastream_count == 0:
            return []

        # Determine how far back we can read
        readable_count = min(datastream_count, self._max_datastream_events)

        # Limit scan depth based on expected data rate and time window.
        time_window_frames  = to_timestamp - from_timestamp
        time_window_seconds = max(time_window_frames / self._frames_per_second, 0.1)
        expected_events     = int(time_window_seconds * MAX_DATASTREAM_EVENTS_PER_SECOND * SCAN_LIMIT_MULTIPLIER)
        scan_limit          = max(MIN_SCAN_LIMIT, expected_events)
        readable_count      = min(readable_count, scan_limit)

        result = []
        index_buf = self._shm_datastream_index.buf
        heap_buf  = self._shm_datastream_heap.buf
        heap_size = self._datastream_heap_size

        for i in range(readable_count):
            # Scan backwards from most recent
            idx          = (datastream_write_idx - 1 - i) % self._max_datastream_events
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
        return result

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

        generation_diff = current_generation - record_generation

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
        if self.write_timestamp >= target_timestamp:
            return True

        start_time_ns = time.perf_counter_ns()

        # Calculate frame timing for sleep/spin approach
        frame_duration_ns = 1_000_000_000 // self._frames_per_second
        # Spin-wait threshold: for waits shorter than this, just spin (more accurate)
        # macOS has ~1ms sleep granularity, so spin for the final 2ms
        spin_threshold_ns = 2_000_000  # 2ms

        while True:
            current_ts = self.write_timestamp
            if current_ts >= target_timestamp:
                return True

            if timeout_seconds is not None and (time.perf_counter_ns() - start_time_ns) * 1e-9 > timeout_seconds:
                return False

            # Detect accelerated mode: if requested_timestamp >= target, producer
            # should be advancing quickly - poll without sleeping
            if self.requested_timestamp >= target_timestamp:
                # Accelerated mode: tight poll, no sleep
                continue

            # Real-time mode: hybrid sleep/spin approach
            frames_remaining = target_timestamp - current_ts
            expected_wait_ns = frames_remaining * frame_duration_ns

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
        segments = {
            self._shm_header,
            self._shm_frames,
            self._shm_spikes,
            self._shm_stims,
            self._shm_datastream_index,
            self._shm_datastream_heap,
        }
        for shm in segments:
            with contextlib.suppress(Exception):
                shm.close()
                if unlink:
                    shm.unlink()
