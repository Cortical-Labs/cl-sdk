"""
Tests for read_datastream_events_since_count.

The method uses a two-stage approach: a seqlock on three header reads (Stage 1)
followed by event iteration outside the seqlock (Stage 2).  See _data_buffer.py
for the full design rationale.
"""
from __future__ import annotations

import logging
import sys
import threading

import pytest

from cl._sim._data_buffer import (
    DEFAULT_MAX_DATASTREAM_EVENTS,
    MAX_DATASTREAM_EVENTS_PER_SECOND,
    DataStreamEventRecord,
    SharedDataBuffer,
)


#
# Helpers
#

def _make_payload(i: int) -> bytes:
    """Return a per-event payload that encodes the event index."""
    return b"event-" + i.to_bytes(4, "little")


def _run_writer(producer: SharedDataBuffer, stop: threading.Event, ts_box: list[int]) -> None:
    """Write datastream events as fast as possible until stop is set."""
    while not stop.is_set():
        producer.write_datastream_event(
            DataStreamEventRecord(
                timestamp   = ts_box[0],
                stream_name = "gamestate",
                data        = _make_payload(ts_box[0]),
            )
        )
        ts_box[0] += 1


#
# Tests
#

def test_read_during_concurrent_writes() -> None:
    """
    Read completes without error while the writer is appending concurrently.

    Regression for: RuntimeError: Failed to read a consistent data stream snapshot.

    Pre-writes MAX_DATASTREAM_EVENTS_PER_SECOND events (the batch to read), then
    starts a bounded writer that adds exactly the same number of extra events.
    Total events = 2 × MAX_DATASTREAM_EVENTS_PER_SECOND < DEFAULT_MAX_DATASTREAM_EVENTS,
    so the ring never fills, oldest_count stays at 0, and the index ring safety
    margin (~1621 slots) is never exhausted regardless of when Stage 1 runs.
    """
    producer  = SharedDataBuffer.create()
    consumer  = SharedDataBuffer.attach(name_prefix=producer.get_name_prefix(), max_retries=5)
    n_backlog = MAX_DATASTREAM_EVENTS_PER_SECOND  # events to pre-write and read
    n_extra   = MAX_DATASTREAM_EVENTS_PER_SECOND  # events written concurrently

    for i in range(n_backlog):
        producer.write_datastream_event(
            DataStreamEventRecord(timestamp=i, stream_name="gamestate", data=_make_payload(i))
        )

    to_count = producer.datastream_count  # == n_backlog; bounds the read batch

    def _bounded_writer() -> None:
        for j in range(n_extra):
            producer.write_datastream_event(
                DataStreamEventRecord(
                    timestamp   = n_backlog + j,
                    stream_name = "gamestate",
                    data        = _make_payload(n_backlog + j),
                )
            )

    writer_thread = threading.Thread(target=_bounded_writer, daemon=True)
    writer_thread.start()

    try:
        events, next_count = consumer.read_datastream_events_since_count(0, to_count)

        assert next_count  == to_count
        assert len(events) == n_backlog
    finally:
        writer_thread.join(timeout=2)
        producer.close(unlink=True)
        consumer.close()


def test_read_returns_correct_events() -> None:
    """
    Events returned have correct stream_name, data payload, and timestamp, both
    for a full-ring read from 0 and for incremental cursor-based reads.

    Writes a full ring of events, reads the first half to advance the cursor,
    then reads the second half, verifying each event in both batches by payload.
    This exercises the cursor-advancing pattern used by the recording process.
    """
    producer = SharedDataBuffer.create()
    consumer = SharedDataBuffer.attach(name_prefix=producer.get_name_prefix(), max_retries=5)

    try:
        n_events = DEFAULT_MAX_DATASTREAM_EVENTS
        for i in range(n_events):
            producer.write_datastream_event(
                DataStreamEventRecord(timestamp=i, stream_name="gamestate", data=_make_payload(i))
            )

        mid = n_events // 2

        # First cursor advance: read events [0, mid).
        events_a, cursor = consumer.read_datastream_events_since_count(0, mid)

        assert len(events_a) == mid
        assert cursor        == mid
        for i, event in enumerate(events_a):
            assert event.stream_name == "gamestate",    f"batch_a[{i}].stream_name={event.stream_name!r}"
            assert event.data      == _make_payload(i), f"batch_a[{i}] has wrong payload"
            assert event.timestamp == i,                f"batch_a[{i}].timestamp={event.timestamp}"

        # Second cursor advance: read events [mid, n_events).
        events_b, cursor = consumer.read_datastream_events_since_count(cursor, None)

        assert len(events_b) == n_events - mid
        assert cursor        == n_events
        for j, event in enumerate(events_b):
            expected_i = mid + j
            assert event.stream_name == "gamestate",             f"batch_b[{j}].stream_name={event.stream_name!r}"
            assert event.data      == _make_payload(expected_i), f"batch_b[{j}] has wrong payload"
            assert event.timestamp == expected_i,                f"batch_b[{j}].timestamp={event.timestamp}"
    finally:
        producer.close(unlink=True)
        consumer.close()


def test_ring_wrap_guard_fires() -> None:
    """
    RuntimeError is raised when the writer laps into the read range during Stage 2.

    Reads the full ring (safety margin = 0) while a fast concurrent writer is
    running.  setswitchinterval is set to near-zero so Stage 2 is interrupted
    frequently, giving the writer time to advance past the safety margin.
    """
    producer = SharedDataBuffer.create()
    consumer = SharedDataBuffer.attach(name_prefix=producer.get_name_prefix(), max_retries=5)
    stop     = threading.Event()
    ts_box   = [0]

    writer_thread = threading.Thread(target=_run_writer, args=(producer, stop, ts_box), daemon=True)
    writer_thread.start()

    import time
    time.sleep(0.05)  # fill the ring

    original_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-9)
    try:
        with pytest.raises(RuntimeError, match="index ring wrapped"):
            consumer.read_datastream_events_since_count(0, None)
    finally:
        sys.setswitchinterval(original_interval)
        stop.set()
        writer_thread.join(timeout=2)
        producer.close(unlink=True)
        consumer.close()


def test_ring_overflow_warning_logged(caplog: pytest.LogCaptureFixture) -> None:
    """
    When the reader is further behind than the ring capacity, a warning is logged
    stating how many events were lost, and reading resumes from oldest_count.

    Writes 2× the ring capacity so from_count=0 is behind oldest_count, then
    verifies: the warning names the number of lost events; the returned batch
    contains exactly the most-recent max_events events with correct payloads;
    and next_count advances to the total write count.
    """
    producer = SharedDataBuffer.create()
    consumer = SharedDataBuffer.attach(name_prefix=producer.get_name_prefix(), max_retries=5)

    try:
        max_events  = producer._max_datastream_events
        total_wrote = max_events * 2 + 1
        for i in range(total_wrote):
            producer.write_datastream_event(
                DataStreamEventRecord(timestamp=i, stream_name="gamestate", data=_make_payload(i))
            )

        with caplog.at_level(logging.WARNING, logger="cl.data_buffer"):
            events, next_count = consumer.read_datastream_events_since_count(0, None)

        n_lost       = total_wrote - max_events
        oldest_count = n_lost  # == total_wrote - max_events

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(str(n_lost) in m for m in warning_messages), (
            f"Expected warning to mention {n_lost} lost events; got: {warning_messages}"
        )

        assert len(events) == max_events
        assert next_count  == total_wrote

        # Events should be the last max_events written, starting at oldest_count.
        for j, event in enumerate(events):
            expected_i = oldest_count + j
            assert event.timestamp == expected_i,         f"event[{j}].timestamp={event.timestamp}"
            assert event.data      == _make_payload(expected_i), f"event[{j}] has wrong payload"
    finally:
        producer.close(unlink=True)
        consumer.close()


def test_heap_previous_generation_payload_overwrite_guard_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    RuntimeError is raised when a previous-generation heap payload is overwritten
    during Stage 2 without the heap generation advancing.

    Uses a tiny 32-byte heap and 8-byte payloads.  Four writes fill the heap and
    advance the heap generation to 1 with write_offset=0, leaving the four readable
    records in the previous generation.  During the first Stage 2 payload read,
    the monkeypatched reader performs one additional producer write, which advances
    write_offset to 8 in the same generation and invalidates the payload at offset 0.
    """
    producer = SharedDataBuffer.create(max_datastream_events=16, datastream_heap_size=32)
    consumer = SharedDataBuffer.attach(name_prefix=producer.get_name_prefix(), max_retries=5)

    try:
        payload_size = 8
        for i in range(4):
            producer.write_datastream_event(
                DataStreamEventRecord(
                    timestamp   = i,
                    stream_name = "gamestate",
                    data        = bytes([i]) * payload_size,
                )
            )

        original_read_heap_data = consumer._read_heap_data
        has_written             = False

        def _read_heap_data_with_overwrite(
            heap_offset       : int,
            data_length       : int,
            record_generation : int,
            current_generation: int,
            heap_write_offset : int,
            heap_buf,
            heap_size         : int,
        ) -> bytes:
            nonlocal has_written

            data = original_read_heap_data(
                heap_offset,
                data_length,
                record_generation,
                current_generation,
                heap_write_offset,
                heap_buf,
                heap_size,
            )
            if not has_written:
                has_written = True
                producer.write_datastream_event(
                    DataStreamEventRecord(
                        timestamp   = 100,
                        stream_name = "gamestate",
                        data        = b"x" * payload_size,
                    )
                )
            return data

        monkeypatch.setattr(consumer, "_read_heap_data", _read_heap_data_with_overwrite)

        with pytest.raises(RuntimeError, match="heap payload overwritten"):
            consumer.read_datastream_events_since_count(0, 4)
    finally:
        producer.close(unlink=True)
        consumer.close()
