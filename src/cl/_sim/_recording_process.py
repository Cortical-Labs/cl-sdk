"""
Recording subprocess that independently reads from the shared data buffer
and writes to an HDF5 file.

Each recording runs as a separate subprocess.Popen worker that attaches to
the SharedDataBuffer as a consumer, periodically polling for new frames,
spikes, stims, and data stream events. This eliminates the dependency on
neurons.read() for data capture and prevents duplicate data when overlapping
reads occur.
"""
from __future__ import annotations

import argparse
import logging
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from queue import Empty
from typing import Any

import numpy as np
import tables

from ._data_buffer import DEFAULT_FRAMES_PER_SECOND, DataStreamEventRecord, SharedDataBuffer
from ._recording_writer import (
    DATA_STREAM_FLUSH_ROWS,
    DataStreamIndexRow,
    DataStreamState,
    SpikeRow,
    StimRow,
)
from ._subprocess import StdoutStatusWriter, read_ipc_message, start_ipc_command_reader

_logger = logging.getLogger("cl.recording_process")

# How often the subprocess polls the shared buffer for new data
POLL_INTERVAL_SECS               = 0.02   # 20ms → ~1000 frames per poll at 50 kHz internal
DATASTREAM_READ_RETRIES          = 3
DATASTREAM_READ_RETRY_DELAY_SECS = 0.001

# ── Commands sent from main thread → recording process ──────────────────────

@dataclass
class InitDataStreamCmd:
    """Initialize a named data stream group in the H5 file."""
    stream_name: str
    attributes : dict[str, Any]

@dataclass
class StopCmd:
    """Stop the recording at a given timestamp with final attributes."""
    stop_timestamp        : int
    final_attributes      : dict[str, Any]
    data_stream_attributes: dict[str, dict[str, Any]]  # stream_name → attributes
    datastream_stop_count : int | None = None

def recording_command_to_message(cmd: InitDataStreamCmd | StopCmd) -> dict[str, Any]:
    """Convert a recording command dataclass into a pipe-safe message."""
    match cmd:
        case InitDataStreamCmd():
            return {
                "type"       : "init_data_stream",
                "stream_name": cmd.stream_name,
                "attributes" : cmd.attributes,
            }
        case StopCmd():
            return {
                "type"                  : "stop",
                "stop_timestamp"        : cmd.stop_timestamp,
                "final_attributes"      : cmd.final_attributes,
                "data_stream_attributes": cmd.data_stream_attributes,
                "datastream_stop_count" : cmd.datastream_stop_count,
            }
        case _:
            raise TypeError(f"Unsupported recording command: {type(cmd).__name__}")

def recording_command_from_message(message: dict[str, Any]) -> InitDataStreamCmd | StopCmd:
    """Convert a decoded pipe message into a recording command dataclass."""
    match message.get("type"):
        case "init_data_stream":
            return InitDataStreamCmd(
                stream_name = message["stream_name"],
                attributes  = message["attributes"],
            )
        case "stop":
            return StopCmd(
                stop_timestamp         = message["stop_timestamp"],
                final_attributes       = message["final_attributes"],
                data_stream_attributes = message["data_stream_attributes"],
                datastream_stop_count  = message.get("datastream_stop_count"),
            )
    raise ValueError(f"Unknown recording command type: {message.get('type')!r}")

# ── Subprocess entry point ──────────────────────────────────────────────────

def recording_process_main(
    buffer_name_prefix      : str,
    file_path               : str,
    channel_count           : int,
    recording_start_ts      : int,
    read_from_ts            : int,
    include_spikes          : bool,
    include_stims           : bool,
    include_raw_samples     : bool,
    include_data_streams    : bool,
    exclude_data_streams    : list[str],
    initial_attributes      : dict[str, Any],
    command_queue           : queue.Queue[InitDataStreamCmd | StopCmd],
    status_writer           : StdoutStatusWriter,
    auto_stop_timestamp     : int | None,
) -> None:
    """Entry point for the recording subprocess."""
    worker = _RecordingProcessWorker(
        buffer_name_prefix       = buffer_name_prefix,
        file_path                = file_path,
        channel_count            = channel_count,
        recording_start_ts       = recording_start_ts,
        read_from_ts             = read_from_ts,
        include_spikes           = include_spikes,
        include_stims            = include_stims,
        include_raw_samples      = include_raw_samples,
        include_data_streams     = include_data_streams,
        exclude_data_streams     = set(exclude_data_streams),
        initial_attributes       = initial_attributes,
        command_queue            = command_queue,
        status_writer            = status_writer,
        auto_stop_timestamp      = auto_stop_timestamp,
    )
    try:
        worker.run()
    except Exception:
        error = traceback.format_exc()
        _logger.error("Recording process failed:\n%s", error)
        status_writer.put({"status": "error", "error": error})
    finally:
        status_writer.put({"status": "stopped"})

# ── Worker (runs inside the subprocess) ─────────────────────────────────────

class _RecordingProcessWorker:
    """Reads from the shared data buffer ring and writes to an H5 file."""

    def __init__(
        self,
        buffer_name_prefix      : str,
        file_path               : str,
        channel_count           : int,
        recording_start_ts      : int,
        read_from_ts            : int,
        include_spikes          : bool,
        include_stims           : bool,
        include_raw_samples     : bool,
        include_data_streams    : bool,
        exclude_data_streams    : set[str],
        initial_attributes      : dict[str, Any],
        command_queue           : queue.Queue[InitDataStreamCmd | StopCmd],
        status_writer           : StdoutStatusWriter,
        auto_stop_timestamp     : int | None,
    ):
        self._buffer_name_prefix   = buffer_name_prefix
        self._file_path            = file_path
        self._channel_count        = channel_count
        self._recording_start_ts   = recording_start_ts    # External 25kHz (for H5 relative timestamps)
        self._read_ts              = read_from_ts          # 25kHz (buffer access, same as external)
        self._stim_read_ts         = read_from_ts          # Separate cursor for stim reads (may lag behind _read_ts)
        self._datastream_read_count: int | None = None     # Separate cursor by event count, not timestamp
        self._include_spikes       = include_spikes
        self._include_stims        = include_stims
        self._include_raw_samples  = include_raw_samples
        self._include_data_streams = include_data_streams
        self._exclude_data_streams = exclude_data_streams
        self._attributes           = initial_attributes
        self._command_queue        = command_queue
        self._status_writer        = status_writer
        self._auto_stop_ts         = auto_stop_timestamp   # 25kHz (for buffer comparison)

        # H5 handles (set in _open_file)
        self._h5_file   : tables.File  | None = None
        self._h5_spikes : tables.Table | None = None
        self._h5_stims  : tables.Table | None = None
        self._h5_samples: tables.EArray| None = None

        # Data-stream state: stream_name → DataStreamState
        self._data_streams              : dict[str, DataStreamState] = {}
        self._pending_data_stream_events: dict[str, list[DataStreamEventRecord]] = {}
        self._final_ds_attrs            : dict[str, dict[str, Any]]  = {}
        self._datastream_stop_count     : int | None = None

        # Shared buffer handle (set in run)
        self._buffer: SharedDataBuffer | None = None

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self) -> None:
        self._buffer = SharedDataBuffer.attach(
            name_prefix                      = self._buffer_name_prefix,
            max_retries                      = 100,
            retry_delay                      = 0.05,
            unregister_from_resource_tracker = True,
        )
        self._open_file()
        self._datastream_read_count = self._buffer.datastream_count

        # Signal that the subprocess is ready to receive data
        self._status_writer.put({"status": "ready"})

        stop_ts: int | None = self._auto_stop_ts

        try:
            while True:
                # 1) Drain the command queue
                stop_cmd = self._process_commands()
                if stop_cmd is not None:
                    stop_ts = stop_cmd.stop_timestamp
                    self._attributes.update(stop_cmd.final_attributes)
                    self._final_ds_attrs = stop_cmd.data_stream_attributes
                    self._datastream_stop_count = stop_cmd.datastream_stop_count

                # 2) Determine how far to read
                buffer_write_ts = self._buffer.write_timestamp
                target_ts       = buffer_write_ts
                if stop_ts is not None:
                    target_ts = min(target_ts, stop_ts)

                # 3) Read and write new data
                self._poll_and_write(target_ts, stopping=(stop_ts is not None))
                self._status_writer.put({"status": "progress", "read_ts": self._read_ts})

                # 4) Check exit condition
                if stop_ts is not None and self._read_ts >= stop_ts:
                    break

                # 5) Wait before next poll
                time.sleep(POLL_INTERVAL_SECS)
        finally:
            self._finalise_and_close()
            if self._buffer is not None:
                self._buffer.close()

    # ── Command processing ──────────────────────────────────────────────

    def _process_commands(self) -> StopCmd | None:
        """Drain all pending commands. Returns StopCmd if one was received."""
        stop_cmd: StopCmd | None = None
        while True:
            try:
                cmd = self._command_queue.get_nowait()
            except Empty:
                break
            if isinstance(cmd, InitDataStreamCmd):
                self._init_data_stream(cmd.stream_name, cmd.attributes)
            elif isinstance(cmd, StopCmd):
                stop_cmd = cmd
        return stop_cmd

    # ── Data reading & writing ──────────────────────────────────────────

    def _poll_and_write(self, target_ts: int, *, stopping: bool = False) -> None:
        """Read data from shared buffer up to *target_ts* and write to H5.

        Buffer timestamps are at 25kHz (external rate) — no decimation or
        timestamp conversion needed.

        When *stopping* is True, stims are read through *target_ts*
        (bypassing stim_write_timestamp) because the main process has finished
        writing all stims before sending the stop command.
        """
        assert self._buffer is not None
        self._poll_data_stream_events(stop_count=self._datastream_stop_count)

        if target_ts <= self._read_ts:
            # Frames are caught up, but stims may still be behind
            # (stim_write_timestamp can advance independently).
            if self._include_stims and self._h5_stims is not None:
                buf_start = self._buffer.start_timestamp
                stim_ceiling = (target_ts + 1) if stopping else min(target_ts, self._buffer.stim_write_timestamp)
                if stim_ceiling > self._stim_read_ts:
                    stim_from = max(self._stim_read_ts, buf_start)
                    stim_records = self._buffer.read_stims(stim_from, stim_ceiling)
                    if stim_records:
                        for rec in stim_records:
                            row = self._h5_stims.row
                            row["timestamp"] = rec.intended_timestamp - self._recording_start_ts
                            row["channel"]   = rec.channel
                            row.append()
                        self._h5_stims.flush()
                    self._stim_read_ts = stim_ceiling
            return  # nothing new for frames/spikes

        # Clamp read_ts to buffer start to avoid reading data that has
        # already been overwritten in the ring buffer.
        buf_start = self._buffer.start_timestamp
        if self._read_ts < buf_start:
            _logger.warning(
                "Recording fell behind ring buffer (cursor=%d, buf_start=%d). "
                "Some data was lost.",
                self._read_ts, buf_start,
            )
            self._read_ts = buf_start

        from_ts = self._read_ts
        frame_count = target_ts - from_ts
        if frame_count <= 0:
            return

        # ── Frames ──────────────────────────────────────────────────
        if self._include_raw_samples and self._h5_samples is not None:
            try:
                frames = self._buffer.read_frames(from_ts, frame_count)
                self._h5_samples.append(frames)
            except ValueError as exc:
                _logger.warning("Failed to read frames: %s", exc)

        # ── Spikes ──────────────────────────────────────────────────
        if self._include_spikes and self._h5_spikes is not None:
            spike_records = self._buffer.read_spikes(from_ts, target_ts)
            if spike_records:
                for rec in spike_records:
                    row = self._h5_spikes.row
                    # Buffer timestamps are already at 25kHz — just make relative
                    row["timestamp"] = rec.timestamp - self._recording_start_ts
                    row["channel"]   = rec.channel
                    row["samples"]   = rec.samples
                    row.append()
                self._h5_spikes.flush()

        # ── Stims ───────────────────────────────────────────────────
        # Stims are written to the ring by the main process (not the producer).
        # Use stim_write_timestamp as the ceiling so we never read a range
        # before the main process has finished writing stims for it.
        # When stopping, the main process has already written all stims, so
        # include stims exactly on the stop boundary.
        if self._include_stims and self._h5_stims is not None:
            stim_ceiling = (target_ts + 1) if stopping else min(target_ts, self._buffer.stim_write_timestamp)
            if stim_ceiling > self._stim_read_ts:
                stim_from = max(self._stim_read_ts, buf_start) if self._stim_read_ts < buf_start else self._stim_read_ts
                stim_records = self._buffer.read_stims(stim_from, stim_ceiling)
                if stim_records:
                    for rec in stim_records:
                        row = self._h5_stims.row
                        # Buffer timestamps are already at 25kHz — just make relative
                        row["timestamp"] = rec.intended_timestamp - self._recording_start_ts
                        row["channel"]   = rec.channel
                        row.append()
                    self._h5_stims.flush()
                self._stim_read_ts = stim_ceiling

        self._read_ts = target_ts

    # ── Data stream events (received via shared memory) ────────────────

    def _poll_data_stream_events(self, stop_count: int | None = None) -> None:
        """Drain newly appended data stream events from shared memory."""
        if self._buffer is None or self._datastream_read_count is None:
            return

        read_result = self._read_datastream_events_with_retries(stop_count)
        if read_result is None:
            return

        events, next_count = read_result
        self._datastream_read_count = next_count

        if not self._include_data_streams:
            return

        for event in events:
            if event.timestamp < self._recording_start_ts:
                continue
            if self._auto_stop_ts is not None and event.timestamp > self._auto_stop_ts:
                continue
            self._write_data_stream_event(event)

    def _read_datastream_events_with_retries(
        self,
        stop_count: int | None,
    ) -> tuple[list[DataStreamEventRecord], int] | None:
        assert self._buffer is not None
        assert self._datastream_read_count is not None

        for attempt in range(DATASTREAM_READ_RETRIES + 1):
            try:
                return self._buffer.read_datastream_events_since_count(
                    self._datastream_read_count,
                    stop_count,
                )
            except RuntimeError as exc:
                if not self._is_datastream_lap_error(exc):
                    raise
                if attempt < DATASTREAM_READ_RETRIES:
                    time.sleep(DATASTREAM_READ_RETRY_DELAY_SECS)
                    continue

                target_count = self._buffer.datastream_count
                if stop_count is not None:
                    target_count = min(target_count, stop_count)
                skipped = max(0, target_count - self._datastream_read_count)
                _logger.warning(
                    "DataStream recording reader was lapped; skipping %d event(s): %s",
                    skipped,
                    exc,
                )
                self._datastream_read_count = target_count
                return None

        raise AssertionError("unreachable")

    @staticmethod
    def _is_datastream_lap_error(exc: RuntimeError) -> bool:
        message = str(exc)
        return (
            "DataStream heap wrapped during Stage 2 read" in message or
            "DataStream index ring wrapped during Stage 2 read" in message or
            "DataStream heap payload overwritten during Stage 2 read" in message
        )

    def _write_data_stream_event(self, event: DataStreamEventRecord) -> None:
        """Write a single data stream event to the H5 file."""
        stream_name = event.stream_name
        if not self._include_data_streams or stream_name in self._exclude_data_streams:
            return
        state = self._data_streams.get(stream_name)
        if state is None:
            self._pending_data_stream_events.setdefault(stream_name, []).append(event)
            return
        timestamp = event.timestamp
        data = event.data
        idx_row = state.index_table.row
        idx_row["timestamp"]   = timestamp - self._recording_start_ts
        idx_row["start_index"] = state.next_data_index
        idx_row["end_index"]   = state.next_data_index + len(data)
        idx_row.append()
        state.next_data_index += len(data)
        state.data_array.append(np.frombuffer(data, dtype=np.uint8))
        state.rows_since_flush += 1
        if state.rows_since_flush >= DATA_STREAM_FLUSH_ROWS:
            state.index_table.flush()
            state.data_array.flush()
            state.rows_since_flush = 0

    # ── H5 file management ──────────────────────────────────────────────

    def _open_file(self) -> None:
        from pathlib import Path
        Path(self._file_path).parent.mkdir(parents=True, exist_ok=True)

        self._h5_file = tables.open_file(self._file_path, mode="w")

        # Write initial attributes
        for key, value in self._attributes.items():
            self._h5_file.root._v_attrs[key] = value

        if self._include_spikes:
            self._h5_spikes = self._h5_file.create_table(
                where="/", name="spikes", description=SpikeRow,
                expectedrows=10_000_000, filters=None,
            )

        if self._include_stims:
            self._h5_stims = self._h5_file.create_table(
                where="/", name="stims", description=StimRow,
                expectedrows=10_000_000, filters=None,
            )

        if self._include_raw_samples:
            self._h5_samples = self._h5_file.create_earray(
                where="/", name="samples", atom=tables.Int16Atom(),
                shape=(0, self._channel_count),
                chunkshape=(256, self._channel_count), filters=None,
            )

        if self._include_data_streams:
            self._h5_file.create_group("/", "data_stream")

    def _init_data_stream(self, stream_name: str, attributes: dict[str, Any]) -> None:
        if self._h5_file is None or stream_name in self._data_streams:
            return
        group = self._h5_file.create_group("/data_stream", stream_name)
        group._v_attrs["name"]        = stream_name
        group._v_attrs["application"] = attributes

        index_table = self._h5_file.create_table(
            where=group, name="index", description=DataStreamIndexRow,
        )
        data_array = self._h5_file.create_earray(
            where=group, name="data", atom=tables.UInt8Atom(),
            shape=(0,), chunkshape=(2**15,),
        )
        self._data_streams[stream_name] = DataStreamState(index_table, data_array)

        pending = self._pending_data_stream_events.pop(stream_name, [])
        for event in pending:
            self._write_data_stream_event(event)

    def _compute_auto_stop_attributes(self) -> None:
        """Compute stop attributes when the subprocess auto-stops without a StopCmd."""
        # Buffer runs at 25kHz — _read_ts is already in external timestamp units
        stop_ts_external = self._read_ts
        fps              = self._attributes.get("frames_per_second", DEFAULT_FRAMES_PER_SECOND)
        elapsed_frames   = stop_ts_external - self._recording_start_ts
        elapsed_secs     = elapsed_frames / fps

        self._attributes["duration_frames"]  = elapsed_frames
        self._attributes["duration_seconds"] = elapsed_secs
        self._attributes["end_timestamp"]    = stop_ts_external

        try:
            created_local = datetime.fromisoformat(self._attributes["created_localtime"])
            ended_local = created_local + timedelta(seconds=elapsed_secs)
            ended_utc   = ended_local.astimezone(timezone.utc)
            self._attributes["ended_localtime"] = ended_local.isoformat()
            self._attributes["ended_utc"]       = ended_utc.isoformat()
        except (KeyError, ValueError):
            pass

    def _finalise_and_close(self) -> None:
        if self._h5_file is None:
            return
        try:
            # If the subprocess auto-stopped (no StopCmd received), compute
            # the duration and timing attributes from available data.
            read_ts_external = self._read_ts
            if self._attributes.get("duration_frames", 0) == 0 and read_ts_external > self._recording_start_ts:
                self._compute_auto_stop_attributes()

            # Build indexes
            if self._h5_spikes is not None:
                self._h5_spikes.cols.timestamp.create_index()
            if self._h5_stims is not None:
                self._h5_stims.cols.timestamp.create_index()
            for state in self._data_streams.values():
                state.index_table.cols.timestamp.create_index()

            # Update data stream attributes with final values
            for stream_name, attrs in self._final_ds_attrs.items():
                if stream_name in self._data_streams and self._h5_file is not None:
                    try:
                        group = self._h5_file.get_node(f"/data_stream/{stream_name}")
                        group._v_attrs["application"] = attrs
                    except tables.NoSuchNodeError:
                        pass

            # Close data stream tables
            for state in self._data_streams.values():
                state.index_table.close()
                state.data_array.close()

            # Write final attributes
            for key, value in self._attributes.items():
                self._h5_file.root._v_attrs[key] = value

            # Close main tables
            if self._h5_spikes is not None:
                self._h5_spikes.close()
            if self._h5_stims is not None:
                self._h5_stims.close()
            if self._h5_samples is not None:
                self._h5_samples.close()

            self._h5_file.close()
            _logger.debug("Recording file closed: %s", self._file_path)
        except Exception as exc:
            _logger.error("Error finalising H5 file: %s", exc)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CL SDK recording subprocess")
    parser.add_argument("--process-name", default="cl-recording")
    parser.parse_args()
    run_from_stdin()

def run_from_stdin() -> None:
    initial_message = read_ipc_message(sys.stdin)
    if initial_message.get("type") != "start":
        raise ValueError(f"Expected start message, got {initial_message.get('type')!r}")
    config = initial_message["config"]

    command_queue: queue.Queue[InitDataStreamCmd | StopCmd] = queue.Queue()
    start_ipc_command_reader(command_queue, decode=recording_command_from_message)

    recording_process_main(
        buffer_name_prefix   = config["buffer_name_prefix"],
        file_path            = config["file_path"],
        channel_count        = config["channel_count"],
        recording_start_ts   = config["recording_start_ts"],
        read_from_ts         = config["read_from_ts"],
        include_spikes       = config["include_spikes"],
        include_stims        = config["include_stims"],
        include_raw_samples  = config["include_raw_samples"],
        include_data_streams = config["include_data_streams"],
        exclude_data_streams = config["exclude_data_streams"],
        initial_attributes   = config["initial_attributes"],
        command_queue        = command_queue,
        status_writer        = StdoutStatusWriter(),
        auto_stop_timestamp  = config["auto_stop_timestamp"],
    )

if __name__ == "__main__":
    main()
