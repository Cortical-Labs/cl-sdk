from __future__ import annotations

import atexit
import contextlib
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote as url_escape
from uuid import uuid4

from . import Spike, Stim, _logger
from ._sim._recording_process import InitDataStreamCmd, StopCmd, recording_command_to_message
from ._sim._recording_writer import RecordingWriter
from ._sim._subprocess import IpcProcess
from .util import AttributesDict, RecordingView

if TYPE_CHECKING:
    from numpy import ndarray

    from . import Neurons
    from .data_stream import DataStream

def _utcdatestring(dt: datetime) -> str:
    """ Returns a formatted datetime string for prefixing recording filenames. """
    dt = dt.astimezone(UTC)
    formatted = dt.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3] + dt.strftime("%z")
    formatted = formatted[:-2] + "-" + formatted[-2:]
    return formatted

class Recording:
    """
    Handles recording functionality by the CL1 system. This is returned when
    calling `Neurons.record()`. Do not create instances of `Recording` directly.

    Note that:
    - In the Simulator, recording data is captured by a background subprocess
      that independently reads from the shared data buffer.
    - Each recording maintains its own read cursor, so multiple recordings can
      run concurrently without interfering with each other or with
      `neurons.read()` calls.
    """

    attributes: AttributesDict
    """
    Attributes that will be written to the recording file and available at
    `Recording.file.root._v_attrs` if using the raw PyTables interaface.
    See `RecordingView.attributes` for details.

    Note that:
    - Simulator recordings can be identified by file_format.version == "SDK".
    - The following attributes are included in the Simulator recording for
      completeness, but the values are empty: `git_hash`, `git_branch`,
      `git_tags`, and `git_status`.
    """

    file: dict[str, str]
    """
    `dict` containing information relating to the recording file.

    Keys:
        name:     Recording file name.
        path:     Absolute path to the recording file.
        uri_path: URL encoded file path.
    """

    start_timestamp: int
    """ Timestamp of the first frame. """

    status: Literal["started", "stopped"]
    """ Indicates the recording status. """

    def __init__(
        self,
        # Simulator only parameters
        _neurons,
        _channel_count     : int,
        _sampling_frequency: int,
        _frames_per_second : int,
        _uV_per_sample_unit: float,
        _data_streams      : dict[str, DataStream],
        _file_path_override: Path | None = None,

        # API parameters
        file_suffix         : str | None            = None,
        file_location       : str | None            = None,
        attributes          : dict[str, Any] | None = None,
        include_spikes      : bool                  = True,
        include_stims       : bool                  = True,
        include_raw_samples : bool                  = True,
        include_data_streams: bool                  = True,
        exclude_data_streams: list[str]             = [],
        stop_after_seconds  : float | None          = None,
        stop_after_frames   : int   | None          = None,

        from_seconds_ago: float | None = None,
        from_frames_ago : int | None   = None,
        from_timestamp  : int | None   = None,
        ):
        """
        Constructor for a `Recording`.

        See `Neurons.recording()` for docs.

        @private -- hide from docs
        """
        self._neurons: Neurons        = _neurons

        self._include_spikes       = include_spikes
        self._include_stims        = include_stims
        self._include_raw_samples  = include_raw_samples
        self._include_data_streams = include_data_streams
        self._exclude_data_streams = set(exclude_data_streams) if exclude_data_streams else set()

        # Timestamps
        self._created_local: datetime = datetime.now().astimezone()
        self._created_utc:   datetime = self._created_local.astimezone(timezone.utc)
        self.start_timestamp          = self._neurons.timestamp()

        if _file_path_override is not None:
            self._file_path = _file_path_override
            file_name = self._file_path.name
        else:
            # File paths, we prepend a datetime string to form the file name
            file_prefix = _utcdatestring(self._created_utc)
            if file_suffix is None:
                file_suffix = "recording"
            file_name = f"{file_prefix}_{file_suffix}.h5"

            if file_location is None:
                file_location = "./"
            self._file_path = Path(file_location) / file_name

        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        # Information about the recording file
        self.file = \
            {
                "name"    : file_name,
                "path"    : str(self._file_path.resolve()),
                "uri_path": url_escape(file_name)
            }

        # Specify default attributes that will be added to recording.root._v_attrs.
        # Some of these will be updated in Recording.stop().
        self.attributes = AttributesDict(
            application        = attributes if isinstance(attributes, dict) else {},
            hostname           = "cl-sdk",
            project_id         = "cl-sdk",
            cell_batch_id      = "cl-sdk",
            created_localtime  = self._created_local.isoformat(),
            created_utc        = self._created_utc.isoformat(),
            ended_localtime    = "",  # Updated in .stop()
            ended_utc          = "",  # Updated in .stop()
            git_hash           = "",  # Ignored in mock
            git_branch         = "",  # Ignored in mock
            git_tags           = "",  # Ignored in mock
            git_status         = "",  # Ignored in mock
            channel_count      = _channel_count,
            sampling_frequency = _sampling_frequency,
            frames_per_second  = _frames_per_second,
            uV_per_sample_unit = _uV_per_sample_unit,
            start_timestamp    = self.start_timestamp,
            end_timestamp      = 0,  # Updated in .stop()
            duration_frames    = 0,  # Updated in .stop()
            duration_seconds   = 0,  # Updated in .stop()
            file_format= {
                "version": "SDK",
                "stim_and_spike_timestamps_relative_to_start": True
                }
            )

        # Store reference to data streams for initialization
        self._data_streams: dict[str, DataStream] = _data_streams

        # Handle callbacks for stopping the recording based on time
        self._scheduled_stop_timestamp: int | None = None
        if stop_after_seconds is not None:
            stop_after_frames = int(stop_after_seconds * self._neurons.get_frames_per_second())
        if stop_after_frames is not None:
            self._scheduled_stop_timestamp = self.start_timestamp + stop_after_frames

        # ── Decide mode: "process" (normal) or "direct" (temp recording) ──
        shared_buffer = getattr(_neurons, '_shared_buffer', None)
        use_process   = _file_path_override is None and shared_buffer is not None

        if use_process:
            self._mode = "process"
            self._writer = None  # not used
            self._atexit_callback_handle = None  # Will store atexit registration

            assert shared_buffer is not None  # guaranteed by use_process condition

            _buf_start_external = shared_buffer.start_timestamp

            # Support from_* parameters by adjusting the initial read cursor (external 25kHz)
            read_from_ts = self.start_timestamp
            if from_timestamp is not None:
                read_from_ts = max(_buf_start_external, from_timestamp)
            elif from_seconds_ago is not None:
                read_from_ts = max(
                    _buf_start_external,
                    self.start_timestamp - int(from_seconds_ago * _frames_per_second),
                )
            elif from_frames_ago is not None:
                read_from_ts = max(
                    _buf_start_external,
                    self.start_timestamp - from_frames_ago,
                )

            # If reading from the past, adjust start_timestamp so relative
            # timestamps in the H5 file begin at 0.
            if read_from_ts < self.start_timestamp:
                self.start_timestamp = read_from_ts
                self.attributes["start_timestamp"] = self.start_timestamp

            # Start the recording subprocess
            buffer_name = shared_buffer.get_name_prefix()
            self._stopped_event = Event()
            self._process_error: str | None = None
            self._process_read_ts = read_from_ts
            ready_event         = Event()

            def _handle_process_status(message: dict[str, Any]) -> None:
                status = message.get("status")
                match status:
                    case "ready":
                        ready_event.set()
                    case "stopped":
                        self._stopped_event.set()
                    case "progress":
                        self._process_read_ts = int(message.get("read_ts", self._process_read_ts))
                    case "error":
                        self._process_error = str(message.get("error", "Recording subprocess failed"))
                        self._stopped_event.set()

            process_name = f"cl-recording-{uuid4().hex[:8]}"
            self._process = IpcProcess(
                target       = "cl._sim._recording_process:run_from_stdin",
                process_name = process_name,
                on_status    = _handle_process_status,
            )
            self._process.start()
            self._process.send_message({
                "type"  : "start",
                "config": {
                    "buffer_name_prefix"     : buffer_name,
                    "file_path"              : str(self._file_path),
                    "channel_count"          : _channel_count,
                    "recording_start_ts"     : self.start_timestamp,
                    "read_from_ts"           : read_from_ts,
                    "include_spikes"         : include_spikes,
                    "include_stims"          : include_stims,
                    "include_raw_samples"    : include_raw_samples,
                    "include_data_streams"   : include_data_streams,
                    "exclude_data_streams"   : list(self._exclude_data_streams),
                    "initial_attributes"     : dict(self.attributes),
                    "auto_stop_timestamp"    : self._scheduled_stop_timestamp,
                },
            })

            # Wait for the subprocess to attach to the buffer and open the file
            # before returning. This prevents the data producer from advancing
            # past the recording's initial read position.
            if not ready_event.wait(timeout=30.0):
                self._process.terminate()
                self._process.join(timeout=5.0)
                if self._process_error is not None:
                    raise RuntimeError(self._process_error)
                raise TimeoutError("Recording subprocess did not start within timeout")

            # Register atexit callback to ensure clean subprocess termination
            self._atexit_callback_handle = atexit.register(self._atexit_cleanup)

            # Initialize existing data streams in the subprocess
            for data_stream in _data_streams.values():
                if data_stream.name not in self._exclude_data_streams:
                    self._send_process_command(InitDataStreamCmd(data_stream.name, data_stream._attributes))
        else:
            # Direct mode: use thread-based RecordingWriter when no shared buffer is attached.
            self._mode                   = "direct"
            self._process                = None
            self._process_error          = None
            self._process_read_ts        = self.start_timestamp
            self._atexit_callback_handle = None      # Not used in direct mode

            self._writer = RecordingWriter(
                file_path            = self._file_path,
                channel_count        = _channel_count,
                start_timestamp      = self.start_timestamp,
                include_spikes       = include_spikes,
                include_stims        = include_stims,
                include_raw_samples  = include_raw_samples,
                include_data_streams = include_data_streams,
                exclude_data_streams = exclude_data_streams,
                initial_attributes   = dict(self.attributes),
            )
            self._writer.start()

            # Initialize any existing data streams in the writer
            for data_stream in _data_streams.values():
                if data_stream.name not in self._exclude_data_streams:
                    self._writer.init_data_stream(data_stream.name, data_stream._attributes)

        # Register the recording
        self._neurons._recordings.append(self)

        self.status = "started"

    # --- Methods for receiving data (direct mode only) ---

    def _write_samples(self, samples: ndarray) -> None:
        """Queue sample frames to be written by the background writer (direct mode only)."""
        if self._mode == "direct" and self.status == "started" and self._include_raw_samples:
            assert self._writer is not None
            self._writer.write_samples(samples)

    def _write_spikes(self, spikes: list[Spike]) -> None:
        """Queue spikes to be written by the background writer (direct mode only)."""
        if self._mode == "direct" and self.status == "started" and self._include_spikes:
            assert self._writer is not None
            self._writer.write_spikes(spikes)

    def _write_stims(self, stims: list[Stim]) -> None:
        """Queue stims to be written by the background writer (direct mode only)."""
        if self._mode == "direct" and self.status == "started" and self._include_stims:
            assert self._writer is not None
            self._writer.write_stims(stims)

    def _write_data_stream_event(self, stream_name: str, timestamp: int, data: bytes) -> None:
        """Queue a data stream event to be written by the background writer (direct mode only)."""
        if self.status != "started" or not self._include_data_streams or stream_name in self._exclude_data_streams:
            return
        if self._mode == "direct":
            assert self._writer is not None
            self._writer.write_data_stream_event(stream_name, timestamp, data)

    def _init_data_stream(self, stream_name: str, attributes: dict[str, Any]) -> None:
        """Initialize a new data stream in the recording."""
        if self.status != "started" or not self._include_data_streams or stream_name in self._exclude_data_streams:
            return
        if self._mode == "direct":
            assert self._writer is not None
            self._writer.init_data_stream(stream_name, attributes)
        else:
            self._send_process_command(InitDataStreamCmd(stream_name, attributes))

    def _send_process_command(self, cmd: InitDataStreamCmd | StopCmd) -> None:
        """Send a control command to the recording subprocess."""
        if self._process is None or not self._process.is_alive():
            return
        try:
            self._process.send_message(recording_command_to_message(cmd))
        except (BrokenPipeError, OSError) as exc:
            self._process_error = str(exc)

    def open(self):
        """
        Return a `RecordingView` of the recoding file.

        Constraints:
        - This can only be performed **after** the recording has stopped.
        """
        if self.status == "stopped":
            return RecordingView(str(self._file_path.resolve()))
        else:
            raise RuntimeError("Cannot open recording file before it has stopped")

    def set_attribute(self, key: str, value: Any):
        """
        Set a single application attribute on the recording. The application attribute
        refers to the attribute dictionary passed to `Neurons.record(attributes)`.

        Args:
            key:   Attribute key.
            value: Attribute value.

        Constraints:
        - This can only be performed **before** the recording is stopped.
        """
        self.update_attributes({key: value})

    def update_attributes(self, attributes: dict[str, Any]):
        """
        Update multiple application attributes on the recording. The application attribute
        refers to the attribute dictionary originally passed to `Neurons.record(attributes)`.

        Args:
            attributes: `dict` containing attribute keys and values to be updated.

        Constraints:
        - This can only be performed **before** the recording is stopped.
        """
        self.attributes["application"].update(attributes)
        if self._mode == "direct":
            assert self._writer is not None
            self._writer.update_attributes({"application": self.attributes["application"]})
        # In process mode, attributes are sent with the StopCmd at close time.

    def stop(self):
        """
        Stop the recording, if not already stopped.
        """
        if self.status == "stopped":
            return

        self.status = "stopped"

        # Determine stop timestamp
        current_timestamp = self._neurons.timestamp()
        stop_timestamp    = current_timestamp
        if self._scheduled_stop_timestamp is not None:
            stop_timestamp = max(current_timestamp, self._scheduled_stop_timestamp)

        shared_buffer = getattr(self._neurons, '_shared_buffer', None)
        if (
            self._scheduled_stop_timestamp is None
            and self._mode == "process"
            and shared_buffer is not None
            and getattr(self._neurons, '_use_accelerated_time', False)
        ):
            last_stim_ts = getattr(self._neurons, "_last_published_stim_timestamp", None)
            if last_stim_ts is not None and last_stim_ts > stop_timestamp:
                stop_timestamp = last_stim_ts

        # Compute final time-based attributes
        self._compute_stop_attributes(stop_timestamp)

        if self._mode == "direct":
            # Direct mode: force a read to capture final data, then stop writer
            assert self._writer is not None
            # _read_timestamp is internal 50kHz; convert to 25kHz for comparison
            read_timestamp_ext = self._neurons._read_timestamp // (self._neurons._frames_per_second // self._neurons.get_frames_per_second())
            unread_frames  = stop_timestamp - read_timestamp_ext
            if unread_frames > 0:
                self._neurons.read(unread_frames, read_timestamp_ext)
                self._neurons._read_spikes(unread_frames * (self._neurons._frames_per_second // self._neurons.get_frames_per_second()), self._neurons._read_timestamp)

            assert self._writer is not None
            self._writer.update_attributes(dict(self.attributes))
            self._writer.stop()
        else:
            # Process mode: ensure the data producer has advanced past stop_timestamp
            if shared_buffer is not None and getattr(self._neurons, '_use_accelerated_time', False):
                # Buffer runs at 25kHz — stop_timestamp is already in buffer units
                if shared_buffer.requested_timestamp < stop_timestamp:
                    shared_buffer.requested_timestamp = stop_timestamp
                shared_buffer.wait_for_timestamp(stop_timestamp, timeout_seconds=30.0)

            # Unregister atexit callback since we're stopping gracefully
            if self._atexit_callback_handle is not None:
                atexit.unregister(self._atexit_callback_handle)
                self._atexit_callback_handle = None

            # Collect final data stream attributes
            ds_attrs = {
                name: ds._attributes
                for name, ds in self._data_streams.items()
                if name not in self._exclude_data_streams
            }

            datastream_stop_count = shared_buffer.datastream_count if shared_buffer is not None else None

            # Send stop command with 25kHz stop timestamp (buffer rate),
            # and final attributes (which contain external 25kHz timestamps) for the subprocess
            self._send_process_command(StopCmd(stop_timestamp, dict(self.attributes), ds_attrs, datastream_stop_count))
            self._stopped_event.wait(timeout=30.0)
            if self._process is not None and self._process.is_alive():
                self._process.join(timeout=5.0)

        # Remove from active recordings list
        with contextlib.suppress(ValueError):
            self._neurons._recordings.remove(self)

        _logger.debug(f"recording stopped, saved to {self._file_path.resolve()!s}")

    def _compute_stop_attributes(self, stop_timestamp: int) -> None:
        """Compute and store the time-related attributes at stop time."""
        frames_per_second = self.attributes["frames_per_second"]
        elapsed_frames    = stop_timestamp - self.start_timestamp
        elapsed_secs      = elapsed_frames / frames_per_second

        ended_local: datetime = self._created_local + timedelta(seconds=elapsed_secs)
        ended_utc:   datetime = ended_local.astimezone(UTC)

        self.attributes["ended_localtime"]  = ended_local.isoformat()
        self.attributes["ended_utc"]        = ended_utc.isoformat()
        self.attributes["duration_frames"]  = elapsed_frames
        self.attributes["duration_seconds"] = elapsed_secs
        self.attributes["end_timestamp"]    = stop_timestamp

    def has_stopped(self):
        """
        Return `True` if the recording has stopped.

        In process mode this also detects auto-stop (via `stop_after_seconds`
        or `stop_after_frames`) and finalises the main-thread state.
        """
        if self.status == "stopped":
            return True
        # In process mode the subprocess may have auto-stopped
        if self._mode == "process" and self._stopped_event.is_set():
            self._finalise_auto_stop()
            return True
        return False

    def _finalise_auto_stop(self) -> None:
        """Called when the recording subprocess finished on its own (auto-stop)."""
        self.status = "stopped"
        stop_ts = self._scheduled_stop_timestamp or self._neurons.timestamp()
        self._compute_stop_attributes(stop_ts)
        with contextlib.suppress(ValueError):
            self._neurons._recordings.remove(self)
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=5.0)

    def _atexit_cleanup(self) -> None:
        """Cleanup handler called by atexit if program terminates unexpectedly."""
        if self.status != "stopped" and self._process is not None and self._process.is_alive():
            # Terminate the subprocess cleanly with a short grace period
            self._process.terminate()
            self._process.join(timeout=5.0)
            # If still alive, force kill
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)

    def wait_until_stopped(self):
        """
        Wait until the recording has stopped.

        Raises `RuntimeError` if the recording was not scheduled to stop automatically.
        """
        if self.has_stopped():
            return

        if self._scheduled_stop_timestamp is None:
            raise RuntimeError("Recording is not scheduled to stop")

        if self._mode == "process":
            # In accelerated mode, ensure the producer advances past the stop point
            shared_buffer = getattr(self._neurons, '_shared_buffer', None)
            if shared_buffer is not None and getattr(self._neurons, '_use_accelerated_time', False):
                # The shared buffer is addressed in external 25kHz units; the
                # scheduled stop timestamp is already in those units.
                if shared_buffer.requested_timestamp < self._scheduled_stop_timestamp:
                    shared_buffer.requested_timestamp = self._scheduled_stop_timestamp

            # Block until the subprocess finishes
            self._stopped_event.wait(timeout=60.0)
            if not self.has_stopped():
                # Subprocess didn't finish in time; fall through to stop()
                self.stop()
        else:
            self.stop()

    def __del__(self):
        if self.status != "stopped":
            self.stop()
