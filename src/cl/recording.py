from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote as url_escape

from . import Spike, Stim, _logger
from ._recording_writer import RecordingWriter
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
    - In the Simulator, data is written incrementally to disk via a background
      thread as it arrives, ensuring non-blocking operation.
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

        # Below are unused in this mock version but included for completeness
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

        # Create and start the background writer immediately
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
            if data_stream.name not in exclude_data_streams:
                self._writer.init_data_stream(data_stream.name, data_stream._attributes)

        # Handle callbacks for stopping the recording based on time
        self._scheduled_stop_timestamp: int | None = None
        if stop_after_seconds is not None:
            stop_after_frames = int(stop_after_seconds * self._neurons.get_frames_per_second())
        if stop_after_frames is not None:
            self._scheduled_stop_timestamp = self.start_timestamp + stop_after_frames
            self._neurons._timed_ops.put((self._scheduled_stop_timestamp, self.stop))

        # Register the recording
        self._neurons._recordings.append(self)

        self.status = "started"

    # --- Methods for receiving data from Neurons ---

    def _write_samples(self, samples: ndarray) -> None:
        """Queue sample frames to be written by the background writer."""
        if self.status == "started" and self._include_raw_samples:
            self._writer.write_samples(samples)

    def _write_spikes(self, spikes: list[Spike]) -> None:
        """Queue spikes to be written by the background writer."""
        if self.status == "started" and self._include_spikes:
            self._writer.write_spikes(spikes)

    def _write_stims(self, stims: list[Stim]) -> None:
        """Queue stims to be written by the background writer."""
        if self.status == "started" and self._include_stims:
            self._writer.write_stims(stims)

    def _write_data_stream_event(self, stream_name: str, timestamp: int, data: bytes) -> None:
        """Queue a data stream event to be written by the background writer."""
        if self.status == "started" and self._include_data_streams and stream_name not in self._exclude_data_streams:
            self._writer.write_data_stream_event(stream_name, timestamp, data)

    def _init_data_stream(self, stream_name: str, attributes: dict[str, Any]) -> None:
        """Initialize a new data stream in the recording."""
        if self.status == "started" and self._include_data_streams and stream_name not in self._exclude_data_streams:
            self._writer.init_data_stream(stream_name, attributes)

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
        # Also update in the writer for when the file is finalized
        self._writer.update_attributes({"application": self.attributes["application"]})

    def stop(self):
        """
        Stop the recording, if not already stopped.
        """
        if self.status == "stopped":
            return

        self.status = "stopped"

        # Local references
        frames_per_second  = self.attributes["frames_per_second"]
        current_timestamp  = self._neurons.timestamp()
        stop_timestamp     = current_timestamp
        if self._scheduled_stop_timestamp is not None:
            stop_timestamp = max(current_timestamp, self._scheduled_stop_timestamp)
        read_timestamp     = self._neurons._read_timestamp
        unread_frames      = stop_timestamp - read_timestamp

        # If there are unread frames, we will read them now.
        # This ensures final data is captured before stopping.
        if unread_frames > 0:
            self._neurons.read(unread_frames, read_timestamp)
            self._neurons._read_spikes(unread_frames, read_timestamp)

        # Update time attributes by checking how many frames have passed
        elapsed_frames = stop_timestamp - self.start_timestamp
        elapsed_secs   = elapsed_frames / frames_per_second

        created_local: datetime = self._created_local
        ended_local:   datetime = created_local + timedelta(seconds=elapsed_secs)
        ended_utc:     datetime = ended_local.astimezone(UTC)

        self.attributes["ended_localtime"]  = ended_local.isoformat()
        self.attributes["ended_utc"]        = ended_utc.isoformat()
        self.attributes["duration_frames"]  = elapsed_frames
        self.attributes["duration_seconds"] = elapsed_secs
        self.attributes["end_timestamp"]    = self.start_timestamp + elapsed_frames

        # Update final attributes in the writer and stop it
        # This will drain the queue and close the H5 file
        self._writer.update_attributes(dict(self.attributes))
        self._writer.stop()

        # Remove from active recordings list to avoid iteration overhead in data stream appends
        with contextlib.suppress(ValueError):
            self._neurons._recordings.remove(self)

        _logger.debug(f"recording stopped, saved to {self._file_path.resolve()!s}")
        return

    def has_stopped(self):
        """ Return `True` if the recording has stopped. """
        return self.status == "stopped"

    def wait_until_stopped(self):
        """
        Wait until the recording has stopped.

        Raises `RuntimeError` if the recording was not scheduled to stop automatically.
        """
        if self.has_stopped():
            return

        if self._scheduled_stop_timestamp is None:
            raise RuntimeError("Recording is not scheduled to stop")

        # Waiting time is handled by the _neurons._timed_ops(stop())
        self.stop()

    def __del__(self):
        if self.status != "stopped":
            self.stop()
