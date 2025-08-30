import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Literal
from pathlib import Path
from urllib.parse import quote as url_escape

import tables

import numpy as np
from numpy import ndarray

from cl import Stim, Spike, _logger
from cl.util import RecordingView
from cl.data_stream import DataStream

MAX_TABLE_ROWS = 10_000_000
""" Maximum number of rows that could be written to the Spikes/Stims table. """

class SpikeRow(tables.IsDescription):
    """ Descriptor for a row of spike data within recording_file.root.spikes table. """

    timestamp = tables.Int64Col(pos=0)
    """ Timestamp of the spike. """

    channel   = tables.UInt8Col(pos=1)
    """ Channel that spiked. """

    samples   = tables.Float32Col(shape=(75,))
    """
    25 samples before + 50 samples from the time of the spike,
    shifted by the channel mean and converted to µV.
    """

class StimRow(tables.IsDescription):
    """ Descriptor for a row of stim data within recording_file.root.stims table. """

    timestamp = tables.Int64Col(pos=0)
    """ Timestamp of the stim. """

    channel   = tables.UInt8Col(pos=1)
    """ Channel that stim was conducted. """

class DataStreamIndexRow(tables.IsDescription):
    timestamp   = tables.Int64Col(pos=0)
    """ Timestamp of the datastream row. """

    start_index = tables.UInt64Col(pos=1)
    """ Index of first byte in data array. """

    end_index   = tables.UInt64Col(pos=2)
    """ Index + 1 of last byte in data array. """

def utcdatestring(dt: datetime) -> str:
    """ Returns a formatted datetime string for prefixing recording filenames. """
    dt = dt.astimezone(timezone.utc)
    formatted = dt.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3] + dt.strftime("%z")
    formatted = formatted[:-2] + "-" + formatted[-2:]
    return formatted

class Recording:
    """
    Handles recording functionality. In this mock, recording data is only saved
    when calling close().
    """

    attributes: dict[str, Any] = {}
    """
    Attributes that will be written to the recording file and available at
    recording_file.root._v_attrs. Below is a list of all the attributes
    that is included in the recording file.

    Note that:
    - Mock recordings can be identified by file_format.version == "MOCK".
    - Attributes marked with (*) are included in the mock recording for
      completeness, but the values are set to None.

    Attributes:
        application        : Application attributes as a user provided dict from
                             the attributes parameter in the Recording constructor.
        created_localtime  : When the recording is created in ISO format in the local timezone.
        created_utc        : When the recording is created in ISO format in UTC timezone.
        ended_localtime    : When the recording ended in ISO format in the local timezone.
        ended_utc          : When the recording ended in ISO format in UTC timezone.
        git_hash *         : Metadata relating to the software version.
        git_branch *       : Metadata relating to the software version.
        git_tags *         : Metadata relating to the software version.
        git_status *       : Metadata relating to the software version.
        channel_count      : Number of channels.
        sampling_frequency : Sampling frequency in Hz.
        frames_per_second  : Number of frames per second, same as sampling frequency.
        uV_per_sample_unit : Multiply the recording sample values by this constant
                             to obtain sample values as microvolts (uV).
        duration_frames    : Duration of this recording in frames.
        duration_seconds   : Duration of this recording in seconds.
        start_timestamp    : Timestamp of the first frame.
        end_timestamp      : Timestamp of the last frame.
        file_format        : Information relating to the format of the recording
                             as a dict. This contains two attributes being
                             "version" and "stim_and_spike_timestamps_relative_to_start".
                             The latter, when True, indicates that the timestamps
                             included for stims and spikes are relative to the
                             start_timestamp of the recording.
    """

    file: dict[str, str]
    """
    Information relating to the recording file.

    Attributes:
        name    : Recording file name.
        path    : Absolute path to the recording file.
        uri_path: URL encoded file path.
    """

    start_timestamp: int
    """ Timestamp of the first frame. """

    status: Literal["started", "stopped"]
    """ Indicates the recording status. """

    def __init__(
        self,
        # Mock only parameters
        _neurons,
        _channel_count     : int,
        _sampling_frequency: int,
        _frames_per_second : int,
        _uV_per_sample_unit: float,
        _recording_spikes  : list[Spike],
        _recording_stims   : list[Stim],
        _recording_samples : list[ndarray],
        _data_streams      : dict[str, DataStream],

        # API parameters
        file_suffix          :str | None            = None,
        file_location        :str | None            = None,
        attributes           :dict[str, Any] | None = None,
        include_spikes       :bool                  = True,
        include_stims        :bool                  = True,
        include_raw_samples  :bool                  = True,
        include_data_streams :bool                  = True,
        exclude_data_streams :list[str]             = [],
        stop_after_seconds   :int | None            = None,
        stop_after_frames    :int | None            = None,

        # Below are unused in this mock version but included for completeness
        from_seconds_ago     :float | None          = None,
        from_frames_ago      :int | None            = None,
        from_timestamp       :int | None            = None,
        ):
        """
        Instantiate a new recording.

        See Neurons.recording() for docs.
        """
        from cl import Neurons
        self._neurons: Neurons        = _neurons

        self._include_spikes          = include_spikes
        self._include_stims           = include_stims
        self._include_raw_samples     = include_raw_samples
        self._include_data_streams    = include_data_streams
        self._exclude_data_streams    = exclude_data_streams

        # Timestamps
        self._created_local: datetime = datetime.now().astimezone()
        self._created_utc:   datetime = self._created_local.astimezone(timezone.utc)
        self.start_timestamp          = self._neurons.timestamp()

        # File paths, we prepend a datetime string to form the file name
        file_prefix = utcdatestring(self._created_utc)
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
        self.attributes = \
            {
                "application"       : attributes if isinstance(attributes, dict) else {},
                "created_localtime" : self._created_local.isoformat(),
                "created_utc"       : self._created_utc.isoformat(),
                "ended_localtime"   : None, # Updated in .close()
                "ended_utc"         : None, # Updated in .close()
                "git_hash"          : None, # Ignored in mock
                "git_branch"        : None, # Ignored in mock
                "git_tags"          : None, # Ignored in mock
                "git_status"        : None, # Ignored in mock
                "channel_count"     : _channel_count,
                "sampling_frequency": _sampling_frequency,
                "frames_per_second" : _frames_per_second,
                "uV_per_sample_unit": _uV_per_sample_unit,
                "start_timestamp"   : self.start_timestamp,
                "end_timestamp"     : None, # Updated in .close()
                "duration_frames"   : None, # Updated in .close()
                "duration_seconds"  : None, # Updated in .close()
                "file_format": {
                    "version": "MOCK",
                    "stim_and_spike_timestamps_relative_to_start": True
                    }
            }

        # Data caches for mock functionality
        self._recording_spikes:  list[Spike]           = _recording_spikes
        self._recording_stims:   list[Stim]            = _recording_stims
        self._recording_samples: list[ndarray]         = _recording_samples
        self._data_streams:      dict[str, DataStream] = _data_streams

        # Handle callbacks for stopping the recording based on time
        self.expected_stop_timestamp: int | None = None
        if stop_after_seconds is not None:
            stop_after_frames = int(stop_after_seconds * self._neurons.get_frames_per_second())
        if stop_after_frames is not None:
            self.expected_stop_timestamp = self.start_timestamp + stop_after_frames
            self._neurons._timed_ops.put((self.expected_stop_timestamp, self.stop))

        # Register the recording
        self._neurons._recordings.append(self)

        self.status = "started"
        return

    def _clear_data_cache(self):
        """ (Mock only) Clear the data caches in the mock Neurons. """
        self._recording_spikes.clear()
        self._recording_stims.clear()
        self._recording_samples.clear()

    def open(self):
        """
        Return a RecordingView of the recoding file.

        Recording files are standard HDF5 files and can be opened with any
        HDF5 viewer or library. A RecordingView provides a more convenient
        way to access the data.
        """
        if self.status == "stopped":
            return RecordingView(str(self._file_path.resolve()))
        else:
            raise RuntimeError("Cannot open recording file before it has stopped")

    def stop(self):
        """
        Stop the recording, if not already stopped.
        """
        if self.status == "stopped":
            return

        self.status = "stopped"

        # Local references
        frames_per_second  = self.attributes["frames_per_second"]
        channel_count      = self.attributes["channel_count"]
        current_timestamp  = self._neurons.timestamp()
        stop_timestamp     = current_timestamp
        if self.expected_stop_timestamp is not None:
            stop_timestamp = max(current_timestamp, self.expected_stop_timestamp)
        read_timestamp     = self._neurons._read_timestamp
        unread_frames      = stop_timestamp - read_timestamp

        # If there are unread frames, we will read them now.
        if unread_frames > 0:
            self._neurons.read(unread_frames, read_timestamp)
            self._neurons._read_spikes(unread_frames, read_timestamp)

        # Update time attributes by checking how many frames have passed
        elapsed_frames = stop_timestamp - self.start_timestamp
        elapsed_secs   = elapsed_frames / frames_per_second

        created_local: datetime = self._created_local
        ended_local:   datetime = created_local + timedelta(seconds=elapsed_secs)
        ended_utc:     datetime = ended_local.astimezone(timezone.utc)

        self.attributes["ended_localtime"]  = ended_local.isoformat()
        self.attributes["ended_utc"]        = ended_utc.isoformat()
        self.attributes["duration_frames"]  = elapsed_frames
        self.attributes["duration_seconds"] = elapsed_secs
        self.attributes["end_timestamp"]    = self.start_timestamp + elapsed_frames

        # Create the H5 file
        h5_file = tables.open_file(str(self._file_path.resolve()), mode="w")
        for key, value in self.attributes.items():
            h5_file.root._v_attrs[key] = value

        # Add spikes
        if self._include_spikes:
            h5_spikes = h5_file.create_table(
                where        = "/",
                name         = "spikes",
                description  = SpikeRow,
                expectedrows = MAX_TABLE_ROWS,
                filters      = None
                )
            h5_spikes.cols.timestamp.create_index()

            for spike in self._recording_spikes:
                row = h5_spikes.row
                row["timestamp"] = spike.timestamp - self.start_timestamp
                row["channel"] = spike.channel
                row["samples"] = spike.samples
                row.append()

            h5_spikes.close()

        # Add stims
        if self._include_stims:
            h5_stims = h5_file.create_table(
                where        = "/",
                name         = "stims",
                description  = StimRow,
                expectedrows = MAX_TABLE_ROWS,
                filters      = None
                )
            h5_stims.cols.timestamp.create_index()

            for stim in self._recording_stims:
                row = h5_stims.row
                row["timestamp"] = stim.timestamp - self.start_timestamp
                row["channel"] = stim.channel
                row.append()

            h5_stims.close()

        # Add datastreams
        if self._include_data_streams and len(self._data_streams) > 0:
            h5_file.create_group("/", "data_stream")

            for data_stream in self._data_streams.values():
                if data_stream.name in self._exclude_data_streams:
                    continue

                data_stream_group          = h5_file.create_group("/data_stream", data_stream.name)
                group_attrs                = data_stream_group._v_attrs
                group_attrs["name"]        = data_stream.name
                group_attrs["application"] = data_stream.attributes

                data_stream_index = h5_file.create_table(
                    where       = data_stream_group,
                    name        = "index",
                    description = DataStreamIndexRow,
                    )
                data_stream_index.cols.timestamp.create_index()

                data_stream_data = h5_file.create_earray(
                    where      = data_stream_group,
                    name       = "data",
                    atom       = tables.UInt8Atom(),
                    shape      = (0,),
                    chunkshape = (2**15,)
                    )
                next_data_index: int = 0
                for timestamp, serialised_data in data_stream._data.items():
                    # Store data index
                    start_index      = next_data_index
                    next_data_index += len(serialised_data)
                    end_index        = next_data_index

                    row = data_stream_index.row
                    row["timestamp"]   = timestamp - self.start_timestamp
                    row["start_index"] = start_index
                    row["end_index"]   = end_index
                    row.append()

                    # Store data
                    data_stream_data.append(np.frombuffer(serialised_data, dtype=np.uint8))

                data_stream_index.close()
                data_stream_data.close()

        if self._include_raw_samples:
            h5_samples = h5_file.create_earray(
                where      = "/",
                name       = "samples",
                atom       = tables.Int16Atom(),
                shape      = (0, channel_count),
                chunkshape = (256, channel_count),
                filters    = None
                )
            for samples in self._recording_samples:
                h5_samples.append(samples)

        # Close, save and clear data cache
        h5_file.close()
        self._clear_data_cache()

        _logger.debug(f"recording stopped, saved to {str(self._file_path.resolve())}")
        return

    def update_attributes(self, attributes: dict[str, Any]):
        """
        Update multiple attributes on the recording.
        """
        self.attributes["application"] = attributes

    def wait_until_stopped(self):
        """
        Wait until the recording has stopped.
        Raises an error if the recording was not scheduled to stop
        automatically.
        """
        if not self.status == "stopped":
            self.stop()

    def __del__(self):
        if not self.status == "stopped":
            self.stop()