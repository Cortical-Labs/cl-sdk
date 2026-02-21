import math

import numpy as np

from scipy.sparse import dok_array

from ..util import RecordingView
from . import Array1DInt, AnalysisMetadata

class _AnalysisDataCache:
    """ Common analysis data structures and functions. """

    def __init__(
        self,
        file_path: str,
        recording: RecordingView
        ) -> None:
        self._recording = recording
        self.metadata   = AnalysisMetadata(
            file_path          = file_path,
            channel_count      = recording.attributes["channel_count"],
            sampling_frequency = recording.attributes["sampling_frequency"],
            duration_frames    = recording.attributes["duration_frames"],
            duration_seconds   = recording.attributes["duration_seconds"]
            )

    #
    # Data caches for analysis
    #

    _spike_count_bin_size: int
    """ Size of each time bin in frames. """

    _spike_count_excluded_channels: list[int]
    """ Channels exluded by _spike_count_per_time_bin. """

    _spike_count_limit_to_max_spike_timestamp: bool
    """ Limit to the max spike timestamp used by _spike_count_per_time_bin. """

    _spike_count_per_time_bin: dok_array[tuple[int, int], int]
    """
    Count of spikes in each time bin as a sparse array of shape (channel_count less exclusions, bin_count).
    - The size of each time bin is given by _spike_count_bin_size.
    - Channel exclusions given by _spike_count_excluded_channels.
    - Trailing time bins may be limited by the maximum spike timestamp in the recording.
    """

    _spike_train: dok_array[tuple[int, int], bool]
    """
    Indicates when (in frames) a spike has occurred for each channel as a boolean sparse array
    of shape (channel_count, duration_frames).
    """

    _stim_train: dok_array[tuple[int, int], bool]
    """
    Indicates when (in frames) a stim has occurred for each channel as a boolean sparse array
    of shape (channel_count, duration_frames).
    """

    #
    # Data cache getters
    #

    def get_spike_count_per_time_bin(
        self,
        bin_size:                     int,
        excluded_channels:            list[int]  = [],
        limit_to_max_spike_timestamp: bool       = False
        ) -> dok_array[tuple[int, int], int]:
        """
        Gets a sparse array representing count of spikes for each time bin,
        caching after the first call if the bin_size does not change.

        Args:
            bin_size:                     Size of each time bin in frames.
            excluded_channels:            Channels to exclude from this data.
            limit_to_max_spike_timestamp: If true, do not use the full recording duration, instead
                                          terminate at the max spike timestamp.

        Returns:
            dok_array[int]: sparse (dictionary of keys) array with shape (channel_count, bin_count).
        """
        if (
            hasattr(self, "_spike_count_bin_size") and
            hasattr(self, "_spike_count_per_time_bin") and
            hasattr(self, "_spike_count_excluded_channels") and
            hasattr(self, "_spike_count_limit_to_max_spike_timestamp") and
            (self._spike_count_bin_size == bin_size) and
            isinstance(self._spike_count_per_time_bin, dok_array) and
            self._spike_count_excluded_channels == excluded_channels and
            self._spike_count_limit_to_max_spike_timestamp == limit_to_max_spike_timestamp
            ):
            return self._spike_count_per_time_bin

        assert self._recording.spikes is not None, "Recording does not contain spikes."

        channel_count      = self._recording.attributes["channel_count"]
        sampling_frequency = self._recording.attributes["sampling_frequency"]

        duration_seconds   = self._recording.attributes["duration_seconds"]
        duration_frames    = duration_seconds * sampling_frequency
        if limit_to_max_spike_timestamp:
            duration_frames = self._recording.spikes[-1]["timestamp"]
        bin_count          = math.ceil(duration_frames / bin_size)
        spike_count_array  = dok_array((channel_count, bin_count), dtype=int)

        for spike in self._recording.spikes:
            channel  = spike["channel"]
            time_bin = spike["timestamp"] // bin_size
            spike_count_array[channel, time_bin] += 1

        channel_mask      = np.array([ ch for ch in range(channel_count) if not ch in excluded_channels ])
        spike_count_array = spike_count_array[channel_mask, :]

        self._spike_count_bin_size                     = bin_size
        self._spike_count_excluded_channels            = excluded_channels
        self._spike_count_limit_to_max_spike_timestamp = limit_to_max_spike_timestamp
        self._spike_count_per_time_bin                 = spike_count_array
        return spike_count_array

    def get_spike_train(self) -> dok_array[bool]:
        """
        Gets a sparse array representing the occurrence of a spike at each frame,
        caching after the first call.

        Returns:
            dok_array[bool]: sparse (dictionary of keys) array with shape (channel_count, duration_frames).
        """
        if (
            hasattr(self, "_spike_train") and
            isinstance(self._spike_train, dok_array)
            ):
            return self._spike_train

        channel_count     = self._recording.attributes["channel_count"]
        duration_frames   = self._recording.attributes["duration_frames"]
        spike_train_array = dok_array((channel_count, duration_frames), dtype=bool)

        assert self._recording.spikes is not None, "Recording does not contain spikes."

        for spike in self._recording.spikes:
            channel   = spike["channel"]
            timestamp = spike["timestamp"]
            spike_train_array[channel, timestamp] = 1

        self._spike_train = spike_train_array
        return spike_train_array

    def get_spike_frames_by_channel(self) -> dict[int, Array1DInt]:
        """
        Gets a view of the _spike_train containing for each channel, the frames
        that a spike has occurred.

        Returns:
            Dictionary where the keys are the channels and the values are sorted
            arrays of the frames at which a spike occurs.
        """
        channel_count = self._recording.attributes["channel_count"]
        spike_train   = self.get_spike_train()
        spike_frames_by_channel = {
            ch : spike_train[ch].nonzero()[0]
            for ch in range(channel_count)
            }
        return spike_frames_by_channel

    def get_stim_train(self) -> dok_array[bool]:
        """
        Gets a sparse array representing the occurrence of a stim at each frame,
        caching after the first call.

        Returns:
            dok_array[bool]: sparse (dictionary of keys) array with shape (channel_count, duration_frames).
        """
        if (
            hasattr(self, "_stim_train") and
            isinstance(self._stim_train, dok_array)
            ):
            return self._stim_train

        channel_count     = self._recording.attributes["channel_count"]
        duration_frames   = self._recording.attributes["duration_frames"]
        stim_train_array  = dok_array((channel_count, duration_frames), dtype=bool)

        assert self._recording.stims is not None, "Recording does not contain stims."

        for stim in self._recording.stims:
            channel   = stim["channel"]
            timestamp = stim["timestamp"]
            stim_train_array[channel, timestamp] = 1

        self._stim_train = stim_train_array
        return stim_train_array

    def get_stim_frames_by_channel(self) -> dict[int, Array1DInt]:
        """
        Gets a view of the _stim_train containing for each channel, the frames
        that a stim has occurred.

        Returns:
            Dictionary where the keys are the channels and the values are sorted
            arrays of the frames at which a stim occurs.
        """
        channel_count = self._recording.attributes["channel_count"]
        stim_train    = self.get_stim_train()
        stim_frames_by_channel = {
            ch : stim_train[ch].nonzero()[0]
            for ch in range(channel_count)
            }
        return stim_frames_by_channel