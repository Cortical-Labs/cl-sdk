from __future__ import annotations
import logging
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Self, overload
from collections.abc import Generator, Sequence

import numpy as np
from numpy import ndarray

_logger = logging.getLogger("cl")
""" (Mock) Logger for debugging purposes. """

@dataclass
class Stim:
    """ A Stim object is created for each stim delivered by the system. """
    timestamp: int
    """ Timestamp the stim was delivered. """

    channel: int
    """ Channel the stim was delivered on. """

    def __lt__(self, other: "Stim") -> bool:
        """ (Mock only) Compare two instances of Stim for neurons._stim_queue. """
        assert isinstance(other, type(self)), \
            f"Cannot compare Stim with {other.__class__.__name__}"
        if self.timestamp == other.timestamp:
            return self.channel < other.channel
        return self.timestamp < other.timestamp

@dataclass
class Spike:
    """ A Spike object is created for each spike detected by the system. """

    timestamp: int
    """ Timestamp of the sample that triggered the detection of the spike. """

    channel: int
    """ Which channel the spike was detected on. """

    channel_mean_sample: float
    """
    The rolling mean value of the channel at the time of the spike. In the mock,
    this is the mean of the samples.
    """

    samples: ndarray[Any, np.dtype[np.float32]]
    """
    Numpy array of 75 floating point µV sample zero-centered values around
    timestamp. This involves 25 samples before the spike and 50 samples
    after the spike.
    """

@dataclass
class DetectionResult:
    """
    A DetectionResult that holds spikes and stims at a given timestamp.
    """
    timestamp: int
    """ Timestamp of the first processed frame in this result. """

    spikes: list[Spike] = field(default_factory=list)
    """ List of detected spikes. """

    stims: list[Stim]   = field(default_factory=list)
    """ List of stims delivered. """

class ChannelSet:
    """
    Stores a set of channels for stimulation.

    Args:
        *channels: One or more channels as int provided as constructor arguments.

    For example:

        # Select channels 8, 9 and 10
        ChannelSet(8, 9, 10)
    """

    _CHANNELS_TOTAL: int = 64
    """ (Mock only) Total number of channels supported by the system. """

    _channels: ndarray[Any, np.dtype[np.bool]]
    """ (Mock only) Current channels in the set. """

    def __init__(self, *args) -> None:
        if len(args) < 1:
            raise TypeError("ChannelSet requires at least one channel")
        self._channels = np.zeros(self._CHANNELS_TOTAL, np.bool)
        for channel in args:
            self._add_channels(channel)

    def _add_channels(self, channel: int):
        """ (Mock only) Adds a channel to this ChannelSet. """
        assert isinstance(channel, int), "Channels must be integers"
        assert 0 <= channel < self._CHANNELS_TOTAL, f"Channel number {channel} out of range"
        self._channels[channel] = True

    def _check_operand_args(self, other: Any) -> "ChannelSet":
        """ (Mock only) Validates the args for ChannelSet operations. """
        if isinstance(other, type(self)):
            return other
        if isinstance(other, int):
            return ChannelSet(other)
        if isinstance(other, list) | isinstance(other, tuple):
            return ChannelSet(*other)
        raise TypeError("Channels must be an int, list or tuple")

    def _iterate_channels(self) -> Generator[int, None, None]:
        """ (Mock only) Iterates over sorted channels in this ChannelSet. """
        for channel in sorted(np.where(self._channels)[0]):
            yield int(channel)

    def __and__(self, other: ChannelSet | Sequence[int]) -> Self:
        """
        Performs an AND operation between the channels between this ChannelSet
        and either another ChannelSet or iterable containing channels.
        """
        other = self._check_operand_args(other)
        self._channels = np.logical_and(self._channels, other._channels)
        return self

    def __or__(self, other: ChannelSet | Sequence[int]) -> Self:
        """
        Performs a OR operation between the channels between this ChannelSet
        and either another ChannelSet or iterable containing channels.
        """
        other = self._check_operand_args(other)
        self._channels = np.logical_or(self._channels, other._channels)
        return self

    def __xor__(self, other: ChannelSet | Sequence[int]) -> Self:
        """
        Performs a XOR operation between the channels between this ChannelSet
        and either another ChannelSet or iterable containing channels.
        """
        other = self._check_operand_args(other)
        self._channels = np.logical_xor(self._channels, other._channels)
        return self

    def __invert__(self) -> Self:
        """
        Inverts the channels within this ChannelSet
        """
        self._channels = ~self._channels
        return self

    def __repr__(self) -> str:
        return f"ChannelSet{tuple(self._iterate_channels())}"

class StimDesign:
    """
    Stores the parameters of a mono, bi, or triphasic stim design by specifying
    2, 4 or 6 pairs of arguments respectively.

    Args:
        duration_us: Pulse width in microseconds (us), must be positive and
            evenly divisible by _DURATION_BIN.
        current_uA : Current in microampere (uA). (Mock only) We validate the
            absolute current up to a recommended safe range (_MAX_CURRENT).

    For example:

        # Monophasic stim with current of -1.0 uA, pulse width of 160 us.
        StimDesign(160, -1.0)

        # Biphasic stim with current of 1.0 uA, pulse width of 160 us and negative leading edge.
        StimDesign(160, -1.0, 160, 1.0)

        # Triphasic stim with current of 1.0 uA, pulse width of 160 us and negative leading edge.
        StimDesign(160, -1.0, 160, 1.0, 160, -1.0)
    """

    _MAX_CURRENT: float = 11.0
    """ (Mock only) Recommended absolute stim current in microampere (uA). """

    _DURATION_BIN: int  = 20
    """ (Mock only) Pulse width granularity in microseconds (us). """

    _total_duration_us: float
    """ (Mock only) Total duration of this stim design. """

    @overload
    def __init__(
        self,
        duration_us_1: int,
        current_uA_1 : float,
        /
    ):
        ...

    @overload
    def __init__(
        self,
        duration_us_1: int,
        current_uA_1 : float,
        duration_us_2: int,
        current_uA_2 : float,
        /
    ):
        ...

    @overload
    def __init__(
        self,
        duration_us_1: int,
        current_uA_1 : float,
        duration_us_2: int,
        current_uA_2 : float,
        duration_us_3: int,
        current_uA_3 : float,
        /
    ):
        ...

    def __init__(self, *args) -> None:
        if not len(args) in [2, 4, 6]:
            raise ValueError("StimDesign requires 2, 4, or 6 arguments.")
        durations = args[ ::2] # args indices [0, 2, 4]
        currents  = args[1::2] # args indices [1, 3, 5]
        self._validate(durations, currents)
        self._total_duration_us = sum(durations)
        self._args              = args

    def _validate(self, durations, currents):
        """ (Mock only) Validate the stim and raise a ValueError if needed. """
        #
        # duration_us
        #
        for i, duration in enumerate(durations):
            if duration < self._DURATION_BIN:
                raise ValueError(
                    f"duration_us_{i+1} "
                    f"must be at least {self._DURATION_BIN}"
                )
            if not (duration % self._DURATION_BIN) == 0:
                raise ValueError(
                    f"duration_us_{i+1} "
                    f"must be evenly divisible by {self._DURATION_BIN}"
                )

        #
        # current_uA
        #
        for i, current in enumerate(currents):
            if not (abs(current) <= self._MAX_CURRENT):
                raise ValueError(
                    f"current_uA_{i} "
                    f"must be less or equal to {self._MAX_CURRENT} uA"
                )
            if (i > 0) and (np.sign(currents[i-1]) == np.sign(currents[i])):
                raise ValueError(
                    f"current_uA_{i} and current_uA_{i+1} "
                    f"must have different polarities"
                )

    def __repr__(self) -> str:
        return f"StimDesign{tuple(self._args)}"

class BurstDesign:
    """
    Stores the parameters of a stimulation burst.

    Args:
        burst_count: Number of stims to perform within this burst.
        burst_hz   : Frequency of stims within this burst.

    For example, a burst containing 10 stims operating at 150 Hz:

        BurstDesign(10, 150)
    """

    _burst_count: int
    """ (Mock only) Number of stims within this burst. """

    _burst_hz: float
    """ (Mock only) Frequency to perform stims for this burst. """

    _burst_interval_frames: int
    """ (Mock only) Number of frames between each stim for this burst. """

    def __init__(self, burst_count: int, burst_hz: float, /):
        self._validate(burst_count, burst_hz)
        self._burst_count       = burst_count
        self._burst_hz          = burst_hz
        self._burst_interval_us = int(1 / burst_hz * 1e6)
        self._args              = (burst_count, burst_hz)

    def _validate(self, burst_count: int, burst_hz: float):
        """ (Mock only) Validate the burst and raise a ValueError if needed. """
        if not (isinstance(burst_count, int) and (burst_count >= 0)):
            raise ValueError("burst_count needs to be a positive int")
        if burst_hz < 0:
            raise ValueError("burst_hz must be positive")

    def __repr__(self) -> str:
        return f"BurstDesign{tuple(self._args)}"

from .closed_loop import Loop
from .neurons import Neurons

@contextmanager
def open(take_control: bool = True, wait_until_recordable: bool = True):
    """
    Open a connection to the device, optionally take and retain control,
    and attempt to start it if necessary. The device will not be stopped
    automatically.

    Will raise a ControlRequestError if start is required and another process
    has control of the device (Exception not simulated in mock).
    """
    with Neurons() as neurons:
        try:
            if take_control:
                neurons.take_control()
                if not neurons.has_started():
                    neurons.start()
            else:
                if not neurons.has_started():
                    neurons.take_control()
                    neurons.start()
                    neurons.release_control()

            # A recently started device will not immediately be readable.
            neurons.wait_until_readable()

            # The background recording system may not be ready immediately.
            if wait_until_recordable:
                neurons.wait_until_recordable()

            yield neurons

        finally:
            # Explicitly close the Neurons object after exiting the context.
            # Otherwise top level code in Jupyter would hold an open Neurons.
            neurons.close()

#
# Manages replay recordings per session
#

_CL_MOCK_REPLAY_PATH: str | None = None
""" (Mock only) Path to the recording to be replayed, persisting each session. """

def _generate_random_recording(
    sample_mean:      int,
    spike_percentile: float,
    duration_sec:     float,
    random_seed:      int
    ) -> str:
    """
    Generate a temporary recording by sampling from a Poisson distribution.
    Spikes are generated when the sample value exceeds a percentile threshold.

    Args:
        sample_mean: Lambda value for the Poisson distribution.
        spike_percentile: Spikes are generated when the sample value exceeds
            this percentile threshold.
        duration_sec: Duration of the recording.
        random_seed: Seed for the random number generator.

    Returns:
        File path to the temporary recording that can be used by cl_mock.
    """
    import atexit
    from tempfile import TemporaryDirectory
    from cl.recording import Recording

    _logger.debug(
        f"generating a temporary {duration_sec:2f} sec recording, "
        f"with mean sample value = {sample_mean}, "
        f"spike percentile = {spike_percentile}"
        )

    # Create a temporary directory and register for it to be automatically cleanedup
    temp_recording_dir = TemporaryDirectory(delete=True)
    atexit.register(temp_recording_dir.cleanup)

    # Define recording attributes
    channel_count      = 64
    frames_per_second  = 25_000
    sampling_frequency = frames_per_second
    uV_per_sample_unit = 0.195

    # Define data cache for spikes and samples
    recording_spikes : list[Spike]   = []
    recording_stims  : list[Stim]    = []
    recording_samples: list[ndarray] = []

    # Define timing attributes
    duration_frames = int(duration_sec * frames_per_second)

    # Here, we need to create a FakeNeurons class to pass to our Recording
    # in order to generate a temporary random recording to use with Neurons.
    class FakeNeurons:

        _timestamp:         int             = 0
        _read_timestamp:    int             = 0
        _frames_per_second: int             = 25_000
        _recordings:        list[Recording] = []

        def timestamp(self) -> int:
            return self._timestamp

        def get_frames_per_second(self) -> int:
            return self._frames_per_second

    fake_neurons = FakeNeurons()

    # Create random number generator
    rng = np.random.RandomState(random_seed)

    # Generate samples by sampling from Poisson distribution
    samples: ndarray = rng.poisson(sample_mean, size=(duration_frames, channel_count)).astype(np.int16)
    recording_samples.append(samples)

    # Generate spikes by sampling from Poisson distribution
    spike_threshold: float = float(np.percentile(samples, spike_percentile))
    spike_frames, spike_channels = np.where(samples > spike_threshold)
    for frame, channel in zip(spike_frames, spike_channels):
        if (frame < 25 or frame > (duration_frames - 50)):
            # Spikes require samples from at least 25 frames before and
            # 50 frames after the spike timestamp
            continue
        i = frame - 25
        j = frame + 50
        spike = Spike(
            timestamp           = fake_neurons.timestamp() +  frame,
            channel             = channel,
            samples             = (samples[i:j, channel] * uV_per_sample_unit).astype(np.float32),
            channel_mean_sample = sample_mean
            )
        recording_spikes.append(spike)

    # Instantiate a new recording
    temp_recording = Recording(
        _neurons            = fake_neurons,
        _channel_count      = channel_count,
        _sampling_frequency = sampling_frequency,
        _frames_per_second  = frames_per_second,
        _uV_per_sample_unit = uV_per_sample_unit,
        _recording_spikes   = recording_spikes,
        _recording_stims    = recording_stims,
        _recording_samples  = recording_samples,
        _data_streams       = {},
        file_location       = temp_recording_dir.name
        )

    # Increment our timestamp then close the recording
    fake_neurons._timestamp     += duration_frames
    fake_neurons._read_timestamp = fake_neurons._timestamp
    temp_recording.stop()
    temp_recording.wait_until_stopped()
    return temp_recording.file["path"]

def _load_h5_recording():
    """
    Loads a H5 recording so that it can be replayed. The path of the recording
    is determined by the CL_MOCK_REPLAY_PATH environment variable that is
    contained with a .env file.

    If CL_MOCK_REPLAY_PATH is not provided, a temporary recording will be
    generated where spikes and samples are sampled from a Poisson distribution.
    The following environment variables can be optionally provided:
    - CL_MOCK_SAMPLE_MEAN     : Mean samples value (default 170).
    - CL_MOCK_SPIKE_PERCENTILE: Percentile threshold of samples values, above
                                which will correspond to a spike (default 99.995).
    - CL_MOCK_DURATION_SEC    : Duration of the temporary recording (default 60).
    - CL_MOCK_RANDOM_SEED     : Random seed (defaults to Unix time).
    """
    import os
    import time
    from pathlib import Path
    from dotenv import load_dotenv

    global _CL_MOCK_REPLAY_PATH

    # Read possible variables from .env file
    load_dotenv(".env")

    # User defined replay path will always take precedence.
    if _CL_MOCK_REPLAY_PATH is None:
        _CL_MOCK_REPLAY_PATH = os.getenv("CL_MOCK_REPLAY_PATH", None)

    # If a replay recording is not provided, we generate a temporary one using random sampling
    if _CL_MOCK_REPLAY_PATH is None:
        sample_mean      = int(os.getenv("CL_MOCK_SAMPLE_MEAN", 170))
        spike_percentile = float(os.getenv("CL_MOCK_SPIKE_PERCENTILE", 99.995))
        duration_sec     = float(os.getenv("CL_MOCK_DURATION_SEC", 60))
        random_seed      = int(os.getenv("CL_MOCK_RANDOM_SEED", time.time()))
        _CL_MOCK_REPLAY_PATH = \
            _generate_random_recording(
                sample_mean      = sample_mean,
                spike_percentile = spike_percentile,
                duration_sec     = duration_sec,
                random_seed      = random_seed
                )

    # Load the replay recording
    assert Path(_CL_MOCK_REPLAY_PATH).exists(), f"Recording not found: {_CL_MOCK_REPLAY_PATH}"
    return

_load_h5_recording()