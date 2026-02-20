"""
"""
from __future__ import annotations

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any, Self, overload

import numpy as np

_logger = logging.getLogger("cl")
""" (Simulator) Logger for debugging purposes. """

class Stim:
    """
    A Stim object is created for each stim delivered by the system.

    This is accessible via `LoopTick.analysis` (which is a `DetectionResult`) when using `Neurons.loop()`
    (see `DetectionResult` for more details). Do not create instances of `Stim` directly.

    For example:

    ```python
    import cl
    with cl.open() as neurons:
        for tick in neurons.loop(ticks_per_second=100, stop_after_ticks=2):
            if tick.iteration == 0:
                # In the first iteration, perform a stim
                neurons.stim(ChannelSet(8, 9), StimDesign(160, -1.0, 160, 1.0))

            for stim in tick.analysis.stims: # Loops through each stim object in the current tick
                print(stim)                  # Print out the stim object
    ```
    """

    timestamp: int
    """ Timestamp the stim was delivered. """

    channel: int
    """ Channel the stim was delivered on. """

    def __init__(self, timestamp: int, channel: int) -> None:
        """
        @private -- hide from docs
        """
        self.timestamp = int(timestamp)
        self.channel   = int(channel)

    def __lt__(self, other: Stim) -> bool:
        """ (Simulator only) Compare two instances of Stim for neurons._stim_queue. """
        assert isinstance(other, type(self)), \
            f"Cannot compare Stim with {other.__class__.__name__}"
        if self.timestamp == other.timestamp:
            return self.channel < other.channel
        return self.timestamp < other.timestamp

    def __repr__(self) -> str:
        return f"Stim(timestamp={self.timestamp}, channel={self.channel})"

    #
    # Necessary for Pytest snapshots
    #

    def __eq__(self, other):
        """
        @private -- hide from docs
        """
        if self.__class__ is other.__class__:
            return self.__dict__ == other.__dict__
        return NotImplemented

class Spike:
    """
    A Spike object is created for each spike detected by the system.

    This is accessible via `LoopTick.analysis` (which is a `DetectionResult`) when using `Neurons.loop()`
    (see `DetectionResult` for more details). Do not create instances of `Spike` directly.

    For example:

    ```python
    import cl
    with cl.open() as neurons:
        for tick in neurons.loop(ticks_per_second=100, stop_after_ticks=2):
            for spike in tick.analysis.spikes: # Loops through each spike object in the current tick
                print(spike)                   # Print out the spike object
    ```
    """

    timestamp: int
    """ Timestamp of the sample that triggered the detection of the spike. """

    channel: int
    """ Which channel the spike was detected on. """

    channel_mean_sample: float
    """
    The rolling mean value of the channel at the time of the spike.

    In the Simulator, this is the mean of `samples`.
    """

    samples: np.ndarray[tuple[int], np.dtype[np.float32]]
    """
    Numpy array of 75 floating point µV sample zero-centered values around
    timestamp. This involves 25 samples before the spike and 50 samples
    after the spike.
    """

    def __init__(
        self,
        timestamp:           int,
        channel:             int,
        channel_mean_sample: float,
        samples:             np.ndarray[tuple[int], np.dtype[np.float32]]
        ) -> None:
        """
        @private -- hide from docs
        """
        self.timestamp           = int(timestamp)
        self.channel             = int(channel)
        self.channel_mean_sample = float(channel_mean_sample)
        self.samples             = samples

    def __repr__(self) -> str:
        return f"Spike(timestamp={self.timestamp}, channel={self.channel})"

    #
    # Necessary for Pytest snapshots
    #

    def __eq__(self, other) -> bool:
        """
        @private -- hide from docs
        """
        if self.__class__ is other.__class__:
            return self.__dict__ == other.__dict__
        return NotImplemented

from .util import deprecated

class DetectionResult:
    """
    A DetectionResult that holds spikes and stims at a given timestamp.

    This is accessible via `LoopTick.analysis` when using `Neurons.loop()`.
    Do not create instances of `DetectionResult` directly.
    """
    start_timestamp: int
    """ Timestamp of the first processed frame in this result. """

    stop_timestamp: int
    """
    Timestamp of the first **not analysed** frame after `DetectionResult.start_timestamp`.
    (i.e. `DetectionResult.start_timestamp + len(LoopTick.frames)`.)
    """

    spikes: list[Spike]
    """ List of detected spikes. """

    stims: list[Stim]
    """ List of stims delivered. """

    @deprecated("tick.analysis.start_timestamp")
    @property
    def timestamp(self) -> int:
        """
        Timestamp of the first processed frame in this result.
        @private -- hide from docs
        """
        return self.start_timestamp

    def __init__(
        self,
        start_timestamp: int,
        stop_timestamp:  int,
        spikes:          list[Spike] = [],
        stims:           list[Stim]  = []
        ) -> None:
        """
        @private -- hide from docs
        """
        self.start_timestamp = start_timestamp
        self.stop_timestamp  = stop_timestamp
        self.spikes          = spikes
        self.stims           = stims

    def __repr__(self) -> str:
        return f"DetectionResult(start_timestamp={self.start_timestamp})"

    #
    # Necessary for Pytest snapshots
    #

    def __eq__(self, other):
        """
        @private -- hide from docs
        """
        if self.__class__ is other.__class__:
            return self.__dict__ == other.__dict__
        return NotImplemented

class ChannelSet:
    """
    Stores a set of channels for stimulation.

    Args:
        *channels: One or more channels as int provided as separate arguments
                   or as a sequence of ints.

    For example:

    ```python
    # Select channels 8, 9 and 10
    ChannelSet(8, 9, 10)
    ```

    Supports convenient manipulation of channels, such as:

    ```python
    print(ChannelSet(8, 9) | ChannelSet(9, 10)) # ChannelSet(8, 9, 10)
    print(ChannelSet(8, 9) & ChannelSet(9, 10)) # ChannelSet(9)
    print(ChannelSet(8, 9) ^ ChannelSet(9, 10)) # ChannelSet(8, 10)
    print(~ChannelSet(8, 9))                    # All channels except 8, 9
    ```
    """

    _CHANNELS_TOTAL: int = 64
    """ (Simulator only) Total number of channels supported by the system. """

    _channels: np.ndarray[Any, np.dtype[np.bool]]
    """ (Simulator only) Current channels in the set. """

    def __init__(self, *channels) -> None:
        """ Constructor for ChannelSet. """
        if len(channels) < 1:
            raise TypeError("ChannelSet requires at least one channel")
        self._channels = np.zeros(self._CHANNELS_TOTAL, np.bool)
        if len(channels) == 1 and isinstance(channels[0], Sequence):
            _channels = channels[0]
        else:
            _channels = channels
        for channel in _channels:
            self._add_channels(channel)

    def _add_channels(self, channel: int) -> None:
        """ (Simulator only) Adds a channel to this ChannelSet. """
        assert isinstance(channel, int), "Channels must be integers"
        assert 0 <= channel < self._CHANNELS_TOTAL, f"Channel number {channel} out of range"
        self._channels[channel] = True

    def _check_operand_args(self, other: Any) -> ChannelSet:
        """ (Simulator only) Validates the args for ChannelSet operations. """
        if isinstance(other, type(self)):
            return other
        if isinstance(other, int):
            return ChannelSet(other)
        if isinstance(other, list) | isinstance(other, tuple):
            return ChannelSet(*other)
        raise TypeError("Channels must be an int, list or tuple")

    def _iterate_channels(self) -> Generator[int]:
        """ (Simulator only) Iterates over sorted channels in this ChannelSet. """
        for channel in sorted(np.where(self._channels)[0]):
            yield int(channel)

    def _tolist(self) -> list[int]:
        """ (Simulator only) Returns the channels in this ChannelSet as a list. """
        return np.flatnonzero(self._channels).tolist()

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
        duration_us: Pulse width in microseconds (us).
        current_uA : Current in microampere (uA).

    Constraints:
    - `duration_us` must be positive and evenly divisible by `20` us.
    - `current_uA` must be less than or equal to `3.0` uA in absolute terms (i.e. range `-3.0` to `3.0`).
    - Total charge must not exceed `3.0` nanocoulombs (nC).

    For example:

    ```python
    # Monophasic stim with current of -1.0 uA, pulse width of 160 us.
    StimDesign(160, -1.0)
    ```

    ```python
    # Biphasic stim with current of 1.0 uA, pulse width of 160 us and negative leading edge.
    StimDesign(160, -1.0, 160, 1.0)
    ```

    ```python
    # Triphasic stim with current of 1.0 uA, pulse width of 160 us and negative leading edge.
    StimDesign(160, -1.0, 160, 1.0, 160, -1.0)
    ```
    """

    _CURRENT_LIMIT_UA: float = 3.0
    """ (Simulator only) Maximum absolute stim current in microampere (uA). """

    _DURATION_BIN_US: int  = 20
    """ (Simulator only) Pulse width granularity in microseconds (us). """

    _PHASE_CHARGE_INJECTION_LIMIT_PC: float  = 3000.0
    """ (Simulator only) Maximum charge delivery across all phases in picocoulombs (pC), where (us * uA = pC). """

    duration_us: int
    """ Total stimulation duration in microseconds (us). """

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
        """ Constructor for StimDesign. """
        if not len(args) in [2, 4, 6]:
            raise ValueError("StimDesign requires 2, 4, or 6 arguments.")
        durations = args[ ::2] # args indices [0, 2, 4]
        currents  = args[1::2] # args indices [1, 3, 5]
        self._validate(durations, currents)
        self.duration_us = sum(durations)
        self._args       = args

    def _validate(self, durations, currents) -> None:
        """ (Simulator only) Validate the stim and raise a ValueError if needed. """
        for i, (duration_us, current_uA) in enumerate(zip(durations, currents)):
            # Total charge
            charge_pC = current_uA * duration_us
            if charge_pC > self._PHASE_CHARGE_INJECTION_LIMIT_PC:
                raise ValueError(
                    f"Charge injection of "
                    f"{duration_us} us x {current_uA} uA = {charge_pC / 1000} nC "
                    f"cannot be greater than {self._PHASE_CHARGE_INJECTION_LIMIT_PC / 1000} nC."
                    )

            # Current
            if not (abs(current_uA) <= self._CURRENT_LIMIT_UA):
                raise ValueError(
                    f"Stim current of {current_uA:.3f} uA "
                    f"cannot be {"less" if current_uA < 0 else "greater"} than "
                    f"{-self._CURRENT_LIMIT_UA if current_uA < 0 else self._CURRENT_LIMIT_UA:.3f} uA."
                    )
            if (i > 0) and (np.sign(currents[i-1]) == np.sign(currents[i])):
                raise ValueError(
                    f"current_uA_{i} and current_uA_{i+1} "
                    f"must have different polarities"
                )

            # Duration
            if duration_us < self._DURATION_BIN_US:
                raise ValueError(
                    f"duration_us_{i+1} "
                    f"must be at least {self._DURATION_BIN_US}"
                )
            if not (duration_us % self._DURATION_BIN_US) == 0:
                raise ValueError(
                    f"duration_us_{i+1} "
                    f"must be evenly divisible by {self._DURATION_BIN_US}"
                )

    def __repr__(self) -> str:
        return f"StimDesign{tuple(self._args)}"

class BurstDesign:
    """
    Stores the parameters of a stimulation burst.

    Args:
        burst_count: Number of stims to perform within this burst.
        burst_hz   : Frequency of stims within this burst.

    Constraints:
    - `burst_hz` must not exceed `200` Hz.

    For example:

    ```python
    # Burst containing 10 stims operating at 150 Hz
    BurstDesign(10, 150)
    ```
    """

    _burst_count: int
    """ (Simulator only) Number of stims within this burst. """

    _burst_requested_hz: float
    """ (Simulator only) Frequency to perform stims for this burst. """

    _burst_interval_us: int
    """ (Simulator only) Amount of time in microseconds (us) between each stim for this burst. """

    _BURST_FREQUENCY_LIMIT_HZ: int = 200
    """ (Simulator only) Maximum allowable burst frequency. """

    def __init__(self, burst_count: int, burst_hz: float, /) -> None:
        """ Constructor for BurstDesign. """
        self._validate(burst_count, burst_hz)
        self._burst_count        = burst_count
        self._burst_requested_hz = burst_hz
        self._burst_interval_us  = int(1 / burst_hz * 1e6)
        self._args               = (burst_count, burst_hz)

    def _validate(self, burst_count: int, burst_hz: float) -> None:
        """ (Simulator only) Validate the burst and raise a ValueError if needed. """
        if not (isinstance(burst_count, int) and (burst_count >= 0)):
            raise ValueError("requires a unsigned integer for burst_count")
        if not (isinstance(burst_hz, float) or isinstance(burst_hz, int)):
            raise ValueError("requires a floating point number for burst_hz")
        if burst_hz < 0:
            raise ValueError("Burst frequency must be positive")
        if burst_hz > self._BURST_FREQUENCY_LIMIT_HZ:
            raise ValueError(f"Burst frequency cannot be greater than {self._BURST_FREQUENCY_LIMIT_HZ}Hz")

    def __repr__(self) -> str:
        return f"BurstDesign{tuple(self._args)}"

from ._closed_loop import Loop, LoopTick
from ._stim_plan import StimPlan
from .neurons import Neurons
from .util import RecordingView
from .recording import Recording
from .data_stream import DataStream

@contextmanager
def open(take_control: bool = True, wait_until_recordable: bool = True) -> Generator[Neurons]:
    """
    Open a connection to the device, optionally take and retain control,
    and attempt to start it if necessary. The device will not be stopped
    automatically. To minimise latency, Python garbage collection is disabled
    while connection is open.

    This is the preferred entry point for the CL API. Do not use `cl.Neurons` directly.

    Args:
        take_control:          Take control of the device. Will raise a `ControlRequestError`
                               if start is required and another process has control of the device.
        wait_until_recordable: Wait (block) until the recording system is ready.

    For example:

    ```python
    import cl

    with cl.open() as neurons:
        # Your code here
        ...
    ```
    """
    import gc
    import os

    with Neurons() as neurons:
        gc_was_enabled = False

        try:
            if take_control:
                # Disable garbage collector if not already disabled
                if gc.isenabled():
                    gc.disable()
                    gc_was_enabled = True

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

            # Start WebSocket server if enabled via environment variable
            if os.getenv("CL_SDK_WEBSOCKET", "0") == "1":
                port = int(os.getenv("CL_SDK_WEBSOCKET_PORT", "1025"))
                host = os.getenv("CL_SDK_WEBSOCKET_HOST", "127.0.0.1")
                neurons._start_websocket_server(port=port, host=host)

            yield neurons

        finally:
            # Explicitly close the Neurons object after exiting the context.
            # This stops WebSocket server and data producer subprocesses.
            neurons.close()

            # Restore the garbage collector to the state it was in before we started
            if gc_was_enabled:
                gc.enable()

def get_system_attributes() -> dict[str, Any]:
    """
    Gets the system attributes that are included in each recording as a dictionary.
    This has the following structure:

    ```python
    {
        'project_id'   : str,
        'chip_id'      : str,
        'cell_batch_id': str,
        'plugin'       : dict[str, Any],   # plugin-specific attributes, with the top level keys being the plugin names
        'system_id'    : str,              # a unique identifier for the system, e.g. "cl1-0123-456"
        'hostname'     : str,              # the hostname of the system
    }
    ```
    """
    import socket
    return {
        "project_id"    : "cl-sdk-project",
        "chip_id"       : "cl-sdk-chip",
        "cell_batch_id" : "cl-sdk-cell-batch",
        "plugin"        : {},
        "system_id"     : "cl1-sdk-000",
        "hostname"      : socket.gethostname(),
    }

#
# Manages replay recordings per session
#

_CL_SDK_REPLAY_PATH: str | None = None
""" (Simulator only) Path to the recording to be replayed, persisting each session. """

def _generate_random_recording(
    sample_mean:      float,
    spike_percentile: float,
    duration_sec:     float,
    random_seed:      int
    ) -> str:
    """
    Generate a temporary recording by sampling from a Poisson distribution.
    Spikes are generated when the sample value exceeds a percentile threshold.

    Args:
        sample_mean:      Lambda value for the Poisson distribution.
        spike_percentile: Spikes are generated when the sample value exceeds
                          this percentile threshold.
        duration_sec:     Duration of the recording.
        random_seed:      Seed for the random number generator.

    Returns:
        File path to the temporary recording that can be used by cl_mock.
    """
    import atexit
    from tempfile import TemporaryDirectory
    from .recording import Recording

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
    samples: np.ndarray = rng.poisson(sample_mean, size=(duration_frames, channel_count)).astype(np.int16) - sample_mean

    # Generate spikes by sampling from Poisson distribution
    spike_threshold: float = float(np.percentile(samples, spike_percentile))
    spike_frames, spike_channels = np.where(samples > spike_threshold)
    generated_spikes: list[Spike] = []
    for frame, channel in zip(spike_frames, spike_channels, strict=True):
        if (frame < 25 or frame > (duration_frames - 50)):
            # Spikes require samples from at least 25 frames before and
            # 50 frames after the spike timestamp
            continue
        i = frame - 25
        j = frame + 50
        spike = Spike(
            timestamp           = fake_neurons.timestamp() + frame,
            channel             = channel,
            samples             = (samples[i:j, channel] * uV_per_sample_unit).astype(np.float32),
            channel_mean_sample = sample_mean
            )
        generated_spikes.append(spike)

    # Instantiate a new recording
    temp_recording = Recording(
        _neurons            = fake_neurons,
        _channel_count      = channel_count,
        _sampling_frequency = sampling_frequency,
        _frames_per_second  = frames_per_second,
        _uV_per_sample_unit = uV_per_sample_unit,
        _data_streams       = {},
        file_location       = temp_recording_dir.name
        )

    # Push the generated data to the recording
    temp_recording._write_samples(samples)
    temp_recording._write_spikes(generated_spikes)

    # Increment our timestamp then close the recording
    fake_neurons._timestamp     += duration_frames
    fake_neurons._read_timestamp = fake_neurons._timestamp
    temp_recording.stop()
    temp_recording.wait_until_stopped()
    return temp_recording.file["path"]

def _load_h5_recording() -> None:
    """
    Loads a H5 recording so that it can be replayed. The path of the recording
    is determined by the CL_SDK_REPLAY_PATH environment variable that is
    contained with a .env file.

    If CL_SDK_REPLAY_PATH is not provided, a temporary recording will be
    generated where spikes and samples are sampled from a Poisson distribution.
    The following environment variables can be optionally provided:
    - CL_SDK_SAMPLE_MEAN     : Mean samples value (default 170).
    - CL_SDK_SPIKE_PERCENTILE: Percentile threshold of samples values, above
                                which will correspond to a spike (default 99.995).
    - CL_SDK_DURATION_SEC    : Duration of the temporary recording (default 60).
    - CL_SDK_RANDOM_SEED     : Random seed (defaults to Unix time).
    """
    import os
    import time
    from pathlib import Path
    from dotenv import load_dotenv

    global _CL_SDK_REPLAY_PATH

    # Read possible variables from .env file
    load_dotenv(".env")

    # User defined replay path will always take precedence.
    if _CL_SDK_REPLAY_PATH is None:
        _CL_SDK_REPLAY_PATH = os.getenv("CL_SDK_REPLAY_PATH", None)

    # If a replay recording is not provided, we generate a temporary one using random sampling
    if _CL_SDK_REPLAY_PATH is None:
        sample_mean      = int(os.getenv("CL_SDK_SAMPLE_MEAN", 170))
        spike_percentile = float(os.getenv("CL_SDK_SPIKE_PERCENTILE", 99.995))
        duration_sec     = float(os.getenv("CL_SDK_DURATION_SEC", 60))
        random_seed      = int(os.getenv("CL_SDK_RANDOM_SEED", time.time()))
        _CL_SDK_REPLAY_PATH = \
            _generate_random_recording(
                sample_mean      = sample_mean,
                spike_percentile = spike_percentile,
                duration_sec     = duration_sec,
                random_seed      = random_seed
                )

    # Load the replay recording
    assert Path(_CL_SDK_REPLAY_PATH).exists(), f"Recording not found: {_CL_SDK_REPLAY_PATH}"
    return

_load_h5_recording()

from . import app
from . import analysis
from . import playback
from . import visualisation

__all__ = [
    "open",
    "get_system_attributes",
    "Neurons",
    "Stim",
    "Spike",
    "DetectionResult",
    "ChannelSet",
    "BurstDesign",
    "StimDesign",
    "StimPlan",
    "Loop",
    "LoopTick",
    "Recording",
    "DataStream",
    "RecordingView",
    "app",
    "analysis",
    "visualisation",
    "playback"
]
