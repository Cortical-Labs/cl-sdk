"""
"""
from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any, Final, overload

import numpy as np
from dotenv import load_dotenv

_FRAME_TIME_US = 20

_logger = logging.getLogger("cl")
""" (Simulator) Logger for debugging purposes. """

# Read possible variables from .env file
load_dotenv()

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

    def __init__(self, *channels: int | Iterable[int]) -> None:
        """ Constructor for ChannelSet. """
        self._channels = np.zeros(self._CHANNELS_TOTAL, np.bool)
        flattened_chs: list[int] = []
        for ch in channels:
            if isinstance(ch, int):
                flattened_chs.append(ch)
            elif isinstance(ch, Iterable):
                for sub_ch in ch:
                    if not isinstance(sub_ch, int):
                        raise TypeError("Channels within tuples and lists must be ints")
                    flattened_chs.append(sub_ch)
            else:
                raise TypeError("channels must be an int, list or tuple")

        for channel in flattened_chs:
            self._add_channels(channel)

    def _add_channels(self, channel: int) -> None:
        """ (Simulator only) Adds a channel to this ChannelSet. """
        if channel < 0 or channel >= self._CHANNELS_TOTAL:
            raise ValueError(f"Channel number {channel} is out of range")
        self._channels[channel] = True

    def _check_operand_args(self, other: Any) -> ChannelSet:
        """ (Simulator only) Validates the args for ChannelSet operations. """
        if isinstance(other, type(self)):
            return other
        if isinstance(other, int):
            return ChannelSet(other)
        try:
            if isinstance(other, Iterable):
                return ChannelSet(*other)
        except TypeError:
            # other is an Iterable but its contents are not valid for ChannelSet construction
            # In this case, we want to raise the below TypeError rather than the one from ChannelSet.__init__ to mimic device behaviour
            pass
        raise TypeError(f"type {type(other)} cannot be added to a ChannelSet")

    def _tolist(self) -> list[int]:
        """ (Simulator only) Returns the channels in this ChannelSet as a list. """
        return np.flatnonzero(self._channels).tolist()

    def _is_empty(self) -> bool:
        """ (Simulator only) Returns True if this ChannelSet is empty. """
        return not np.any(self._channels)

    def _count(self) -> int:
        """ (Simulator only) Returns the number of channels in this ChannelSet. """
        return int(np.sum(self._channels))

    @classmethod
    def _from_array(cls, channels: np.ndarray) -> ChannelSet:
        """ (Simulator only) Constructs a ChannelSet directly from a bool array, bypassing argument validation. """
        instance = object.__new__(cls)
        instance._channels = channels
        return instance

    def __and__(self, other: ChannelSet | Iterable[int]) -> ChannelSet:
        """
        Performs an AND operation between the channels between this ChannelSet
        and either another ChannelSet or iterable containing channels.
        """
        other = self._check_operand_args(other)
        return ChannelSet._from_array(np.logical_and(self._channels, other._channels))

    def __iand__(self, other: ChannelSet | Iterable[int]) -> ChannelSet:
        """
        Performs an in-place AND operation between the channels between this ChannelSet
        and either another ChannelSet or iterable containing channels.
        """
        other = self._check_operand_args(other)
        np.logical_and(self._channels, other._channels, out=self._channels)
        return self

    def __or__(self, other: ChannelSet | Iterable[int]) -> ChannelSet:
        """
        Performs a OR operation between the channels between this ChannelSet
        and either another ChannelSet or iterable containing channels.
        """
        other = self._check_operand_args(other)
        return ChannelSet._from_array(np.logical_or(self._channels, other._channels))

    def __ior__(self, other: ChannelSet | Iterable[int]) -> ChannelSet:
        """
        Performs an in-place OR operation between the channels between this ChannelSet
        and either another ChannelSet or iterable containing channels.
        """
        other = self._check_operand_args(other)
        np.logical_or(self._channels, other._channels, out=self._channels)
        return self

    def __xor__(self, other: ChannelSet | Iterable[int]) -> ChannelSet:
        """
        Performs a XOR operation between the channels between this ChannelSet
        and either another ChannelSet or iterable containing channels.
        """
        other = self._check_operand_args(other)
        return ChannelSet._from_array(np.logical_xor(self._channels, other._channels))

    def __ixor__(self, other: ChannelSet | Iterable[int]) -> ChannelSet:
        """
        Performs an in-place XOR operation between the channels between this ChannelSet
        and either another ChannelSet or iterable containing channels.
        """
        other = self._check_operand_args(other)
        np.logical_xor(self._channels, other._channels, out=self._channels)
        return self

    def __invert__(self) -> ChannelSet:
        """
        Inverts the channels within this ChannelSet
        """
        return ChannelSet._from_array(np.logical_not(self._channels))

    def __repr__(self) -> str:
        return f"ChannelSet({', '.join(map(str, self))})"

    def __iter__(self) -> Generator[int]:
        """ Iterates over channels in this ChannelSet. """
        for channel in sorted(np.where(self._channels)[0]):
            yield int(channel)

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
        if len(args) not in {2, 4, 6}:
            raise ValueError("StimDesign requires 2, 4, or 6 arguments.")
        durations: tuple[int, ...]   = args[ ::2]  # args indices [0, 2, 4]
        currents:  tuple[float, ...] = args[1::2]  # args indices [1, 3, 5]
        self._validate(durations, currents)
        self.duration_us = sum(durations)
        self._args       = args

        self._num_phases = len(durations)
        self._padded_durations = (
            durations[0],
            durations[1] if self._num_phases > 1 else 0,
            durations[2] if self._num_phases > 2 else 0,
        )
        self._padded_currents  = (
            currents[0],
            currents[1] if self._num_phases > 1 else 0.0,
            currents[2] if self._num_phases > 2 else 0.0,
        )

    def _validate(self, durations, currents) -> None:
        """ (Simulator only) Validate the stim and raise a ValueError if needed. """
        for i, (duration_us, current_uA) in enumerate(zip(durations, currents)):
            # Total charge
            charge_pC = current_uA * duration_us
            if abs(charge_pC) > self._PHASE_CHARGE_INJECTION_LIMIT_PC:
                raise ValueError(
                    f"Charge injection of "
                    f"{duration_us} us x {current_uA} uA = {charge_pC / 1000} nC "
                    f"cannot be greater than {self._PHASE_CHARGE_INJECTION_LIMIT_PC / 1000} nC."
                    )

            # Current
            if not (abs(current_uA) <= self._CURRENT_LIMIT_UA):
                raise ValueError(
                    f"Stim current of {current_uA:.3f} uA "
                    f"cannot be {'less' if current_uA < 0 else 'greater'} than "
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

    def _get_padded(self) -> tuple[tuple[int, int, int], tuple[float, float, float], int]:
        """ (Simulator only) Returns the durations and currents padded to 3 phases, and the number of phases. """
        return self._padded_durations, self._padded_currents, self._num_phases

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

    _BURST_FREQUENCY_LIMIT_HZ: Final[int] = 200
    """ (Simulator only) Maximum allowable burst frequency. """

    def __init__(self, burst_count: int, burst_hz: float, /) -> None:
        """ Constructor for BurstDesign. """
        self._validate(burst_count, burst_hz)
        self._burst_count        = burst_count
        self._burst_requested_hz = burst_hz
        self._burst_interval_us  = int((1_000_000 / burst_hz / _FRAME_TIME_US) + 0.5) * _FRAME_TIME_US
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

from ._sim._closed_loop import Loop, LoopTick
from ._sim._stim_plan import StimPlan
from .neurons import Neurons
from .util import RecordingView
from .recording import Recording
from .data_stream import DataStream
from .error import (
    ControlRequestError,
    ControlRequiredError,
    TransactionRejected,
    ChannelQueueFull,
    SyncLimitExceeded,
    RunTimestampOrderError,
    DeferredInterruptLimitExceeded,
    RecordingFailedError,
    WsApiError,
    UnsafeOperationError
)

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

    with Neurons._get_instance() as neurons:
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

def is_simulator() -> bool:
    """
    Returns True if running in the simulator environment, False if running on a real device.
    """
    return True

from . import app
from . import analysis
from . import sim
from . import playback
from . import visualisation

__all__ = [
    "open",
    "get_system_attributes",
    "is_simulator",
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
    "ControlRequestError",
    "ControlRequiredError",
    "TransactionRejected",
    "ChannelQueueFull",
    "SyncLimitExceeded",
    "RunTimestampOrderError",
    "DeferredInterruptLimitExceeded",
    "RecordingFailedError",
    "WsApiError",
    "UnsafeOperationError",
    "app",
    "analysis",
    "sim",
    "visualisation",
    "playback",
]
