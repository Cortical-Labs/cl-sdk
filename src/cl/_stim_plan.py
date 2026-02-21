import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import BurstDesign, ChannelSet, StimDesign

if TYPE_CHECKING:
    from . import Neurons

FROZEN_ERROR_MSG = "Cannot modify a StimPlan after it has been used."

@dataclass
class Operation:
    op  : Callable
    args: Sequence[Any]

class StimPlan:
    """
    Allows building and executing a sequence of stim operations that can be run on demand. The `StimPlan`
    cannot be modified further after it has been run once. Stim plans are created with `Neurons.create_stim_plan()`,
    do not create `StimPlan` instances directly.

    A `RuntimeError` will be raised if any modification method is called after the first `run()`.

    For example:

    ```python
    import cl

    with cl.open() as neurons:
        # Create a stim plan
        stim_plan = neurons.create_stim_plan()
    ```
    """

    _frozen: bool
    """
    When `True`, this `StimPlan` cannot be further modified.
    In this state, only `run()` can be called, and any attempt to modify it will raise a `RuntimeError`.
    """

    _channels_to_interrupt: ChannelSet | None = None
    """ Channels to interrupt when this plan is run. """

    def __init__(self, neurons) -> None:
        """
        Initializes a new `StimPlan` instance.

        @private -- hide from docs
        """
        self._neurons: Neurons = neurons
        self._frozen           = False
        self._operations: list[Operation] = []

    def stim(
        self,
        channel_set:  ChannelSet  | int,
        stim_design:  StimDesign  | float,
        /,
        burst_design: BurstDesign | None   = None,
        lead_time_us: int                  = 80
        ) -> None:
        """
        Enqueues the same operation as `Neurons.stim()` onto this `StimPlan`.

        Args:
            channel_set : A `ChannelSet` object with one or more channels, or a single channel to stimulate.
            stim_design : A `StimDesign` object or a scalar current in microamperes. Use of a `StimDesign` is preferred.
                          A scalar current is the equivalent of a symmetric biphasic, negative-first pulse with a pulse width of `160`
                          microseconds, i.e., `StimDesign(160, -value, 160, value)`.
            burst_design: An optional `BurstDesign` object specifying the burst count and frequency. If unspecified, a single pulse will be delivered.
            lead_time_us: The lead time in microseconds before the stimulation starts.

        Constraints:
            - The minimum `lead_time_us` is `80`.
            - `lead_time_us` must be evenly divisible by `40`.

        For example:

        ```python
        import cl
        from cl import ChannelSet, StimDesign, BurstDesign

        # Predefine stimulation parameters
        channel_set_1 = ChannelSet(1, 2, 3)
        channel_set_2 = ChannelSet(8, 9, 10)
        stim_design   = StimDesign(160, -1.0, 160, 1.0)
        burst_design  = BurstDesign(5, 100)

        with cl.open() as neurons:
            stim_plan = neurons.create_stim_plan()

            # Queue a burst stimulation on channel_set_1
            stim_plan.stim(channel_set_1, stim_design, burst_design)

            # Queue a single stimulation on channel_set_2 with 0.5 µA current
            stim_plan.stim(channel_set_2, 0.5)  # Using scalar current

            # Optionally interrupt these channels when running this plan to ensure they are free
            stim_plan.channels_to_interrupt = channel_set_1 | channel_set_2

            # Finalise and run the stim plan
            # The above stimulations will run concurrently, due to no overlapping channels
            stim_plan.run()
        ```
        """
        if self._frozen:
            raise RuntimeError(FROZEN_ERROR_MSG)
        self._operations.append(Operation(
            op   = self._neurons._queue_stims,
            args = (channel_set, stim_design, burst_design, lead_time_us)
            ))

    def interrupt(self, channel_set: ChannelSet | int, /) -> None:
        """
        Enqueues the same operation as `Neurons.interrupt()` onto this `StimPlan`.

        Args:
            channel_set: A `ChannelSet` object with one or more channels, or a single channel to interrupt.

        @private -- hide from docs
        """
        warnings.warn("StimPlan interrupt() is deprecated. Use the channels_to_interrupt property instead.")
        if self._frozen:
            raise RuntimeError(FROZEN_ERROR_MSG)
        self._operations.append(Operation(
            op   = self._neurons._interrupt_queued_stims,
            args = (channel_set,)
            ))

    def interrupt_then_stim(
        self,
        channel_set:  ChannelSet  | int,
        stim_design:  StimDesign  | float,
        /,
        burst_design: BurstDesign | None   = None,
        lead_time_us: int                  = 80
        ) -> None:
        """
        Enqueues the same operation as `Neurons.interrupt_then_stim()` onto this `StimPlan`. This is equivalent to
        calling `interrupt()` followed by `stim()`, on the same set of channels.

        Constraints:
            - The minimum `lead_time_us` is `80`.
            - `lead_time_us` must be evenly divisible by `40`.

        Args:
            channel_set:  A `ChannelSet` object with one or more channels, or a single channel to stimulate.
            stim_design:  A `StimDesign` object or a floating point current in microamperes. As with `stim()`, use of a `StimDesign` is preferred.
            burst_design: A `BurstDesign` object specifying the burst count and frequency.
            lead_time_us: The lead time in microseconds before the stimulation starts.

        @private -- hide from docs
        """
        warnings.warn("StimPlan interrupt_then_stim() is deprecated. Use the channels_to_interrupt property and stim() instead.")
        if self._frozen:
            raise RuntimeError(FROZEN_ERROR_MSG)
        self._operations.append(Operation(
            op   = self._neurons._interrupt_queued_stims,
            args = (channel_set,)
            ))
        self._operations.append(Operation(
            op   = self._neurons._queue_stims,
            args = (channel_set, stim_design, burst_design, lead_time_us)
            ))

    def sync(self, channel_set: ChannelSet, /) -> None:
        """ Enqueues the same operation as `Neurons.sync()` onto this `StimPlan`. """
        if self._frozen:
            raise RuntimeError(FROZEN_ERROR_MSG)
        self._operations.append(Operation(
            op   = self._neurons._sync_channels,
            args = (channel_set,)
            ))

    @property
    def channels_to_interrupt(self) -> ChannelSet | None:
        """
        Allows specification of channels to interrupt when this plan is run.

        For example:

        ```python
        # Set multiple channels with a ChannelSet object
        stim_plan.channels_to_interrupt = ChannelSet(8, 9, 10)
        ```

        ```python
        # Set a single channel with an integer.
        stim_plan.channels_to_interrupt = 8
        ```

        ```python
        # Clear previously set channels.
        stim_plan.channels_to_interrupt = None
        ```
        """
        return self._channels_to_interrupt

    @channels_to_interrupt.setter
    def channels_to_interrupt(self, channel_set: ChannelSet | int | None, /):
        """ Setter for channels_to_interrupt. """
        if self._frozen:
            raise RuntimeError(FROZEN_ERROR_MSG)
        elif channel_set is None:
            self._channels_to_interrupt = None
            return
        elif isinstance(channel_set, ChannelSet):
            self._channels_to_interrupt = channel_set
            return
        elif isinstance(channel_set, int):
            self._channels_to_interrupt = ChannelSet(channel_set)
        else:
            raise TypeError(f"channel_set must be a ChannelSet object or an int, not {type(channel_set)}")

    def run(self, at_timestamp: int | None = None) -> None:
        """
        Execute the queued operations in the `StimPlan`. After this method is called,
        the `StimPlan` is frozen and cannot be modified.

        If `StimPlan.channels_to_interrupt` has been set, interrupt will be called
        on the specified channels before executing enqueued commands.

        Args:
            at_timestamp: Optionally, execute this `StimPlan` at a specified timestamp.
                          `StimPlan` will run immediately if timestamp is in the past.
        """
        now          = self._neurons.timestamp()
        timestamp    = at_timestamp if (at_timestamp is not None) and (at_timestamp >= now) else now
        self._frozen = True
        operations   = self._operations

        if self.channels_to_interrupt is not None:
            # Insert an interrupt command at the start of the operation list
            operations.insert(0, Operation(
                op   = self._neurons._interrupt_queued_stims,
                args = (self.channels_to_interrupt,)
                ))

        for operation in operations:
            operation.op(timestamp, *operation.args)
