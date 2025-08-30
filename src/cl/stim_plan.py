from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence, Callable

from cl import ChannelSet, StimDesign, BurstDesign

@dataclass
class Operation:
    op  : Callable
    args: Sequence[Any]

class StimPlan:
    """
    Allows building and executing a sequence of stim operations. The StimPlan
    cannot be modified after it has been run.
    """

    frozen: bool
    """ When True, the StimPlan cannot be modified. """

    _operations: list[Operation]
    """ List of operations that make up this StimPlan. """

    def __init__(self, neurons) -> None:
        from cl import Neurons
        self._neurons: Neurons = neurons
        self.frozen            = False
        self._operations       = []

    def stim(
        self,
        channel_set:  ChannelSet  | int,
        stim_design:  StimDesign  | float,
        burst_design: BurstDesign | None   = None,
        lead_time_us: int                  = 80
        ) -> None:
        """ Performs the same function as Neurons.stim(). """
        if self.frozen:
            raise TypeError("Cannot modify a StimPlan after it has been used.")
        self._operations.append(Operation(
            op   = self._neurons._queue_stims,
            args = (channel_set, stim_design, burst_design, lead_time_us)
            ))

    def interrupt(self, channel_set: ChannelSet | int):
        """ Performs the same function as Neurons.interrupt(). """
        if self.frozen:
            raise TypeError("Cannot modify a StimPlan after it has been used.")
        self._operations.append(Operation(
            op   = self._neurons._interrupt_queued_stims,
            args = (channel_set,)
            ))

    def interrupt_then_stim(
        self,
        channel_set:  ChannelSet  | int,
        stim_design:  StimDesign  | float,
        burst_design: BurstDesign | None   = None,
        lead_time_us: int                  = 80
        ) -> None:
        """ Performs the same function as Neurons.interrupt_then_stim(). """
        if self.frozen:
            raise TypeError("Cannot modify a StimPlan after it has been used.")
        self._operations.append(Operation(
            op   = self._neurons._interrupt_queued_stims,
            args = (channel_set,)
            ))
        self._operations.append(Operation(
            op   = self._neurons._queue_stims,
            args = (channel_set, stim_design, burst_design, lead_time_us)
            ))

    def sync(self, channels: ChannelSet):
        """ Performs the same function as Neurons.sync(). """
        if self.frozen:
            raise TypeError("Cannot modify a StimPlan after it has been used.")
        self._operations.append(Operation(
            op   = self._neurons.sync,
            args = (channels,)
            ))

    def run(self):
        """
        Execute the queued operations in the StimPlan.

        After this method is called, the StimPlan is frozen and cannot be modified.
        """
        self.run_at_timestamp(timestamp=self._neurons.timestamp())

    def run_at_timestamp(self, timestamp: int):
        """
        Execute the queued operations in the StimPlan at a specified timestamp.

        After this method is called, the StimPlan is frozen and cannot be modified.
        """
        self.frozen = True
        for operation in self._operations:
            operation.op(timestamp, *operation.args)
