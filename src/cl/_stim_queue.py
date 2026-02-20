"""
Optimized stim queue data structure for efficient interrupt handling.

This module provides a channel-indexed stim queue that supports:
- O(log n) insertion per stim
- O(k log k) interrupt for k stims on interrupted channels (vs O(n) drain-rebuild)
- O(k) iteration for stims in a timestamp range

The key insight is that interrupts target specific channels, so by indexing
stims by channel, we can efficiently filter without touching unrelated channels.
"""
from __future__ import annotations

import bisect
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

@dataclass(slots=True)
class StimEntry[T]:
    """A stim entry with timestamp and payload."""
    timestamp: int
    payload  : T

    def __lt__(self, other: StimEntry) -> bool:
        return self.timestamp < other.timestamp

class ChannelStimQueue[T]:
    """
    An efficient stim queue indexed by channel for fast interrupt operations.

    Instead of a single PriorityQueue that must be drained and rebuilt on
    every interrupt, this uses per-channel sorted lists. This allows:

    - Interrupts to only touch the affected channels: O(k) where k = stims on channel
    - Insertions remain O(log n) per stim via bisect.insort
    - Iteration over timestamp ranges is O(m) where m = stims in range

    The trade-off is slightly more memory for the channel dict, but this is
    negligible compared to the O(n²) worst-case of repeated drain-rebuild.
    """

    __slots__ = ('_channel_stims', '_total_count')

    def __init__(self) -> None:
        # Dict mapping channel -> sorted list of StimEntry (sorted by timestamp)
        self._channel_stims: dict[int, list[StimEntry[T]]] = {}
        self._total_count  : int                           = 0

    def put(self, timestamp: int, channel: int, payload: T) -> None:
        """
        Insert a stim entry for the given channel at the specified timestamp.

        Time complexity: O(log k) where k = number of stims on this channel.
        """
        entry = StimEntry(timestamp, payload)
        if channel not in self._channel_stims:
            self._channel_stims[channel] = [entry]
        else:
            bisect.insort(self._channel_stims[channel], entry)
        self._total_count += 1

    def interrupt_channels(
        self,
        channels      : Iterable[int],
        from_timestamp: int,
    ) -> list[tuple[int, int, T]]:
        """
        Remove and return stims on the specified channels at or after from_timestamp.

        Stims before from_timestamp are kept. Returns list of (timestamp, channel, payload)
        for removed stims.

        Time complexity: O(c * log k) where c = number of channels, k = stims per channel.
        """
        removed: list[tuple[int, int, T]] = []

        for channel in channels:
            if channel not in self._channel_stims:
                continue

            stim_list = self._channel_stims[channel]
            if not stim_list:
                continue

            # Binary search for first stim >= from_timestamp
            cutoff_idx = bisect.bisect_left(stim_list, StimEntry(from_timestamp, None))

            # Stims at and after cutoff are removed
            if cutoff_idx < len(stim_list):
                for entry in stim_list[cutoff_idx:]:
                    removed.append((entry.timestamp, channel, entry.payload))
                    self._total_count -= 1
                # Keep only stims before the cutoff
                self._channel_stims[channel] = stim_list[:cutoff_idx]

        return removed

    def pop_until(self, to_timestamp: int) -> list[tuple[int, int, T]]:
        """
        Remove and return all stims with timestamp < to_timestamp, sorted by timestamp.

        Time complexity: O(m log m) where m = number of stims popped.
        """
        result: list[tuple[int, int, T]] = []

        for channel, stim_list in self._channel_stims.items():
            if not stim_list:
                continue

            # Binary search for first stim >= to_timestamp
            cutoff_idx = bisect.bisect_left(stim_list, StimEntry(to_timestamp, None))

            if cutoff_idx > 0:
                # Pop stims before cutoff
                for entry in stim_list[:cutoff_idx]:
                    result.append((entry.timestamp, channel, entry.payload))
                    self._total_count -= 1
                # Keep only stims from cutoff onwards
                self._channel_stims[channel] = stim_list[cutoff_idx:]

        # Sort by timestamp for correct ordering
        result.sort(key=operator.itemgetter(0))
        return result

    def peek_min_timestamp(self) -> int | None:
        """
        Return the minimum timestamp across all channels, or None if empty.

        Time complexity: O(c) where c = number of channels with stims.
        """
        min_ts: int | None = None
        for stim_list in self._channel_stims.values():
            if stim_list:
                ts = stim_list[0].timestamp
                if min_ts is None or ts < min_ts:
                    min_ts = ts
        return min_ts

    def get_stims_before(self, to_timestamp: int) -> list[tuple[int, int, T]]:
        """
        Return (but don't remove) all stims with timestamp < to_timestamp.

        Time complexity: O(m) where m = number of matching stims.
        """
        result: list[tuple[int, int, T]] = []

        for channel, stim_list in self._channel_stims.items():
            if not stim_list:
                continue

            cutoff_idx = bisect.bisect_left(stim_list, StimEntry(to_timestamp, None))
            result.extend((entry.timestamp, channel, entry.payload) for entry in stim_list[:cutoff_idx])

        result.sort(key=operator.itemgetter(0))
        return result

    def iter_channel(self, channel: int) -> Iterator[tuple[int, T]]:
        """Iterate over stims for a specific channel in timestamp order."""
        if channel in self._channel_stims:
            for entry in self._channel_stims[channel]:
                yield entry.timestamp, entry.payload

    def get_last_timestamp_for_channel(self, channel: int) -> int | None:
        """Get the last (maximum) timestamp for a specific channel."""
        if self._channel_stims.get(channel):
            return self._channel_stims[channel][-1].timestamp
        return None

    def get_last_entry_before(self, channel: int, before_timestamp: int) -> tuple[int, T] | None:
        """Get the last stim entry (timestamp, payload) before the given timestamp for a channel."""
        if channel not in self._channel_stims or not self._channel_stims[channel]:
            return None

        stim_list = self._channel_stims[channel]
        # Binary search for first stim >= before_timestamp
        cutoff_idx = bisect.bisect_left(stim_list, StimEntry(before_timestamp, None))

        # If there are stims before the cutoff, return the last one
        if cutoff_idx > 0:
            last_entry = stim_list[cutoff_idx - 1]
            return (last_entry.timestamp, last_entry.payload)
        return None

    def clear_channel(self, channel: int) -> int:
        """Clear all stims for a channel. Returns count of removed stims."""
        if channel in self._channel_stims:
            count = len(self._channel_stims[channel])
            self._total_count -= count
            if self._channel_stims[channel]:
                self._channel_stims[channel].clear()
            else:
                self._channel_stims[channel] = []
            return count
        return 0

    def qsize(self) -> int:
        """Return total number of stims across all channels."""
        return self._total_count

    def empty(self) -> bool:
        """Return True if queue is empty."""
        return self._total_count == 0

    def __len__(self) -> int:
        return self._total_count

    def __bool__(self) -> bool:
        return self._total_count > 0
