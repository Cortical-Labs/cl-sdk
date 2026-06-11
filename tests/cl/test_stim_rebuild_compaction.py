"""
Regression test for _compact_stim_history():

Without the fix, _stim_op_records and _interrupt_records grow to O(N) where
N is the total number of interrupt_then_stim calls across the session. Each
call to _rebuild_stim_queue() then scans all N records, producing O(N^2)
behaviour that degrades performance over long runs.

With the fix, records are pruned to the ring-buffer retention window,
bounding rebuild cost to constant time, independent of total session duration.

Uses CL_SDK_ACCELERATED_TIME=1 so no real-time delay is needed.
"""
import os

os.environ["CL_SDK_VISUALISATION"]   = "0"
os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

import cl
import pytest
from cl._sim._data_buffer import DEFAULT_BUFFER_SIZE_FRAMES

_STIM_CHANNELS  = (0, 1)    # multi-channel → forces rebuild on every interrupt
_STIM_DURATION  = 160       # µs
_STIM_CURRENT   = 1         # µA

_TICKS_PER_SECOND = 100     # Hz — one interrupt_then_stim call per tick
_STOP_AFTER_SECS  = 20      # simulated seconds ≈ 4× the ring-buffer window
_WINDOW_FRAMES    = DEFAULT_BUFFER_SIZE_FRAMES
_TOLERANCE        = int(_WINDOW_FRAMES * 0.1)

def test_stim_rebuild_history_is_bounded() -> None:
    """
    After simulating well past the ring-buffer window, _stim_op_records and
    _interrupt_records must be bounded to approximately window_seconds × call_rate,
    not growing linearly to the total number of calls.
    """
    total_calls = _TICKS_PER_SECOND * _STOP_AFTER_SECS  # 2 000 without compaction

    with cl.open() as neurons:
        fps         = neurons.get_frames_per_second()
        channel_set = cl.ChannelSet(*_STIM_CHANNELS)
        stim_design = cl.StimDesign(_STIM_DURATION, _STIM_CURRENT)

        for _ in neurons.loop(ticks_per_second = _TICKS_PER_SECOND, stop_after_seconds = _STOP_AFTER_SECS):
            neurons.interrupt_then_stim(channel_set, stim_design)

        n_ops  = len(neurons._stim_op_records)
        n_ints = len(neurons._interrupt_records)

    # Records should plateau near window_s × ticks_per_second.
    window_s    = _WINDOW_FRAMES / fps
    upper_bound = int(window_s * _TICKS_PER_SECOND) + _TOLERANCE

    assert n_ops < total_calls, (
        f"_stim_op_records grew to {n_ops} — compaction is not running (expected < {total_calls})"
    )
    assert n_ints < total_calls, (
        f"_interrupt_records grew to {n_ints} — compaction is not running (expected < {total_calls})"
    )
    assert n_ops <= upper_bound, (
        f"_stim_op_records={n_ops} exceeded upper bound of {upper_bound}"
    )
    assert n_ints <= upper_bound, (
        f"_interrupt_records={n_ints} exceeded upper bound of {upper_bound}"
    )
