from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

from . import DetectionResult
from .util import deprecated, frames_to_approximate_seconds

if TYPE_CHECKING:
    import numpy as np
    from numpy import ndarray

    from . import Neurons

_MIN_JITTER_TOLERANCE: int = 20
""" (Simulator only) Allow a small amount of jitter as desktop OS will likely have higher latency than CL1. """

class LoopTick:
    """
    Contains spikes, stims and frames collected during a loop iteration.
    (Relates to `Neurons.loop()` and `Loop`.)

    The tick object itself is only valid for the duration of the
    loop iteration. If you need to keep a reference to instance
    variables (such as analysis) beyond the end of the loop body,
    copy them to another variable.

    This is accessible via `Neurons.loop()` and yielded by the `Loop` iterator.
    Do not create instances of `LoopTick` directly.
    """
    loop: Loop
    """ A reference to the running Loop. """

    iteration: int
    """ Iteration count of this LoopTick within the Loop. """

    iteration_timestamp: int
    """ Timestamp of the loop body. This is equivalent to `DetectionResult.stop_timestamp`. """

    iteration_next_timestamp: int
    """
    Timestamp of the **next** loop body.
    This is equivalent to `LoopTick.iteration_timestamp + len(LoopTick.frames)`.
    """

    _timestamp: int
    """ (Simulator only) The start timestamp of the tick period. """

    @property
    @deprecated("tick.analysis.start_timestamp")
    def timestamp(self) -> int:
        """
        The start timestamp of the tick period.
        @private -- hide from docs
        """
        return self._timestamp

    analysis: DetectionResult
    """ Contains the spikes and stims analysis of the frames read during the tick. """

    frames: ndarray[tuple[int, int], np.dtype[np.int16]]
    """ The frames read during the tick period, with shape (duration_frames, channel_count). """

    def __init__(self, loop: Loop) -> None:
        """
        @private -- hide from docs
        """
        self.loop      = loop
        self.iteration = 0

class Loop:
    """
    Iterator that yields a `LoopTick`. (Relates to `LoopTick` and `Neurons.loop()`.)

    This is made available through the `Neurons.loop()` interface.
    Do not create instances of `Loop` directly.
    """
    def __init__(
        self,
        neurons,
        ticks_per_second:        float,
        stop_after_seconds:      float | None = None,
        stop_after_ticks:        int   | None = None,
        ignore_jitter:           bool         = False,
        jitter_tolerance_frames: int          = 0,
        ) -> None:
        """
        @private -- hide from docs
        """
        if ticks_per_second <= 0:
            raise ValueError("ticks_per_second must be greater than zero")

        # Determine the loop end point
        if stop_after_seconds is not None:
            if stop_after_seconds <= 0:
                raise ValueError("stop_after_seconds must be greater than zero")
            if stop_after_ticks is not None:
                raise ValueError("Cannot set both stop_after_seconds and stop_after_ticks")
            self._stop_after_ticks = math.ceil(stop_after_seconds * ticks_per_second)
        elif stop_after_ticks is not None:
            if stop_after_ticks <= 0:
                raise ValueError("stop_after_ticks must be greater than zero")
            self._stop_after_ticks = stop_after_ticks
        else:
            # In practical terms, this is the same as running indefinitely
            # and removes a branch from the loop body.
            self._stop_after_ticks = 2**63 - 1

        self._neurons: Neurons          = neurons
        self._tick                      = LoopTick(self)
        self._ticks_per_second          = ticks_per_second
        self._frames_per_tick           = int(neurons.get_frames_per_second() // ticks_per_second)
        self._jitter_tolerance_frames   = max(int(2**31 - 1 if ignore_jitter else jitter_tolerance_frames), _MIN_JITTER_TOLERANCE)

        # This is later updated to the timestamp of the first loop iteration.
        self._start_timestamp: int | str = "invalid timestamp"

    @property
    def start_timestamp(self) -> int:
        """ Return the timestamp of the first loop iteration. """
        return int(self._start_timestamp)

    @property
    def duration_ticks(self) -> int:
        """ Return the current duration of the loop, in ticks. """
        return self._tick.iteration + 1

    @property
    def duration_frames(self) -> int:
        """ Return the current duration of the loop, in frames. """
        return (self._tick.iteration + 1) * self._frames_per_tick

    @property
    def frames_per_tick(self) -> int:
        """ Return the number of frames in each tick """
        return self._frames_per_tick

    def approximate_duration_seconds(self) -> float:
        """
        Return an approximate duration of the closed loop in seconds.
        """
        return frames_to_approximate_seconds(
            frames            = self.duration_frames,
            frames_per_second = self._neurons.get_frames_per_second()
            )

    def __iter__(self) -> Generator[LoopTick]:
        """
        For each tick, yield a `LoopTick` object containing spikes, stims and frames collected during
        the previous iteration.
        @public
        """
        # Make local references
        neurons                 = self._neurons
        tick                    = self._tick
        timestamp               = neurons.timestamp
        frames_per_tick         = self._frames_per_tick
        frames_per_second       = neurons.get_frames_per_second()
        frames_per_ns           = frames_per_second / 1_000_000_000
        jitter_tolerance_frames = self._jitter_tolerance_frames
        read_frames             = neurons.read
        read_spikes             = neurons._read_spikes
        read_stims              = neurons._read_and_reset_stim_cache

        use_accelerated_time    = neurons._use_accelerated_time

        # Timing variables
        start_ts                = timestamp()
        next_ts                 = start_ts
        next_deadline_ts        = next_ts + frames_per_tick + jitter_tolerance_frames
        self._start_timestamp   = start_ts

        # Mark that we're in a loop and set initial deadline
        neurons._in_loop = True
        neurons._loop_deadline_ts = next_deadline_ts
        # Base timestamp for stim scheduling is the start of the NEXT tick
        # (i.e., the end of the current tick), since user code runs during the tick
        neurons._loop_tick_timestamp = start_ts + frames_per_tick

        # In accelerated mode, request the first batch of data from the producer
        # Request data for first tick only to avoid running too far ahead
        if use_accelerated_time and neurons._shared_buffer is not None:
            neurons._shared_buffer.requested_timestamp = start_ts + frames_per_tick

        neurons._last_stim_read_ts = start_ts

        # Wall clock timing for accelerated mode jitter detection
        # Will be reset after first iteration to exclude initialization overhead
        wall_start = time.perf_counter_ns()

        print("Warning: Jitter detection is currently not supported in cl-sdk. This may lead to unexpected loop timing behaviour if your loop body takes a long time to execute.", file=sys.stderr)

        while tick.iteration < self._stop_after_ticks:
            now = timestamp()

            # For jitter checking, we need different approaches for different modes:
            # - Real-time mode: use producer's timestamp only, since it runs in a separate
            #   process and keeps real time independently. Mixing wall clock would cause
            #   false positives due to IPC latency.
            # - Accelerated mode: use wall clock to detect slow Python loop bodies (e.g.,
            #   time.sleep()). Add 10% of elapsed simulated time as extra tolerance to allow
            #   for normal processing overhead that accumulates over long runs.
            if use_accelerated_time:
                # Skip wall clock jitter check for first iteration (allows initialization overhead)
                if tick.iteration == 0:
                    effective_deadline = next_deadline_ts
                else:
                    wall_elapsed_ns = time.perf_counter_ns() - wall_start
                    wall_elapsed_frames = int(wall_elapsed_ns * frames_per_ns)
                    now = max(now, start_ts + wall_elapsed_frames)
                    # Add proportional tolerance: allow wall time to be up to 2x simulated time.
                    # This allows for normal Python processing overhead that accumulates over
                    # long runs (e.g., assertions, logging), while still catching deliberate
                    # sleeps that would make wall time >>> simulated time.
                    simulated_elapsed = timestamp() - start_ts
                    effective_deadline = next_deadline_ts + simulated_elapsed  # 100% tolerance
            else:
                effective_deadline = next_deadline_ts

            #
            # Handle jitter and jitter recovery states
            #
            loop_is_late = now > effective_deadline
            if not self._jitter_recovery_enabled:
                # Normal loop operation without jitter recovery, which may lead to
                # jitter failure if late
                if loop_is_late:
                    self._handle_jitter_failure(start_ts, next_ts, frames_per_tick, now, tick)

            # Jitter recovery is enabled, handle all situations below
            elif not loop_is_late:
                # Loop has caught up timing-wise, but only exit recovery if we've reached
                # the target iteration (to prevent early exit due to timing variations)
                if self._jitter_recovery_target_iteration == 0 or tick.iteration >= self._jitter_recovery_target_iteration:
                    self._jitter_recovery_reset()

            elif self._jitter_recovery_timeout_timestamp == 0:
                # First iteration with jitter recovery enabled
                # Calculate which iteration we should resume at based on how far behind we are
                # This makes recovery deterministic regardless of producer timing variations
                self._jitter_recovery_target_iteration = (now - start_ts) // frames_per_tick

                # Calculate and set time limit for loop to catch up
                self._jitter_recovery_timeout_timestamp = \
                        int(now + (self._jitter_recovery_timeout_sec * self._neurons._frames_per_second))

            elif now >= self._jitter_recovery_timeout_timestamp:
                # Jitter recovery is running but we have exceeded timeout before recovering
                raise TimeoutError(
                    "Loop fell too far behind and jitter recovery "
                    f"could not complete within {self._jitter_recovery_timeout_sec:.3f} seconds."
                    )

            #
            # Read the next set of frames, spikes and stims
            #
            tick._timestamp               = next_ts
            tick.iteration_timestamp      = next_ts + frames_per_tick
            tick.iteration_next_timestamp = next_ts + (2 * frames_per_tick)

            # Update the jitter deadline and tick timestamp for this iteration
            neurons._loop_deadline_ts = next_deadline_ts
            # Base timestamp for stim scheduling is the start of the NEXT tick
            # (i.e., the end of the current tick), since user code runs during the tick
            neurons._loop_tick_timestamp = next_ts + frames_per_tick

            # In accelerated mode, wait for producer to write data for this tick
            # (data was requested at end of previous iteration)
            if use_accelerated_time and neurons._shared_buffer is not None:
                target_ts = next_ts + frames_per_tick
                for _ in range(1000):  # Max 100ms wait
                    current_write = neurons._shared_buffer.write_timestamp
                    if current_write >= target_ts:
                        # write_timestamp is updated after frames but before stims
                        # Sleep briefly to ensure stims are also written
                        time.sleep(0.0001)  # 100μs to allow stim writes to complete
                        break
                    time.sleep(0.0001)  # 100μs

            tick.frames    = read_frames(frames_per_tick, next_ts)
            tick.analysis  = \
                DetectionResult(
                    start_timestamp = next_ts,
                    stop_timestamp  = next_ts + frames_per_tick,
                    spikes          = read_spikes(frames_per_tick, next_ts),
                    stims           = read_stims(),
                )

            #
            # Handle loop tick behaviour
            #
            if self._jitter_recovery_enabled:
                # In recovery mode - call callback or skip to catch up
                if self._jitter_recovery_callback is not None:
                    self._jitter_recovery_callback(tick)
                # else: skip tick to allow loop to catch up

            else:
                # Normal operation - yield tick
                yield tick

            # After user code runs, request next tick's data from producer
            # This triggers producer to process commands (including queued stims)
            # Prepare for the next tick (increment counters first)
            next_ts += frames_per_tick
            next_deadline_ts += frames_per_tick
            tick.iteration += 1

            if use_accelerated_time and neurons._shared_buffer is not None:
                # Request data for next tick only to keep producer one tick ahead
                neurons._shared_buffer.requested_timestamp = next_ts + frames_per_tick
        # Loop finished - pause producer in accelerated mode to prevent it racing ahead
        if use_accelerated_time and neurons._shared_buffer is not None:
            neurons._shared_buffer.requested_timestamp = 0

        # Clear loop flag and deadline
        neurons._in_loop             = False
        neurons._loop_tick_timestamp = None
        neurons._loop_deadline_ts    = None

        return

    def run(self, loop_body_callback: Callable[[LoopTick], None]) -> None:
        """
        Run the closed loop, calling `loop_body_callback` for each tick.

        The callback is passed a `LoopTick` object containing detected
        spikes and other relevant information. The loop body can stop the loop
        by calling `LoopTick.loop.stop()`.

        For example:

        ```python
        TICKS_PER_SECOND = 2

        def loop_body_callback(tick: LoopTick):
            # Do something ...
            tick.loop.stop()

        with cl.open() as neurons:
            loop = neurons.loop(TICKS_PER_SECOND)
            loop.run(loop_body_callback)
        ```
        """
        for tick in self:
            loop_body_callback(tick)

    def stop(self) -> None:
        """
        Stop the `Loop` iterations.

        Typically called via `LoopTick.loop` in a loop body in cases where a simple
        `break` is not convenient, such as when using the `Loop.run()` syntax.
        """
        self._stop_after_ticks = self._tick.iteration

    def recover_from_jitter(
        self,
        handle_recovery_tick: Callable[[LoopTick], None] | None = None,
        timeout_seconds:      float                             = 5.0
        ) -> None:
        """
        Call to enable jitter recovery for potentially long running operations within
        a `Loop` iteration that otherwise might trigger a `TimeoutError`.

        This effectively skips execution of code in the loop body until iterations
        catches up to the expected iteration in realtime. Data in the skipped iterations
        can be accessed through the `handle_recovery_tick` callback.

        Args:
            handle_recovery_tick: Optional callback function that accepts a `LoopTick` as the only argument.
            timeout_seconds:      Number of seconds to allow for the recovery, defaults to `5` seconds if `None`.

        Constraints:
        - Users need to be careful that the `handle_recovery_tick` callback does not take too long
          otherwise the loop will never catch up.
        - `TimeoutError` will be raised if loop has not caught up within `timeout_seconds`.

        In the following example, the loop from iteration `2` will keep reading data but
        not yield any ticks until the loop catches up in iteration `7`.

        ```python
        TICKS_PER_SECOND = 100
        STOP_AFTER_TICKS = 10

        def handle_recovery_tick(tick: LoopTick):
            # Optionally do something with tick data during recovery
            ...

        with cl.open() as neurons:
            for tick in neurons.loop(TICKS_PER_SECOND, stop_after_ticks=STOP_AFTER_TICKS):
                print(f"{tick.iteration}=")
                if (tick.iteration == 1):
                    tick.loop.recover_from_jitter(handle_recovery_ticks)
                    time.sleep(0.05)

        # Expected output:
        # tick.iteration=0
        # tick.iteration=1
        # tick.iteration=7
        # tick.iteration=8
        # tick.iteration=9
        ```
        """
        if handle_recovery_tick is not None and not isinstance(handle_recovery_tick, Callable):
            raise TypeError(f"Callback {handle_recovery_tick.__repr__} passed to jitter_recovery_callback is not callable")
        if not isinstance(timeout_seconds, float) or isinstance(timeout_seconds, int):
            raise TypeError("timeout_seconds argument must be floating point number.")
        self._jitter_recovery_callback    = handle_recovery_tick
        self._jitter_recovery_timeout_sec = float(timeout_seconds)
        self._jitter_recovery_enabled     = True

    #
    # (Private) Jitter handling and recovery
    #

    _jitter_recovery_enabled: bool = False
    """ (Simulator only) Indicates whether the loop is in jitter recovery mode. """

    _jitter_recovery_callback: Callable[[LoopTick], None] | None = None
    """ (Simulator only) Function that gets called when in recovery mode. """

    _jitter_recovery_timeout_sec: float
    """ (Simulator only) Number of seconds to allow for the recovery. """

    _jitter_recovery_timeout_timestamp: int = 0
    """ (Simulator only) Maximum allowable timestamp for loop to catch up during jitter recovery. """

    _jitter_recovery_target_iteration: int = 0
    """ (Simulator only) Target iteration at which to resume normal operation after recovery. """

    def _jitter_recovery_reset(self) -> None:
        """ (Simulator only) Resets the jitter recovery states. """
        self._jitter_recovery_enabled           = False
        self._jitter_recovery_timeout_timestamp = 0
        self._jitter_recovery_target_iteration  = 0

    def _handle_jitter_failure(
        self,
        start_ts:        int,
        next_ts:         int,
        frames_per_tick: int,
        now:             int,
        tick:            LoopTick
        ) -> None:
        """
        Handles higher jitter scenarios by raising a TimeoutError.

        Args:
            start_ts: Start tick timestamp.
            next_ts: Next tick timestamp.
            frames_per_tick: Number of expected frames per tick.
            now: Current timestamp.
            tick: Current tick object.
        """
        late_frames = now - (next_ts + frames_per_tick)
        late_us     = late_frames * self._neurons.get_frame_duration_us()

        def frames_str(frame_count):
            return f"{frame_count} {'frame' if frame_count == 1 else 'frames'}"

        # TODO: raise TimeoutError when we can better approximate jitter on user's systems
        # raise TimeoutError(
        #     f"Loop fell behind by {frames_str(late_frames)} ({late_us} µs) "
        #     f"when entering the {ordinal(tick.iteration + 1)}\n"
        #     f"iteration. Jitter tolerance is currently set to "
        #     f"{frames_str(self._jitter_tolerance_frames)}. Ideally - optimise\n"
        #     f"the worst-case performance of your loop body. "
        #     f"You may also use recover_from_jitter() or adjust the jitter\n"
        #     f"tolerance via jitter_tolerance_frames={late_frames}."
        #     )
