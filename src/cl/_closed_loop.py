import time
from numpy import ndarray
from collections.abc import Callable, Generator
from dataclasses import dataclass
import math

from cl import DetectionResult
from cl.util import frames_to_approximate_seconds, ordinal

@dataclass
class LoopTick:
    """
    Contains spikes and frames detected during a loop iteration.

    The tick object itself is only valid for the duration of the
    loop iteration. If you need to keep a reference to instance
    variables (such as analysis) beyond the end of the loop body,
    copy them to another variable.
    """
    loop: "Loop"
    """ A reference to the running Loop. """

    iteration: int = 0
    """ Iteration count of this LoopTick within the Loop. """

    timestamp: int = -1
    """ The start timestamp of the tick period. """

    analysis: DetectionResult | None = None
    """ Contains the spikes and stims analysis of the frames read during the tick. """

    frames: ndarray | None = None
    """ The frames read during the tick period. """

class Loop:

    def __init__(
        self,
        neurons,
        ticks_per_second:        int,
        stop_after_seconds:      float | None = None,
        stop_after_ticks:        int   | None = None,
        ignore_jitter:           bool         = False,
        jitter_tolerance_frames: int          = 0,
        ):
        """
        Instantiate a new closed loop.

        See Neurons.loop() for docs.
        """
        from cl import Neurons

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
        self._jitter_tolerance_frames   = int(2**31 - 1 if ignore_jitter else jitter_tolerance_frames)

        # This is later updated to the timestamp of the first loop iteration.
        self.start_timestamp: int | str = "invalid timestamp"

    @property
    def duration_ticks(self):
        """ Return the current duration of the loop, in ticks """
        return self._tick.iteration + 1

    @property
    def duration_frames(self):
        """ Return the current duration of the loop, in frames """
        return (self._tick.iteration + 1) * self._frames_per_tick

    def run(self, loop_body_callback: Callable):
        """
        Run the closed loop, calling loop_body_callback for each tick.

        The callback is passed a LoopTick object containing detected
        spikes and other relevant information. The loop body can stop the loop
        by calling tick.loop.stop().
        """
        for tick in self:
            loop_body_callback(tick)

    def approximate_duration_seconds(self):
        """
        Return an approximate duration of the closed loop in seconds.
        """
        return frames_to_approximate_seconds(
            frames            = self.duration_frames,
            frames_per_second = self._neurons.get_frames_per_second()
        )

    def _handle_jitter_failure(
        self,
        start_ts:        int,
        next_ts:         int,
        frames_per_tick: int,
        now:             int,
        tick:            LoopTick
        ):
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

        raise TimeoutError(
            f"Loop fell behind by {frames_str(late_frames)} ({late_us} µs) "
            f"when entering the {ordinal(tick.iteration + 1)}\n"
            f"iteration. Jitter tolerance is currently set to "
            f"{frames_str(self._jitter_tolerance_frames)}. Ideally - optimise\n"
            f"the worst-case performance of your loop body. "
            f"You may also adjust the jitter\n"
            f"tolerance via jitter_tolerance_frames={late_frames}, "
            f"or ignore jitter entirely via\n"
            f"ignore_jitter=True."
        )

    def stop(self):
        """
        Stop the closed loop.

        Typically called via a tick.loop in a loop body
        in cases where a simple "break" is not convenient.
        """
        self._stop_after_ticks = self._tick.iteration

    def __iter__(self) -> Generator[LoopTick, None, None]:
        """ For each tick, yield a LoopTick object containing spikes and stims. """
        # Make local references
        neurons                 = self._neurons
        tick                    = self._tick
        timestamp               = neurons.timestamp
        frames_per_tick         = self._frames_per_tick
        frames_per_second       = neurons.get_frames_per_second()
        jitter_tolerance_frames = self._jitter_tolerance_frames
        read_frames             = neurons.read
        read_spikes             = neurons._read_spikes
        read_stims              = neurons._read_and_reset_stim_cache

        # Timing variables
        start_ts                = neurons.timestamp()
        next_ts                 = start_ts
        next_deadline_ts        = next_ts + frames_per_tick + jitter_tolerance_frames
        self.start_timestamp    = start_ts

        # We reset the stim cache so that the stims obtained through the loop
        # iterations only contain stims conducted in the previous iteration
        neurons._tick_stims.clear()

        # The mock Neurons API, does not track passage of actual time to allow
        # replay to occur at maximum speed. However, we want to simulate jitter
        # failure due to slow user operation between each tick. To do this, we
        # need awareness of actual passage of time.
        real_next_time = time.time()
        now            = timestamp()

        while tick.iteration < self._stop_after_ticks:
            # When considering jitter failure, we take the maximum number of
            # frames elapsed during a loop iteration between simulated and real
            real_now         = time.time()
            real_tick_secs   = real_now - real_next_time
            real_tick_frames = int(real_tick_secs) * frames_per_second
            real_next_time   = real_now
            now              = max(timestamp(), now + real_tick_frames)

            if now > next_deadline_ts:
                self._handle_jitter_failure(start_ts, next_ts, frames_per_tick, now, tick)

            # Read the next set of frames, spikes and stims
            tick.timestamp = next_ts
            tick.frames    = read_frames(frames_per_tick, next_ts)
            tick.analysis  = \
                DetectionResult(
                    timestamp = next_ts,
                    spikes    = read_spikes(frames_per_tick, next_ts),
                    stims     = read_stims()
                    )

            yield tick

            # Prepare for the next tick
            next_ts          += frames_per_tick
            next_deadline_ts += frames_per_tick
            tick.iteration   += 1

        return
