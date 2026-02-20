from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from queue import PriorityQueue
from random import randint
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from dotenv import load_dotenv
from numpy import ndarray

from . import BurstDesign, ChannelSet, Loop, Spike, Stim, StimDesign, StimPlan, _logger
from ._data_producer import DataProducer
from ._stim_queue import ChannelStimQueue
from .data_stream import DataStream
from .recording import Recording
from .util import RecordingView

if TYPE_CHECKING:
    from collections.abc import Callable

class Neurons:
    """
    The `Neurons` class provides the main interface with the CL1 hardware. This should
    always be accessed via the `cl.open()` context manager and should **not** be
    used in isolation. This functionality includes:
    - Perform `stim()` and `create_stim_plan()`,
    - Access to device information such as `timestamp()`,
    - Create a tightly timed `loop()` to detect spikes and execute code,
    - `record()` data to file,
    - `read()` data from the MEA,
    - and more.

    If you are using the Simulator:
    - This simulates the behaviour of the CL API by either generating
      random data (default) or replaying data from a H5 recording (replay_file). The
      recording to use is controlled by the `CL_SDK_REPLAY_PATH` environment
      variable, which can be set by a `.env` file.
    - This operates on wall-clock time by default to maintain parity with the CL1 device. For
      advanced users, it is possible to switch to accelerated mode by setting the environment
      variable `CL_SDK_ACCELERATED_TIME=1`.
    - The starting position of the replay recording will be randomised every time `cl.open()` is called.
      This can be overriden by setting `CL_SDK_REPLAY_START_OFFSET`, where a value of `0` indicates
      the first frame of the recording.
    """
    def __init__(self):
        _logger.debug("using Cortical Labs Mock API")

        self._read_lock = Lock()

        self._loop_deadline_ts    = None
        self._loop_tick_timestamp = None
        self._websocket_server    = None

    def __enter__(self):
        """ (Simulator only) Open a H5 recording and set required attributes. """
        from . import _CL_SDK_REPLAY_PATH

        def load_replay_file() -> RecordingView:
            from . import _CL_SDK_REPLAY_PATH
            assert _CL_SDK_REPLAY_PATH is not None and Path(_CL_SDK_REPLAY_PATH).exists(), \
                f"Recording not found: {_CL_SDK_REPLAY_PATH}"
            _logger.debug(f"simulating from recording: {_CL_SDK_REPLAY_PATH}")
            return RecordingView(_CL_SDK_REPLAY_PATH)

        self._replay_file       = load_replay_file()
        attrs                   = self._replay_file.attributes

        self._start_timestamp   = int(attrs["start_timestamp"])
        self._read_timestamp    = self._start_timestamp
        self._channel_count     = int(attrs["channel_count"])
        self._frames_per_second = int(attrs["frames_per_second"])
        self._duration_frames   = int(attrs["duration_frames"])
        self._frame_duration_us = int(1_000_000 / self._frames_per_second)
        self._elapsed_frames    = 0

        self._recordings                  = []
        self._tick_stims                  = []
        self._in_loop                     = False
        self._stim_queue                  = ChannelStimQueue()
        self._stim_channel_available_from = np.full((self._channel_count,), fill_value=self._start_timestamp, dtype=int)

        # Per-instance mutable state (NOT class-level to avoid persistence across instances)
        self._timed_ops                   = PriorityQueue()
        self._data_streams                = {}

        self._start_walltime_ns           = time.perf_counter_ns()
        self._prev_walltime_ns            = self._start_walltime_ns

        load_dotenv(".env")
        self._use_accelerated_time        = os.getenv("CL_SDK_ACCELERATED_TIME", "0") == "1"
        if not self._use_accelerated_time:
            _logger.debug("time policy: wall clock time")
        else:
            _logger.debug("time policy: accelerated")

        self._replay_start_offset         = int(os.getenv("CL_SDK_REPLAY_START_OFFSET", "-1"))
        if self._replay_start_offset < 0:
            self._replay_start_offset     = randint(0, self._duration_frames)

        # Prepare the data producer (but don't start yet - lazy initialization)
        assert _CL_SDK_REPLAY_PATH is not None, "Replay path must be set"
        self._data_producer = DataProducer(
            replay_file_path    = _CL_SDK_REPLAY_PATH,
            start_timestamp     = self._start_timestamp,
            replay_start_offset = self._replay_start_offset,
            channel_count       = self._channel_count,
            frames_per_second   = self._frames_per_second,
            duration_frames     = self._duration_frames,
            accelerated_time    = self._use_accelerated_time,
        )
        self._producer_started = False
        self._producer_lock = Lock()  # Thread-safe producer startup
        self._shared_buffer = None

        # Track timestamps for spike/stim reads (to avoid re-reading same data)
        self._last_spike_read_ts = self._start_timestamp
        self._last_stim_read_ts  = self._start_timestamp

        # Heartbeat thread for debugger detection
        self._heartbeat_thread = None
        self._heartbeat_stop_event = None

        return self

    def _ensure_producer_started(self) -> None:
        """Start the producer subprocess if not already started (thread-safe)."""
        # Quick check without lock (common case)
        if self._producer_started:
            return

        # Acquire lock for thread-safe startup
        with self._producer_lock:
            # Double-check after acquiring lock
            if self._producer_started:
                return

            assert self._data_producer is not None

            # Reset wall-clock reference when producer actually starts
            # This ensures _sleep_until() calculations are accurate
            self._start_walltime_ns = time.perf_counter_ns()
            self._prev_walltime_ns = self._start_walltime_ns

            # Start heartbeat thread for debugger detection
            self._start_heartbeat_thread()

            self._data_producer.start()
            self._shared_buffer = self._data_producer.buffer
            self._producer_started = True

            # In accelerated mode, wait for the first batch to be produced
            # so that timestamp() returns a consistent value
            if self._use_accelerated_time:
                assert self._shared_buffer is not None
                # Wait for producer to produce at least one batch
                for _ in range(1000):  # 100ms max wait
                    if self._shared_buffer.write_timestamp > self._start_timestamp:
                        break
                    time.sleep(0.0001)

    def _start_websocket_server(self, port: int = 1025, host: str = "127.0.0.1") -> None:
        """
        Start WebSocket server subprocess for visualization.

        Args:
            port: Port for WebSocket server
            host: Host address for WebSocket server
        """
        if self._websocket_server is not None:
            _logger.info("WebSocket server already running")
            return

        # Ensure producer is running first (WebSocket reads from shared buffer)
        self._ensure_producer_started()

        # Lazy import so we don't need to load websocket dependencies if not using websocket visualization
        from .visualisation._websocket_subprocess import WebSocketProcessManager

        # Get the unique shared memory prefix from the producer's buffer
        buffer_name_prefix = self._data_producer.buffer.get_name_prefix() if self._data_producer and self._data_producer.buffer else ""

        self._websocket_server = WebSocketProcessManager(
            buffer_name       = buffer_name_prefix,
            frames_per_second = self.get_frames_per_second(),
            channel_count     = self.get_channel_count(),
            port              = port,
            host              = host,
            app_html          = Neurons._app_html,
        )
        self._websocket_server.start()
        _logger.info(f"WebSocket subprocess started on {host}:{port}")
        if self._websocket_server.web_url:
            print(f"Data visualiser: {self._websocket_server.web_url}", flush=True)
        if self._websocket_server.app_url:
            print(f"Application visualiser: {self._websocket_server.app_url}", flush=True)

    def _stop_websocket_server(self) -> None:
        """Stop WebSocket server subprocess if running."""
        if self._websocket_server is not None:
            self._websocket_server.stop()
            self._websocket_server = None
            _logger.info("WebSocket subprocess stopped")

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
        return False

    def __del__(self):
        self.close()

    def stim(
        self,
        channel_set:  ChannelSet  | int,
        stim_design:  StimDesign  | float,
        /,
        burst_design: BurstDesign | None   = None,
        lead_time_us: int                  = 80
        ) -> None:
        """
        Stimulate one or more channels.

        Args:
            channels    : A `ChannelSet` object with one or more channels, or a single channel to stimulate.
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

        with cl.open() as neurons:

            # Deliver a single biphasic stim with current of 1.0 uA, pulse width
            # of 160 us and negative leading edge on channels 8, 9, 10
            channel_set = ChannelSet(8, 9, 10)
            stim_design = StimDesign(160, -1.0, 160, 1.0)
            neurons.stim(channel_set, stim_design)

            # Deliver the same stim as a burst of 10 at 40 Hz
            burst_design = BurstDesign(10, 40)
            neurons.stim(channel_set, stim_design, burst_design)
        ```
        """

        self._queue_stims(
            from_timestamp = self.timestamp(),
            channel_set    = channel_set,
            stim_design    = stim_design,
            burst_design   = burst_design,
            lead_time_us   = lead_time_us
            )

    def interrupt(self, channel_set: ChannelSet | int, /) -> None:
        """
        Interrupt existing and clear any pending stimulation for the specified channels.

        Args:
            channels: A `ChannelSet` object with one or more channels, or a single channel to interrupt.
        """
        self._interrupt_queued_stims(
            from_timestamp = self.timestamp(),
            channel_set    = channel_set
            )

    def interrupt_then_stim(
        self,
        channel_set:  ChannelSet  | int,
        stim_design:  StimDesign  | float,
        /,
        burst_design: BurstDesign | None    = None,
        lead_time_us: int                   = 80
        ) -> None:
        """
        Interrupt existing and cancel queued stimulation, then send a stim burst. This is equivalent to
        calling `interrupt()` followed by `stim()`, on the same set of channels.

        Constraints:
            - The minimum `lead_time_us` is `80`.
            - `lead_time_us` must be evenly divisible by `40`.

        Args:
            channel_set:    A `ChannelSet` object with one or more channels, or a single channel to stimulate.
            stim_design:    A `StimDesign` object or a floating point current in microamperes.
            burst_design:   A `BurstDesign` object specifying the burst count and frequency.
            lead_time_us:   The lead time in microseconds before the stimulation starts.
        """
        self.interrupt(channel_set)
        self.stim(channel_set, stim_design, burst_design, lead_time_us)

    def sync(
        self,
        channel_set:          ChannelSet,
        /,
        wait_for_frame_start: bool        = True
        ) -> None:
        """
        Prevent further queued stimulation until all channels have reached this sync point.

        If `wait_for_frame_start` is `True`, the sync will wait until the start of the next frame.
        This is generally preferred as it allows subsequent stimcode to be generated in a more
        efficient form.

        If `False`, it will not wait for the next frame start, allowing zero latency immediate
        continuation of the stimulation plan. However, subsequent stimcode on these channels
        within this plan be generated in a less efficient form. This option is useful for
        switching between stim frequencies without sometimes adding an additional half-frame
        of latency at the switch point.

        Args:
            channel_set:            One or more channels to sync.
            wait_for_frame_start:   Whether to wait for the next frame start before continuing.

        For example:

        ```python
        with cl.open() as neurons:

            stim_design    = StimDesign(160, -1.0, 160, 1.0)

            channel_set_1  = ChannelSet(8)
            burst_design_1 = BurstDesign(2, 100)    # Interval of 250 frames

            channel_set_2  = ChannelSet(10)
            burst_design_2 = BurstDesign(2, 20)     # Interval of 1250 frames

            group_1_stims = []
            for tick in neurons.loop(ticks_per_second=10, stop_after_ticks=11):

                if tick.iteration == 0:
                    # Group 1
                    neurons.stim(channel_set_1, stim_design, burst_design_1)
                    neurons.stim(channel_set_2, stim_design, burst_design_2)

                    # Group 2
                    neurons.sync(channel_set_1 | channel_set_2)
                    neurons.stim(channel_set_1, stim_design)

                for stim in tick.analysis.stims:
                    if stim.channel == 8:
                        group_1_stims.append(stim.timestamp)

            group_gap = group_1_stims[-1] - group_1_stims[0]
            # Group gap is expected to be > 1250 being the interval of the slowest frequency
        ```
        """
        self._sync_channels(
            self._loop_tick_timestamp if (self._in_loop and self._loop_tick_timestamp is not None) else self.timestamp(),
            channel_set,
            wait_for_frame_start = wait_for_frame_start
            )

    def create_stim_plan(self) -> StimPlan:
        """
        Create a new `StimPlan` object to build a stimcode plan.

        Stim plans which are reusable stimulation instructions that can be created
        at the beginning of an application to run on demand and contain the same
        stimulation interface, such as `StimPlan.stim()`, etc.

        For example:

        ```python
        import cl
        from cl import ChannelSet, StimDesign, BurstDesign

        with cl.open() as neurons:

            # Create a stim plan with a single biphasic stim with current of
            # 1.0 uA, pulse width of 160 us and negative leading edge on
            # two sets of channels
            my_stim_plan  = neurons.create_stim_plan()
            channel_set_1 = ChannelSet(8, 9)
            channel_set_2 = ChannelSet(10, 11)
            stim_design   = StimDesign(160, -1.0, 160, 1.0)
            my_stim_plan.stim(channel_set_1, stim_design)
            my_stim_plan.stim(channel_set_2, stim_design)

            # ... Do something else

            # Execute the stim plan at any stage of your script
            my_stim_plan.run()
        ```
        """
        return StimPlan(self)

    def loop(
        self,
        ticks_per_second:        float,
        stop_after_seconds:      float | None = None,
        stop_after_ticks:        int   | None = None,
        ignore_jitter:           bool         = False,
        jitter_tolerance_frames: int          = 0,
        ) -> Loop:
        """
        Periodically detect spikes and execute code. (Relates to `Loop` and `LoopTick`.)

        Intended for use as an iterator:

        ```python
        TICKS_PER_SECOND = 100

        with cl.open() as neurons:
            for tick in neurons.loop(TICKS_PER_SECOND):
                # tick                      is a `LoopTick` object
                # tick.iteration            is the count of this tick within the loop
                # tick.iteration_timestamp  is the timestamp of the loop body
                # tick.frames               is a numpy array of processed electrode samples
                # tick.analysis.spikes      is a list of any detected spikes
                # tick.analysis.stims       is a list of any stimulation
                # tick.loop                 is the running loop object
        ```

        Or by passing a callback to `Loop.run()`:

        ```python
        TICKS_PER_SECOND = 100

        def handle_tick(tick: LoopTick):
            # Do something ...

            # When ready to stop ...
            tick.loop.stop()

        neurons.loop(TICKS_PER_SECOND).run(handle_tick)
        ```

        **Jitter**

        As `Loop` is intended for realtime operation, by default it will raise a
        `TimeoutError` if the loop body does not finish before data beyond the next
        tick is available.

        This can be relaxed by setting `jitter_tolerance_frames` to a non-zero value,
        or ignored entirely by setting `ignore_jitter` to `True`. We do **not** recommend
        the general use of these parameters to handle jitter. Instead consider
        explicit jitter recovery with `Loop.recover_from_jitter()`.

        Otherwise, the loop will continue indefinitely unless `stop_after_seconds`
        or `stop_after_ticks` is passed at loop creation time, `LoopTick.loop.stop()`
        is called during the tick, or a break statement is used to exit the for loop.

        **Timestamps**

        Since `Loop` operates in realtime, there are a few key considerations if
        precise timing is desired. This can be very important for executing synchronised
        stims and event logging.

        - Data accessible during each loop tick via `tick.analysis` (which is a type of
          `DetectionResult`) is collected in the previous tick and is bounded by
          `DetectionResult.start_timestamp` and `DetectionResult.stop_timestamp`.
        - System timestamp when entering the loop body is accessible by `LoopTick.iteration_timestamp`,
          and is equivalent to the end of the data collection period.
          (i.e. `LoopTick.iteration_timestamp == DetectionResult.stop_timestamp`.)

        ```python
        import cl
        from cl import ChanelSet, StimDesign

        with cl.open() as neurons:
            stim_plan_A = neurons.create_stim_plan()
            stim_plan_A.stim(ChannelSet(8, 9), StimDesign(160, -1.0, 160, 1.0))

            stim_plan_B = neurons.create_stim_plan()
            stim_plan_B.stim(ChannelSet(16, 17), StimDesign(160, -1.0, 160, 1.0))

            data_stream = neurons.create_data_stream("stim_events")

            for tick in neurons.loop(ticks_per_second=10, stop_after_seconds=2):
                # The system timestamp will be slightly later than the
                # starting timestamp of the current loop body
                assert neurons.timestamp() >= tick.iteration_timestamp

                # Stim plans executed at the tick.iteration_timestamp will be
                # executed as soon as possible, as it is slightly in the past
                # and is not guaranteed to be at the same time
                stim_plan_A.run(at_timestamp=iteration_timestamp)
                stim_plan_B.run(at_timestamp=iteration_timestamp)

                # ... and will be equivalent to
                stim_plan_A.run()
                stim_plan_B.run()

                # Users seeking to execute synchronised stims could
                # take advantage of tick.iteration_next_timestamp
                stim_plan_A.run(at_timestamp=tick.iteration_next_timestamp)
                stim_plan_B.run(at_timestamp=tick.iteration_next_timestamp)

                # Using tick.iteration_next_timestamp is also helpful to ensure
                # that stim events are correctly aligned when logging events
                data_stream.append(tick.iteration_next_timestamp, "Stim Happened!")
        ```

        Args:
            ticks_per_second:        How often the loop should return a result.
            stop_after_seconds:      How long to run the closed loop for in seconds.
                                     (default: `None`, i.e. loop indefinitely)
            stop_after_ticks:        How long to run the closed loop for in number of ticks.
                                     (default: `None`, i.e. loop indefinitely)
            ignore_jitter:           If True, the loop will not raise a `TimeoutError`.
            jitter_tolerance_frames: How far the loop can fall behind (in frames)
                                     before it raises a `TimeoutError`.

        Constraints:
        - `ticks_per_second` must not exceed the system sampling rate of 25,000 Hz.
        """
        return \
            Loop(
                neurons                 = self,
                ticks_per_second        = ticks_per_second,
                stop_after_seconds      = stop_after_seconds,
                stop_after_ticks        = stop_after_ticks,
                ignore_jitter           = ignore_jitter,
                jitter_tolerance_frames = jitter_tolerance_frames
                )

    def record(
        self,
        file_suffix         : str   | None            = None,
        file_location       : str   | None            = None,
        from_seconds_ago    : float | None            = None,
        from_frames_ago     : int   | None            = None,
        from_timestamp      : int   | None            = None,
        stop_after_seconds  : float | None            = None,
        stop_after_frames   : int   | None            = None,
        attributes          : dict[str, Any] | None   = None,
        include_spikes      : bool                    = True,
        include_stims       : bool                    = True,
        include_raw_samples : bool                    = True,
        include_data_streams: bool                    = True,
        exclude_data_streams: list[str]               = []
        ) -> Recording:
        """
        Start a new HDF5 recording.

        Args:
            file_suffix:            The suffix to append to the filename, before the `.h5` extension.
            file_location:          An absolute path to the directory where the file should be saved,
                                    or relative path (relative to the default recording location).
            from_seconds_ago:       The number of seconds ago to start recording from, if possible.
            from_frames_ago:        The number of frames ago to start recording from, if possible.
            from_timestamp:         The timestamp to start recording from, if possible.
            stop_after_seconds:     The number of seconds to record for.
            stop_after_frames:      The number of frames to record.
            attributes:             A dictionary of attributes to add to the recording.
            include_spikes:         Whether to include detected spikes in the recording.
            include_stims:          Whether to include stimulation events in the recording.
            include_raw_samples:    Whether to include frames of raw samples in the recording.
            include_data_streams:   Pass `True` to record all data streams, False to record no data streams,
                                    or a list of specific data stream names to record.
            exclude_data_streams:   A list of application data streams to exclude from the recording.

        Specific to the Simulator:
        - Recording data is kept in system memory and only saved to disk when calling `close()`.
        - Recording from the past using `from_*` parameters are not used.
        - Recordings can be identified by the attribute `file_format.version == "SDK"`.
        - The following attributes are included in the Simulator recording for
          completeness, but the values are empty: `git_hash`, `git_branch`,
          `git_tags`, and `git_status`.

        Typical usage example:

        ```python
        with cl.open() as neurons:
            recording = neurons.record()
            # Your code here ...
            recording.stop()
        ```

        Example for stopping recording after a duration of time:

        ```python
        with cl.open() as neurons:
            recording = neurons.record(stop_after_seconds=3)
            recording.wait_until_stopped()
        ```
        """
        return \
            Recording(
                file_suffix          = file_suffix,
                file_location        = file_location,
                from_seconds_ago     = from_seconds_ago,
                from_frames_ago      = from_frames_ago,
                from_timestamp       = from_timestamp,
                stop_after_seconds   = stop_after_seconds,
                stop_after_frames    = stop_after_frames,
                attributes           = attributes,
                include_spikes       = include_spikes,
                include_stims        = include_stims,
                include_raw_samples  = include_raw_samples,
                include_data_streams = include_data_streams,
                exclude_data_streams = exclude_data_streams,

                # Simulator only parameters
                _neurons             = self,
                _channel_count       = self._replay_file.attributes["channel_count"],
                _sampling_frequency  = self._replay_file.attributes["sampling_frequency"],
                _frames_per_second   = self._replay_file.attributes["frames_per_second"],
                _uV_per_sample_unit  = self._replay_file.attributes["uV_per_sample_unit"],
                _data_streams        = self._data_streams
                )

    def create_data_stream(
        self,
        name:       str,
        attributes: dict[str, Any] | None = None
        ) -> DataStream:
        """
        Publish a named stream of (timesamp, serialised_data) for recordings and visualisation.

        See `RecordingView.data_streams` for how to use data streams saved in a recording.

        Args:
            name:       Datastream name.
            attributes: A dictionary of attributes to add to the datastream.

        For example:

        ```python
        with cl.open() as neurons:
            # Create a named data stream - by default, it will be added to any active or future recordings.
            data_stream = neurons.create_data_stream(
                name       = 'example_data_stream',
                attributes = { 'score': 0, 'another_attrbute': [0, 1, 2, 3] }
                )

            # Start a recording
            recording = neurons.record(stop_after_seconds=1)

            timestamp = neurons.timestamp()

            # Add some data stream entries with unique, ascending timestamps:
            data_stream.append(timestamp + 0, { 'arbitrary': 'data' })
            data_stream.append(timestamp + 1, ['of', 'arbitrary', 'size'])
            data_stream.append(timestamp + 2, 'and type.')
            data_stream.append(timestamp + 3, numpy.array([2**64 - 1, 2**64 - 2, 2**64 - 3], dtype=numpy.uint64))

            # Update a single attribute
            data_stream.set_attribute('score', 1)

            # Update multiple attributes at once
            data_stream.update_attributes({ 'score': 2, 'new_attribute': 9.9 })

            recording.wait_until_stopped()
        ```
        """
        data_stream = DataStream(
            neurons    = self,
            name       = name,
            attributes = attributes
        )

        # Initialize this data stream in any active recordings
        for recording in self._recordings:
            recording._init_data_stream(name, data_stream._attributes)

        return data_stream

    def get_channel_count(self) -> int:
        """
        Get the number of channels (electrodes) the device supports.
        A frame is a single sample from each channel.
        """
        return self._channel_count

    def get_frames_per_second(self) -> int:
        """
        Get the number of frames per second the device is configured to produce.
        A frame is a single sample from each channel.
        """
        return self._frames_per_second

    def get_frame_duration_us(self) -> float:
        """ Get the duration of a frame in microseconds. """
        return 1e6 / self.get_frames_per_second()

    def timestamp(self) -> int:
        """
        Get the current timestamp of the device.
        The timestamp sequence resets when the device is restarted.
        """
        # Ensure producer is started (lazy initialization)
        self._ensure_producer_started()

        assert self._shared_buffer is not None

        # Return timestamp from shared buffer (producer is source of truth)
        return self._shared_buffer.write_timestamp

    def read(
        self,
        frame_count:    int,
        from_timestamp: int | None = None,
        /
    ) -> ndarray[tuple[int, int], np.dtype[np.int16]]:
        """
        Read `frame_count` frames from the neurons, starting at `from_timestamp`
        if supplied.

        This method will block until the requested frames are available.
        If `from_timestamp` is `None`, the current timestamp minus one will be
        used, which ensures that a single frame read will return without
        blocking.

        Args:
            frame_count:    Number of frames to return.
            from_timestamp: Read from a specific timestamp. If None, return
                            from the current timestamp.

        Returns:
            Frames as an array with shape (frame_count, channel_count).
        """
        # Ensure producer is started (lazy initialization)
        self._ensure_producer_started()

        assert self._shared_buffer is not None

        # Calculate required timestamps
        now = self._shared_buffer.write_timestamp
        if from_timestamp is None:
            from_timestamp = now
        to_timestamp = from_timestamp + frame_count

        # In loop mode with accelerated time, check if read would exceed jitter tolerance
        # This prevents user code from advancing time too far ahead
        if self._use_accelerated_time and self._in_loop:
            # Get current deadline_timestamp from shared buffer (set by loop)
            # If reading beyond deadline would trigger jitter failure, raise TimeoutError
            if self._loop_deadline_ts is not None and to_timestamp > self._loop_deadline_ts:
                raise TimeoutError(
                    f"Read request would exceed loop jitter tolerance "
                    f"(requested up to {to_timestamp}, deadline is {self._loop_deadline_ts})"
                )

        # The system will allow reading from up to ~ 5 secs in the past (shared buffer size)
        if from_timestamp < (now - self._shared_buffer.buffer_duration_frames):
            raise Exception(f"Requested read from past timestamp (from={from_timestamp}, now={now}, buf={self._shared_buffer.buffer_duration_frames}, req={frame_count}) exceeds buffer capacity")

        # For large reads in accelerated mode that might exceed buffer capacity,
        # read in chunks to avoid buffer wraparound issues
        max_chunk_size = self._shared_buffer.buffer_duration_frames // 2  # Read half buffer at a time
        if self._use_accelerated_time and frame_count > max_chunk_size:
            result = np.empty((frame_count, self._channel_count), dtype=np.int16)
            chunks_read = 0

            while chunks_read < frame_count:
                chunk_size = min(max_chunk_size, frame_count - chunks_read)
                chunk_from_ts = from_timestamp + chunks_read
                chunk_to_ts = chunk_from_ts + chunk_size

                # Tell producer to advance to this chunk's end
                # In loop mode, already checked deadline above
                self._shared_buffer.requested_timestamp = chunk_to_ts

                # Wait for chunk data
                if not self._shared_buffer.wait_for_timestamp(chunk_to_ts, timeout_seconds=30.0):
                    raise TimeoutError(f"Timeout waiting for timestamp {chunk_to_ts}")

                # Read chunk
                try:
                    chunk_data = self._shared_buffer.read_frames(chunk_from_ts, chunk_size)
                    result[chunks_read:chunks_read + chunk_size] = chunk_data
                except ValueError as e:
                    raise Exception(f"Failed to read frames at chunk {chunks_read}: {e}") from e

                chunks_read += chunk_size

            read_frames = result
        else:
            # Normal single read for small requests or real-time mode
            # In accelerated mode, tell the producer to advance to the required timestamp
            # In loop mode, deadline check was already done above
            if self._use_accelerated_time:
                self._shared_buffer.requested_timestamp = to_timestamp

            # Wait for data to be available if reading into the future
            if to_timestamp > now and not self._shared_buffer.wait_for_timestamp(to_timestamp, timeout_seconds=30.0):
                raise TimeoutError(f"Timeout waiting for timestamp {to_timestamp}")

            try:
                read_frames = self._shared_buffer.read_frames(from_timestamp, frame_count)
            except ValueError as e:
                # Data not available - might be too old or not yet produced
                raise Exception(f"Failed to read frames: {e}") from e

        # Update _elapsed_frames for backward compatibility
        new_elapsed = to_timestamp - self._start_timestamp
        self._elapsed_frames = max(self._elapsed_frames, new_elapsed)

        # Push samples to all active recordings
        for recording in self._recordings:
            recording._write_samples(read_frames)

        self._read_timestamp = max(self._read_timestamp, to_timestamp)

        return read_frames

    async def read_async(
        self,
        frame_count:    int,
        from_timestamp: int | None = None
        ) -> ndarray[tuple[int, int], np.dtype[np.int16]]:
        """ Asynchronous version of read(). """
        return self.read(frame_count, from_timestamp)

    #
    # All non-passive functionality requires that the calling process
    # has taken "control" of the device. We only allow a single process
    # to take control at a time.
    #

    def has_control(self) -> bool:
        """
        Indicates whether control has been obtained.

        @private -- hide from docs
        """
        return True

    def take_control(self) -> None:
        """
        Take control of the device. Only one process can take control at a time.

        @private -- hide from docs
        """
        ...

    def release_control(self) -> None:
        """
        Release control of the device.

        @private -- hide from docs
        """
        ...

    #
    # Methods that indicate the device readiness.
    #

    def is_readable(self) -> bool:
        """
        Returns `True` if the device can be read from.

        @private -- hide from docs
        """
        return True

    def wait_until_readable(self, timeout_seconds: float = 15):
        """
        Blocks until the device can be read from, raising a `TimeoutError` if the
        timeout is exceeded.

        Args:
            timeout_seconds: Number of seconds to wait before timeout.

        @private -- hide from docs
        """
        ...

    def is_recordable(self) -> bool:
        """
        Return `True` if the device is recordable.

        @private -- hide from docs
        """
        return True

    def wait_until_recordable(self, timeout_seconds: float = 15):
        """
        Blocks until the recording system is ready, raising a `TimeoutError` if
        the timeout is exceeded.

        Args:
            timeout_seconds: Number of seconds to wait before timeout.

        @private -- hide from docs
        """
        ...

    #
    # Methods below here require that that the calling process has taken control.
    #

    def start(self) -> None:
        """
        Start the device if has not already started.

        @private -- hide from docs
        """
        self._is_running = True

    def has_started(self) -> bool:
        """
        Returns `True` if the device has started.

        @private -- hide from docs
        """
        return self._is_running

    def restart(
        self,
        timeout_seconds      : int = 15,
        wait_until_recordable: int = True
        ) -> None:
        """
        Restart the device and wait until it is readable, and optionally, recordable.

        @private -- hide from docs
        """
        self._elapsed_frames = 0
        # Reset wall-clock reference so _sleep_until() works correctly after restart
        self._start_walltime_ns = time.perf_counter_ns()
        self._prev_walltime_ns = self._start_walltime_ns

    def stop(self) -> None:
        """
        Stop the device if it has started.

        @private -- hide from docs
        """
        self._is_running = False

    def close(self) -> None:
        """
        Closes the connection to the CL1. If we have control, ensure stimulation is off,
        then release control. This is called automatically when using the `with cl.open()`
        context manager interface.

        @private -- hide from docs
        """
        if self.has_control():
            self.release_control()

        if self.has_started():
            self.stop()

        # Stop WebSocket server subprocess first (it reads from the buffer)
        self._stop_websocket_server()

        # Stop heartbeat thread
        self._stop_heartbeat_thread()

        # Stop the data producer subprocess (always, even if not started via has_started)
        if hasattr(self, '_data_producer') and self._data_producer is not None:
            self._data_producer.stop()
            self._data_producer = None
            self._shared_buffer = None
            self._producer_started = False

        # Stop any recordings
        for recording in self._recordings:
            recording.stop()

        # Close the H5 recording
        self._replay_file.close()

    #
    # Simulator specific functionality, do not use these in your applications.
    #

    _is_running: bool = False
    """ (Simulator only) Indicates the current status. """

    _replay_file: RecordingView
    """ (Simulator only) The recording file to replay. """

    _replay_start_offset: int
    """ (Simulator only) Offset the starting index of the replay file. """

    _start_timestamp: int
    """ (Simulator only) Start timestamp of the recording. """

    _read_timestamp: int
    """ (Simulator only) Timestamp that the system was read up to. """

    _start_walltime_ns: int
    """ (Simulator only) Starting system wall time in nanoseconds. """

    _prev_walltime_ns: int
    """ (Simulator only) Last seen system wall time in nanoseconds. """

    _use_accelerated_time: bool
    """ (Simulator only) When True, use system accelerated time, otherwise, use wall clock time. """

    _channel_count: int
    """ (Simulator only) Number of channels used in the recording. """

    _frames_per_second: int
    """ (Simulator only) Sampling frequency of the recording. """

    _duration_frames: int
    """ (Simulator only) Duration of the recording in frames. """

    _stim_queue: ChannelStimQueue[_StimOp]
    """ (Simulator only) Queued stims to be delivered at specific timestamps. Indexed by channel for efficient interrupt handling. """

    _stim_frequency_bin_duration_us: int = 20
    """
    (Simulator only) Duration of the smallest frequency bin for generating stim bursts
    that is supported by the system in microseconds (us).
    """

    _frame_duration_us: int
    """ (Simulator only) Time interval between frames in microseconds (us) based on _frames_per_second. """

    _tick_stims: list[Stim]
    """ (Simulator only) Record of stims during ticks, will be reset when read. """

    _stim_channel_available_from: ndarray[tuple[int], np.dtype[np.int_]]
    """ (Simulator only) Timestamps each channel will be available from. """

    _recordings: list[Recording]
    """ (Simulator only) Keep track of active recordings. """

    _elapsed_frames: int
    """ (Simulator only) Keep track of how many frames have elapsed, to inform timestamp(). """

    _timed_ops: PriorityQueue[tuple[int, Callable]]
    """
    (Simulator only) A queue of operations to be called at specific timestamps. This
    can be useful for things like stopping recordings at a given timestamp.
    """

    _data_streams: dict[str, DataStream]
    """ (Simulator only) Record of all DataStreams in use. """

    _loop_deadline_ts: int | None = None
    """ (Simulator only) Deadline timestamp for the current loop tick, or None if not in loop. """

    _loop_tick_timestamp: int | None = None
    """ (Simulator only) Timestamp of the current loop tick, or None if not in loop. """

    _sleep_latency_buffer_secs: float = 0.1
    """
    (Simulator only) Buffer to account for latency when waking up from time.sleep()
    that has been tested on a number of systems.
    """

    _app_html: ClassVar[str | None] = None
    """ (Simulator only) HTML for visualisation of an application run. """

    def _advance_elapsed_frames(self, frame_count: int = 0) -> None:
        """
        (Simulator only) Advances the _elapsed_frames counter one frame at a time to
        simulate passage of time. We use this opportunity to apply time
        dependent tasks like performing stims.

        Args:
            frame_count: Number of frames to advance. When this is zero and
                we are in not in accelerated time mode, we will advance the
                _elapsed_frames by the real passage of time.
        """
        blocking_mode = True

        if frame_count == 0 and not self._use_accelerated_time:
            # Here, we allow the frame counter to catch up to wall clock time
            current_walltime_ns    = time.perf_counter_ns()
            elapsed_walltime_ns    = current_walltime_ns - self._prev_walltime_ns
            frame_count            = int(elapsed_walltime_ns * self._frames_per_second / 1e9)
            blocking_mode          = False

        # Advance elapsed_frames iteratively while performing queued stims and/or ops.
        now              = self._start_timestamp + self._elapsed_frames
        timestamp_target = now + frame_count
        stim_queue       = self._stim_queue
        ops_queue        = self._timed_ops

        while now < timestamp_target:

            # Advance to the timestamp of the next queued stim or operation
            next_stim_timestamp   = stim_queue.peek_min_timestamp()
            next_stim_timestamp   = timestamp_target if next_stim_timestamp is None else next_stim_timestamp
            next_op_timestamp     = timestamp_target if ops_queue.qsize()  < 1 else ops_queue.queue[0][0]
            next_timestamp        = min(next_stim_timestamp, next_op_timestamp, timestamp_target)
            self._elapsed_frames += (next_timestamp - now)
            now                   = next_timestamp

            # Perform any stims in the queue up to current timestamp
            stim_ch_msg: dict[int, list[int]] = defaultdict(list)
            popped_stims = stim_queue.pop_until(now + 1)  # +1 because pop_until is exclusive
            for stim_ts, _stim_channel, stim_op in popped_stims:
                if isinstance(stim_op, _StimOp):
                    # Push stim to all active recordings
                    for recording in self._recordings:
                        recording._write_stims([stim_op.stim])
                    self._tick_stims.append(stim_op.stim)
                    stim_ch_msg[stim_ts].append(stim_op.stim.channel)

            # Potentially hot path, so explicitly check to avoid unnecessary overhead
            if _logger.isEnabledFor(logging.DEBUG) and stim_ch_msg:
                # This is for a verbose message to let the user know we've performed a stim
                for stim_ts, stim_chs in stim_ch_msg.items():
                    _logger.debug("Stim at %d on channels %s", stim_ts, stim_chs)

            # Perform any operations in the queue
            while (ops_queue.qsize() > 0):
                if ops_queue.queue[0][0] > now:
                    break
                _, op = ops_queue.get()
                op()

        # Here, we block the thread for the requested frame_count in wall clock time.
        if blocking_mode and not self._use_accelerated_time:
            self._sleep_until(timestamp_target)

        # Update the wall clock time before leaving
        self._prev_walltime_ns = time.perf_counter_ns()

    def _sleep_until(self, timestamp: int) -> None:
        """
        (Simulator only) Block the thread until the specified timestamp is reached.

        Args:
            timestamp: wake up the system at this timestamp
        """
        assert self._shared_buffer is not None

        # Calculate how many frames we need to wait
        current_timestamp = self._shared_buffer.write_timestamp
        frames_to_wait    = timestamp - current_timestamp

        if frames_to_wait <= 0:
            return  # Already past target timestamp

        wait_secs = frames_to_wait / self._frames_per_second

        # We wake up the system earlier with a buffer to increase accuracy
        end = time.perf_counter() + wait_secs
        if wait_secs > self._sleep_latency_buffer_secs:
            # This is power efficient but may have a variable amount of latency
            time.sleep(wait_secs - self._sleep_latency_buffer_secs)
        while time.perf_counter() < end:
            # This is to increase accuracy but not power efficient
            pass

    def _read_spikes(
        self,
        frame_count:    int,
        from_timestamp: int | None
        ) -> list[Spike]:
        """
        (Simulator only) Read spikes from the shared buffer that are found in the next
        frame_count frames, starting at from_timestamp if supplied.

        Args:
            frame_count: Number of frames to consider for reading spikes.
            from_timestamp: Read from a specific timestamp. If None, return
                from the current timestamp.

        Returns:
            List of spikes found within the given number of frames.
        """
        # Calculate required timestamps
        now               = self.timestamp()
        from_timestamp    = now if from_timestamp is None else from_timestamp
        to_timestamp      = from_timestamp + frame_count

        # Read from shared buffer
        if self._shared_buffer is not None:
            spike_records = self._shared_buffer.read_spikes(from_timestamp, to_timestamp)
            if not spike_records:
                return []
            read_spikes = [
                Spike(
                    timestamp           = rec.timestamp,
                    channel             = rec.channel,
                    samples             = rec.samples,
                    channel_mean_sample = rec.channel_mean_sample,
                )
                for rec in spike_records
            ]
            # Push spikes to all active recordings
            for recording in self._recordings:
                recording._write_spikes(read_spikes)
            return read_spikes

        # Fallback to legacy direct replay file access
        assert self._replay_file.spikes is not None, "Replay file does not contain spikes"
        replay_spikes = self._replay_file.spikes

        op_timestamp                     = from_timestamp - self._start_timestamp + self._replay_start_offset
        op_end_timestamp                 = to_timestamp   - self._start_timestamp + self._replay_start_offset
        start_idx                        = op_timestamp   % self._duration_frames
        read_spikes      : list[Spike]   = []
        while op_timestamp < op_end_timestamp:
            remaining_frames = op_end_timestamp - op_timestamp
            end_idx          = min(self._duration_frames, start_idx + remaining_frames)
            for i in replay_spikes.get_where_list(f"(timestamp > {start_idx}) & (timestamp <= {end_idx})"):
                replay_spike = replay_spikes[i]
                spike_timestamp = int(
                    replay_spike["timestamp"]
                    - start_idx
                    + op_timestamp
                    - self._replay_start_offset
                    + self._start_timestamp
                    )

                assert spike_timestamp > from_timestamp and spike_timestamp <= to_timestamp
                read_spikes.append(Spike(
                    timestamp           = spike_timestamp,
                    channel             = int(replay_spike["channel"]),
                    samples             = replay_spike["samples"],
                    channel_mean_sample = float(replay_spike["samples"].mean())
                    ))

            op_timestamp += (end_idx - start_idx)
            start_idx = end_idx % self._duration_frames

        # Push spikes to all active recordings
        for recording in self._recordings:
            recording._write_spikes(read_spikes)
        return read_spikes

    def _read_stims(
        self,
        from_timestamp: int,
        to_timestamp  : int
        ) -> list[Stim]:
        """
        (Simulator only) Read stims from the shared buffer within a timestamp range.

        Args:
            from_timestamp: Start of range (inclusive).
            to_timestamp: End of range (exclusive).

        Returns:
            List of stims found within the given timestamp range.
        """
        if hasattr(self, '_shared_buffer') and self._shared_buffer is not None:
            stim_records = self._shared_buffer.read_stims(from_timestamp, to_timestamp)
            if not stim_records:
                return []
            read_stims = [
                Stim(timestamp=rec.timestamp, channel=rec.channel)
                for rec in stim_records
            ]
            # Push stims to all active recordings
            for recording in self._recordings:
                recording._write_stims(read_stims)
            return read_stims

        # Fallback - no stims available without shared buffer
        return []

    def _read_and_reset_stim_cache(self) -> list[Stim]:
        """
        (Simulator only) Read stims since the last read and update tracking.

        This is the primary interface for getting stims in a loop iteration.
        """
        if hasattr(self, '_shared_buffer') and self._shared_buffer is not None:
            # In a loop, use tick timestamps to avoid skipping stims scheduled
            # "late" during a tick (when producer is already ahead)
            if self._in_loop and self._loop_tick_timestamp is not None:
                # Read stims from last tick to current tick (not to producer position)
                from_ts = self._last_stim_read_ts
                to_ts   = self._loop_tick_timestamp         # End of current tick
                stims   = self._read_stims(from_ts, to_ts)
                self._last_stim_read_ts = to_ts
            else:
                # Non-loop: use producer's write position
                now   = self.timestamp()
                stims = self._read_stims(self._last_stim_read_ts, now)
                self._last_stim_read_ts = now
            return stims

        # Fallback to legacy behavior
        stims = self._tick_stims.copy()
        self._tick_stims.clear()
        return stims

    def _determine_burst_interval_us(self, frequency_hz: float) -> int:
        """ (Simulator only) Determines the interval of stims in bursts as microseconds (us). """
        assert frequency_hz > 0, "Burst frequency must be positive"
        interval_frames = int((1_000_000 / frequency_hz / self._stim_frequency_bin_duration_us) + 0.5)
        interval_us     = int(interval_frames * self._stim_frequency_bin_duration_us)
        return interval_us

    def _queue_stims(
        self,
        from_timestamp: int,
        channel_set:    ChannelSet  | int,
        stim_design:    StimDesign  | float,
        burst_design:   BurstDesign | None   = None,
        lead_time_us:   int                  = 80,
        ) -> None:
        """
        (Simulator only). Queues stims on one or more channels at the specified timestamp.

        from_timestamp: Timestamp of the first stim.
        channels      : One or more channels to stimulate.
        stim_design   : A StimDesign object or a scalar current in microamperes.
        burst_design  : A BurstDesign object specifying the burst count and frequency (default: None).
        lead_time_us  : The lead time in microseconds before the stimulation starts (default: 80).
        """
        # Check and build ChannelSet
        if isinstance(channel_set, ChannelSet):
            pass
        elif isinstance(channel_set, int):
            channel_set = ChannelSet(channel_set)
        else:
            raise ValueError(
                f"channel_set must be "
                f"ChannelSet object or an int, "
                f"not {channel_set.__class__.__name__}"
                )

        # Check and build StimDesign
        if isinstance(stim_design, StimDesign):
            pass
        elif (isinstance(stim_design, (int, float))):
            # Default StimDesign is biphasic with negative leading edge and 160 us pulse width
            stim_design = StimDesign(160, -stim_design, 160, stim_design)
        else:
            raise ValueError(
                f"stim_design must be "
                f"StimDesign object or a float, "
                f"not {stim_design.__class__.__name__}"
                )

        # Check and build BurstDesign
        if isinstance(burst_design, BurstDesign):
            pass
        elif burst_design is None:
            burst_design = BurstDesign(1, 100)  # burst_hz does not matter for burst of one
        else:
            raise ValueError(
                f"burst_design must be "
                f"BurstDesign object, "
                f"not {burst_design.__class__.__name__}"
                )

        # Specify stimulation constraints
        lead_time_us_bins         = 40
        minimum_lead_time_us      = 80
        minimum_lead_time_frames  = int(minimum_lead_time_us / lead_time_us_bins)
        minimum_burst_interval_us = minimum_lead_time_us + stim_design.duration_us

        # Check that stimulation constraints have been met
        if lead_time_us < minimum_lead_time_us:
            raise ValueError(f"lead_time_us must be at least {minimum_lead_time_us}")

        if not lead_time_us % lead_time_us_bins == 0:
            raise ValueError(f"lead_time_us must be evenly divisible by {lead_time_us_bins}")

        if burst_design._burst_interval_us < minimum_burst_interval_us:
            raise ValueError(
                f"Burst interval {burst_design._burst_interval_us} us "
                f"must be at least {minimum_lead_time_us} us "
                f"+ duration {stim_design.duration_us}"
                )

        stim_duration_us        = stim_design.duration_us
        stim_duration_frames    = int(stim_duration_us / 1e6 * self._frames_per_second)
        lead_time_frames        = int(lead_time_us     / 1e6 * self._frames_per_second)

        # Calculate burst intervals
        # Some frequencies can have a slightly longer interval due to how it lines up
        # with the frequency bins, i.e. 96 Hz will have interval frames as [260, 261, 260, 261 ...]
        burst_interval_us           = self._determine_burst_interval_us(burst_design._burst_requested_hz)
        total_burst_duration_us     = burst_design._burst_count * burst_interval_us
        burst_times_us              = np.arange(0, total_burst_duration_us, step=burst_interval_us)
        burst_timestamps, remainder = np.divmod(burst_times_us, self._frame_duration_us)
        burst_timestamps           += lead_time_frames
        has_trailing_offset         = remainder[-1] != 0

        for stim_channel in channel_set._iterate_channels():
            free_ts                  = self._stim_channel_available_from[stim_channel]
            is_available             = from_timestamp > free_ts
            start_offset             = from_timestamp if is_available else free_ts
            channel_burst_timestamps = burst_timestamps + start_offset
            for i, stim_start_ts in enumerate(channel_burst_timestamps):
                stim_end_ts       = stim_start_ts + stim_duration_frames
                next_available_ts = stim_end_ts

                if i < burst_design._burst_count - 1:
                    # Add a delay if we are in a middle of a burst, which is
                    # equivalent to a direct swap should a new stim command
                    # be called immediately following interrupt. In this case,
                    # minimum lead time need to be subtracted.
                    next_available_ts = channel_burst_timestamps[i + 1] - minimum_lead_time_frames

                elif i == burst_design._burst_count - 1 and has_trailing_offset:
                    # Add an extra frame due to frequency bin alignment
                    next_available_ts += 1

                self._stim_queue.put(
                    timestamp = stim_start_ts,
                    channel   = stim_channel,
                    payload   = _StimOp(
                        stim          = Stim(timestamp=stim_start_ts, channel=stim_channel),
                        end_timestamp = next_available_ts
                    ),
                )

                # We mark the channel busy from the stim timestamp for the
                # amount of time it takes to perform the stim
                self._stim_channel_available_from[stim_channel] = next_available_ts

                # Send stim command to the data producer subprocess
                if hasattr(self, '_data_producer') and self._data_producer is not None:
                    self._data_producer.queue_stim(
                        timestamp     = int(stim_start_ts),
                        channel       = int(stim_channel),
                        end_timestamp = int(next_available_ts),
                    )

    def _interrupt_queued_stims(
        self,
        from_timestamp: int,
        channel_set:    ChannelSet | int
        ) -> None:
        """
        (Simulator only). Interrupt existing and clear queued stims from a specified timestamp.

        Args:
            from_timestamp: Timestamp after which queue stims should be cleared.
            channels      : One or more channels to interrupt.
        """
        if isinstance(channel_set, ChannelSet):
            pass
        elif isinstance(channel_set, int):
            channel_set = ChannelSet(channel_set)
        else:
            raise ValueError(
                f"channel_set must be "
                f"ChannelSet object or an int, "
                f"not {channel_set.__class__.__name__}"
                )

        interrupt_channels = channel_set._tolist()

        # Before removing stims, get the last kept stim's end timestamp for each channel
        # This preserves channel availability info needed for sync()
        for channel in interrupt_channels:
            last_kept = self._stim_queue.get_last_entry_before(channel, from_timestamp)
            if last_kept is not None:
                _, stim_op = last_kept
                # Use the end timestamp of the last kept stim
                self._stim_channel_available_from[channel] = stim_op.end_timestamp
            else:
                # No stims kept, channel is available at interrupt time
                self._stim_channel_available_from[channel] = from_timestamp

        # Use efficient channel-indexed removal: O(c * log k) instead of O(n) drain-rebuild
        # This removes stims at or after from_timestamp for the specified channels
        self._stim_queue.interrupt_channels(interrupt_channels, from_timestamp)

        # Send interrupt command to the data producer subprocess
        if hasattr(self, '_data_producer') and self._data_producer is not None:
            self._data_producer.interrupt_channels(
                timestamp = from_timestamp,
                channels  = interrupt_channels,
            )

    def _sync_channels(
        self,
        from_timestamp:       int,
        channel_set:          ChannelSet,
        /,
        wait_for_frame_start: bool        = True
        ) -> None:
        """
        (Simulator only) Align channel availability to a common timestamp, which is
        the latest (maximum) availability timestamp of the specified channels.

        Args:
            from_timestamp:         Timestamp after which sync should be performed.
            channel_set:            One or more channels to sync.
            wait_for_frame_start:   Whether to wait for the next frame start before continuing.
                                    This has no effect in mock, since only full-frames are used.
        """
        sync_channels  = np.array(channel_set._tolist())
        sync_timestamp = max(self._stim_channel_available_from[sync_channels].max(), from_timestamp)
        self._stim_channel_available_from[sync_channels] = sync_timestamp

    def _start_heartbeat_thread(self) -> None:
        """
        (Simulator only) Start background thread that continuously updates heartbeat.

        This thread runs every 50ms to update the heartbeat timestamp in shared memory.
        When the debugger pauses the process, this thread also pauses, causing the
        heartbeat to go stale and triggering subprocesses to pause.
        """
        if self._heartbeat_thread is not None:
            return  # Already running

        self._heartbeat_stop_event = Event()
        self._heartbeat_thread     = Thread(
            target = self._heartbeat_loop,
            daemon = True,
            name   = "HeartbeatThread"
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat_thread(self) -> None:
        """(Simulator only) Stop the heartbeat thread."""
        if self._heartbeat_thread is None or self._heartbeat_stop_event is None:
            return

        self._heartbeat_stop_event.set()
        if self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=0.5)
        self._heartbeat_thread = None
        self._heartbeat_stop_event = None

    def _heartbeat_loop(self) -> None:
        """
        (Simulator only) Background thread loop that updates heartbeat timestamp.

        Runs every 50ms to keep the heartbeat fresh. When the debugger pauses
        the process, this thread also pauses, causing the heartbeat to become stale.
        """
        if self._heartbeat_stop_event is None:
            return

        while not self._heartbeat_stop_event.is_set():
            try:
                if hasattr(self, '_shared_buffer') and self._shared_buffer is not None:
                    self._shared_buffer.main_process_heartbeat_ns = time.perf_counter_ns()
            except Exception:
                pass  # Ignore errors in background thread

            # Sleep for 50ms (updates heartbeat at 20Hz)
            self._heartbeat_stop_event.wait(timeout=0.05)

class _StimOp:
    """ (Simulator only) Object representing a Stim Operation used in Neurons._stim_queue. """

    stim: Stim
    """ A Stim object. """

    end_timestamp: int
    """
    Expected end timestamp of the stim, which is:
    1. Duration of the StimDesign if single stim, and
    2. Frequency delay in the case of stim bursts.
    """

    def __init__(self, stim: Stim, end_timestamp: int) -> None:
        self.stim          = stim
        self.end_timestamp = end_timestamp

    def __repr__(self) -> str:
        return f"StimOp(stim={self.stim}, end_timestamp={self.end_timestamp})"

    def __lt__(self, other: _StimOp) -> bool:
        """ (Simulator only) Compare two instances of StimOp for neurons._stim_queue. """
        assert isinstance(other, type(self)), \
            f"Cannot compare StimOp with {other.__class__.__name__}"
        if self.stim.timestamp == other.stim.timestamp:
            return self.stim.channel < other.stim.channel
        return self.stim.timestamp < other.stim.timestamp
