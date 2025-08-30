import logging
from queue import PriorityQueue
from typing import Any
from pathlib import Path
from collections.abc import Sequence, Callable
import os
import time
from random import randint

from dotenv import load_dotenv

import numpy as np
from numpy import ndarray

from cl import (
    Loop,
    Stim,
    Spike,
    ChannelSet,
    StimDesign,
    BurstDesign,
    _logger
)
from cl.stim_plan import StimPlan
from cl.recording import Recording
from cl.util import RecordingView, AttributesView, more_accurate_sleep
from cl.data_stream import DataStream

class Neurons:
    """
    A mock Neurons class that simulates the behaviour of the real "cl" API by
    replaying spikes and raw samples from a H5 recording (replay_file). The
    recording to use is controlled by the CL_MOCK_REPLAY_PATH environment
    variable, which can be set by a .env file.

    Note on timing:
    - In order to facilitate rapid testing, the internal frame counter
        is not coupled to actual time. Instead, it advances when either
        neurons.read() is called or when using neurons.loop(). When exiting
        the context manager, neurons.read() will be automatically called if
        there are queued events, such as stims.
    - Actual passage of time is simulated in neurons.loop() to facilitate
        the generation of TimeoutError in high jitter situations (i.e. when
        the loop iterations falls behind). This can help to identify and
        optimise computation within loop iterations.
    """

    def __init__(self):
        _logger.debug("using Cortical Labs Mock API")

    def __enter__(self):
        """ (Mock only) Open a H5 recording and set required attributes. """

        def load_replay_file() -> RecordingView:
            from cl import _CL_MOCK_REPLAY_PATH
            assert _CL_MOCK_REPLAY_PATH is not None and Path(_CL_MOCK_REPLAY_PATH).exists(), \
                f"Recording not found: {_CL_MOCK_REPLAY_PATH}"
            _logger.debug(f"simulating from recording: {_CL_MOCK_REPLAY_PATH}")
            return RecordingView(_CL_MOCK_REPLAY_PATH)

        self._replay_file: RecordingView  = load_replay_file()
        attrs:             AttributesView = self._replay_file.attributes

        self._start_timestamp   = int(attrs["start_timestamp"])
        self._read_timestamp    = self._start_timestamp
        self._channel_count     = int(attrs["channel_count"])
        self._frames_per_second = int(attrs["frames_per_second"])
        self._duration_frames   = int(attrs["duration_frames"])
        self._elapsed_frames    = randint(0, self._duration_frames)

        self._recordings                  = []
        self._recording_stims             = []
        self._recording_spikes            = []
        self._recording_samples           = []
        self._tick_stims                  = []
        self._stim_queue                  = PriorityQueue()
        self._stim_channel_available_from = [self._start_timestamp] * self._channel_count

        load_dotenv(".env")
        self._use_accelerated_time        = os.getenv("CL_MOCK_ACCELERATED_TIME", "0") == "1"
        self._start_walltime_ns           = time.time_ns()
        self._prev_walltime_ns            = self._start_walltime_ns
        if not self._use_accelerated_time:
            _logger.debug("time policy: wall clock time")
        else:
            _logger.debug("time policy: accelerated")
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
        return False

    def __del__(self):
        self.close()

    def close(self):
        """
        If we have control, ensure stimulation is off, then release control.
        Then close this connection to the device. This is called automatically
        when used in a "with" statement.
        """
        if self.has_control():
            self.release_control()

        # Perform housekeeping before closing
        if self._use_accelerated_time:
            # Advance the frame counter to perform queued stims, since we are
            # decoupled from actual passing of time.
            if self._stim_queue.qsize() > 0:
                final_ts, _ = self._stim_queue.queue[-1]
                frame_count = final_ts - self.timestamp() + 1

                self.read(frame_count, None)
                self._read_spikes(frame_count, None)
                self._read_and_reset_stim_cache()
        else:
            # Advance the frame counter
            self._advance_elapsed_frames()

        # Stop any recordings
        for recording in self._recordings:
            recording.stop()

        # Close the H5 recording
        self._replay_file.close()
        return

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

    def has_started(self) -> bool:
        """ Returns True if the device has started. """
        return True

    def is_readable(self) -> bool:
        """ Returns True if the device can be read from. """
        return True

    def wait_until_readable(self, timeout_seconds: float = 15):
        """
        Blocks until the device can be read from, raising a TimeoutError if the
        timeout is exceeded.
        """
        ...

    def is_recordable(self) -> bool:
        """ Return True if the device is recordable. """
        return True

    def wait_until_recordable(self, timeout_seconds: float = 15):
        """
        Blocks until the recording system is ready, raising a TimeoutError if
        the timeout is exceeded.
        """
        ...

    def timestamp(self) -> int:
        """
        Get the current timestamp of the device.
        The timestamp sequence resets when the device is restarted.
        """
        self._advance_elapsed_frames()
        return self._start_timestamp + self._elapsed_frames

    def read(
        self,
        frame_count:    int,
        from_timestamp: int | None,
        /
    ) -> ndarray[Any, np.dtype[np.int16]]:
        """
        Read frame_count frames from the neurons, starting at from_timestamp
        if supplied.

        This method will block until the requested frames are available.
        If from_timestamp is None, the current timestamp minus one will be
        used, which ensures that a single frame read will return without
        blocking.

        Args:
            frame_count: Number of frames to return.
            from_timestamp: Read from a specific timestamp. If None, return
                from the current timestamp.

        Returns:
            Frames as an array with shape (frame_count, channel_count).
        """
        assert frame_count <= self._duration_frames, \
            "Requested frame_count exceeds the duration of the replay_file"

        # This can be adjusted based on if from_timestamp is in the past
        op_timestamp = self.timestamp()
        frame_count_to_advance = frame_count
        if from_timestamp is None:
            from_timestamp = op_timestamp
        to_timestamp = from_timestamp + frame_count

        # The system will allow reading from up to 5 secs in the past
        if from_timestamp < op_timestamp - int(5 * self._frames_per_second):
            raise Exception("read did not succeed")

        # If the from_timestamp is in the future, block and wait.
        while op_timestamp < from_timestamp:
            self._advance_elapsed_frames(frame_count=1)
            op_timestamp = self.timestamp()

        # If the from_timestamp is in the past, adjust frame_count
        timestamp_diff = from_timestamp - op_timestamp
        if timestamp_diff < 0:
            frame_count_to_advance = max(0, frame_count + timestamp_diff)

        # We advance the system time by the requested number of frames to
        # facilitate frame read.
        self._advance_elapsed_frames(frame_count=frame_count_to_advance)
        op_timestamp = self.timestamp()

        # We will retrieve the frame samples from the replay_file, which
        # is an array with shape (duration_frames, channel_count). If the
        # requested frame_count is longer than duration_frames, then we
        # will wrap from the beginning of the recording.
        end_relative_frame = min(to_timestamp, op_timestamp) - self._start_timestamp
        if end_relative_frame < 0:
            end_relative_frame = self._duration_frames + end_relative_frame
        end_idx            = end_relative_frame % self._duration_frames
        start_idx          = end_idx - frame_count

        assert self._replay_file.samples is not None, "Replay file does not contain samples"
        replay_frames = self._replay_file.samples

        if start_idx >= 0:
            # No wrapping is needed
            read_frames = replay_frames[start_idx:end_idx, :]
        else:
            # Wrapping is needed
            start_idx += self._duration_frames
            read_frames: ndarray = np.concat([
                replay_frames[start_idx:        , :],
                replay_frames[         :end_idx , :],
                ])

        self._recording_samples.append(read_frames)
        self._read_timestamp = op_timestamp
        return read_frames

    async def read_async(
        self,
        frame_count:    int,
        from_timestamp: int | None
        ) -> ndarray[Any, np.dtype[np.int16]]:
        """ Asynchronous version of read(). """
        return self.read(frame_count, from_timestamp)

    def create_stim_plan(self) -> StimPlan:
        """
        Create a new StimPlan object to build a stimcode plan.

        This allows the creation of complex stimulation plans with control
        over stim alignment between channels.

        For example:

            from cl import ChannelSet, StimDesign, BurstDesign

                # Create a stim plan with a single biphasic stim with current of
                # 1.0 uA, pulse width of 160 us and negative leading edge on
                # two sets of channels
                my_stim_plan = neurons.create_stim_plan()
                channel_set_1 = ChannelSet(8, 9)
                channel_set_2 = ChannelSet(10, 11)
                stim_design = StimDesign(160, -1.0, 160, 1.0)
                my_stim_plan.stim(channel_set_1, stim_design)
                my_stim_plan.stim(channel_set_2, stim_design)

                # ... Do something else

                # Execute the stim plan at any stage of your script
                my_stim_plan.run()
        """
        return StimPlan(self)

    def stim(
        self,
        channel_set:  ChannelSet  | int,
        stim_design:  StimDesign  | float,
        burst_design: BurstDesign | None   = None,
        lead_time_us: int                  = 80
        ):
        """
        Stimulate one or more channels.

        channels    : One or more channels to stimulate.
        stim_design : A StimDesign object or a scalar current in microamperes.
        burst_design: A BurstDesign object specifying the burst count and frequency (default: None).
        lead_time_us: The lead time in microseconds before the stimulation starts (default: 80).

        For example:

            from cl import ChannelSet, StimDesign, BurstDesign

            # Deliver a single biphasic stim with current of 1.0 uA, pulse width
            # of 160 us and negative leading edge on channels 8, 9, 10
            channel_set = ChannelSet(8, 9, 10)
            stim_design = StimDesign(160, -1.0, 160, 1.0)
            neurons.stim(channel_set, stim_design)

            # Deliver the same stim as a burst of 10 at 40 Hz
            burst_design = BurstDesign(10, 40)
            neurons.stim(channel_set, stim_design, burst_design)
        """
        self._queue_stims(
            from_timestamp = self.timestamp(),
            channel_set    = channel_set,
            stim_design    = stim_design,
            burst_design   = burst_design,
            lead_time_us   = lead_time_us
            )

    def interrupt(self, channel_set: ChannelSet | int):
        """
        Interrupt existing and clear queued stimulation.

        This will stop existing stimulation and clear any pending stimulation for the specified channels.

        Interruption either occurs at the end of a stim, or immediately if the channel is waiting for
        a specific timestamp. Notably, if the channel is between stims within a burst, it will not
        interrupt until the next stim within the burst has completed. This is to allow cleaner switching
        between frequencies.

        channels:   One or more channels to interrupt.
        """
        self._interrupt_queued_stims(
            from_timestamp = self.timestamp(),
            channel_set    = channel_set
            )

    def interrupt_then_stim(
        self,
        channels:     ChannelSet | int,
        stim_design:  StimDesign | float,
        burst_design: BurstDesign,
        lead_time_us: int = 80
        ):
        """
        Interrupt existing and cancel queued stimulation, then send a stim burst.

        channels:       One or more channels to stimulate.
        stim_design:    A StimDesign object or an integer current in microamperes.
        burst_design:   A BurstDesign object specifying the burst count and frequency.
        lead_time_us:   The lead time in microseconds before the stimulation starts (default: 80).
        """
        self.interrupt(channels)
        self.stim(channels, stim_design, burst_design, lead_time_us)

    def sync(
        self,
        channels:             ChannelSet,
        wait_for_frame_start: bool        = True
        ):
        """
        Prevent further queued stimulation until all channels have reached this sync point.

        If wait_for_frame_start is True, the sync will wait until the start of the next frame.
        This is generally preferred as it allows subsequent stimcode to be generated in a more
        efficient form.

        If False, it will not wait for the next frame start, allowing zero latency immediate
        continuation of the stimulation plan. However, subsequent stimcode on these channels
        within this plan be generated in a less efficient form. This option is useful for
        switching between stim frequencies without sometimes adding an additional half-frame
        of latency at the switch point.

        channels:               One or more channels to sync.
        wait_for_frame_start:   Whether to wait for the next frame start before continuing (default: True).
        """
        raise NotImplementedError("sync() is not yet implemented")

    def stimulate(
        self,
        channels         : Sequence[int] | int,
        current_uA       : float | None         = None,
        burst_count      : int                  = 1,
        burst_frequency  : int                  = 0,
        lead_time_us     : int                  = 80,
        pulse_duration_us: int                  = 160,
        stim_design      : StimDesign | None    = None
        ):
        """
        Stimulate one or more channels.

        Deprecated: Use stim() instead.
        """
        raise NotImplementedError("Deprecated, use stim()")

    def loop(
        self,
        ticks_per_second:        int,
        stop_after_seconds:      float | None = None,
        stop_after_ticks:        int | None   = None,
        ignore_jitter:           bool         = True, # TODO: set to False when loop timing is fixed
        jitter_tolerance_frames: int          = 0,
        ):
        """
        Periodically detect spikes and execute code.

        Can be used as an iterator:

            TICKS_PER_SECOND = 100

            for tick in neurons.loop(TICKS_PER_SECOND):
                # tick.timestamp is the timestamp of the first processed frame
                # tick.frames is a numpy array of processed electrode samples
                # tick.analysis.spikes is a list of any detected spikes
                # tick.analysis.stims is a list of any stimulation
                # tick.loop is the running loop object

        Or by passing a callback to run():

            TICKS_PER_SECOND = 100

            def handle_tick(tick):
                # ...

            neurons.loop(TICKS_PER_SECOND).run(handle_tick)

        As Loop is intended for realtime operation, by default it will raise a
        TimeoutError if the loop body does not finish before data beyond the next
        tick is available.

        This can be relaxed by setting jitter_tolerance_frames to a non-zero value,
        or ignored entirely by setting ignore_jitter to True.

        Otherwise, the loop will continue indefinitely unless stop_after_seconds
        or stop_after_ticks is passed at loop creation time, tick.loop.stop()
        is called during the tick, or a break statement is used to exit the for loop.

        ticks_per_second:        How often the loop should return a result.
        stop_after_seconds:      How long to run the closed loop for. (default:
                                 None, i.e. loop indefinitely)
        stop_after_seconds:      How long to run the closed loop for. (default:
                                 None, i.e. loop indefinitely)
        ignore_jitter:           If True, the loop will not raise a
                                 TimeoutError. (default: False)
        jitter_tolerance_frames: How far the loop can fall behind (in frames)
                                 before it raises a TimeoutError. (default: 0)
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
        file_suffix          :str | None            = None,
        file_location        :str | None            = None,
        from_seconds_ago     :float | None          = None,
        from_frames_ago      :int | None            = None,
        from_timestamp       :int | None            = None,
        stop_after_seconds   :int | None            = None,
        stop_after_frames    :int | None            = None,
        attributes           :dict[str, Any] | None = None,
        include_spikes       :bool                  = True,
        include_stims        :bool                  = True,
        include_raw_samples  :bool                  = True,
        include_data_streams :bool                  = True,
        exclude_data_streams :list[str]             = []
        ) -> Recording:
        """
        Start a new HDF5 recording.

        file_suffix:            The suffix to append to the filename, before the .h5 extension.
        file_location:          An absolute path to the directory where the file should be saved,
                                or relative path (relative to the default recording location).
        from_seconds_ago:       The number of seconds ago to start recording from, if possible.
        from_frames_ago:        The number of frames ago to start recording from, if possible.
        from_timestamp:         The timestamp to start recording from, if possible.
        stop_after_seconds:     The number of seconds to record for.
        stop_after_frames:      The number of frames to record.
        attributes:             A dictionary of attributes to add to the recording.
        include_spikes:         Whether to include detected spikes in the recording. (default: True)
        include_stims:          Whether to include stimulation events in the recording. (default: True)
        include_raw_samples:    Whether to include frames of raw samples in the recording. (default: True)
        include_data_streams:   Pass True to record all data streams, False to record no data streams,
                                or a list of specific data stream names to record. (Default: True)
        exclude_data_streams:   A list of application data streams to exclude from the recording. (Default: [])
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

                # Mock only parameters
                _neurons             = self,
                _channel_count       = self._replay_file.attributes["channel_count"],
                _sampling_frequency  = self._replay_file.attributes["sampling_frequency"],
                _frames_per_second   = self._replay_file.attributes["frames_per_second"],
                _uV_per_sample_unit  = self._replay_file.attributes["uV_per_sample_unit"],
                _recording_spikes    = self._recording_spikes,
                _recording_stims     = self._recording_stims,
                _recording_samples   = self._recording_samples,
                _data_streams        = self._data_streams
                )

    def create_data_stream(self, name, attributes=None) -> DataStream:
        """
        Publish a named stream of (timesamp, serialised_data) for recordings and visualisation.

        If added to a recording, the dataset path will be /data_stream/<name>.
        """
        return \
            DataStream(
                neurons    = self,
                name       = name,
                attributes = attributes
                )

    #
    # All non-passive functionality requires that the calling process
    # has taken "control" of the device. We only allow a single process
    # to take control at a time.
    #

    def has_control(self) -> bool:
        return True

    def take_control(self):
        """ Take control of the device. Only one process can take control at a time. """
        ...

    def release_control(self):
        """ Release control of the device. """
        ...

    #
    # Methods below here require that that the calling process has taken control.
    #

    def start(self):
        """ Start the device if has not already started. """
        ...

    def restart(
        self,
        timeout_seconds      : int = 15,
        wait_until_recordable: int = True
    ):
        """ Restart the device and wait until it is readable, and optionally, recordable. """
        ...

    def stop(self):
        """ Stop the device if it has started. """
        ...

    #
    # Mock specific functionality, do not use these in your applications.
    #

    _replay_file: RecordingView
    """ (Mock only) The recording file to replay. """

    _start_timestamp: int
    """ (Mock only) Start timestamp of the recording. """

    _read_timestamp: int
    """ (Mock only) Timestamp that the system was read up to. """

    _start_walltime_ns: int
    """ (Mock only) Starting system wall time in nanoseconds. """

    _prev_walltime_ns: int
    """ (Mock only) Last seen system wall time in nanoseconds. """

    _use_accelerated_time: bool
    """ (Mock only) When True, use system accelerated time, otherwise, use wall clock time. """

    _channel_count: int
    """ (Mock only) Number of channels used in the recording. """

    _frames_per_second: int
    """ (Mock only) Sampling frequency of the recording. """

    _duration_frames: int
    """ (Mock only) Duration of the recording in frames. """

    _stim_queue: PriorityQueue[tuple[int, Stim]]
    """ (Mock only) Queued stims to be delivered at specific timestamps. """

    _tick_stims: list[Stim]
    """ (Mock only) Record of stims during ticks, will be reset when read. """

    _stim_channel_available_from: list[int]
    """ (Mock only) Timestamps each channel will be available from. """

    _recordings: list[Recording]
    """ (Mock only) Keep track of recordings. """

    _recording_stims: list[Stim]
    """ (Mock only) Record of all stims conducted for mock recording. """

    _recording_spikes: list[Spike]
    """ (Mock only) Record of all spikes observed for mock recording. """

    _recording_samples: list[ndarray]
    """ (Mock only) Record of all samples observed for mock recording. """

    _elapsed_frames: int
    """
    (Mock only) Keep track of how many frames have elapsed, which is used to
    inform timestamp. Since this is a mock API, we don't sync this
    to the actual passage of time. Note that Loop iterations are time
    aware managed with standalone functionality (See Loop.__iter__()).
    """

    _timed_ops: PriorityQueue[tuple[int, Callable]] = PriorityQueue()
    """
    (Mock only) A queue of operations to be called at specific timestamps. This
    can be useful for things like stopping recordings at a given timestamp.
    """

    _data_streams: dict[str, DataStream] = {}
    """ (Mock only) Record of all DataStreams in use. """

    def _advance_elapsed_frames(self, frame_count: int = 0):
        """
        (Mock only) Advances the _elapsed_frames counter one frame at a time to
        simulate passage of time. We use this opportunity to apply time
        dependent tasks like performing stims.

        Args:
            frame_count: Number of frames to advance. When this is zero and
                we are in not in accelerated time mode, we will advance the
                _elapsed_frames by the real passage of time.
        """
        if not self._use_accelerated_time:
            if frame_count < 1:
                # If we are using wall time, we will override the frame_count
                # with the actual number of frames passed based on wall clock time.
                current_walltime_ns    = time.time_ns()
                elapsed_walltime_ns    = current_walltime_ns - self._prev_walltime_ns
                frame_count            = int(elapsed_walltime_ns * self._frames_per_second / 1e9)
                self._prev_walltime_ns = current_walltime_ns
            else:
                # Here, we block the thread for the requested frame_count in
                # wall clock time.
                wait_secs = frame_count / self._frames_per_second
                more_accurate_sleep(wait_secs)
                self._prev_walltime_ns = time.time_ns()

        for _ in range(frame_count):
            self._elapsed_frames += 1
            now                   = self._start_timestamp + self._elapsed_frames

            # We check whether we have reached the timestamp for the next
            # stim in the stim_queue. If so, simulate a stim by moving the stim
            # from the stim_queue to stims_conducted.
            stim_queue             = self._stim_queue
            stim_ch_msg: list[int] = []
            while (stim_queue.qsize() > 0):
                if stim_queue.queue[0][0] > now:
                    break
                stim_ts, stim = self._stim_queue.get()
                self._recording_stims.append(stim)
                self._tick_stims.append(stim)
                stim_ch_msg.append(stim.channel)

            # This is for a verbose message to let the user know we've
            # performed a stim
            if len(stim_ch_msg) > 0:
                _logger.debug(f"stim at {now} on channels {stim_ch_msg}")
                stim_ch_msg.clear()

            # We we also call any _timed_ops when reaching the appropriate timestamp.
            ops_queue = self._timed_ops
            while (ops_queue.qsize() > 0):
                if ops_queue.queue[0][0] > now:
                    break
                op_ts, op = ops_queue.get()
                op()

    def _read_spikes(
        self,
        frame_count:    int,
        from_timestamp: int | None
        ) -> list[Spike]:
        """
        (Mock only) Read spikes from the replay_file that are found in the next
        frame_count frames, starting at from_timestamp if supplied.

        Args:
            frame_count: Number of frames to consider for reading spikes.
            from_timestamp: Read from a specific timestamp. If None, return
                from the current timestamp.

        Returns:
            List of spikes found within the given number of frames.
        """
        assert frame_count <= self._duration_frames, "Requested frame_count exceeds the duration of the replay_file"

        # Calculate the starting frame number
        start_frame = self._elapsed_frames if from_timestamp is None else (from_timestamp - self._start_timestamp)
        if start_frame < 0:
            raise ValueError("Can't read from before the start timestamp")

        # We will retrieve the spikes from the replay_file. If the
        # requested frame_count is longer than duration_frames, then we
        # will wrap from the beginning of the recording.
        replay_start_ts = start_frame % self._duration_frames
        replay_end_ts   = replay_start_ts + frame_count

        assert self._replay_file.spikes is not None, "Replay file must contain spikes"
        replay_spikes = self._replay_file.spikes

        # Spikes in the replay_file are relative to the start timestamp
        # of the replay_file. We need to calculate an offset so that the
        # spikes have actual timestamps.
        times_wrapped = start_frame // self._duration_frames

        def ts_offset(times_wrapped: int) -> int:
            return (times_wrapped * self._duration_frames) + self._start_timestamp

        spikes: list[Spike]
        if replay_end_ts <= self._duration_frames:
            # No wrapping is needed
            spikes = [
                Spike(
                    timestamp           = int(replay_spikes[i]["timestamp"] + ts_offset(times_wrapped)),
                    channel             = int(replay_spikes[i]["channel"]),
                    samples             = replay_spikes[i]["samples"],
                    channel_mean_sample = float(replay_spikes[i]["samples"].mean())
                    )
                for i in replay_spikes.get_where_list(
                    f"(timestamp > {replay_start_ts}) & "
                    f"(timestamp <= {replay_end_ts})"
                    )
            ]

        else:
            # Wrapping is needed
            replay_end_ts = replay_end_ts % self._duration_frames
            spikes = [
                Spike(
                    timestamp           = int(replay_spikes[i]["timestamp"] + ts_offset(times_wrapped)),
                    channel             = int(replay_spikes[i]["channel"]),
                    samples             = replay_spikes[i]["samples"],
                    channel_mean_sample = float(replay_spikes[i]["samples"].mean())
                    )
                for i in replay_spikes.get_where_list(f"(timestamp > {replay_start_ts})")
            ]
            spikes.extend([
                Spike(
                    timestamp=int(
                        replay_spikes[i]["timestamp"]
                        + ts_offset(times_wrapped + 1)
                    ),
                    channel=int(replay_spikes[i]["channel"]),
                    samples=replay_spikes[i]["samples"],
                    channel_mean_sample=float(replay_spikes[i]["samples"].mean())
                )
                for i in replay_spikes.get_where_list(f"(timestamp <= {replay_end_ts})")
            ])

        self._recording_spikes.extend(spikes)
        return spikes

    def _read_and_reset_stim_cache(self) -> list[Stim]:
        """ (Mock only) Read and clear the stim cache. """
        stims = self._tick_stims.copy()
        self._tick_stims.clear()
        return stims

    def _queue_stims(
        self,
        from_timestamp: int,
        channel_set:    ChannelSet  | int,
        stim_design:    StimDesign  | float,
        burst_design:   BurstDesign | None   = None,
        lead_time_us:   int                  = 80,
        ):
        """
        (Mock only). Queues stims on one or more channels at the specified timestamp.

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
        elif (isinstance(stim_design, int) or isinstance(stim_design, float)):
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
            burst_design = BurstDesign(1, 100) # burst_hz does not matter for burst of one
        else:
            raise ValueError(
                f"burst_design must be "
                f"BurstDesign object, "
                f"not {burst_design.__class__.__name__}"
                )

        # Specify stimulation constraints
        minimum_lead_time_us      = 80
        lead_time_us_bins         = 40
        minimum_burst_interval_us = minimum_lead_time_us + stim_design._total_duration_us

        # Check that stimulation constraints have been met
        if lead_time_us < minimum_lead_time_us:
            raise ValueError(f"lead_time_us must be at least {minimum_lead_time_us}")

        if not lead_time_us % lead_time_us_bins == 0:
            raise ValueError(f"lead_time_us must be evenly divisible by {lead_time_us_bins}")

        if burst_design._burst_interval_us < minimum_burst_interval_us:
            raise ValueError(
                f"Burst interval {burst_design._burst_interval_us} us "
                f"must be at least {minimum_lead_time_us} us "
                f"+ duration {stim_design._total_duration_us}"
                )

        # burst_interval_frames = burst_design._burst_interval_frames
        burst_interval_frames = int(1 / burst_design._burst_hz * self._frames_per_second)
        stim_duration_us      = stim_design._total_duration_us
        stim_duration_frames  = int(stim_duration_us / 1e6 * self._frames_per_second)
        lead_time_frames      = int(lead_time_us     / 1e6 * self._frames_per_second)
        next_burst_ts         = from_timestamp

        for _ in range(burst_design._burst_count):
            for stim_channel in channel_set._iterate_channels():
                free_ts        = self._stim_channel_available_from[stim_channel]
                is_available   = next_burst_ts > free_ts
                stim_start_ts  = next_burst_ts if is_available else free_ts
                stim_start_ts += lead_time_frames
                stim_end_ts    = stim_start_ts + stim_duration_frames

                # Request a stim to be performed in the stim queue
                self._stim_queue.put((
                    stim_start_ts,  # queue priority
                    Stim(           # queue entry
                        timestamp = stim_start_ts,
                        channel   = stim_channel
                        )
                    ))

                # We mark the channel busy from the stim timestamp for the
                # amount of time it takes to perform the stim
                self._stim_channel_available_from[stim_channel] = stim_end_ts

            # Prepare for next burst iteration
            next_burst_ts += burst_interval_frames

    def _interrupt_queued_stims(
        self,
        from_timestamp: int,
        channel_set: ChannelSet | int
        ):
        """
        (Mock only). Interrupt existing and clear queued stims from a specified timestamp.

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

        interrupt_channels = set(channel_set._iterate_channels())

        # Clear the stim queue by draining the queue and only adding back
        # either the stims before from_timestamp or stims not in the interrupt channels
        stims_to_keep = PriorityQueue()
        while self._stim_queue.qsize() > 0:
            stim_ts, stim = self._stim_queue.get()
            if (stim_ts < from_timestamp) or (not stim.channel in interrupt_channels):
                stims_to_keep.put((stim_ts, stim))
        self._stim_queue = stims_to_keep

        # Align the interrupt channels to be available from the from_timestamp
        for channel in interrupt_channels:
            channel_available_ts = self._stim_channel_available_from[channel]
            if channel_available_ts > from_timestamp:
                self._stim_channel_available_from[channel] = from_timestamp

    def _conduct_stims(self):
        ...
