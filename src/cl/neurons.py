from __future__ import annotations

import heapq
import os
import time
import warnings
import atexit
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from queue import PriorityQueue
from random import randint
from threading import Event, Lock, RLock, Thread
from typing import TYPE_CHECKING, Any, ClassVar, overload, Literal

import numpy as np
from dotenv import load_dotenv
from numpy import ndarray

from . import _FRAME_TIME_US, BurstDesign, ChannelSet, Loop, Spike, Stim, StimDesign, StimPlan, _logger, DetectionResult
from ._sim._data_buffer import DEFAULT_FRAMES_PER_SECOND, StimRecord
from ._sim._data_producer import DataProducer
from .sim._data_source import (
    DEFAULT_DURATION_FRAMES,
    default_data_source_config,
    get_configured_data_source_config,
    load_data_source_metadata,
)
from ._sim._stim_queue import ChannelStimQueue
from .data_stream import DataStream
from .recording import Recording

if TYPE_CHECKING:
    from collections.abc import Callable

_MINIMUM_LEAD_TIME_US = 80

_INTERNAL_RATE_MULTIPLIER   = 2   # 50kHz internal / 25kHz external
_EXTERNAL_FRAMES_PER_SECOND = DEFAULT_FRAMES_PER_SECOND

_HEARTBEAT_INTERVAL_NS          = 10_000_000  # 10ms heartbeat interval for debugger detection
_HEARTBEAT_INTERVAL_S           = _HEARTBEAT_INTERVAL_NS / 1e9
_STIM_PUBLISH_ACTIVE_WINDOW_S   = 0.002   # 2ms
_STIM_PUBLISH_ACTIVE_INTERVAL_S = 0.0001  # 100us

def _to_external_ts(internal_ts: int) -> int:
    """Convert internal 50kHz timestamp to external 25kHz timestamp."""
    return internal_ts // _INTERNAL_RATE_MULTIPLIER

def _to_internal_ts(external_ts: int) -> int:
    """Convert external 25kHz timestamp to internal 50kHz timestamp."""
    return external_ts * _INTERNAL_RATE_MULTIPLIER

_NON_STIMMABLE_CHANNELS = ChannelSet(0, 7, 56, 63)

_NOTICE_STEPS         = _MINIMUM_LEAD_TIME_US // _FRAME_TIME_US  # 4
_CHANNEL_COMPLETE     = -1
_GROUNDED_CHANNEL_SET = set(_NON_STIMMABLE_CHANNELS._tolist())
_SYNC_ACTIVE          = -2

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

        self._stim_lock = RLock()

        self._loop_deadline_ts    = None
        self._loop_tick_timestamp = None
        self._in_loop             = False
        self._websocket_server    = None

        self._data_producer        = None
        self._producer_lock        = Lock()  # Thread-safe producer startup
        self._shared_buffer        = None
        self._recordings           = []
        self._heartbeat_thread     = None
        self._heartbeat_stop_event = None
        self._stim_publish_event   = Event()
        self._buffer_name_prefix   = None

        load_dotenv(".env")
        configured_source = get_configured_data_source_config()
        if configured_source is None:
            self._data_source_config = default_data_source_config()
            source_metadata          = load_data_source_metadata(self._data_source_config)

            self._replay_start_offset = int(os.getenv("CL_SDK_REPLAY_START_OFFSET", "-1"))
            if self._replay_start_offset < 0:
                replay_duration = source_metadata.duration_frames or DEFAULT_DURATION_FRAMES
                # _replay_start_offset is in 25kHz file frame units
                self._replay_start_offset = randint(0, replay_duration)
            self._data_source_config["replay_start_offset"] = self._replay_start_offset
            _logger.debug(
                "simulating from %s data source",
                self._data_source_config.get("kind"),
            )
        else:
            self._replay_start_offset = 0
            self._data_source_config = configured_source
            _logger.debug("simulating from custom data source: %s", configured_source.get("factory"))
            source_metadata = load_data_source_metadata(self._data_source_config)
        self._use_accelerated_time = os.getenv("CL_SDK_ACCELERATED_TIME", "0") == "1"
        if self._use_accelerated_time and not source_metadata.supports_accelerated:
            raise ValueError("Configured simulator data source does not support accelerated time")
        self._data_source_config["metadata"] = asdict(source_metadata)

        duration_frames = source_metadata.duration_frames or DEFAULT_DURATION_FRAMES
        self._replay_attrs = {
            "start_timestamp"   : int(source_metadata.start_timestamp),
            "channel_count"     : int(source_metadata.channel_count),
            "sampling_frequency": int(source_metadata.frames_per_second),
            "frames_per_second" : int(source_metadata.frames_per_second),
            "uV_per_sample_unit": float(source_metadata.uV_per_sample_unit),
            "duration_frames"   : int(duration_frames),
        }

        self._start_timestamp   = int(self._replay_attrs["start_timestamp"]) * _INTERNAL_RATE_MULTIPLIER
        self._read_timestamp    = self._start_timestamp
        self._channel_count     = int(self._replay_attrs["channel_count"])
        self._frames_per_second = int(self._replay_attrs["frames_per_second"]) * _INTERNAL_RATE_MULTIPLIER
        self._duration_frames   = int(self._replay_attrs["duration_frames"]) * _INTERNAL_RATE_MULTIPLIER
        self._frame_duration_us = 1_000_000 // self._frames_per_second
        self._elapsed_frames    = 0

        self._recordings = []
        self._stim_queue = ChannelStimQueue()

        self._stim_channel_available_from = np.full((self._channel_count,), fill_value=self._start_timestamp, dtype=int)

        # Operation tracking for rebuild-on-interrupt
        self._stim_op_records             : list[_StimOpRecord | _SyncOpRecord] = []
        self._interrupt_records           : list[tuple[int, list[int], int]]    = []     # (from_timestamp, channels, stim_op_index)
        self._rebuilding                  : bool                                = False
        self._rebuild_pending             : bool                                = False
        self._stim_history_has_rebuild_ops: bool                                = False
        self._rebuild_affected_channels   : set[int]                            = set()  # channels in sync/multi-channel ops
        self._pending_interrupts          : list[tuple[int, list[int]]]         = []     # interrupts not yet applied during rebuild
        self._rebuild_checkpoint_avail    : np.ndarray | None                   = None   # channel availability snapshot for incremental rebuilds

        self._timed_ops    = PriorityQueue()
        self._data_streams = {}

        if not self._use_accelerated_time:
            _logger.debug("time policy: wall clock time")
        else:
            _logger.debug("time policy: accelerated")

        self._start_walltime_ns = time.perf_counter_ns()
        self._prev_walltime_ns = self._start_walltime_ns

        # Prepare the data producer (but don't start yet - lazy initialization)
        # The producer and shared buffer run at the external 25kHz rate.
        # Internal 50kHz timestamps are only used for stim timing math.
        self._data_producer = DataProducer(
            data_source_config = self._data_source_config,
            start_timestamp    = _to_external_ts(self._start_timestamp),
            channel_count      = self._channel_count,
            frames_per_second  = _EXTERNAL_FRAMES_PER_SECOND,
            duration_frames    = self._duration_frames // _INTERNAL_RATE_MULTIPLIER,
            accelerated_time   = self._use_accelerated_time,
        )
        self._producer_started = False

        # Track timestamps for spike/stim reads (to avoid re-reading same data)
        self._last_spike_read_ts    = self._start_timestamp
        self._last_stim_read_ts     = self._start_timestamp
        self._stim_buffer_write_ts  = self._start_timestamp  # High-water mark for buffer stim writes
        self._last_published_stim_timestamp: int | None = None

        # Cadence marker for bounding the rebuild-history records (_stim_op_records /
        # _interrupt_records) even when no rebuild is triggered. See _read_stims.
        self._last_history_compact_ts = self._start_timestamp

        # Heartbeat thread for debugger detection
        self._heartbeat_thread     = None
        self._heartbeat_stop_event = None

        self._buffer_name_prefix = None

        # Whether to use the visualisation server
        self._use_websocket_server = os.getenv("CL_SDK_VISUALISATION", os.getenv("CL_SDK_WEBSOCKET", "1")) == "1"     # CL_SDK_WEBSOCKET is deprecated
        self._websocket_host       = os.getenv("CL_SDK_VISUALISATION_HOST", os.getenv("CL_SDK_WEBSOCKET_HOST", None)) # CL_SDK_WEBSOCKET_HOST is deprecated
        if os.getenv("CL_SDK_VISUALISATION") is None and os.getenv("CL_SDK_WEBSOCKET") is not None:
            warnings.warn("CL_SDK_WEBSOCKET is deprecated. Please use CL_SDK_VISUALISATION instead.", DeprecationWarning)
        if self._use_accelerated_time and self._use_websocket_server:
            _logger.warning("Visualisation service is not compatible with accelerated time. Disabling Visualisation service.")
            os.environ["CL_SDK_VISUALISATION"] = "0"
            os.environ["CL_SDK_WEBSOCKET"]     = "0"
            self._use_websocket_server         = False

    def __enter__(self):
        """ (Simulator only) Open a H5 recording and set required attributes. """

        self._recordings.clear()
        self._in_loop = False
        self._data_streams.clear()
        self._start_simulator_services()
        self._reset_stim_state()

        return self

    def _reset_stim_state(self) -> None:
        """Reset simulator stim scheduling state between open sessions."""
        with self._stim_lock:
            self._stim_queue.clear()
            self._stim_channel_available_from.fill(self._start_timestamp)
            self._stim_op_records.clear()
            self._interrupt_records.clear()
            self._rebuilding = False
            self._rebuild_pending = False
            self._stim_history_has_rebuild_ops = False
            self._rebuild_affected_channels.clear()
            self._pending_interrupts.clear()
            self._rebuild_checkpoint_avail = None
            self._last_published_stim_timestamp = None
            self._stim_publish_event.clear()

    def _start_simulator_services(self):
        """Starts simulator services like the producer and websocket for visualisations."""
        self._ensure_producer_started()
        self._start_websocket_server(host=self._websocket_host)
        atexit.register(Neurons._clear_instance, force=True)

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
            self._prev_walltime_ns  = self._start_walltime_ns

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
                for _ in range(2500):  # 100ms max wait
                    if self._shared_buffer.write_timestamp > _to_external_ts(self._start_timestamp):
                        break
                    time.sleep(0.00004)  # 40us sleep per iteration to avoid busy wait

    def _start_websocket_server(self, host: str | None = None, port: int | None = None) -> None:
        """
        (Simulator only) Start WebSocket server subprocess for visualization.

        Args:
            host: Host address for WebSocket server
            port: Port for WebSocket server (None for auto-find)
        """
        if self._websocket_server is not None:
            _logger.info("WebSocket server already running")
            return

        _use_websocket_server = os.getenv("CL_SDK_VISUALISATION", os.getenv("CL_SDK_WEBSOCKET", "1")) == "1"
        if not _use_websocket_server:
            return

        # Ensure producer is running first (WebSocket reads from shared buffer)
        self._ensure_producer_started()

        # Lazy import so we don't need to load websocket dependencies if not using websocket visualization
        from ._sim.visualisation._websocket_subprocess import WebSocketProcessManager

        # Get the unique shared memory prefix from the producer's buffer
        self._buffer_name_prefix = self._data_producer.buffer.get_name_prefix() if self._data_producer and self._data_producer.buffer else ""

        self._websocket_server = WebSocketProcessManager(
            buffer_name       = self._buffer_name_prefix,
            frames_per_second = self.get_frames_per_second(),
            channel_count     = self.get_channel_count(),
            port              = port,
            host              = host,
            app_html          = Neurons._app_html,
        )
        self._websocket_server.start()
        _logger.info(f"Visualisation service started on {host}:{self._websocket_server.port}")
        if self._websocket_server.web_url:
            print(f"Data visualiser: {self._websocket_server.web_url}", flush=True)
        if self._websocket_server.app_url:
            print(f"Application visualiser: {self._websocket_server.app_url}", flush=True)

    def _stop_websocket_server(self) -> None:
        """(Simulator only) Stop WebSocket server subprocess if running."""
        if self._websocket_server is not None:
            self._websocket_server.stop()
            self._websocket_server = None
            _logger.info("Visualisation service stopped")

    @classmethod
    def _get_http_port(cls) -> int | None:
        """(Simulator only) Get the HTTP port number for the visualiser."""
        neurons = cls._get_instance()
        if neurons._websocket_server is None or not neurons._websocket_server.is_alive():
            return None
        return neurons._websocket_server.web_port

    @classmethod
    def _get_ws_port(cls) -> int | None:
        """(Simulator only) Get the WebSocket port number for the visualiser."""
        neurons = cls._get_instance()
        if neurons._websocket_server is None or not neurons._websocket_server.is_alive():
            return None
        return neurons._websocket_server.port

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
        return False

    def __del__(self):
        self.close()

        # Stop WebSocket server subprocess first (it reads from the buffer)
        self._stop_websocket_server()

        # Stop heartbeat thread
        self._stop_heartbeat_thread()

        # Stop the data producer subprocess (always, even if not started via has_started)
        if self._data_producer is not None:
            self._data_producer.stop()
            self._data_producer = None
            self._shared_buffer = None
            self._producer_started = False

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
            from_timestamp = self._stim_command_timestamp(),
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
            from_timestamp = self._stim_command_timestamp(),
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
        from_timestamp = self._stim_command_timestamp()
        self._interrupt_queued_stims(
            from_timestamp = from_timestamp,
            channel_set    = channel_set
        )
        self._queue_stims(
            from_timestamp = from_timestamp,
            channel_set    = channel_set,
            stim_design    = stim_design,
            burst_design   = burst_design,
            lead_time_us   = lead_time_us
        )

    def sync(
        self,
        channel_set: ChannelSet,
        /
        ) -> None:
        """
        Cause all channel_set channels to wait until all are ready to continue together.

        The sync operation allows you ensure that subsequent operations on different channels
        begin at the same time - after all previously queued operations on those channels have
        completed.

        Args:
            channel_set: One or more channels to sync.

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
            self._stim_command_timestamp(),
            channel_set
            )

    def create_stim_plan(self) -> StimPlan:
        """
        Create a new `StimPlan` object to build a stimulation plan.

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
        assert self._replay_attrs is not None
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
                _channel_count       = self._replay_attrs["channel_count"],
                _sampling_frequency  = self._replay_attrs["frames_per_second"],
                _frames_per_second   = self._replay_attrs["frames_per_second"],
                _uV_per_sample_unit  = self._replay_attrs["uV_per_sample_unit"],
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
        return _EXTERNAL_FRAMES_PER_SECOND

    def get_frame_duration_us(self) -> float:
        """ Get the duration of a frame in microseconds. """
        return 1e6 / self.get_frames_per_second()

    def _internal_timestamp(self) -> int:
        """
        (Simulator only) Get the current internal 50kHz timestamp.
        The shared buffer runs at 25kHz; this converts to 50kHz.
        """
        self._ensure_producer_started()
        assert self._shared_buffer is not None
        return _to_internal_ts(self._shared_buffer.write_timestamp)

    def timestamp(self) -> int:
        """
        Get the current timestamp of the device.
        The timestamp sequence resets when the device is restarted.
        """
        self._ensure_producer_started()
        assert self._shared_buffer is not None
        return self._shared_buffer.write_timestamp

    def _stim_command_timestamp(self) -> int:
        """Return the causal timestamp for a user stim command."""
        if self._in_loop and self._loop_tick_timestamp is not None:
            return self._loop_tick_timestamp
        return self._internal_timestamp()

    @overload
    def read(
        self,
        frame_count:    int,
        from_timestamp: int | None      = None,
        /,
        *,
        analysis:       Literal[False]  = False
        ) -> ndarray[tuple[int, int], np.dtype[np.int16]]:
        ...

    @overload
    def read(
        self,
        frame_count:    int,
        from_timestamp: int | None      = None,
        /,
        *,
        analysis:       Literal[True]
        ) -> DetectionResult:
        ...

    def read(
        self,
        frame_count:    int,
        from_timestamp: int | None = None,
        /,
        *,
        analysis:       bool       = False
        ) -> ndarray[tuple[int, int], np.dtype[np.int16]] | DetectionResult:
        """
        Read `frame_count` frames from the neurons, starting at `from_timestamp`
        if supplied.

        This method will block until the requested frames are available.
        If `from_timestamp` is `None`, the current timestamp minus one will be
        used, which ensures that a single frame read will return without
        blocking.

        Args:
            frame_count:    Number of frames to return (at 25kHz).
            from_timestamp: Read from a specific timestamp (at 25kHz). If None, return
                            from the current timestamp.
            analysis:       When `True`, return `DetectionResult` instead of raw frames.

        Returns:
            Frames as an array with shape (frame_count, channel_count) if
            `analysis=False` or `DetectionResult` if `analysis=True`.
        """
        # Ensure producer is started (lazy initialization)
        self._ensure_producer_started()

        assert self._shared_buffer is not None

        # Buffer operates at external 25kHz — use parameters directly
        now = self._shared_buffer.write_timestamp
        if from_timestamp is None:
            buf_from = now
        else:
            buf_from = from_timestamp
        buf_to = buf_from + frame_count

        # In loop mode with accelerated time, check if read would exceed jitter tolerance
        # Compare in internal 50kHz space where _loop_deadline_ts lives
        if self._use_accelerated_time and self._in_loop:
            internal_to = _to_internal_ts(buf_to)
            if self._loop_deadline_ts is not None and internal_to > self._loop_deadline_ts:
                raise TimeoutError(
                    f"Read request would exceed loop jitter tolerance "
                    f"(requested up to {buf_to}, deadline is {_to_external_ts(self._loop_deadline_ts)})"
                )

        # The system will allow reading from up to ~ 5 secs in the past (shared buffer size)
        if buf_from < (now - self._shared_buffer.buffer_duration_frames):
            raise Exception(f"Requested read from past timestamp (from={buf_from}, now={now}, buf={self._shared_buffer.buffer_duration_frames}, req={frame_count}) exceeds buffer capacity")

        # For large reads in accelerated mode that might exceed buffer capacity,
        # read in chunks to avoid buffer wrap around issues
        max_chunk_size = self._shared_buffer.buffer_duration_frames // 2  # Read half buffer at a time
        if self._use_accelerated_time and frame_count > max_chunk_size:
            read_frames = np.empty((frame_count, self._channel_count), dtype=np.int16)
            read_spikes = []
            read_stims  = []
            chunks_read = 0

            while chunks_read < frame_count:
                chunk_size    = min(max_chunk_size, frame_count - chunks_read)
                chunk_from_ts = buf_from + chunks_read
                chunk_to_ts   = chunk_from_ts + chunk_size

                # Publish stims before advancing producer so source callbacks are causal.
                self._publish_stims_to_buffer_until(_to_internal_ts(chunk_to_ts))
                # Tell producer to advance to this chunk's end
                # In loop mode, already checked deadline above
                self._shared_buffer.requested_timestamp = chunk_to_ts

                # Wait for chunk data
                if not self._shared_buffer.wait_for_timestamp(chunk_to_ts, timeout_seconds=30.0):
                    raise TimeoutError(f"Timeout waiting for timestamp {chunk_to_ts}")

                # Read chunk
                try:
                    chunk_frames = self._shared_buffer.read_frames(chunk_from_ts, chunk_size)
                    # Spike/stim reads use internal 50kHz timestamps
                    internal_chunk_from = _to_internal_ts(chunk_from_ts)
                    internal_chunk_to   = _to_internal_ts(chunk_to_ts)
                    chunk_spikes = self._read_spikes(internal_chunk_to - internal_chunk_from, internal_chunk_from)
                    chunk_stims  = self._read_stims(internal_chunk_from, internal_chunk_to, write_to_buffer=not self._in_loop)

                    read_frames[chunks_read:chunks_read + chunk_size] = chunk_frames
                    read_spikes.extend(chunk_spikes)
                    read_stims.extend(chunk_stims)
                except ValueError as e:
                    raise Exception(f"Failed to read frames at chunk {chunks_read}: {e}") from e

                chunks_read += chunk_size

        else:
            # Normal single read for small requests or real-time mode
            # In accelerated mode, tell the producer to advance to the required timestamp
            # In loop mode, deadline check was already done above
            if self._use_accelerated_time:
                self._publish_stims_to_buffer_until(_to_internal_ts(buf_to))
                self._shared_buffer.requested_timestamp = buf_to

            # Wait for data to be available if reading into the future
            if buf_to > now and not self._shared_buffer.wait_for_timestamp(buf_to, timeout_seconds=30.0):
                raise TimeoutError(f"Timeout waiting for timestamp {buf_to}")

            try:
                read_frames = self._shared_buffer.read_frames(buf_from, frame_count)
                # Spike/stim reads use internal 50kHz timestamps
                internal_from = _to_internal_ts(buf_from)
                internal_to   = _to_internal_ts(buf_to)
                read_spikes = self._read_spikes(internal_to - internal_from, internal_from)
                read_stims  = self._read_stims(internal_from, internal_to, write_to_buffer=not self._in_loop)
            except ValueError as e:
                # Data not available - might be too old or not yet produced
                raise Exception(f"Failed to read frames: {e}") from e

        # Update _elapsed_frames for backward compatibility (internal 50kHz)
        new_elapsed = _to_internal_ts(buf_to) - self._start_timestamp
        self._elapsed_frames = max(self._elapsed_frames, new_elapsed)

        self._read_timestamp = max(self._read_timestamp, _to_internal_ts(buf_to))

        if analysis:
            # Convert internal timestamps to external 25kHz for user-facing API
            return DetectionResult(
                start_timestamp = buf_from,
                stop_timestamp  = buf_to,
                spikes          = read_spikes,
                stims           = read_stims
                )
        else:
            return read_frames

    def _read_loop_frames(
        self,
        frame_count: int,
        from_timestamp: int,
    ) -> ndarray[tuple[int, int], np.dtype[np.int16]]:
        """Read loop frames without also materializing analysis."""
        assert self._shared_buffer is not None

        buf_from = from_timestamp
        buf_to   = buf_from + frame_count
        now      = self._shared_buffer.write_timestamp

        if buf_from < (now - self._shared_buffer.buffer_duration_frames):
            raise Exception(
                f"Requested read from past timestamp (from={buf_from}, now={now}, "
                f"buf={self._shared_buffer.buffer_duration_frames}, req={frame_count}) exceeds buffer capacity"
            )

        if buf_to > now and not self._shared_buffer.wait_for_timestamp(buf_to, timeout_seconds=30.0):
            raise TimeoutError(f"Timeout waiting for timestamp {buf_to}")

        try:
            read_frames = self._shared_buffer.read_frames(buf_from, frame_count)
        except ValueError as e:
            raise Exception(f"Failed to read frames: {e}") from e

        new_elapsed = _to_internal_ts(buf_to) - self._start_timestamp
        self._elapsed_frames = max(self._elapsed_frames, new_elapsed)
        self._read_timestamp = max(self._read_timestamp, _to_internal_ts(buf_to))
        return read_frames

    @overload
    async def read_async(
        self,
        frame_count:    int,
        from_timestamp: int | None      = None,
        /,
        *,
        analysis:       Literal[False]  = False
        ) -> ndarray[tuple[int, int], np.dtype[np.int16]]:
        ...

    @overload
    async def read_async(
        self,
        frame_count:    int,
        from_timestamp: int | None      = None,
        /,
        *,
        analysis:       Literal[True]
        ) -> DetectionResult:
        ...

    async def read_async(
        self,
        frame_count:    int,
        from_timestamp: int | None = None,
        /,
        *,
        analysis:       bool       = False
        ) -> ndarray[tuple[int, int], np.dtype[np.int16]] | DetectionResult:
        """ Asynchronous version of read(). """
        return self.read(frame_count, from_timestamp, analysis=analysis)

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

        # Stop any recordings
        for recording in self._recordings:
            recording.stop()

    #
    # Simulator specific functionality, do not use these in your applications.
    #

    _is_running: bool = False
    """ (Simulator only) Indicates the current status. """

    _replay_attrs: dict[str, Any] | None = None
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

    _stim_frequency_bin_duration_us: int = _FRAME_TIME_US
    """
    (Simulator only) Duration of the smallest frequency bin for generating stim bursts
    that is supported by the system in microseconds (us).
    """

    _frame_duration_us: int
    """ (Simulator only) Time interval between frames in microseconds (us) based on _frames_per_second. """

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

    _instance: ClassVar[Neurons | None] = None
    """ (Simulator only) Singleton instance of Neurons for the simulator. """

    @classmethod
    def _get_instance(cls) -> Neurons:
        """ (Simulator only) Get the singleton instance of Neurons for the simulator. """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def _clear_instance(cls, force: bool = False) -> None:
        """
        (Simulator only) Clear the singleton instance of Neurons.

        Refuses to tear down an instance that is currently inside a
        `cl.open()` / `with Neurons(...)` context, since doing so would pull
        the running data-producer subprocess out from under any in-flight
        reads/stims. Pass `force=True` only from process-shutdown hooks
        (e.g. `atexit`) where the context cannot be exited cleanly.
        """
        if cls._instance is None:
            return

        instance = cls._instance
        if instance.has_started() and not force:
            raise RuntimeError(
                "Cannot clear the active Neurons instance while inside a "
                "cl.open()/Neurons context. Exit the context first "
                "(e.g. leave the `with cl.open() as neurons:` block) before "
                "reconfiguring the simulator (e.g. via "
                "cl.sim.set_simulator_data_source / clear_simulator_data_source)."
            )

        instance.close()

        # Stop WebSocket server subprocess first (it reads from the buffer)
        instance._stop_websocket_server()

        # Stop heartbeat thread
        instance._stop_heartbeat_thread()

        # Stop the data producer subprocess (always, even if not started via has_started)
        if instance._data_producer is not None:
            instance._data_producer.stop()
            instance._data_producer    = None
            instance._shared_buffer    = None
            instance._producer_started = False

        del cls._instance
        cls._instance = None

        atexit.unregister(cls._clear_instance)

    def _sleep_until(self, timestamp: int) -> None:
        """
        (Simulator only) Block the thread until the specified timestamp is reached.

        Args:
            timestamp: wake up the system at this timestamp (external 25kHz)
        """
        assert self._shared_buffer is not None

        # Calculate how many frames we need to wait (in external 25kHz space)
        current_timestamp = self.timestamp()
        frames_to_wait    = timestamp - current_timestamp

        if frames_to_wait <= 0:
            return  # Already past target timestamp

        wait_secs = frames_to_wait / _EXTERNAL_FRAMES_PER_SECOND

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

        All parameters are in internal 50kHz timestamps. Returned spikes have
        timestamps converted to external 25kHz.

        Args:
            frame_count: Number of internal frames to consider for reading spikes.
            from_timestamp: Read from a specific internal timestamp. If None, return
                from the current timestamp.

        Returns:
            List of spikes found within the given number of frames.
        """
        # Calculate required timestamps (internal 50kHz)
        now               = self._internal_timestamp()
        from_timestamp    = now if from_timestamp is None else from_timestamp
        to_timestamp      = from_timestamp + frame_count

        # Convert internal 50kHz → buffer 25kHz for the shared buffer read
        assert self._shared_buffer is not None
        spike_records = self._shared_buffer.read_spikes(
            _to_external_ts(from_timestamp), _to_external_ts(to_timestamp)
        )
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
        return read_spikes

    def _stim_record_from_op(self, stim_op: _StimOp, ext_ts: int) -> StimRecord:
        phase_durations_us, phase_currents_uA, phase_count = stim_op.stim_design._get_padded()
        return StimRecord(
            timestamp          = ext_ts,
            intended_timestamp = ext_ts,
            channel            = stim_op.channel,
            phase_count        = phase_count,
            phase_durations_us = phase_durations_us,
            phase_currents_uA  = phase_currents_uA,
        )

    def _publish_stims_to_buffer_until(self, to_timestamp: int) -> None:
        """Publish queued stims up to an internal 50 kHz timestamp."""
        if self._shared_buffer is None:
            return
        if to_timestamp <= self._stim_buffer_write_ts:
            return
        self._read_stims(
            self._stim_buffer_write_ts,
            to_timestamp,
            write_to_buffer=True,
        )

    def _read_stims(
        self,
        from_timestamp  : int,
        to_timestamp    : int,
        write_to_buffer : bool = False,
        ) -> list[Stim]:
        """
        (Simulator only) Read stims within a timestamp range.

        All parameters and internal state use internal 50kHz timestamps.
        Returned stims have timestamps converted to external 25kHz.

        When `write_to_buffer` is True, stims whose internal timestamp is at or
        past `_stim_buffer_write_ts` are written to the shared buffer's stim
        ring, and the high-water mark advances to prevent double-writes from
        overlapping `read()` calls.

        Retention: the queue is pruned to match the shared buffer's 5-second
        ring window so memory stays bounded and backward reads behave
        consistently with the old shared-buffer implementation.
        """
        # Fast path: nothing queued, no rebuild pending. Avoid lock acquire + prune scan
        # which is otherwise paid every accelerated-loop tick.
        if (
            not self._rebuild_pending
            and self._stim_queue._total_count == 0
        ):
            if not write_to_buffer or self._shared_buffer is None:
                return []
            with self._stim_lock:
                if (
                    not self._rebuild_pending
                    and self._stim_queue._total_count == 0
                ):
                    ext_to = _to_external_ts(to_timestamp)
                    if ext_to > self._shared_buffer.stim_write_timestamp:
                        self._shared_buffer.stim_write_timestamp = ext_to
                    if to_timestamp > self._stim_buffer_write_ts:
                        self._stim_buffer_write_ts = to_timestamp
                    return []

        with self._stim_lock:
            # If a rebuild was deferred (by interrupt()), run it now — before any
            # stim data is returned — so that the current plan's stim/sync
            # operations are included in the mini-sim.
            if self._rebuild_pending:
                self._rebuild_pending = False
                self._rebuild_stim_queue()
            # Prune stims that have fallen outside the shared buffer's retention window,
            # and collect the in-range entries — in a single pass over the channel dict.
            # Buffer timestamps are 25kHz; stim queue timestamps are internal 50kHz.
            if self._shared_buffer is not None:
                oldest_valid = _to_internal_ts(self._shared_buffer.write_timestamp - self._shared_buffer.buffer_duration_frames)
                prune_before = max(oldest_valid, self._start_timestamp)
            else:
                prune_before = self._start_timestamp

            # Bound the rebuild-history records even when no rebuild fires. A
            # rebuild (and the compaction it triggers) only runs for multi-channel
            # stims or interrupts; a workload of purely single-channel stims would
            # otherwise grow _stim_op_records / _interrupt_records for the entire
            # session. Each retained record holds a ChannelSet (a 64-element numpy
            # array), so unbounded growth leaks memory and steadily lengthens the
            # main process's gen-2 GC pauses — the gradual slow-down seen over long
            # sessions. Records older than the ring retention window can never be
            # the target of a future interrupt-driven rebuild, so pruning them is
            # safe. Cadenced to once per simulated second to keep per-tick cost
            # negligible.
            if (
                not self._rebuilding
                and self._stim_op_records
                and to_timestamp - self._last_history_compact_ts >= self._frames_per_second
            ):
                self._last_history_compact_ts = to_timestamp
                self._compact_stim_history(prune_before)

            entries = self._stim_queue.prune_and_get_range(prune_before, from_timestamp, to_timestamp)
            if not entries:
                # Still advance the stim write cursor so the recording process
                # knows it's safe to read up to this point (no stims to write).
                if write_to_buffer and self._shared_buffer is not None:
                    ext_to = _to_external_ts(to_timestamp)
                    if ext_to > self._shared_buffer.stim_write_timestamp:
                        self._shared_buffer.stim_write_timestamp = ext_to
                    if to_timestamp > self._stim_buffer_write_ts:
                        self._stim_buffer_write_ts = to_timestamp
                return []

            # Convert internal 50kHz stim timestamps to external 25kHz and
            # optionally write StimRecords to the shared buffer.
            read_stims           : list[Stim]       = []
            stim_records_to_write: list[StimRecord] = []
            for _ts, _ch, stim_op in entries:
                if not isinstance(stim_op, _StimOp):
                    continue
                ext_ts = _to_external_ts(stim_op.timestamp)
                read_stims.append(Stim(timestamp=ext_ts, channel=stim_op.channel))
                if write_to_buffer and self._shared_buffer is not None:
                    # Only write stims we haven't written before (high-water mark)
                    if _ts >= self._stim_buffer_write_ts:
                        stim_records_to_write.append(
                            self._stim_record_from_op(stim_op, ext_ts)
                        )

            if stim_records_to_write and self._shared_buffer is not None:
                self._shared_buffer.write_stims(stim_records_to_write)
                last_published_ts = max(rec.intended_timestamp for rec in stim_records_to_write)
                if (
                    self._last_published_stim_timestamp is None
                    or last_published_ts > self._last_published_stim_timestamp
                ):
                    self._last_published_stim_timestamp = last_published_ts

            # Advance the stim write cursor so the recording process knows
            # all stims up to this timestamp are in the ring buffer.
            if write_to_buffer and self._shared_buffer is not None:
                ext_to = _to_external_ts(to_timestamp)
                if ext_to > self._shared_buffer.stim_write_timestamp:
                    self._shared_buffer.stim_write_timestamp = ext_to
                if to_timestamp > self._stim_buffer_write_ts:
                    self._stim_buffer_write_ts = to_timestamp

            return read_stims

    def _compute_burst_params(
        self,
        stim_duration_us: int,
        burst_design:     BurstDesign | None,
    ) -> tuple[BurstDesign, np.ndarray, bool]:
        """
        (Simulator only) Compute burst timing parameters from stim duration and optional burst design.

        Returns:
            burst_design:        Resolved BurstDesign (original or synthesized single-burst).
            burst_timestamps:    Array of burst boundary frames relative to stim start.
            has_trailing_offset: True if final burst has sub-frame remainder.
        """
        if burst_design is not None:
            total_burst_duration_us     = (burst_design._burst_count + 1) * burst_design._burst_interval_us
            burst_times_us              = np.arange(0, total_burst_duration_us, step=burst_design._burst_interval_us)
            burst_timestamps, remainder = np.divmod(burst_times_us, self._frame_duration_us)
            has_trailing_offset         = remainder[-1] != 0
        else:
            total_stim_duration_us = stim_duration_us + _MINIMUM_LEAD_TIME_US
            single_burst_hz        = min(1_000_000 / total_stim_duration_us, BurstDesign._BURST_FREQUENCY_LIMIT_HZ)
            burst_design           = BurstDesign(1, single_burst_hz)
            burst_timestamps       = np.array([0, burst_design._burst_interval_us // self._frame_duration_us])
            has_trailing_offset    = False
        return burst_design, burst_timestamps, has_trailing_offset

    def _compute_burst_schedule(
        self,
        stim_duration_us: int,
        burst_design:     BurstDesign | None,
    ) -> tuple[int, int]:
        """
        Return the burst count and interval without allocating every boundary
        timestamp. The rebuild simulator still uses _compute_burst_params().
        """
        if burst_design is not None:
            return burst_design._burst_count, burst_design._burst_interval_us

        total_stim_duration_us = stim_duration_us + _MINIMUM_LEAD_TIME_US
        single_burst_hz        = min(1_000_000 / total_stim_duration_us, BurstDesign._BURST_FREQUENCY_LIMIT_HZ)
        burst_interval_us      = int((1_000_000 / single_burst_hz / _FRAME_TIME_US) + 0.5) * _FRAME_TIME_US
        return 1, burst_interval_us

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
        if isinstance(channel_set, int):
            if channel_set < 0 or channel_set >= ChannelSet._CHANNELS_TOTAL:
                raise ValueError(f"Channel number {channel_set} is out of range")
            if channel_set in _GROUNDED_CHANNEL_SET:
                return
            unfiltered_channels = (channel_set,)
            stimmable_channels  = (channel_set,)
        elif isinstance(channel_set, ChannelSet):
            unfiltered_channels = tuple(channel_set._tolist())
            stimmable_channels  = tuple(ch for ch in unfiltered_channels if ch not in _GROUNDED_CHANNEL_SET)
            if not stimmable_channels:
                return
        else:
            raise ValueError(
                f"channel_set must be "
                f"ChannelSet object or an int, "
                f"not {channel_set.__class__.__name__}"
                )

        channel_count             = len(stimmable_channels)
        unfiltered_channel_count  = len(unfiltered_channels)
        unfiltered_channel_set    = ChannelSet(*unfiltered_channels)

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

        # Check BurstDesign
        if burst_design is not None and not isinstance(burst_design, BurstDesign):
            raise ValueError(
                f"burst_design must be a "
                f"BurstDesign object, "
                f"not {burst_design.__class__.__name__}"
                )

        # Specify stimulation constraints
        minimum_lead_time_frames  = _MINIMUM_LEAD_TIME_US // _FRAME_TIME_US
        minimum_burst_interval_us = _MINIMUM_LEAD_TIME_US + stim_design.duration_us

        # Check that stimulation constraints have been met
        if lead_time_us < _MINIMUM_LEAD_TIME_US:
            raise ValueError(f"lead_time_us must be at least {_MINIMUM_LEAD_TIME_US}")

        if not lead_time_us % _FRAME_TIME_US == 0:
            raise ValueError(f"lead_time_us must be evenly divisible by {_FRAME_TIME_US}")

        if burst_design is not None and burst_design._burst_interval_us < minimum_burst_interval_us:
            raise ValueError(
                f"Burst interval {burst_design._burst_interval_us} us "
                f"must be at least {_MINIMUM_LEAD_TIME_US} us "
                f"+ duration {stim_design.duration_us}"
                )

        stim_duration_us = stim_design.duration_us
        lead_time_frames = lead_time_us // self._frame_duration_us

        with self._stim_lock:
            # Record this operation for potential rebuild-on-interrupt.
            # Store the original (unfiltered) channel_set so the rebuild can add
            # sync opcodes for grounded channels, matching the reference simulator
            # which adds syncs for ALL channels but only stim ops for stimmable ones.
            if not self._rebuilding:
                if unfiltered_channel_count > 1:
                    self._stim_history_has_rebuild_ops = True
                    self._rebuild_affected_channels.update(unfiltered_channels)
                self._stim_op_records.append(_StimOpRecord(
                    from_timestamp = from_timestamp,
                    channel_set    = unfiltered_channel_set,
                    stim_design    = stim_design,
                    burst_design   = burst_design,
                    lead_time_us   = lead_time_us,
                ))

            burst_count, burst_interval_us = self._compute_burst_schedule(stim_duration_us, burst_design)

            # If channel_set has more than one channel, insert a sync() before the stims.
            # Use from_timestamp (the plan's timestamp) for the sync lower bound, rather than
            # the current wall-clock time.
            if channel_count > 1:
                self._sync_channels(from_timestamp, ChannelSet(*stimmable_channels), record=False)

            for stim_channel in stimmable_channels:
                free_ts                  = self._stim_channel_available_from[stim_channel]
                is_available             = from_timestamp > free_ts
                start_offset             = (from_timestamp if is_available else free_ts) + lead_time_frames
                next_available_ts        = int(free_ts)
                queued_stims: list[tuple[int, _StimOp]] = []
                for i in range(burst_count):
                    stim_start_ts = start_offset + (i * burst_interval_us) // self._frame_duration_us
                    # Add a delay if we are in a burst, which is equivalent to a
                    # direct swap should a new stim command be called immediately
                    # following interrupt. The minimum lead time is subtracted.
                    next_available_ts = (
                        start_offset
                        + ((i + 1) * burst_interval_us) // self._frame_duration_us
                        - minimum_lead_time_frames
                    )

                    queued_stims.append((
                        stim_start_ts,
                        _StimOp(
                            timestamp     = stim_start_ts,
                            channel       = stim_channel,
                            end_timestamp = next_available_ts,
                            stim_design   = stim_design,
                        ),
                    ))

                if queued_stims:
                    self._stim_queue.put_many_sorted(stim_channel, queued_stims)

                # We mark the channel busy from the stim timestamp for the
                # amount of time it takes to perform the stim
                self._stim_channel_available_from[stim_channel] = next_available_ts

        self._notify_stim_publisher()

    def _compact_stim_history(self, prune_before: int) -> None:
        """
        Prune stim-op and interrupt records whose outputs have been consumed.

        After each rebuild the stim queue reflects all operations to date, so
        records whose from_timestamp is older than the ring-buffer retention
        window are no longer needed for future rebuilds. Pruning bounds the
        list lengths to ~(event_rate × window_seconds) regardless of how long
        the simulation runs, eliminating the O(N^2) rebuild-scan growth.

        Saves the current channel-availability snapshot as a checkpoint so the
        next rebuild can start from the post-prune state instead of _start_timestamp.

        Constraints:
        - Must only be called while holding _stim_lock.
        - Must only be called immediately after _rebuild_stim_queue() completes,
          so _stim_channel_available_from reflects the current rebuilt state.
        - prune_before must be <= current simulation timestamp.
        """
        n_pruned = 0
        for record in self._stim_op_records:
            if record.from_timestamp < prune_before:
                n_pruned += 1
            else:
                break  # records are in chronological append order

        if n_pruned == 0:
            return

        # Save the post-rebuild availability as the new starting point for
        # future rebuilds, but clamp each channel to prune_before. The retained
        # records (from_timestamp >= prune_before) are re-simulated on top of
        # this checkpoint by the next rebuild; if the checkpoint still carried
        # their cumulative availability, those records would be applied twice and
        # the inflated floor would drag a not-yet-published burst forward past
        # the publish head, re-writing it as a "phantom" duplicate. Clamping to
        # prune_before keeps only the pruned records' contribution: any retained
        # op (and any interrupt at/after prune_before) re-establishes the correct
        # availability during re-simulation, while present-time operations are
        # unaffected because they always start after prune_before.
        self._rebuild_checkpoint_avail = np.minimum(
            self._stim_channel_available_from, prune_before
        )

        del self._stim_op_records[:n_pruned]

        # Drop interrupt records older than prune_before and re-index the rest.
        # All retained records have from_timestamp >= prune_before and therefore
        # stim_op_index >= n_pruned, so subtracting n_pruned gives a valid
        # non-negative index into the pruned list.
        self._interrupt_records = [
            (ts, chs, max(0, idx - n_pruned))
            for ts, chs, idx in self._interrupt_records
            if ts >= prune_before
        ]
        _logger.debug(
            "_compact_stim_history: pruned %d ops, %d interrupt records remain",
            n_pruned,
            len(self._interrupt_records),
        )

    def _rebuild_stim_queue(self) -> None:
        """
        (Simulator only) Rebuild the stim queue by simulating per-channel opcode
        queues with shared sync barriers, matching the reference simulator's lazy
        barrier resolution semantics.

        This is called after an interrupt changes a channel's availability,
        which may invalidate sync timestamps that were eagerly resolved.
        """
        # Phase 1: Determine which channels are affected and build their opcode queues
        channel_queues, interrupt_opcodes, affected_chs = self._build_rebuild_opcodes()

        if not affected_chs:
            # No channels affected — nothing to rebuild
            return

        # Phase 2: Event-driven simulation only on affected channels
        result_stims, ch_completed_at = self._run_rebuild_simulation(
            channel_queues, interrupt_opcodes, affected_chs
        )

        # Phase 3: Queue computed stims and sync data producer — only for affected channels
        _checkpoint = self._rebuild_checkpoint_avail
        for ch in affected_chs:
            stim_list = self._stim_queue._channel_stims.get(ch)
            if stim_list:
                self._stim_queue._total_count -= len(stim_list)
                self._stim_queue._channel_stims[ch] = []
            # Use checkpoint as the starting floor so the simulation result is
            # interpreted relative to the state captured after the last prune,
            # not relative to the very beginning of the session.
            self._stim_channel_available_from[ch] = (
                int(_checkpoint[ch]) if _checkpoint is not None else self._start_timestamp
            )

        result_stims.sort(key=lambda item: (item[0], item[1], item[2]))
        for stim_start_ts, channel, next_available_ts, stim_design in result_stims:
            self._stim_queue.put_sorted(
                timestamp = stim_start_ts,
                channel   = channel,
                payload   = _StimOp(
                    timestamp     = stim_start_ts,
                    channel       = channel,
                    end_timestamp = next_available_ts,
                    stim_design   = stim_design,
                ),
            )
            self._stim_channel_available_from[channel] = max(
                self._stim_channel_available_from[channel],
                next_available_ts,
            )

        for ch in affected_chs:
            self._stim_channel_available_from[ch] = max(
                self._stim_channel_available_from[ch],
                ch_completed_at[ch],
            )

        # Compact history: prune records that are now behind the ring buffer's
        # retention window and save a checkpoint so the next rebuild can start
        # from the post-prune availability state instead of _start_timestamp.
        if self._shared_buffer is not None:
            _oldest_valid = _to_internal_ts(
                self._shared_buffer.write_timestamp - self._shared_buffer.buffer_duration_frames
            )
            _prune_before = max(_oldest_valid, self._start_timestamp)
            self._compact_stim_history(_prune_before)
        else:
            # No shared buffer (unit tests / offline use) — just save the
            # checkpoint without pruning so future rebuilds use it as the floor.
            self._rebuild_checkpoint_avail = self._stim_channel_available_from.copy()

        # After a rebuild all sync barriers have been resolved — the queue now
        # contains stims at definite timestamps.  Future single-channel
        # interrupts only replace individual stim entries and cannot break a
        # sync barrier, so they do not need a rebuild.  Clearing the set
        # prevents high-frequency single-channel stim updates (rate encoding)
        # on channels that ALSO appear in multi-channel feedback operations
        # from triggering a rebuild every tick.  The set is repopulated by
        # _queue_stims() / _sync_channels() when the next multi-channel
        # operation is submitted.
        self._rebuild_affected_channels.clear()

    def _get_affected_channels(self) -> set[int]:
        """
        Determine which channels are affected by recorded interrupts.

        A channel is affected if it is directly interrupted, or if it participates
        in an operation (stim, sync) that shares channels with a directly or
        transitively affected channel. This is a transitive closure over shared
        operations.
        """
        # Start with directly interrupted channels, but only those that
        # participate in sync/multi-channel operations.  Interrupts on channels
        # that have never been part of such operations are fully handled by the
        # direct stim queue modification in _interrupt_queued_stims() and do
        # not need rebuild processing.  Including them here would pull all
        # their stim records into the rebuild, degrading performance over time
        # (e.g. high-frequency single-channel stim updates in rate encoding).
        dirty: set[int] = set()
        for _, int_channels, _ in self._interrupt_records:
            dirty.update(ch for ch in int_channels if ch in self._rebuild_affected_channels)

        if not dirty and self._stim_history_has_rebuild_ops:
            for record in self._stim_op_records:
                channels = record.channel_set._tolist()
                if isinstance(record, _SyncOpRecord) or len(channels) > 1:
                    dirty.update(channels)

        if not dirty:
            return dirty

        # Expand through operations that share channels with dirty set
        # Each operation's channel set is a group — if any member is dirty, all are dirty
        op_channel_groups: list[set[int]] = []
        for record in self._stim_op_records:
            op_channel_groups.append(set(record.channel_set._tolist()))

        # Transitive closure: keep expanding until stable
        changed = True
        while changed:
            changed = False
            for group in op_channel_groups:
                if group & dirty and not group <= dirty:
                    dirty   |= group
                    changed  = True

        return dirty

    def _build_rebuild_opcodes(self) -> tuple[list[deque], list[_RebuildInterrupt], set[int]]:
        """
        Phase 1 of stim queue rebuild: build per-channel opcode queues from
        recorded stim operations and interrupt records.

        Only builds queues for affected channels (those directly or transitively
        connected to interrupted channels via shared operations).

        Returns:
            channel_queues:    per-channel opcode deques (only populated for affected channels)
            interrupt_opcodes: list of interrupt opcodes
            affected_chs:      set of channel indices that need rebuilding
        """
        affected_chs = self._get_affected_channels()

        channel_queues   : list[deque]             = [deque() for _ in range(64)]
        interrupt_opcodes: list[_RebuildInterrupt] = []

        if not affected_chs:
            return channel_queues, interrupt_opcodes, affected_chs

        interrupt_iter = iter(self._interrupt_records)
        next_int       = next(interrupt_iter, None)

        prev_from_ts   : dict[int, int | None] = {}
        current_plan_ts: int | None            = None

        for op_idx, record in enumerate(self._stim_op_records):
            # Insert any interrupts that fire before this operation
            while next_int is not None and next_int[2] <= op_idx:
                int_from_ts, int_channels, _ = next_int
                # Only create interrupt opcodes for channels in the affected set.
                # Non-affected channel interrupts are handled directly by
                # _interrupt_queued_stims() and don't need rebuild simulation.
                # Including them would cause Phase 2 to visit every interrupt
                # timestamp, re-processing sync-blocked channels at each one.
                affected_int_chs = set(ch for ch in int_channels if ch in affected_chs)
                if affected_int_chs:
                    int_opcode = _RebuildInterrupt(int_from_ts, affected_int_chs)
                    interrupt_opcodes.append(int_opcode)
                    for ch in affected_int_chs:
                        if prev_from_ts.get(ch) != int_from_ts:
                            channel_queues[ch].append(_RebuildWaitUntil(int_from_ts))
                            prev_from_ts[ch] = int_from_ts
                        channel_queues[ch].append(int_opcode)
                current_plan_ts = int_from_ts
                next_int        = next(interrupt_iter, None)

            channels = record.channel_set._tolist()

            # Skip operations that don't involve any affected channel
            if not any(ch in affected_chs for ch in channels):
                continue

            effective_from = max(current_plan_ts, record.from_timestamp) if current_plan_ts is not None else record.from_timestamp
            for ch in channels:
                if ch not in affected_chs:
                    continue
                if prev_from_ts.get(ch) != effective_from:
                    channel_queues[ch].append(_RebuildWaitUntil(effective_from))
                    prev_from_ts[ch] = effective_from

            if isinstance(record, _SyncOpRecord):
                affected_in_sync = [ch for ch in channels if ch in affected_chs]
                sync_op = _RebuildSync(len(affected_in_sync))
                for ch in affected_in_sync:
                    channel_queues[ch].append(sync_op)
            else:
                self._append_rebuild_stim_ops(channel_queues, channels, record, affected_chs)

        # Insert any remaining interrupts
        while next_int is not None:
            int_from_ts, int_channels, _ = next_int
            affected_int_chs = set(ch for ch in int_channels if ch in affected_chs)
            if affected_int_chs:
                int_opcode = _RebuildInterrupt(int_from_ts, affected_int_chs)
                interrupt_opcodes.append(int_opcode)
                for ch in affected_int_chs:
                    if prev_from_ts.get(ch) != int_from_ts:
                        channel_queues[ch].append(_RebuildWaitUntil(int_from_ts))
                        prev_from_ts[ch] = int_from_ts
                    channel_queues[ch].append(int_opcode)
            next_int = next(interrupt_iter, None)

        return channel_queues, interrupt_opcodes, affected_chs

    def _append_rebuild_stim_ops(
        self,
        channel_queues: list[deque],
        channels:       list[int],
        record:         _StimOpRecord,
        affected_chs:   set[int],
    ) -> None:
        """Append stim opcodes to per-channel queues for a single stim operation record."""
        stim_duration_us     = record.stim_design.duration_us
        stim_duration_frames = (stim_duration_us // self._frame_duration_us // _INTERNAL_RATE_MULTIPLIER) * _INTERNAL_RATE_MULTIPLIER
        lead_time_frames     = record.lead_time_us // self._frame_duration_us

        burst_design, burst_timestamps, _ = self._compute_burst_params(
            stim_duration_us, record.burst_design
        )
        burst_count = burst_design._burst_count

        # Multi-channel stim → implicit sync for ALL affected channels (including
        # grounded), matching the reference simulator which adds a sync
        # to every channel but only adds stim ops to stimmable ones.
        affected_in_op = [ch for ch in channels if ch in affected_chs]
        if len(channels) > 1 and affected_in_op:
            sync_op = _RebuildSync(len(affected_in_op))
            for ch in affected_in_op:
                channel_queues[ch].append(sync_op)

        # Stim ops only for stimmable (non-grounded) affected channels
        stimmable_channels = [ch for ch in channels if ch not in _GROUNDED_CHANNEL_SET and ch in affected_chs]
        lead_delay_steps   = lead_time_frames - _NOTICE_STEPS
        burst_period_steps = int(burst_timestamps[1] - burst_timestamps[0])
        _notice = _RebuildStimNotice()  # reuse single instance
        for ch in stimmable_channels:
            if lead_delay_steps > 0:
                channel_queues[ch].append(_RebuildDelay(lead_delay_steps))

            for _ in range(burst_count):
                channel_queues[ch].append(_notice)
                channel_queues[ch].append(_RebuildStim(
                    duration_steps = burst_period_steps - _NOTICE_STEPS,
                    stim_duration  = stim_duration_frames,
                    channel        = ch,
                    stim_design    = record.stim_design,
                ))

    def _run_rebuild_simulation(
        self,
        channel_queues:    list[deque],
        interrupt_opcodes: list[_RebuildInterrupt],
        affected_chs:      set[int],
    ) -> tuple[list[tuple[int, int, int, StimDesign]], list[int]]:
        """
        Phase 2 of stim queue rebuild: event-driven simulation of per-channel
        opcode queues with shared sync barriers.

        Uses a min-heap to advance directly to the next active time step,
        skipping idle channels.  Sync-blocked channels are re-checked each
        step until their barrier resolves.

        Only simulates channels in affected_chs.

        Returns:
            result_stims:    list of (stim_start_ts, channel, next_available_ts)
            ch_completed_at: per-channel final timeline position
        """
        start_ts   = self._start_timestamp
        _checkpoint = self._rebuild_checkpoint_avail

        # If a checkpoint exists (post-prune availability snapshot), start each
        # channel's timeline from the checkpoint rather than _start_timestamp.
        # This is equivalent to re-simulating from scratch because the checkpoint
        # was saved immediately after the previous full rebuild.
        ch_completed_at = [
            int(_checkpoint[ch]) if _checkpoint is not None else start_ts
            for ch in range(64)
        ]
        ch_can_int      = [True]  * 64

        ch_int_to:      list[_RebuildInterrupt | None] = [None] * 64
        ch_active_sync: list[_RebuildSync      | None] = [None] * 64

        result_stims: list[tuple[int, int, int, StimDesign]] = []

        # Build interrupt index: timestamp → list of (index, opcode)
        interrupt_at_ts: defaultdict[int, list[tuple[int, _RebuildInterrupt]]] = defaultdict(list)
        for idx, int_op in enumerate(interrupt_opcodes):
            interrupt_at_ts[int_op.time_step].append((idx, int_op))
        fired_interrupts: set[int] = set()

        # Sorted unique interrupt timestamps for advancing time
        sorted_int_ts = sorted(set(int_op.time_step for int_op in interrupt_opcodes))
        int_ptr = 0

        # -- Event scheduling --
        # ch_wake[ch] tracks the channel's current state:
        #   >= 0:  scheduled to wake at that time step
        #   _CHANNEL_COMPLETE (-1): finished all opcodes
        #   _SYNC_ACTIVE (-2):     blocked on a sync barrier
        ch_wake     : list[int]             = [_CHANNEL_COMPLETE] * 64
        event_heap  : list[tuple[int, int]] = []
        sync_blocked: set[int]              = set()

        for ch in affected_chs:
            if channel_queues[ch]:
                ch_start    = ch_completed_at[ch]  # checkpoint floor or start_ts
                ch_wake[ch] = ch_start
                heapq.heappush(event_heap, (ch_start, ch))

        # -- Inner helpers (closures over simulation state) --

        def process_oob(ch: int, time_step: int) -> int | None:
            queue = channel_queues[ch]
            new_ts: int | None = None

            if ch_int_to[ch] is not None and ch_can_int[ch]:
                target = ch_int_to[ch]
                while queue:
                    opcode = queue.popleft()
                    if opcode is target:
                        ch_int_to[ch]      = None
                        ch_active_sync[ch] = None
                        new_ts             = time_step
                        break
                    if isinstance(opcode, _RebuildSync):
                        opcode.channel_count -= 1

            while queue:
                front = queue[0]
                match front:
                    case _RebuildSync():
                        queue.popleft()
                        front.channel_count -= 1
                        ch_active_sync[ch]   = front
                    case _RebuildWaitUntil():
                        queue.popleft()
                        if front.time_step > time_step:
                            return front.time_step
                    case _:
                        break

            return new_ts

        def step_ch(ch: int, time_step: int) -> tuple[int, _RebuildStim | None]:
            queue = channel_queues[ch]

            if (sync := ch_active_sync[ch]) is not None:
                if sync.channel_count > 0:
                    return _SYNC_ACTIVE, None
                ch_active_sync[ch] = None

            if not queue:
                ch_completed_at[ch] = time_step
                return _CHANNEL_COMPLETE, None

            opcode = queue.popleft()
            ch_can_int[ch] = False

            match opcode:
                case _RebuildStimNotice():
                    return time_step + _NOTICE_STEPS, None
                case _RebuildStim():
                    ch_can_int[ch] = True
                    return time_step + opcode.duration_steps, opcode
                case _RebuildDelay():
                    ch_can_int[ch] = True
                    return time_step + opcode.duration_steps, None
                case _:
                    raise ValueError(f"Unexpected opcode in step: {opcode}")

        # -- Main simulation loop --
        prev_time_step = start_ts - 1

        while True:
            # Purge stale heap entries (from channels that were rescheduled)
            while event_heap and ch_wake[event_heap[0][1]] != event_heap[0][0]:
                heapq.heappop(event_heap)

            # Determine next time step from heap and unfired interrupts
            heap_ts = event_heap[0][0] if event_heap else None

            while int_ptr < len(sorted_int_ts) and sorted_int_ts[int_ptr] <= prev_time_step:
                int_ptr += 1
            next_int_ts = sorted_int_ts[int_ptr] if int_ptr < len(sorted_int_ts) else None

            candidates = [t for t in (heap_ts, next_int_ts) if t is not None]
            if not candidates:
                break

            time_step      = min(candidates)
            prev_time_step = time_step

            # Pop all channels waking at this time step
            active: list[int] = []
            while event_heap and event_heap[0][0] == time_step:
                _, ch = heapq.heappop(event_heap)
                if ch_wake[ch] == time_step:
                    active.append(ch)

            # Fire interrupts at this time step
            if time_step in interrupt_at_ts:
                for idx, int_op in interrupt_at_ts[time_step]:
                    if idx not in fired_interrupts:
                        fired_interrupts.add(idx)
                        for ch in int_op.channels:
                            if ch in affected_chs:
                                ch_int_to[ch] = int_op

            # Channels to process: heap-woken + sync-blocked
            process_chs = active + list(sync_blocked)

            # -- OOB pass --
            oob_deferred: set[int] = set()
            for ch in process_chs:
                new_nts = process_oob(ch, time_step)
                if new_nts is not None and new_nts > time_step:
                    oob_deferred.add(ch)
                    sync_blocked.discard(ch)
                    ch_wake[ch] = new_nts
                    heapq.heappush(event_heap, (new_nts, ch))

            # -- Step pass --
            new_sync: set[int] = set()
            for ch in process_chs:
                if ch in oob_deferred:
                    continue

                new_nts, stim_op = step_ch(ch, time_step)
                if stim_op is not None:
                    result_stims.append((
                        time_step,
                        stim_op.channel,
                        time_step + stim_op.duration_steps,
                        stim_op.stim_design,
                    ))

                if new_nts == _SYNC_ACTIVE:
                    ch_wake[ch] = _SYNC_ACTIVE
                    new_sync.add(ch)
                elif new_nts == _CHANNEL_COMPLETE:
                    ch_wake[ch] = _CHANNEL_COMPLETE
                else:
                    ch_wake[ch] = new_nts
                    heapq.heappush(event_heap, (new_nts, ch))

            sync_blocked = new_sync

        return result_stims, ch_completed_at

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

        with self._stim_lock:
            # Record this interrupt for rebuild, but only after stim operations
            # have been queued (skip setup/reset interrupts that clear channels
            # before any stim operations exist).
            # Store the current stim_op_records length so the rebuild can interleave
            # this interrupt at the correct position relative to stim operations.
            if not self._rebuilding and self._stim_op_records:
                self._interrupt_records.append((from_timestamp, interrupt_channels, len(self._stim_op_records)))

            # Before removing stims, get the last kept stim's end timestamp for each channel
            for channel in interrupt_channels:
                last_kept = self._stim_queue.get_last_entry_before(channel, from_timestamp)
                if last_kept is not None:
                    _, stim_op = last_kept
                    self._stim_channel_available_from[channel] = stim_op.end_timestamp
                else:
                    self._stim_channel_available_from[channel] = from_timestamp

            # Remove stims at or after from_timestamp for the specified channels
            self._stim_queue.interrupt_channels(interrupt_channels, from_timestamp, return_removed=False)

            # Defer the rebuild until the next read() so that all stim/sync
            # operations from the current plan are included.  This allows the
            # mini-sim to process consecutive syncs in a single pass, matching
            # the reference simulator which consumes sequences of OpcodeSync
            # opcodes as a batch and only blocks the channel on the last one.
            #
            # The rebuild is only needed when sync operations or multi-channel
            # stims exist — these can have eagerly-resolved timestamps that
            # become invalid after an interrupt changes channel availability.
            # For simple single-channel stims, _queue_stims places stims
            # correctly using real-time availability, so no rebuild is needed.
            if not self._rebuilding and self._stim_op_records and self._stim_history_has_rebuild_ops:
                if self._rebuild_affected_channels & set(interrupt_channels):
                    self._rebuild_pending = True
        self._notify_stim_publisher()

    def _sync_channels(
        self,
        from_timestamp: int,
        channel_set   : ChannelSet,
        /,
        record        : bool = True
        ) -> None:
        """
        (Simulator only) Align channel availability to a common timestamp, which is
        the latest (maximum) availability timestamp of the specified channels.

        Args:
            from_timestamp: Timestamp after which sync should be performed.
            channel_set:    One or more channels to sync.
            record:         Whether to record this sync operation for rebuild.
        """
        with self._stim_lock:
            # Record standalone sync operations for rebuild.  Syncs that are
            # implicit within multi-channel stims (_queue_stims) pass _record=False
            # because they are captured by the _StimOpRecord instead.
            if record and not self._rebuilding:
                self._stim_history_has_rebuild_ops = True
                self._rebuild_affected_channels.update(channel_set._tolist())
                self._stim_op_records.append(_SyncOpRecord(
                    from_timestamp = from_timestamp,
                    channel_set    = channel_set,
                ))
            sync_channels  = channel_set._tolist()

            # Find the max availability timestamp across the channels to sync, but only include channels within sync_channels that have stims scheduled
            sync_timestamp = from_timestamp

            for ch in sync_channels:
                if self._stim_queue.get_last_timestamp_for_channel(ch) is not None:
                    available = int(self._stim_channel_available_from[ch])
                    if available > sync_timestamp:
                        sync_timestamp = available

            for ch in sync_channels:
                self._stim_channel_available_from[ch] = sync_timestamp

            self._stim_channel_available_from[sync_channels] = sync_timestamp

    def _start_heartbeat_thread(self) -> None:
        """
        (Simulator only) Start background thread that continuously updates heartbeat.

        This thread updates the heartbeat timestamp and publishes due stim events.
        When the debugger pauses the process, this thread also pauses, causing the
        heartbeat to go stale and triggering subprocesses to pause.
        """
        if self._heartbeat_thread is not None:
            return  # Already running

        self._heartbeat_stop_event = Event()
        self._stim_publish_event.clear()
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
        self._stim_publish_event.set()
        if self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=0.5)
        self._heartbeat_thread     = None
        self._heartbeat_stop_event = None

    def _heartbeat_loop(self) -> None:
        """
        (Simulator only) Background thread loop that updates heartbeat timestamp.

        Runs periodically to keep the heartbeat fresh. When the debugger pauses
        the process, this thread also pauses, causing the heartbeat to become stale.
        """
        if self._heartbeat_stop_event is None:
            return

        while not self._heartbeat_stop_event.is_set():
            try:
                if self._shared_buffer is not None:
                    self._shared_buffer.main_process_heartbeat_ns = time.perf_counter_ns()
                    # The loop owns stim publication while it is active so
                    # realtime consumers never see half-settled tick updates.
                    if not self._in_loop:
                        self._publish_due_stims_to_buffer()
            except Exception:
                _logger.exception("Error in heartbeat loop")
                pass  # Ignore errors in background thread

            self._stim_publish_event.clear()
            wait_s = self._stim_publish_wait_interval()
            self._stim_publish_event.wait(timeout=wait_s)

    def _publish_due_stims_to_buffer(self) -> None:
        """Publish queued stims that are due according to the shared buffer clock."""
        if self._shared_buffer is None:
            return

        self._publish_stims_to_buffer_until(_to_internal_ts(self._shared_buffer.write_timestamp))

    def _notify_stim_publisher(self) -> None:
        """Wake the heartbeat/publish thread after stim queue changes."""
        if not self._in_loop:
            self._stim_publish_event.set()

    def _stim_publish_wait_interval(self) -> float:
        """Return the next wait interval for heartbeat/stim publishing."""
        if self._shared_buffer is None or self._in_loop:
            return _HEARTBEAT_INTERVAL_S

        with self._stim_lock:
            next_stim_ts = self._stim_queue.peek_min_timestamp_from(self._stim_buffer_write_ts)

        if next_stim_ts is None:
            return _HEARTBEAT_INTERVAL_S

        current_ts = _to_internal_ts(self._shared_buffer.write_timestamp)
        frames_until = next_stim_ts - current_ts
        if frames_until <= 0:
            return 0.0

        seconds_until = frames_until / self._frames_per_second
        if seconds_until <= _STIM_PUBLISH_ACTIVE_WINDOW_S:
            return _STIM_PUBLISH_ACTIVE_INTERVAL_S

        return min(
            _HEARTBEAT_INTERVAL_S,
            max(_STIM_PUBLISH_ACTIVE_INTERVAL_S, seconds_until - _STIM_PUBLISH_ACTIVE_WINDOW_S),
        )

@dataclass(slots=True)
class _StimOpRecord:
    """
    (Simulator only) Records a _queue_stims call so it can be replayed
    when an interrupt changes channel availability.
    """
    from_timestamp: int
    channel_set:    ChannelSet
    stim_design:    StimDesign
    burst_design:   BurstDesign | None
    lead_time_us:   int

@dataclass(slots=True)
class _SyncOpRecord:
    """
    (Simulator only) Records a standalone _sync_channels call so it can be
    replayed during rebuild-on-interrupt.
    """
    from_timestamp: int
    channel_set:    ChannelSet

# -- Rebuild mini-simulator opcodes -------------------------------------------
# These lightweight opcode types mirror the reference simulator's opcode model
# and are used only during _rebuild_stim_queue to compute correct stim timestamps
# with lazy barrier sync resolution.

class _RebuildStimNotice:
    """Stim notice period (minimum lead time)."""
    __slots__ = ()

@dataclass(slots=True)
class _RebuildStim:
    """A stim opcode that records channel, duration and design for result collection."""
    duration_steps: int
    stim_duration:  int
    channel:        int
    stim_design:    StimDesign

@dataclass(slots=True)
class _RebuildDelay:
    """A delay opcode."""
    duration_steps: int

@dataclass(slots=True)
class _RebuildSync:
    """Shared mutable sync barrier — same object referenced by all participating channels."""
    channel_count: int

@dataclass(slots=True)
class _RebuildWaitUntil:
    """Wait until a specific time step before proceeding."""
    time_step: int

@dataclass(slots=True)
class _RebuildInterrupt:
    """An interrupt that fires at a specific time step on a set of channels."""
    time_step: int
    channels:  set[int]

class _StimOp:
    """ (Simulator only) Object representing a Stim Operation used in Neurons._stim_queue. """

    __slots__ = ("timestamp", "channel", "end_timestamp", "stim_design")

    timestamp: int
    """ Timestamp the stim is scheduled for in internal 50kHz time. """

    channel: int
    """ Channel the stim is scheduled on. """

    end_timestamp: int
    """
    Expected end timestamp of the stim, which is:
    1. Duration of the StimDesign if single stim, and
    2. Frequency delay in the case of stim bursts.
    """

    stim_design: StimDesign
    """StimDesign used for this individual stim event."""

    def __init__(self, timestamp: int, channel: int, end_timestamp: int, stim_design: StimDesign) -> None:
        self.timestamp     = int(timestamp)
        self.channel       = int(channel)
        self.end_timestamp = end_timestamp
        self.stim_design   = stim_design

    @property
    def stim(self) -> Stim:
        return Stim(self.timestamp, self.channel)

    def __repr__(self) -> str:
        return f"StimOp(timestamp={self.timestamp}, channel={self.channel}, end_timestamp={self.end_timestamp})"

    def __lt__(self, other: _StimOp) -> bool:
        """ (Simulator only) Compare two instances of StimOp for neurons._stim_queue. """
        assert isinstance(other, type(self)), \
            f"Cannot compare StimOp with {other.__class__.__name__}"
        if self.timestamp == other.timestamp:
            return self.channel < other.channel
        return self.timestamp < other.timestamp
