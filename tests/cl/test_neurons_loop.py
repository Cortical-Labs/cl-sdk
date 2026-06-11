import time
import os

import numpy as np

import pytest

from ..conftest import cleanup

# Disable websocket for tests, needs to be set before importing cl
os.environ["CL_SDK_VISUALISATION"] = "0"

import cl
from cl import Loop, LoopTick, DetectionResult

def test_neurons_timestamp():
    """
    Tests neurons.timestamp() when running in realtime mode.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0
        wait_secs = 1.0
        start_ts  = neurons.timestamp()
        neurons.read(int(neurons.get_frames_per_second() * wait_secs), start_ts)
        end_ts    = neurons.timestamp()
        duration_sec = (end_ts - start_ts) / neurons.get_frames_per_second()
        np.testing.assert_allclose(wait_secs, duration_sec, atol=0.1)

def test_neurons_timestamp_accelerated():
    """
    Tests neurons.timestamp() when running in accelerated mode. Here, timestamp
    should not advance simply by waiting.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0
        wait_secs = 1.0
        start_ts = neurons.timestamp()
        time.sleep(wait_secs)
        end_ts   = neurons.timestamp()
        duration_sec = (end_ts - start_ts) / neurons.get_frames_per_second()
        assert duration_sec == 0

# Read tolerance is 0 as it should always read the correct number of frames.
READ_FRAMES_TOL = 0

def test_neurons_read():
    """
    Tests neurons.read() and resulting timestamp alignment, which is is central
    to replaying a recording file using neurons.loop().

    In this test, we consider:
    - A normal read;
    - A read from > 5 secs in the past that will fail.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    with cl.open() as neurons:
        neurons.restart()

        # Test 1: Normal read
        frames_to_read = 2500
        start_ts       = neurons.timestamp()
        frames         = neurons.read(frames_to_read, None)

        calculated = start_ts + len(frames)
        expected   = start_ts + frames_to_read
        delta      = abs(calculated - expected)
        print(f"\nTest 1 - Normal read: {calculated=}, {expected=}, {delta=}")
        assert delta <= READ_FRAMES_TOL

        # Test 2: Reading from > 5 secs in the past
        with pytest.raises(Exception):
            neurons.read(frames_to_read, int(neurons.timestamp() - 5.5 * neurons.get_frames_per_second()))

        # Test 3: Reading from past
        ts_offset      = -2600
        frames_to_read = 2500

        start_ts       = neurons.timestamp()
        from_ts        = start_ts + ts_offset
        frames         = neurons.read(frames_to_read, from_ts)

        calculated = start_ts + frames_to_read + ts_offset
        expected   = from_ts + frames_to_read
        delta      = abs(calculated - expected)
        print(f"Test 3 - Reading from past: {calculated=}, {expected=}, {delta=}")
        assert delta <= READ_FRAMES_TOL

        # Test 4: Reading from future
        ts_offset      = 1000
        frames_to_read = 2500

        start_ts       = neurons.timestamp()
        from_ts        = start_ts + ts_offset
        frames         = neurons.read(frames_to_read, from_ts)

        calculated = start_ts + frames_to_read + ts_offset
        expected   = from_ts + frames_to_read
        delta      = abs(calculated - expected)
        print(f"Test 4 - Reading from future: {calculated=}, {expected=}, {delta=}")
        assert delta <= READ_FRAMES_TOL

def test_neurons_read_accelerated():
    """
    Tests neurons.read() and resulting timestamp alignment, which is is central
    to replaying a recording file using neurons.loop().

    In this test, we consider:
    - A normal read;
    - A read from > 5 secs in the past that will fail.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    with cl.open() as neurons:
        # Test 1: Normal read - verify timestamp advances by frames_to_read
        start_ts       = neurons.timestamp()
        frames_to_read = 2500
        neurons.read(frames_to_read, None)
        end_ts         = neurons.timestamp()

        # In accelerated mode, timestamp should advance exactly by frames read
        assert end_ts == start_ts + frames_to_read

        # Test 2: Reading from > 5 secs in the past (exceeds buffer capacity)
        with pytest.raises(Exception):
            neurons.read(frames_to_read, int(neurons.timestamp() - 5.5 * neurons.get_frames_per_second()))

        # Test 3: Reading from past (within buffer)
        start_ts       = neurons.timestamp()
        ts_offset      = -2600
        from_ts        = start_ts + ts_offset
        frames_to_read = 2500
        frames         = neurons.read(frames_to_read, from_ts)

        # Reading from past shouldn't advance timestamp since data already exists
        end_ts = neurons.timestamp()
        assert end_ts >= start_ts  # Timestamp should not go backward

        # Test 4: Reading from future
        start_ts       = neurons.timestamp()
        ts_offset      = 1000
        from_ts        = start_ts + ts_offset
        frames_to_read = 2500
        frames         = neurons.read(frames_to_read, from_ts)
        end_ts         = neurons.timestamp()

        # In accelerated mode, reading into future advances producer to meet demand
        assert end_ts >= from_ts + frames_to_read

def test_neurons_read_with_analysis():
    """
    This tests Neurons.loop(analysis=True) which returns DetectionResult
    """
    def _test():
        with cl.open() as neurons:
            stim_plan = neurons.create_stim_plan()
            stim_plan.stim(3, -1.0)

            timestamp        = neurons.timestamp()
            offset_frames    = 500
            num_frames       = offset_frames * 2
            stim_lead_frames = 2
            stim_plan_run_ts = timestamp + 1000
            expected_stim_ts = stim_plan_run_ts + stim_lead_frames

            stim_plan.run(at_timestamp=stim_plan_run_ts)

            # To get both frames and analysis, call neurons.read twice
            read_frames   = neurons.read(num_frames, stim_plan_run_ts)
            read_analysis = neurons.read(num_frames, stim_plan_run_ts, analysis=True)

            assert isinstance(read_frames, np.ndarray)
            assert read_frames.shape[0] == num_frames

            assert isinstance(read_analysis, DetectionResult)
            assert len(read_analysis.stims) == 1
            assert read_analysis.stims[0].timestamp == expected_stim_ts

    # Test in both normal and accelerated modes
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    _test()
    cleanup()  # Ensure clean state before next test
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    _test()

def test_neurons_loop():
    """
    Tests neurons.loop(), such as:
    1. Ticks per second
    2. Stops after specified number of ticks
    3. Stops after specified duration
    4. LoopTick contains accurate information, including frames and timestamps
    5. High jitter failure from excessive frames requested in neurons.read().
    6. High jitter failure from slow Python loop operation.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0
        ticks_per_second   = 100
        stop_after_ticks   = 5
        stop_after_seconds = 3
        frames_per_second  = neurons.get_frames_per_second()
        frames_per_tick    = frames_per_second // ticks_per_second
        jitter_frames      = 5
        replay_channels    = neurons._channel_count

        # Test stop_after_ticks and tick timestamps
        neurons_loop: Loop = neurons.loop(
            ticks_per_second = ticks_per_second,
            stop_after_ticks = stop_after_ticks
            )
        tick = None
        for tick in neurons_loop:
            # Tick timestamps is always one iteration behind actual time
            np.testing.assert_allclose(neurons.timestamp(), tick.iteration_timestamp, rtol=1000)
            assert tick.iteration_timestamp == tick.analysis.stop_timestamp
        assert tick is not None and tick.iteration == stop_after_ticks

        # Test stop_after_seconds
        neurons_loop: Loop = neurons.loop(
            ticks_per_second   = ticks_per_second,
            stop_after_seconds = stop_after_seconds
            )
        start_time_sec = time.perf_counter()
        for tick in neurons_loop:
            pass
        stop_time_sec  = time.perf_counter()
        assert tick.iteration == (stop_after_seconds * ticks_per_second)
        np.testing.assert_allclose(stop_time_sec - start_time_sec, stop_after_seconds, atol=0.1)

        # Test LoopTick
        neurons_loop: Loop = neurons.loop(ticks_per_second=ticks_per_second, stop_after_seconds=stop_after_seconds)
        for i, tick in enumerate(neurons_loop):
            assert tick.iteration < neurons_loop._stop_after_ticks
            assert tick.iteration == i
            np.testing.assert_allclose(tick.analysis.start_timestamp, (int(neurons_loop.start_timestamp) + (i * frames_per_tick)), atol=1000)

            assert tick.frames is not None
            assert tick.frames.shape == (frames_per_tick, replay_channels)

            assert tick.analysis is not None
            for spike in tick.analysis.spikes:
                assert spike.timestamp >= tick.analysis.start_timestamp
                assert spike.timestamp <= tick.analysis.stop_timestamp

        # Test jitter failure from neurons.read()
        # TODO: We need to implement a robust way to examine user execution time of the loop body
        # neurons_loop: Loop = neurons.loop(
        #     ticks_per_second        = ticks_per_second,
        #     jitter_tolerance_frames = jitter_frames,
        #     stop_after_ticks        = 2
        #     )
        # with pytest.raises(TimeoutError):
        #     for tick in neurons_loop:
        #         neurons.read(frames_per_tick + jitter_frames + 50, None)

        # Test jitter failure from slow loop operation
        # neurons_loop: Loop = neurons.loop(
        #     ticks_per_second        = ticks_per_second,
        #     jitter_tolerance_frames = jitter_frames
        #     )
        # with pytest.raises(TimeoutError):
        #     for tick in neurons_loop:
        #         time.sleep((1 / ticks_per_second) * 1.5)
        #         if tick.iteration > 0:
        #             break

def test_neurons_loop_run():
    """
    Tests neurons.loop() with Loop.run() syntax.
    """
    import cl

    TICKS_PER_SECOND = 2
    iterations       = []

    def callback(tick):
        iterations.append(tick.iteration)
        if tick.iteration > 0:
            tick.loop.stop()

    with cl.open() as neurons:
        loop = neurons.loop(TICKS_PER_SECOND)
        loop.run(callback)

    assert iterations == [0, 1]

def test_neurons_loop_accelerated():
    """
    Tests neurons.loop() basic functionality:
    1. Ticks per second
    2. Stops after specified number of ticks
    3. Stops after specified duration
    4. LoopTick contains accurate information, including frames and timestamps
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0
        ticks_per_second   = 100
        stop_after_ticks   = 10
        stop_after_seconds = 10
        frames_per_second  = neurons.get_frames_per_second()
        frames_per_tick    = frames_per_second // ticks_per_second
        replay_duration    = neurons._duration_frames
        replay_start_ts    = neurons._start_timestamp
        replay_channels    = neurons._channel_count

        # Test stop_after_ticks and tick timestamps
        neurons_loop: Loop = neurons.loop(
            ticks_per_second = ticks_per_second,
            stop_after_ticks = stop_after_ticks
            )
        tick = None
        for tick in neurons_loop:
            # Tick timestamps is always one iteration behind actual time
            assert neurons.timestamp() == tick.iteration_timestamp
        assert tick is not None and tick.iteration == stop_after_ticks

        # Test stop_after_seconds
        neurons_loop: Loop = neurons.loop(
            ticks_per_second   = ticks_per_second,
            stop_after_seconds = stop_after_seconds
            )
        tick = None
        for tick in neurons_loop:
            pass
        assert tick is not None and tick.iteration == (stop_after_seconds * ticks_per_second)

        # Test LoopTick
        # We allow the loop tick to run for 2.5 times the duration of the
        # replay file, so as to test wrapping functionality
        neurons_loop: Loop = neurons.loop(ticks_per_second=ticks_per_second)
        for i, tick in enumerate(neurons_loop):
            assert tick.iteration < neurons_loop._stop_after_ticks
            assert tick.iteration == i
            assert tick.analysis.start_timestamp == \
                (int(neurons_loop.start_timestamp) + (i * frames_per_tick))

            assert tick.frames is not None
            assert tick.frames.shape == (frames_per_tick, replay_channels)

            assert tick.analysis is not None
            for spike in tick.analysis.spikes:
                assert spike.timestamp >= tick.analysis.start_timestamp
                assert spike.timestamp <= tick.analysis.stop_timestamp

            if tick.iteration_timestamp >= (replay_start_ts + (2.5 * replay_duration)):
                # Here, we also test the stop functionality which can be
                # used instead of "break"
                tick.loop.stop()

@pytest.mark.skip(reason="Jitter failure detection is currently not supported in cl-sdk. This test will be re-enabled once we have a robust way to examine user execution time of the loop body.")
def test_neurons_loop_jitter_failures_accelerated():
    """
    Tests neurons.loop() jitter failures in accelerated mode:
    1. High jitter failure from excessive frames requested in neurons.read().
    2. High jitter failure from slow Python loop operation.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    # Test jitter failure from neurons.read() - use fresh instance
    with cl.open() as neurons:
        ticks_per_second = 100
        frames_per_second = neurons.get_frames_per_second()
        frames_per_tick = frames_per_second // ticks_per_second
        jitter_frames = 5

        neurons_loop: Loop = neurons.loop(
            ticks_per_second        = ticks_per_second,
            jitter_tolerance_frames = jitter_frames
            )
        with pytest.raises(TimeoutError):
            for tick in neurons_loop:
                neurons.read(frames_per_tick + jitter_frames + 1, None)

    # Test jitter failure from slow loop operation - use fresh instance
    with cl.open() as neurons:
        ticks_per_second = 100
        jitter_frames = 5

        neurons_loop: Loop = neurons.loop(
            ticks_per_second        = ticks_per_second,
            jitter_tolerance_frames = jitter_frames
            )
        with pytest.raises(TimeoutError):
            for tick in neurons_loop:
                time.sleep((1 / ticks_per_second) + 1)
                if tick.iteration > 0:
                    break

def _warm_up_producer(neurons, settle_seconds: float = 0.5) -> None:
    """
    Block until the real-time data producer is tracking wall-clock time 1:1.

    The simulator's data producer runs in a subprocess and may perform heavy
    one-shot work at startup (e.g. the random source calibrates a spike
    threshold and generates its first second-long data block). Until that work
    completes the producer runs behind real-time and then catches up faster than
    real-time, so its timestamp-vs-wall-clock relationship is non-uniform. This
    waits until the producer's timestamp advances at (approximately) the
    real-time frame rate, indicating it has settled into steady-state pacing.
    """
    buffer = neurons._shared_buffer
    assert buffer is not None
    frames_per_second = neurons.get_frames_per_second()
    deadline_ns       = time.perf_counter_ns() + int(settle_seconds * 1_000_000_000)
    probe_seconds     = 0.02
    consecutive_ok    = 0
    while time.perf_counter_ns() < deadline_ns:
        start_ts  = buffer.write_timestamp
        start_ns  = time.perf_counter_ns()
        time.sleep(probe_seconds)
        advanced  = buffer.write_timestamp - start_ts
        elapsed_s = (time.perf_counter_ns() - start_ns) / 1_000_000_000
        rate      = advanced / elapsed_s if elapsed_s > 0 else 0
        # Steady state: producer advances at ~real-time (not catching up faster).
        if abs(rate - frames_per_second) <= frames_per_second * 0.05:
            consecutive_ok += 1
            if consecutive_ok >= 3:
                return
        else:
            consecutive_ok = 0

def test_neurons_loop_jitter_recovery():
    """
    This tests:
    1. Jitter recovery catch up
    2. Jitter recovery with callback
    """
    # 1. Jitter recovery catch up
    # 2. Jitter recovery with callback
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    TICKS_PER_SECOND = 100
    FRAMES_PER_TICK  = 250
    STOP_AFTER_TICKS = 10
    RESUME_AT_TICK   = 7

    tick_iterations         = []
    callback_tick_iteration = []
    first_tick_timestamp    = 0
    last_tick_timestamp     = 0

    def handle_recovery_tick(tick: LoopTick):
        callback_tick_iteration.append(tick.iteration)
        # Resume the (briefly frozen) producer once recovery is underway so the
        # resumed ticks can read their data.
        assert neurons._shared_buffer is not None
        neurons._shared_buffer.pause_flag = False

    with cl.open() as neurons:
        neurons.restart()
        # Let the data producer reach steady-state real-time before starting the
        # timing-sensitive loop. The random data source does heavy one-shot work
        # at startup (spike-threshold calibration + first-block generation), so
        # the producer begins behind real-time and catches up at a non-uniform
        # rate. If the loop captures its start timestamp during that catch-up
        # phase, the jitter-recovery resume point becomes timing-dependent.
        # Waiting until the producer is tracking wall-clock 1:1 makes the test
        # deterministic (file replay needs no warm-up as its startup is trivial).
        _warm_up_producer(neurons)
        for tick in neurons.loop(TICKS_PER_SECOND, stop_after_ticks=STOP_AFTER_TICKS):
            tick_iterations.append(tick.iteration)
            if (tick.iteration == 0):
                first_tick_timestamp = tick.analysis.start_timestamp
            elif (tick.iteration == 1):
                tick.loop.recover_from_jitter(handle_recovery_tick=handle_recovery_tick)
                # Deterministically place the producer in the MIDDLE of
                # RESUME_AT_TICK's window, then freeze it. The recovery target
                # iteration is computed as
                # (write_timestamp - start_ts) // frames_per_tick, so any drift
                # of the producer between here and when the loop next samples its
                # timestamp could shift the resume point by a tick. Because the
                # producer is warmed up (advancing at real-time, not sprinting to
                # catch up), we busy-spin until it crosses the mid-point of the
                # target tick and immediately freeze it there, pinning the
                # computed target to RESUME_AT_TICK regardless of scheduling
                # jitter. The producer is released again from the recovery
                # callback once recovery is underway.
                buffer           = neurons._shared_buffer
                assert buffer is not None
                resume_timestamp = first_tick_timestamp + (RESUME_AT_TICK * FRAMES_PER_TICK) + (FRAMES_PER_TICK // 2)
                while buffer.write_timestamp < resume_timestamp:
                    pass
                buffer.pause_flag = True
                # Confirm the producer has actually halted before the loop reads
                # its timestamp, so the resume target cannot drift.
                prev = -1
                while buffer.write_timestamp != prev:
                    prev = buffer.write_timestamp
                    time.sleep(0.002)
            elif (tick.iteration == STOP_AFTER_TICKS - 1):
                last_tick_timestamp = tick.analysis.start_timestamp

    print(f"{tick_iterations=}")
    print(f"{callback_tick_iteration=}")
    print(f"{first_tick_timestamp=}, {last_tick_timestamp=}")
    assert tick_iterations         == [0, 1, 7, 8, 9]
    assert callback_tick_iteration == [2, 3, 4, 5, 6]
    assert last_tick_timestamp     == first_tick_timestamp + ((STOP_AFTER_TICKS - 1) * FRAMES_PER_TICK)

def test_neurons_loop_jitter_recovery_slow_callback():
    """
    This tests:
        1. Jitter recovery with slow callback triggering TimeoutError
    """

    # 1. Jitter recovery with slow callback triggering TimeoutError
    #    - Recovery should exceed timeout of 0.2 secs after 4 calls, each adding 0.05 secs
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    TICKS_PER_SECOND = 100
    STOP_AFTER_TICKS = 10

    with pytest.raises(Exception):

        def slow_recovery_callback(tick: LoopTick):
            print("recovery call:", tick.iteration)
            time.sleep(0.05)

        with cl.open() as neurons:
            for tick in neurons.loop(TICKS_PER_SECOND, stop_after_ticks=STOP_AFTER_TICKS):
                print("loop iteration:", tick.iteration)
                if tick.iteration == 1:
                    tick.loop.recover_from_jitter(handle_recovery_tick=slow_recovery_callback, timeout_seconds=0.2)
                    time.sleep(0.05)