import time
import os
import pathlib

import numpy as np

import pytest

import cl
from cl import Loop, LoopTick


@pytest.fixture(autouse=True)
def cleanup_shared_memory():
    """Clean up any leaked shared memory before and after each test."""
    def _cleanup():
        import time
        import subprocess
        import sys
        from multiprocessing.shared_memory import SharedMemory

        # Kill any orphaned producer processes
        # On Unix-like systems, use pkill to target the specific process by name
        if sys.platform != 'win32':
            try:
                subprocess.run(['pkill', '-9', '-f', 'cl-data-producer'],
                              capture_output=True, timeout=2)
            except Exception:
                pass
        # On Windows: the OS will clean up child processes when the parent (pytest) exits.

        # Give any lingering processes time to die
        time.sleep(0.2)

        # Clean up all shared memory segments matching the cl_sdk_ pattern
        # These now have dynamic names with prefixes: cl_sdk_{prefix}_{segment}

        # On Unix-like systems, shared memory files are in /dev/shm
        if sys.platform != 'win32':
            try:
                shm_dir = pathlib.Path('/dev/shm')
                if shm_dir.exists():
                    for shm_file in shm_dir.glob('cl_sdk_*'):
                        try:
                            shm = SharedMemory(name=shm_file.name)
                            shm.close()
                            shm.unlink()
                        except FileNotFoundError:
                            pass
                        except Exception:
                            pass
            except Exception:
                pass
        # On Windows: shared memory cleanup is handled automatically when
        # processes exit, so we don't need to do manual cleanup

    _cleanup()
    yield
    _cleanup()


def test_neurons_timestamp():
    """
    Tests neurons.timestamp() when running in realtime mode.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0
        wait_secs = 1.0
        start_ts  = neurons.timestamp()
        time.sleep(wait_secs)
        end_ts    = neurons.timestamp()
        duration_sec = (end_ts - start_ts) / neurons._frames_per_second
        assert np.allclose(wait_secs, duration_sec, atol=0.1)

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
        duration_sec = (end_ts - start_ts) / neurons._frames_per_second
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
            neurons.read(frames_to_read, int(neurons.timestamp() - 5.5 * neurons._frames_per_second))

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
            neurons.read(frames_to_read, int(neurons.timestamp() - 5.5 * neurons._frames_per_second))

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
            assert np.allclose(neurons.timestamp(), tick.iteration_timestamp, rtol=1000)
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
        assert np.allclose(stop_time_sec - start_time_sec, stop_after_seconds, atol=0.1)

        # Test LoopTick
        neurons_loop: Loop = neurons.loop(ticks_per_second=ticks_per_second, stop_after_seconds=stop_after_seconds)
        for i, tick in enumerate(neurons_loop):
            assert tick.iteration < neurons_loop._stop_after_ticks
            assert tick.iteration == i
            assert np.allclose(tick.analysis.start_timestamp, (int(neurons_loop.start_timestamp) + (i * frames_per_tick)), atol=1000)

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

    tick_iterations         = []
    callback_tick_iteration = []
    first_tick_timestamp    = 0
    last_tick_timestamp     = 0

    def handle_recovery_tick(tick: LoopTick):
        callback_tick_iteration.append(tick.iteration)

    with cl.open() as neurons:
        neurons.restart()
        for tick in neurons.loop(TICKS_PER_SECOND, stop_after_ticks=STOP_AFTER_TICKS):
            tick_iterations.append(tick.iteration)
            if (tick.iteration == 0):
                first_tick_timestamp = tick.analysis.start_timestamp
            elif (tick.iteration == 1):
                tick.loop.recover_from_jitter(handle_recovery_tick=handle_recovery_tick)
                time.sleep(0.05)
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