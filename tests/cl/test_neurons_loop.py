import time
import os

import numpy as np

import pytest

import cl
from cl import Loop

def test_neurons_timestamp():
    """
    Tests neurons.timestamp() when running in realtime mode.
    """
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "0"
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
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "1"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0
        wait_secs = 1.0
        start_ts = neurons.timestamp()
        time.sleep(wait_secs)
        end_ts   = neurons.timestamp()
        duration_sec = (end_ts - start_ts) / neurons._frames_per_second
        assert duration_sec == 0

def test_neurons_read():
    """
    Tests neurons.read() and resulting timestamp alignment, which is is central
    to replaying a recording file using neurons.loop().

    In this test, we consider:
    - A normal read;
    - A read from > 5 secs in the past that will fail.
    """
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "0"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0
        # Test 1: Normal read
        start_ts       = neurons.timestamp()
        frames_to_read = 2500
        neurons.read(frames_to_read, None)

        calculated = start_ts + neurons._elapsed_frames
        expected   = start_ts + frames_to_read
        assert np.allclose(calculated, expected, rtol=0.1)

        # Test 2: Reading from > 5 secs in the past
        with pytest.raises(Exception):
            #
            neurons.read(frames_to_read, int(neurons.timestamp() - 5.1 * neurons._frames_per_second))

        # Test 3: Reading from past
        start_ts       = neurons.timestamp()
        ts_offset      = -2600
        from_ts        = start_ts + ts_offset
        frames_to_read = 2500
        frames         = neurons.read(frames_to_read, from_ts)

        calculated = start_ts + frames_to_read + ts_offset
        expected   = from_ts + frames_to_read
        assert np.allclose(calculated, expected, rtol=0.1)

        # Test 4: Reading from future
        start_ts       = neurons.timestamp()
        ts_offset      = 1000
        from_ts        = start_ts + ts_offset
        frames_to_read = 2500
        frames         = neurons.read(frames_to_read, from_ts)

        calculated = start_ts + frames_to_read + ts_offset
        expected   = from_ts + frames_to_read
        assert np.allclose(calculated, expected, rtol=0.1)

def test_neurons_read_accelerated():
    """
    Tests neurons.read() and resulting timestamp alignment, which is is central
    to replaying a recording file using neurons.loop().

    In this test, we consider:
    - A normal read;
    - A read from > 5 secs in the past that will fail.
    """
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "1"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0
        # Test 1: Normal read
        start_ts       = neurons.timestamp()
        frames_to_read = 2500
        neurons.read(frames_to_read, None)

        calculated = start_ts + neurons._elapsed_frames
        expected   = start_ts + frames_to_read
        assert calculated == expected

        # Test 2: Reading from > 5 secs in the past
        with pytest.raises(Exception):
            #
            neurons.read(frames_to_read, int(neurons.timestamp() - 5.1 * neurons._frames_per_second))

        # Test 3: Reading from past
        start_ts       = neurons.timestamp()
        ts_offset      = -2600
        from_ts        = start_ts + ts_offset
        frames_to_read = 2500
        frames         = neurons.read(frames_to_read, from_ts)

        calculated = start_ts + frames_to_read + ts_offset
        expected   = from_ts + frames_to_read
        assert calculated == expected

        # Test 4: Reading from future
        start_ts       = neurons.timestamp()
        ts_offset      = 1000
        from_ts        = start_ts + ts_offset
        frames_to_read = 2500
        frames         = neurons.read(frames_to_read, from_ts)

        calculated = start_ts + frames_to_read + ts_offset
        expected   = from_ts + frames_to_read
        assert calculated == expected

@pytest.mark.skip(reason="Await fixes to loop timing")
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
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "0"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0
        ticks_per_second   = 100
        stop_after_ticks   = 5
        stop_after_seconds = 5
        frames_per_second  = neurons.get_frames_per_second()
        frames_per_tick    = frames_per_second // ticks_per_second
        jitter_frames      = 5
        replay_channels    = neurons._channel_count

        # Test stop_after_ticks and tick timestamps
        neurons_loop: Loop = neurons.loop(
            ticks_per_second = ticks_per_second,
            stop_after_ticks = stop_after_ticks
            )
        for tick in neurons_loop:
            # Tick timestamps is always one iteration behind actual time
            assert np.allclose(neurons.timestamp(), tick.timestamp + frames_per_tick, rtol=1000)
        assert tick.iteration == stop_after_ticks

        # Test stop_after_seconds
        neurons_loop: Loop = neurons.loop(
            ticks_per_second   = ticks_per_second,
            stop_after_seconds = stop_after_seconds
            )
        for tick in neurons_loop:
            pass
        assert tick.iteration == (stop_after_seconds * ticks_per_second)

        # Test LoopTick
        neurons_loop: Loop = neurons.loop(ticks_per_second=ticks_per_second, stop_after_seconds=stop_after_seconds)
        for i, tick in enumerate(neurons_loop):
            assert tick.iteration < neurons_loop._stop_after_ticks
            assert tick.iteration == i
            assert np.allclose(tick.timestamp, (int(neurons_loop.start_timestamp) + (i * frames_per_tick)), atol=1000)

            assert tick.frames is not None
            assert tick.frames.shape == (frames_per_tick, replay_channels)

            assert tick.analysis is not None
            for spike in tick.analysis.spikes:
                assert spike.timestamp >= tick.timestamp
                assert spike.timestamp <= tick.timestamp + frames_per_tick

        # Test jitter failure from neurons.read()
        neurons_loop: Loop = neurons.loop(
            ticks_per_second        = ticks_per_second,
            jitter_tolerance_frames = jitter_frames
            )
        with pytest.raises(TimeoutError):
            for tick in neurons_loop:
                neurons.read(frames_per_tick + jitter_frames + 1, None)

        # Test jitter failure from slow loop operation
        neurons_loop: Loop = neurons.loop(
            ticks_per_second        = ticks_per_second,
            jitter_tolerance_frames = jitter_frames
            )
        with pytest.raises(TimeoutError):
            for tick in neurons_loop:
                time.sleep((1 / ticks_per_second) + 1)
                if tick.iteration > 0:
                    break

@pytest.mark.skip(reason="Await fixes to loop timing")
def test_neurons_loop_accelerated():
    """
    Tests neurons.loop(), such as:
    1. Ticks per second
    2. Stops after specified number of ticks
    3. Stops after specified duration
    4. LoopTick contains accurate information, including frames and timestamps
    5. High jitter failure from excessive frames requested in neurons.read().
    6. High jitter failure from slow Python loop operation.
    """
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "1"
    with cl.open() as neurons:
        neurons._elapsed_frames = 0
        ticks_per_second   = 100
        stop_after_ticks   = 10
        stop_after_seconds = 10
        frames_per_second  = neurons.get_frames_per_second()
        frames_per_tick    = frames_per_second // ticks_per_second
        jitter_frames      = 5
        replay_duration    = neurons._duration_frames
        replay_start_ts    = neurons._start_timestamp
        replay_channels    = neurons._channel_count

        # Test stop_after_ticks and tick timestamps
        neurons_loop: Loop = neurons.loop(
            ticks_per_second = ticks_per_second,
            stop_after_ticks = stop_after_ticks
            )
        for tick in neurons_loop:
            # Tick timestamps is always one iteration behind actual time
            assert neurons.timestamp() == tick.timestamp + frames_per_tick
        assert tick.iteration == stop_after_ticks

        # Test stop_after_seconds
        neurons_loop: Loop = neurons.loop(
            ticks_per_second   = ticks_per_second,
            stop_after_seconds = stop_after_seconds
            )
        for tick in neurons_loop:
            pass
        assert tick.iteration == (stop_after_seconds * ticks_per_second)

        # Test LoopTick
        # We allow the loop tick to run for 2.5 times the duration of the
        # replay file, so as to test wrapping functionality
        neurons_loop: Loop = neurons.loop(ticks_per_second=ticks_per_second)
        for i, tick in enumerate(neurons_loop):
            assert tick.iteration < neurons_loop._stop_after_ticks
            assert tick.iteration == i
            assert tick.timestamp == \
                (int(neurons_loop.start_timestamp) + (i * frames_per_tick))

            assert tick.frames is not None
            assert tick.frames.shape == (frames_per_tick, replay_channels)

            assert tick.analysis is not None
            for spike in tick.analysis.spikes:
                assert spike.timestamp >= tick.timestamp
                assert spike.timestamp <= tick.timestamp + frames_per_tick

            if tick.timestamp >= (replay_start_ts + (2.5 * replay_duration)):
                # Here, we also test the stop functionality which can be
                # used instead of "break"
                neurons_loop.stop()

        # Test jitter failure from neurons.read()
        neurons_loop: Loop = neurons.loop(
            ticks_per_second        = ticks_per_second,
            jitter_tolerance_frames = jitter_frames
            )
        with pytest.raises(TimeoutError):
            for tick in neurons_loop:
                neurons.read(frames_per_tick + jitter_frames + 1, None)

        # Test jitter failure from slow loop operation
        neurons_loop: Loop = neurons.loop(
            ticks_per_second        = ticks_per_second,
            jitter_tolerance_frames = jitter_frames
            )
        with pytest.raises(TimeoutError):
            for tick in neurons_loop:
                time.sleep((1 / ticks_per_second) + 1)
                if tick.iteration > 0:
                    break