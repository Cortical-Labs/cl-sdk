import os
import time
import numpy as np

import cl

def test_sleep():
    """ Test our sleep function to make sure that latency is within 1 frame. """
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "0"
    sampling_frequency = 25_000
    sleep_durations    = np.array([1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    sleep_frames       = (sleep_durations * sampling_frequency).astype(int)
    tolerance          = 1 / sampling_frequency
    with cl.open() as neurons:
        for duration, frames in zip(sleep_durations, sleep_frames):
            start_timestamp = neurons.timestamp()
            start_secs      = time.perf_counter()
            neurons._sleep_until(start_timestamp + frames)
            assert np.allclose(time.perf_counter() - start_secs, duration, atol=tolerance)

def test_read_latency():
    """ Test our read timing latency is within 100 frames. """
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "0"
    with cl.open() as neurons:
        neurons.restart()
        ts = [neurons.timestamp()]
        time.sleep(1)
        ts.append(neurons.timestamp())
        neurons.read(12500, neurons.timestamp())
        ts.append(neurons.timestamp())
        neurons.read(12500, neurons.timestamp() - 12500 // 2)
        ts.append(neurons.timestamp())
        neurons.read(12500, neurons.timestamp() - 25000)
        ts.append(neurons.timestamp())

        for i, t in enumerate(ts[1:]):
            test_ts = t - ts[i]
            match i:
                case 0:
                    assert np.allclose(test_ts, 25_000, atol=500) # time.sleep can be unpredictable
                case 1:
                    assert np.allclose(test_ts, 12_500, atol=5)   # we require tight timing
                case 2:
                    assert np.allclose(test_ts, 6_250, atol=5)    # we require tight timing
                case 3:
                    assert np.allclose(test_ts, 0, atol=50)       # time to copy data

            print(f"ts {t} (+{test_ts})")

def test_op_timing():
    """ Test advance_elapsed_times and whether ops are called at the correct timestamp. """
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "1"
    with cl.open() as neurons:
        expected_timestamp = 137
        def timed_operation(neurons=neurons):
            actual_timestamp = neurons.timestamp()
            assert actual_timestamp == expected_timestamp
        neurons._timed_ops.put((expected_timestamp, timed_operation))
        neurons.read(250, None)