import os
import time
import numpy as np

import cl

def test_sleep():
    """ Test our sleep function to make sure that latency is within reasonable tolerance. """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    sampling_frequency = 25_000
    # In non-accelerated mode, only test sleeps >= 10ms due to OS scheduler variability
    sleep_durations    = np.array([1e0, 1e-1, 1e-2])  # 1s, 100ms, 10ms
    # Use realistic tolerances for real-time mode (OS scheduling has ~1-2ms variability)
    tolerances = np.array([0.01, 0.005, 0.002])  # 10ms, 5ms, 2ms
    sleep_frames       = (sleep_durations * sampling_frequency).astype(int)
    for duration, frames, tolerance in zip(sleep_durations, sleep_frames, tolerances):
        with cl.open() as neurons:
            start_timestamp = neurons.timestamp()
            start_secs      = time.perf_counter()
            neurons._sleep_until(start_timestamp + frames)
            assert np.allclose(time.perf_counter() - start_secs, duration, atol=tolerance)

FIRST_READ_TOL = 500
READ_TOL = 100

def test_read_latency():
    """ Test our read timing latency is within 100 frames. """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    with cl.open() as neurons:
        neurons.restart()
        tss = [neurons.timestamp()]
        time.sleep(1)
        ts = neurons.timestamp()
        tss.append(ts)
        neurons.read(12500, ts)
        ts = neurons.timestamp()
        tss.append(ts)
        neurons.read(12500, ts - 12500 // 2)
        ts = neurons.timestamp()
        tss.append(ts)
        neurons.read(12500, ts - 25000)
        ts = neurons.timestamp()
        tss.append(ts)

        for i, t in enumerate(tss[1:]):
            test_ts = t - tss[i]
            match i:
                case 0:
                    assert np.allclose(test_ts, 25_000, atol=FIRST_READ_TOL)  # time.sleep can be unpredictable
                case 1:
                    assert np.allclose(test_ts, 12_500, atol=READ_TOL)        # producer overhead + inter-test cleanup
                case 2:
                    assert np.allclose(test_ts,  6_250, atol=READ_TOL)        # producer overhead + inter-test cleanup
                case 3:
                    assert np.allclose(test_ts,      0, atol=READ_TOL)        # time to copy data

            print(f"ts {t} (+{test_ts})")

def test_op_timing():
    """ Test advance_elapsed_times and whether ops are called at the correct timestamp. """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    with cl.open() as neurons:
        expected_timestamp = 137
        def timed_operation(neurons=neurons):
            actual_timestamp = neurons.timestamp()
            assert actual_timestamp == expected_timestamp
        neurons._timed_ops.put((expected_timestamp, timed_operation))
        neurons.read(250, None)