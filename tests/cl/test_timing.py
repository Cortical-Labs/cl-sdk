import os
import time
import numpy as np

# Disable websocket for tests, needs to be set before importing cl
os.environ["CL_SDK_VISUALISATION"] = "0"

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
            np.testing.assert_allclose(time.perf_counter() - start_secs, duration, atol=tolerance)

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