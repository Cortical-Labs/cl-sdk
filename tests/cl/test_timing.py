import os
import time
import numpy as np

import cl
from cl.util import more_accurate_sleep

def test_sleep():
    """ Test our sleep function to make sure that latency is within 1 frame. """
    sleep_durations = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    tolerance       = 1 / 25_000
    for duration in sleep_durations:
        start = time.perf_counter()
        more_accurate_sleep(duration, buffer_secs=1e-1)
        assert np.allclose(time.perf_counter() - start, duration, atol=tolerance)

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