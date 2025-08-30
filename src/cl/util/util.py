import time
import numpy as np
import msgpack
import msgpack_numpy

#
# Misc utilities
#

SECONDS_FRAMES_CONVERSION_PRECISION_DP = 5

def frames_to_approximate_seconds(frames, frames_per_second) -> float:
    """
    Convert frames to approximate seconds.
    Due to floating point error, the result for frame counts larger than
    52,428,800,015 (24.27 days at 25kHz) may not be perfectly accurate.
    """
    return round(
        number  = frames / frames_per_second,
        ndigits = SECONDS_FRAMES_CONVERSION_PRECISION_DP
        )

def ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = { 1: "st", 2: "nd", 3: "rd" }.get(n % 10, "th")

    return f"{n}{suffix}"

def from_msgpacked(data):
    """ Deserialise python built-in and and numpy types using an extended msgpack. """
    return msgpack.unpackb(data, object_hook=msgpack_numpy.decode, raw=False)

def _serialise_more_types_msgpack(obj):
    """
    Adds support for generic objects and numpy types to msgpack.packb(..., default=_serialise_more_types_msgpack).
    """
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.dtype('O'):
            # Avoid msgpack_numpy using pickle to serialise an object array.
            # Thankfully, numpy tolist is not recursive, so we can use it here
            # without 'tolist'ing values within the array.
            return msgpack_numpy.encode(obj.tolist())
        else:
            # msgpack_numpy can handle other numpy array types just fine.
            return msgpack_numpy.encode(obj)
    elif isinstance(obj, np.generic):
        # Also use msgpack_numpy to directly handle numpy scalars
        return msgpack_numpy.encode(obj)

    if hasattr(obj, "__dict__"):
        # Generic python objects store their attributes in __dict__
        return obj.__dict__

    if hasattr(obj, "__slots__"):
        # Performance focused python objects don't use __dict__, but store attribute keys in __slots__.
        return { key: getattr(obj, key) for key in obj.__slots__ }

    #
    # We couldn't convert it to a serialisable type,
    # so we return the object and let the encoding fail.
    #

    return obj

def to_msgpacked(obj):
    """
    Serialise arbitrary objects and numpy types using an extended msgpack.

    Note: objects from custom classes are converted to dicts.
    """
    return msgpack.packb(obj, default=_serialise_more_types_msgpack, use_bin_type=True)

def binary_search(haystack, needle, make_key=None):
    """
    Perform a binary search on a sorted array.

    If the value is found, returns the index of the value, otherwise None.
    """
    if make_key is None:
        make_key = lambda x: x

    left  = 0
    right = len(haystack)

    while left < right:
        index = (left + right) // 2
        key   = make_key(haystack[index])
        if key < needle:
            left = index + 1
        elif key > needle:
            right = index
        else:
            return index

    return None

def sorted_insert_position_before(haystack, needle, make_key=None):
    """
    Find the index that would result in needle being inserted immediately before all equal or larger keys.

    Haystack must be sorted.
    """
    if make_key is None:
        make_key = lambda x: x

    left  = 0
    right = len(haystack)

    while left < right:
        index = (left + right) // 2
        key = make_key(haystack[index])
        if key < needle:
            left = index + 1
        elif key > needle:
            right = index
        else:
            # it matched, now make sure we have the left-most match
            while index > 0 and make_key(haystack[index - 1]) == needle:
                index -= 1
            return index

    return left

def binary_search_range(haystack, start_needle, end_needle, make_key=None):
    """
    Return (start, end) indicies for a range of values in a sorted list.

    The start index will be of the first value that is equal to or larger than start_needle.
    The end index will be of the first value that is equal to or larger than end_needle.

    Either or both of start_needle and end_needle can be None, in which case the range
    will be unbounded in that direction.

    If there are no values in the range, the start and end will be the same index.

    Example:

        haystack = [0, 2, 2, 4, 5]

        assert (1, 1) == binary_search_range(haystack, 1, 1) # no matching values in range
        assert (1, 1) == binary_search_range(haystack, 1, 2) # no matching values in range
        assert (1, 3) == binary_search_range(haystack, 1, 3) # two matching values (at index 1 and 2)
        assert (1, 4) == binary_search_range(haystack, 2, 5) # three matching values (at index 1, 2, and 3)

        # Print all values in the range 2 to 4 (inclusive), i.e prints 2, 2, 4
        index, end = binary_search_range(haystack, 2, 5)
        while index < end:
            print(haystack[index])
            index += 1
    """
    if start_needle is not None and end_needle is not None and start_needle > end_needle:
        raise ValueError("start_needle must be less than or equal to end_needle")

    if start_needle is None:
        start = 0
    else:
        start = sorted_insert_position_before(haystack, start_needle, make_key)

    if end_needle is None:
        end = len(haystack)
    else:
        end = sorted_insert_position_before(haystack, end_needle, make_key)

    return start, end

#
# Performance benchmarking
#

_benchmark_enabled: bool = True
""" Whether Benchmarking is enabled. """

def benchmark_enable():
    """ Enables the Benchmarking. """
    global _benchmark_enabled
    _benchmark_enabled = True

def benchmark_disable():
    """ Disables the Benchmarking. """
    global _benchmark_enabled
    _benchmark_enabled = False

class Benchmark:
    """ A utility to measure code execution time. """

    def __init__(self, name: str = "Benchmark", report_threshold_us: int = 0):
        """
        Args:
            name: Benchmark name.
            report_threshold_us: Print code execution time if threshold is exceeded.
        """
        self.name                = name
        self.report_threshold_ns = report_threshold_us * 1_000

    def __enter__(self):
        if _benchmark_enabled:
            self.start_time_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _benchmark_enabled:
            end_time_ns = time.perf_counter_ns()
            duration_ns = end_time_ns - self.start_time_ns
            if duration_ns >= self.report_threshold_ns:
                duration_us = duration_ns / 1000.0
                print(f"{self.name}: took {duration_us:.3f} µs")

def more_accurate_sleep(seconds):
    """
    Attempt a more accurate sleep by wasting CPU
    """
    end = time.perf_counter() + seconds
    # if seconds > 0.1:
    #     time.sleep(seconds - 0.1)
    while time.perf_counter() < end:
        pass