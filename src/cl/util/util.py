import os
import time
import json
import traceback
import types

import numpy as np
import msgpack
import msgpack_numpy
import functools
import contextlib

from math import ceil

#
# Misc utilities
#

SECONDS_FRAMES_CONVERSION_PRECISION_DP = 5

def seconds_to_minimum_frames(seconds, frames_per_second):
    """
    Return the minimum number of frames needed to cover a duration.
    Fractional frames are rounded up.
    """
    return ceil(round(seconds * frames_per_second, SECONDS_FRAMES_CONVERSION_PRECISION_DP))

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

def is_under_path(test_path, parent_path):
    """Returns True if test_path is under parent_path after resolving symbolic links."""
    return os.path.realpath(f"{test_path}{os.path.sep}").startswith(os.path.realpath(f"{parent_path}{os.path.sep}"))

def is_iterable(obj):
    """Returns True if the object is iterable, False otherwise."""
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def read_utf8_file(file_path):
    """ Read a utf-8 text file as a string. """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_binary_file(file_path):
    """ Read an arbitrary file as bytes. """
    with open(file_path, "rb") as f:
        return f.read()

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

class Event(object):
    """
    A simple event class that can be used to invoke multiple handlers when something happens.

    Usage:
        event = Event()

        # Add some handlers to the event.
        event += some_handler
        event += some_other_handler

        # Remove a handler from the event.
        event -= some_handler

        # Call all current event handlers with positional and keyword arguments.
        event(...)
    """
    def __init__(self):
        self._handlers = []

    def __iadd__(self, handler):
        """Add a handler to the event."""
        self._handlers.append(handler)
        return self

    def __isub__(self, handler):
        """Remove a handler from the event."""
        if handler in self._handlers:
            self._handlers.remove(handler)
        return self

    def __call__(self, *args, **kwargs):
        """Call all current event handlers with positional and keyword arguments."""
        for handler in self._handlers:
            handler(*args, **kwargs)

def to_msgpacked(obj) -> bytes:
    """
    Serialise arbitrary objects and numpy types using an extended msgpack.

    Note: objects from custom classes are converted to dicts.
    """
    return msgpack.packb(obj, default=_serialise_more_types_msgpack, use_bin_type=True)  # type: ignore

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
    Return (start, end) indices for a range of values in a sorted list.

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
# JSON serialisation
#

def to_json(obj, prettyish=False):
    """
    Serialise arbitrary objects and their properties to JSON.

    The default Python json.dumps(...) method doesn't support serialising the attributes of arbitrary objects,
    so this method adds support for that. It also adds specific support for numpy types which otherwise serialise
    vast amounts of internal state.

    First attempts to use a fast JSON serialisation method that just adds support for a few types to json.dumps(...).
    If that fails due to circular references or other issues, it falls back to a slower method that first recursively builds
    a serialisable version of the object and its properties, then passes that to json.dumps(...). In the event of a fallback,
    a warning is printed to the console and the fast method is disabled for future serialisations.
    """

    if not hasattr(to_json, '_to_json_use_slow'):
        try:
            # faster, doesn't support ignoring circular refs
            return _to_json_fast(obj, prettyish)
        except ValueError as e:
            to_json._to_json_use_slow = True
            print(f"Falling back to slower serialisation method for all future dataset data: {e}")
            # slower, ignores circular refs
            return _to_json_slow(obj, prettyish)
    else:
        return _to_json_slow(obj, prettyish)

class PrettyishEncoder(json.JSONEncoder):
    def iterencode(self, obj, _one_shot=False):
        encoder = super().iterencode(obj, False)

        # Encode lists without any newlines
        is_encoding_list = False
        for chunk in encoder:
            if chunk.startswith('['):
                list_buffer = '[' + chunk[1:].strip()
                if chunk.endswith(']'):
                    yield(list_buffer)
                else:
                    is_encoding_list = True
            elif chunk.endswith(']'):
                yield list_buffer + chunk
                is_encoding_list = False
            elif is_encoding_list:
                if chunk.strip() == '':
                    continue
                list_buffer += ', ' + chunk[1:].strip()
            else:
                yield chunk

def _to_json_slow(obj, prettyish=False):
    if prettyish:
        return json.dumps(_to_serialisable(obj), cls=PrettyishEncoder, separators=(',', ':'))
    else:
        return json.dumps(_to_serialisable(obj), separators=(',', ':'))

def _to_json_fast(obj, prettyish=False):
    if prettyish:
        return json.dumps(obj, default=_serialise_more_types_json, cls=PrettyishEncoder, separators=(',', ':'))
    else:
        return json.dumps(obj, default=_serialise_more_types_json, separators=(',', ':'))

def _to_serialisable(obj, _visited=None, _in_progress=None):
    """
    Converts an arbitrary object to something serialisable by json.dumps(...) and similar.

    This approach is 2-3x slower than passing default=_serialise_more_types to json.dumps(...),
    but it won't choke on object hierarchies with circular references.
    """

    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj

    if callable(obj):
        return None

    _id = id(obj)

    if _visited is None:
        _visited = {}
    elif _id in _visited:
        return _visited[_id]

    if _in_progress is None:
        _in_progress = set()
    _in_progress.add(_id)

    if isinstance(obj, (np.generic, np.ndarray)):
        result = obj.tolist()
    elif isinstance(obj, dict):
        result = { key: _to_serialisable(value, _visited, _in_progress) for key, value in obj.items() if id(value) not in _in_progress}
    elif isinstance(obj, (list, tuple, set)) or is_iterable(obj):
        result = [ _to_serialisable(item, _visited, _in_progress) for item in obj if id(item) not in _in_progress]
    elif hasattr(obj, '__dict__'):
        result = _to_serialisable(obj.__dict__, _visited, _in_progress)
    elif hasattr(obj, '__slots__'):
        result = { key: value for key in obj.__slots__ if id(value := getattr(obj, key)) not in _in_progress }
    else:
        result = repr(obj)

    _in_progress.remove(_id)

    _visited[_id] = result
    return result

def _serialise_more_types_json(obj):
    """
    Adds support for generic objects and numpy types to json.dumps(default=_serialise_more_types).
    """

    # Let numpy handle its own types. Yes, numpy numbers support .tolist().
    if isinstance(obj, (np.generic, np.ndarray)):
        return obj.tolist()

    # Most Python objects store their properties in a __dict__ attribute,
    # which *.dumps(...) can handle just fine.
    if hasattr(obj, '__dict__'):
        return obj.__dict__

    # If the class uses slots, it won't have a __dict__ so we need to make one.
    if hasattr(obj, '__slots__'):
        return { key: getattr(obj, key) for key in obj.__slots__ }

    return obj

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

class _BenchmarkNoop:
    """
    A no-op benchmark that does nothing.

    This may seem pointless, but 'None' does not support
    the 'with' statement, so we need a dummy class to use
    when benchmarks are disabled.
    """
    def __init__(self, name: str | None = None, report_threshold_us: int = 0):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type: type[Exception] | None, exc_val: Exception | None, exc_tb: types.TracebackType | None):
        pass

class _PythonBenchmark:
    def __init__(self, name: str | None = None, report_threshold_us: int = 0):
        """
        A utility to measure code execution time.

        Args:
            name:                Benchmark name.
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
                print(f"{self.name or 'Benchmark'}: took {duration_us:.3f} µs")

try:
    import _cl
    _cl_Benchmark = getattr(_cl, 'Benchmark', None)
except ImportError:
    _cl_Benchmark = None

def Benchmark(name: str | None = None, report_threshold_us: int = 0):
    if not _benchmark_enabled:
        return _BenchmarkNoop(name, report_threshold_us)
    if _cl_Benchmark is not None:
        return _cl_Benchmark(name, report_threshold_us)
    return _PythonBenchmark(name, report_threshold_us)

#
# Deprecation warnings
#

def deprecated(replacement=None):
    """ Decorator that marks a method as deprecated, and prints a warning on first use. """
    def decorator(method):
        @functools.wraps(method) # Required to expose method signature like __doc__
        def wrapper(*args, **kwargs):
            if not hasattr(method, '_warned_deprecation'):
                if replacement:
                    print(f"{method.__name__} is deprecated, use {replacement} instead")
                else:
                    print(f"{method.__name__} is deprecated")
                method._warned_deprecation = True
            return method(*args, **kwargs)
        return wrapper
    return decorator

def in_ipython():
    try:
        # get_ipython is defined by IPython; if not, this will raise NameError
        ipy = get_ipython()
        return ipy is not None
    except NameError:
        return False

def in_vscode():
    """Returns True if running inside VSCode's Jupyter environment."""
    return in_ipython() and "VSCODE_CWD" in os.environ

#
# Security patches for Pickle
# This is needed since PyTables natively use pickle to handle dict within attributes
#

@contextlib.contextmanager
def restricted_unpickling():
    """
    Monkey patch pickle.loads to block unpickling of non-builtin types, non-scalar numpy types, when unpickling data.
    It takes particular care to prevent unpickling of numpy object dtype, which can be used to execute arbitrary code.
    This is a security measure to prevent potential code execution from maliciously crafted pickle data.
    """
    import io
    import pickle

    from ..error import UnsafeOperationError

    try:
        import _pickle
    except ImportError:
        _pickle = None

    # Backup original pickle.loads and _pickle.loads (if it exists) so we can restore them later
    old_pickle_loads    = pickle.loads
    old_Unpickler       = pickle.Unpickler
    old_c_loads         = getattr(_pickle, "loads", None) if _pickle else None

    # Flag to track if we encountered any unsafe pickle data during loading.
    # Unfortunately, pytables will swallow pickle loading errors, so we use
    # this to raise errors on exiting the context if we encountered any unsafe data.
    blocked_type_error = None

    def _block(message):
        nonlocal blocked_type_error
        error = UnsafeOperationError(message)
        # Track the first blocked type error we encountered
        if blocked_type_error is None:
            blocked_type_error = error
        raise error

    def _restricted_scalar_dtype(value, *args, **kwargs):
        # Block unpickling of object dtype, since those can be used to execute arbitrary code.
        dtype = np.dtype(value, *args, **kwargs)

        if dtype.hasobject:
            _block(f"Blocked NumPy dtype containing Python objects: {dtype!r}")
        if dtype.fields is not None:
            _block(f"Blocked structured NumPy dtype: {dtype!r}")
        if dtype.subdtype is not None:
            _block(f"Blocked subarray NumPy dtype: {dtype!r}")
        if dtype.kind not in {"b", "i", "u", "f", "c"}:
            _block(f"Blocked NumPy scalar dtype kind {dtype.kind!r}: {dtype!r}")

        return dtype

    def _restricted_numpy_scalar(_dtype, value):
        dtype = _restricted_scalar_dtype(_dtype)

        if isinstance(value, memoryview):
            value = value.tobytes()
        if not isinstance(value, (bytes, bytearray)):
            _block(f"Blocked NumPy scalar payload type: {type(value)!r}")

        return np._core.multiarray.scalar(dtype, value)

    def _restricted_dtype(*args, **kwargs):
        # Block unpickling of object dtype, since those can be used to execute arbitrary code.
        dtype = np.dtype(*args, **kwargs)
        if dtype.hasobject:
            _block(f"Blocked NumPy dtype containing Python objects: {dtype!r}")
        return dtype

    def _restricted_numpy_reconstruct(subtype, shape, _dtype):
        dtype = _restricted_dtype(_dtype)

        # Do not allow ndarray subclasses from untrusted pickle.
        if subtype is _restricted_ndarray:
            subtype = np.ndarray
        if subtype is not np.ndarray:
            _block(f"Blocked NumPy ndarray subtype reconstruction: {subtype!r}")

        return np._core.multiarray._reconstruct(subtype, shape, dtype)

    def _restricted_ndarray(*args, **kwargs):
        dtype = kwargs.get("dtype", None)
        if len(args) >= 2:
            dtype = args[1]
        if dtype is None:
            dtype = float

        dtype = _restricted_dtype(dtype)

        # Disallow arbitrary external buffers from pickle.
        buffer = kwargs.get("buffer", None)
        if len(args) >= 3:
            buffer = args[2]
        if buffer is not None:
            _block("Blocked numpy.ndarray reconstruction with external buffer")

        return np.ndarray(*args, **kwargs)

    # Allow loading of whitelisted objects
    import numpy as np
    import _codecs
    _UNPICKLE_WHITELIST = {
        # Numpy scalar
        ("numpy.core.multiarray",  "scalar"):       _restricted_numpy_scalar,
        ("numpy._core.multiarray", "scalar"):       _restricted_numpy_scalar,
        ("numpy",                  "dtype"):        _restricted_scalar_dtype,
        ("_codecs",                "encode"):       _codecs.encode,

        # Numpy ndarray
        ("numpy.core.multiarray",  "_reconstruct"): _restricted_numpy_reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): _restricted_numpy_reconstruct,
        ("numpy",                  "ndarray"):      _restricted_ndarray,
    }
    _WHITELIST_SENTINEL = object()

    class BuiltinsOnlyUnpickler(old_Unpickler):
        def find_class(self, module, name):
            nonlocal blocked_type_error

            expected_obj = _UNPICKLE_WHITELIST.get((module, name), _WHITELIST_SENTINEL)
            if expected_obj is not _WHITELIST_SENTINEL:
                return expected_obj

            _block(f"{module}.{name} not in restricted unpickling whitelist")

    def _restricted_pickle_loads(data, /, **kwargs):
        return BuiltinsOnlyUnpickler(io.BytesIO(data)).load()

    # Monkey patch pickle.loads and _pickle.loads (if it exists) to our restricted version
    pickle.loads     = _restricted_pickle_loads
    pickle.Unpickler = BuiltinsOnlyUnpickler
    if _pickle and hasattr(_pickle, "loads"):
        _pickle.loads = _restricted_pickle_loads

    try:
        # Enter the context where unpickling is restricted to builtin types only.
        yield
        # After the block, if we attempted to unpickle any non-builtin types, raise an error to alert the caller.
        if blocked_type_error:
            raise blocked_type_error
    finally:
        # Restore original pickle loading functions
        pickle.loads     = old_pickle_loads
        pickle.Unpickler = old_Unpickler
        if _pickle and old_c_loads is not None:
            _pickle.loads = old_c_loads

#
# Debugging
#

def allow_debugging():
    import signal
    import debugpy

    def start_debugger(signum, frame):
        print("Starting debugpy ...")

        DEFAULT_PORT = 5678
        for port in range(DEFAULT_PORT, DEFAULT_PORT + 4):
            try:
                debugpy.listen(("0.0.0.0", port))
                break
            except RuntimeError as e:
                if '[Errno 98]' in str(e): # yuck
                    print(f"Port {port} in use")
                else:
                    raise e

        # Wait for the debugger client to attach
        print(f"Waiting for debugger to attach to port {port} ...")
        debugpy.wait_for_client()
        print("Debugger now attached, will attempt to break immediately.")

        #
        # Break immediately, leaving control with the debugger client.
        #

        # HACK: we need to wait a while before we can expect debugpy.breakpoint() to actually work.
        time.sleep(3)

        debugpy.breakpoint()

    # Attach the signal handler to SIGUSR1
    signal.signal(signal.SIGUSR1, start_debugger)

    print(f"To debug this process, run 'kill -USR1 {os.getpid()}' and then attach a remote debugger to the logged port.")

#
# Exception printing
#

def clean_uncaught_exception_printing():
    def clean_traceback(tb):
        """ Remove unhelpful stack frames from uncaught exception tracebacks. """
        filtered = []
        while tb:
            frame           = tb.tb_frame
            frame_package   = frame.f_globals.get('__package__', '')

            if frame_package not in ['cl', '_cl', 'IPython.core']:
                filtered.append(tb)

            # move to the next frame
            tb = tb.tb_next

        if not filtered:
            return None

        filtered[-1].tb_next = None
        for i in range(len(filtered) -2, -1, -1):
            filtered[i].tb_next = filtered[i + 1]

        return filtered[0]

    def excepthook_ipython(shell, etype, evalue, tb, tb_offset=None):
        shell.showtraceback((etype, evalue, clean_traceback(tb)), tb_offset=0)

    def excepthook(exc_type, exc_value, exc_tb):
        traceback.print_tb(clean_traceback(exc_tb))
        print(str(exc_value))

    if in_ipython():
        # Override default exception printing for jupyter
        ipy = get_ipython()
        if not ipy.custom_exceptions:
            ipy.set_custom_exc((Exception,), excepthook_ipython)
