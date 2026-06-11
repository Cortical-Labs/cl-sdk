import os
import copyreg
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import tables as tb

import pytest
from pytest_mock import MockerFixture
from inline_snapshot import snapshot

# Disable websocket for tests, needs to be set before importing cl
os.environ["CL_SDK_VISUALISATION"] = "0"

import cl
import cl.neurons
from cl import ChannelSet, StimDesign
from cl.error import UnsafeOperationError
from cl.recording import _utcdatestring
from cl.util import RecordingView, AttributesView


# ── Helpers ──────────────────────────────────────────────────────────────────

def _open_recording(path: Path) -> RecordingView:
    """Open a recording and return the RecordingView."""
    return RecordingView(str(path))


def _write_minimal_recording_attributes(attrs):
    """Populate the minimum standard recording attrs needed by RecordingView."""
    attrs.application = {}
    attrs.created_localtime = ""
    attrs.created_utc = ""
    attrs.ended_localtime = ""
    attrs.ended_utc = ""
    attrs.channel_count = 1
    attrs.sampling_frequency = 25_000
    attrs.frames_per_second = 25_000
    attrs.uV_per_sample_unit = 1.0
    attrs.start_timestamp = 0
    attrs.end_timestamp = 0
    attrs.duration_frames = 0
    attrs.duration_seconds = 0.0
    attrs.file_format = {
        "version": 1,
        "stim_and_spike_timestamps_relative_to_start": True,
    }


def _write_minimal_recording(path: Path):
    """Write a minimal HDF5 file for RecordingView tests."""
    with tb.open_file(str(path), mode="w") as h5:
        _write_minimal_recording_attributes(h5.root._v_attrs)


def _write_recording_attr(path: Path, value: object, *, node_name: str | None = None):
    """Write a single attribute into a minimal HDF5 file for RecordingView tests."""
    with tb.open_file(str(path), mode="w") as h5:
        _write_minimal_recording_attributes(h5.root._v_attrs)
        node = h5.root if node_name is None else h5.create_group("/", node_name)
        node._v_attrs.payload = value


def _assert_recording_blocked(path: Path, message_fragment: str | None = None):
    """Assert that RecordingView refuses to open an unsafe HDF5 file."""
    try:
        view = _open_recording(path)
    except UnsafeOperationError as exc:
        if message_fragment is not None:
            assert message_fragment in str(exc)
    else:
        view.close()
        pytest.fail("Expected RecordingView to reject unsafe HDF5 file")


class _GetPidOnLoad:
    """A harmless stand-in for an arbitrary callable executed during unpickling."""

    def __reduce__(self):
        return (os.getpid, ())


class _DirectObjectDtypeArray:
    """Force direct numpy.ndarray reconstruction with object dtype."""

    def __reduce__(self):
        return (np.ndarray, ((0,), "O"))


def _make_ext2_reduce_payload(code: int) -> bytes:
    """Build PROTO 2 + EXT2 + EMPTY_TUPLE + REDUCE + STOP."""
    return b"\x80\x02\x83" + code.to_bytes(2, "little") + b")R."


def _assert_valid_recording_attributes(attrs: AttributesView):
    """Assert that a recording has all the standard attributes."""
    for key in (
        "application", "created_localtime", "created_utc",
        "ended_localtime", "ended_utc",
        "channel_count", "sampling_frequency", "frames_per_second",
        "uV_per_sample_unit", "start_timestamp", "end_timestamp",
        "duration_frames", "duration_seconds", "file_format",
    ):
        assert key in attrs, f"Missing attribute: {key}"
    assert "version" in attrs["file_format"]
    assert "stim_and_spike_timestamps_relative_to_start" in attrs["file_format"]


def _assert_sample_shape_matches_attributes(view: RecordingView, *, exact: bool = True):
    """Assert sample array shape matches the attributes."""
    attrs = view.attributes
    assert view.samples is not None
    sample_frames, sample_channels = view.samples.shape
    if exact:
        assert attrs["duration_frames"] == sample_frames
    else:
        # Allow small delta for non-accelerated mode timing imprecision
        assert abs(attrs["duration_frames"] - sample_frames) < 50, \
            f"duration_frames={attrs['duration_frames']} vs samples={sample_frames}"
    assert attrs["channel_count"]   == sample_channels


def _assert_timestamps_relative_to_start(view: RecordingView):
    """Assert that spike and stim timestamps are relative to start and within duration."""
    duration = view.attributes["duration_frames"]
    if view.spikes is not None:
        for spike in view.spikes:
            assert 0 <= spike["timestamp"] <= duration, \
                f"Spike timestamp {spike['timestamp']} out of range [0, {duration}]"
    if view.stims is not None:
        for stim in view.stims:
            assert 0 <= stim["timestamp"] <= duration, \
                f"Stim timestamp {stim['timestamp']} out of range [0, {duration}]"

def test_recording(mocker: MockerFixture, tmp_path: Path):
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    # Fix the datetime since the output file depends on it
    mock_datetime = mocker.patch("cl.recording.datetime")
    mock_datetime.now.return_value = datetime(2022, 12, 7, 12, 0, 0)

    # Main operation to test
    duration_sec = 20.0
    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        recording = neurons.record(file_location=str(tmp_path))
        timestamp = recording.start_timestamp

        for tick in neurons.loop(ticks_per_second=100, stop_after_seconds=duration_sec):
            neurons.stim(ChannelSet(8, 9), StimDesign(160, -1, 160, 1))

        data_stream_data_dict  = { "foo": "bar" }
        data_stream_data_list  = [ 1, 2, 3 ]
        data_stream_data_str   = "test_string"
        data_stream_data_array = np.array([ 1, 2, 3 ])

        data_stream = neurons.create_data_stream(
            name="test_data_stream",
            attributes={ "hello": "world" }
        )
        data_stream.append(timestamp + 0, data_stream_data_dict)
        data_stream.append(timestamp + 1, data_stream_data_list)
        data_stream.append(timestamp + 2, data_stream_data_str)
        data_stream.append(timestamp + 3, data_stream_data_array)
        data_stream.set_attribute("score", 1)
        data_stream.update_attributes({ "score": 2, "new_attribute": 9.9 })

        with pytest.raises(RuntimeError):
            recording.open()

        recording.stop()
        recording.wait_until_stopped()

    # Check that the recording was created successfully
    expected_fname_prefix  = _utcdatestring(mock_datetime.now())
    expected_fname_postfix = "recording"
    expected_fname         = f"{expected_fname_prefix}_{expected_fname_postfix}.h5"
    expected_fpath         = tmp_path / expected_fname
    assert expected_fpath.exists()

    # Load and check the recording as a RecordingView
    recording_view: RecordingView = RecordingView(str(expected_fpath))

    assert hasattr(recording_view, "spikes")
    assert hasattr(recording_view, "stims")
    assert hasattr(recording_view, "samples")
    assert hasattr(recording_view, "attributes")

    # Check the recording attributes
    attributes: AttributesView = recording_view.attributes
    assert "application" in attributes
    assert "created_localtime" in attributes
    assert "created_utc" in attributes
    assert "ended_localtime" in attributes
    assert "ended_utc" in attributes
    assert "channel_count" in attributes
    assert "sampling_frequency" in attributes
    assert "frames_per_second" in attributes
    assert "uV_per_sample_unit" in attributes
    assert "start_timestamp" in attributes
    assert "end_timestamp" in attributes
    assert "duration_frames" in attributes
    assert "duration_seconds" in attributes
    assert "file_format" in attributes
    assert "version" in attributes["file_format"]
    assert "stim_and_spike_timestamps_relative_to_start" in attributes["file_format"]

    # Check the duration
    recording_duration = attributes["duration_seconds"]
    start_timestamp    = attributes["start_timestamp"]
    end_timestamp      = attributes["end_timestamp"]
    duration_frames    = attributes["duration_frames"]
    assert_allclose(recording_duration, duration_sec, rtol=0.1)
    assert end_timestamp == start_timestamp + duration_frames

    # Check sample shape
    recording_frames   = attributes["duration_frames"]
    recording_channels = attributes["channel_count"]

    assert recording_view.samples is not None
    sample_frames, sample_channels = recording_view.samples.shape
    assert recording_frames   == sample_frames
    assert recording_channels == sample_channels

    # Check spike timestamps are relative to recording start_timestamp
    assert recording_view.spikes is not None
    for spike in recording_view.spikes:
        assert 0 <= spike["timestamp"] <= duration_frames

    # Check stim timestamps are are relative to recording start_timestamp
    assert recording_view.stims is not None
    for stim in recording_view.stims:
        assert 0 <= stim["timestamp"] <= duration_frames

    # Check datastreams
    assert recording_view.data_streams is not None
    assert hasattr(recording_view.data_streams, "test_data_stream")

    test_data_stream = recording_view.data_streams.test_data_stream
    assert test_data_stream.attributes["name"] == "test_data_stream"
    assert test_data_stream.attributes["application"] == snapshot({
        "hello": "world",
        "score": 2,
        "new_attribute": 9.9
    })
    recording_start_timestamp = recording_view.attributes["start_timestamp"]
    assert list(test_data_stream.keys()) == snapshot([
        timestamp - recording_start_timestamp + 0,
        timestamp - recording_start_timestamp + 1,
        timestamp - recording_start_timestamp + 2,
        timestamp - recording_start_timestamp + 3
    ])
    actual_values = list(test_data_stream.values())
    expected_values = [
        data_stream_data_dict,
        data_stream_data_list,
        data_stream_data_str,
        data_stream_data_array
    ]
    for actual, expected in zip(actual_values, expected_values):
        if isinstance(expected, np.ndarray):
            np.testing.assert_allclose(actual, expected)
        else:
            assert actual == expected

def test_recording_frame_correctness(tmp_path: Path):
    """ Tests that neurons.read() is accurate, especially with wrapping replay file. """
    from tests.conftest import generate_recording

    os.environ["CL_SDK_ACCELERATED_TIME"]    = "1"
    os.environ["CL_SDK_REPLAY_START_OFFSET"] = "0"

    # Generate a recording from the on-the-fly random source, then replay it
    # back as a file so we can verify neurons.read() reproduces it exactly.
    os.environ.pop("CL_SDK_REPLAY_PATH", None)
    replay_path = str(generate_recording(tmp_path, duration_sec=2))
    os.environ["CL_SDK_REPLAY_PATH"] = replay_path
    try:
        with cl.open() as neurons:
            replay_view     = RecordingView(replay_path)
            replay_samples  = replay_view.samples
            replay_duration = int(replay_view.attributes["duration_frames"])  # 25kHz file frames
            start_ts        = neurons.timestamp()
            wrap_times      = 2
            # Read from start_timestamp explicitly to align with replay_offset
            frames          = neurons.read(replay_duration * wrap_times, start_ts)
            step            = 100

            assert replay_samples is not None
            assert frames.shape[0] == replay_samples.shape[0] * wrap_times

            for i in range(0, frames.shape[0], step):
                t = (i + neurons._replay_start_offset) % replay_duration
                np.testing.assert_allclose(replay_samples[t : t+step, :], frames[i : i + step, :]), f"{frames.shape[0]=} {i=}, {t=}"
            print("Pass!")
            replay_view.close()
    finally:
        os.environ.pop("CL_SDK_REPLAY_PATH", None)



# ─────────────────────────────────────────────────────────────────────────────
# Non-accelerated recording tests
# ─────────────────────────────────────────────────────────────────────────────

def test_recording_non_accelerated_with_sleep(tmp_path: Path):
    """
    Recording in real-time mode for >5 seconds using time.sleep(), verifying
    that the subprocess writes data continuously (not just at close time).
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    duration_sec = 6.0

    with cl.open() as neurons:
        neurons.restart()

        recording = neurons.record(
            file_location=str(tmp_path),
            stop_after_seconds=duration_sec,
        )

        # Sleep for the recording duration + margin; the data producer runs
        # in real time and the recording subprocess polls independently.
        time.sleep(duration_sec + 1.0)

        recording.wait_until_stopped()

    # Verify the recording
    h5_files = list(tmp_path.glob("*.h5"))
    assert len(h5_files) == 1, f"Expected 1 recording file, got {len(h5_files)}"

    view = _open_recording(h5_files[0])
    try:
        _assert_valid_recording_attributes(view.attributes)
        _assert_sample_shape_matches_attributes(view, exact=False)

        # Duration should match what we asked for, within tolerance
        assert_allclose(view.attributes["duration_seconds"], duration_sec, rtol=0.15)

        # Must be >5s of data (exceeding ring buffer size)
        assert view.attributes["duration_seconds"] > 5.0
    finally:
        view.close()


def test_recording_non_accelerated_with_loop(tmp_path: Path):
    """
    Recording in non-accelerated mode using neurons.loop() for >5 seconds,
    verifying data is captured continuously.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    duration_sec = 6.0

    with cl.open() as neurons:
        neurons.restart()

        recording = neurons.record(file_location=str(tmp_path))

        for tick in neurons.loop(ticks_per_second=10, stop_after_seconds=duration_sec):
            pass

        recording.stop()

    h5_files = list(tmp_path.glob("*.h5"))
    assert len(h5_files) == 1

    view = _open_recording(h5_files[0])
    try:
        _assert_valid_recording_attributes(view.attributes)
        _assert_sample_shape_matches_attributes(view, exact=False)
        assert_allclose(view.attributes["duration_seconds"], duration_sec, rtol=0.15)
        assert view.attributes["duration_seconds"] > 5.0
    finally:
        view.close()


# ─────────────────────────────────────────────────────────────────────────────
# Multiple overlapping recordings
# ─────────────────────────────────────────────────────────────────────────────

def test_multiple_overlapping_recordings_accelerated(tmp_path: Path):
    """
    Two recordings running simultaneously in accelerated mode.
    Both should capture the full overlapping period independently.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    duration_sec = 5.0

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        rec_a = neurons.record(file_location=str(dir_a), file_suffix="alpha")

        # Run a loop so some frames are produced before the second recording starts
        for tick in neurons.loop(ticks_per_second=100, stop_after_seconds=1.0):
            pass

        overlap_start_ts = neurons.timestamp()
        rec_b = neurons.record(file_location=str(dir_b), file_suffix="beta")

        # Run more frames while both recordings are active
        for tick in neurons.loop(ticks_per_second=100, stop_after_seconds=duration_sec - 1.0):
            neurons.stim(ChannelSet(8), StimDesign(160, -1, 160, 1))

        overlap_end_ts = neurons.timestamp()

        rec_a.stop()
        rec_b.stop()

    files_a = list(dir_a.glob("*.h5"))
    files_b = list(dir_b.glob("*.h5"))
    assert len(files_a) == 1 and len(files_b) == 1

    view_a = _open_recording(files_a[0])
    view_b = _open_recording(files_b[0])
    try:
        _assert_valid_recording_attributes(view_a.attributes)
        _assert_valid_recording_attributes(view_b.attributes)
        _assert_sample_shape_matches_attributes(view_a)
        _assert_sample_shape_matches_attributes(view_b)
        _assert_timestamps_relative_to_start(view_a)
        _assert_timestamps_relative_to_start(view_b)

        # Recording A started earlier, so it should be longer
        assert view_a.attributes["duration_frames"] > view_b.attributes["duration_frames"]

        # Both should have stims (from the overlapping period)
        assert view_a.stims is not None and len(view_a.stims) > 0
        assert view_b.stims is not None and len(view_b.stims) > 0

        # Recording B started later, so all its stims should also be in A
        assert len(view_a.stims) >= len(view_b.stims)
    finally:
        view_a.close()
        view_b.close()


def test_multiple_overlapping_recordings_non_accelerated(tmp_path: Path):
    """
    Two recordings running simultaneously in real-time mode.
    Both should capture the full overlapping period.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    overlap_sec = 2.0

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    with cl.open() as neurons:
        neurons.restart()

        rec_a = neurons.record(file_location=str(dir_a), file_suffix="alpha")
        time.sleep(1.0)

        rec_b = neurons.record(file_location=str(dir_b), file_suffix="beta")
        time.sleep(overlap_sec)

        rec_b.stop()
        rec_a.stop()

    files_a = list(dir_a.glob("*.h5"))
    files_b = list(dir_b.glob("*.h5"))
    assert len(files_a) == 1 and len(files_b) == 1

    view_a = _open_recording(files_a[0])
    view_b = _open_recording(files_b[0])
    try:
        _assert_valid_recording_attributes(view_a.attributes)
        _assert_valid_recording_attributes(view_b.attributes)
        _assert_sample_shape_matches_attributes(view_a, exact=False)
        _assert_sample_shape_matches_attributes(view_b, exact=False)

        # A started earlier, so it should have more frames
        assert view_a.attributes["duration_frames"] > view_b.attributes["duration_frames"]

        # Both should have real data (not empty)
        assert view_a.attributes["duration_seconds"] > 2.5
        assert view_b.attributes["duration_seconds"] > 1.5
    finally:
        view_a.close()
        view_b.close()


# ─────────────────────────────────────────────────────────────────────────────
# Consecutive cl.open() contexts (no data leaks)
# ─────────────────────────────────────────────────────────────────────────────

def test_consecutive_contexts_no_leak_accelerated(tmp_path: Path):
    """
    Recordings in consecutive cl.open() contexts should not have data
    from the other context written to them (accelerated mode).
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    dir_a = tmp_path / "ctx1"
    dir_b = tmp_path / "ctx2"
    dir_a.mkdir()
    dir_b.mkdir()

    # Context 1: record 3 seconds
    with cl.open() as neurons:
        neurons._elapsed_frames = 0
        rec_a = neurons.record(
            file_location=str(dir_a),
            stop_after_seconds=3.0,
        )
        rec_a.wait_until_stopped()

    # Context 2: record 3 seconds
    with cl.open() as neurons:
        neurons._elapsed_frames = 0
        rec_b = neurons.record(
            file_location=str(dir_b),
            stop_after_seconds=3.0,
        )
        rec_b.wait_until_stopped()

    files_a = list(dir_a.glob("*.h5"))
    files_b = list(dir_b.glob("*.h5"))
    assert len(files_a) == 1 and len(files_b) == 1

    view_a = _open_recording(files_a[0])
    view_b = _open_recording(files_b[0])
    try:
        _assert_valid_recording_attributes(view_a.attributes)
        _assert_valid_recording_attributes(view_b.attributes)
        _assert_sample_shape_matches_attributes(view_a)
        _assert_sample_shape_matches_attributes(view_b)

        # Both should be ~3 seconds
        assert_allclose(view_a.attributes["duration_seconds"], 3.0, rtol=0.1)
        assert_allclose(view_b.attributes["duration_seconds"], 3.0, rtol=0.1)

        # File sizes should be similar since durations are the same;
        # a leak would cause one to be larger.
        size_a = files_a[0].stat().st_size
        size_b = files_b[0].stat().st_size
        assert_allclose(size_a, size_b, rtol=0.2)
    finally:
        view_a.close()
        view_b.close()


def test_consecutive_contexts_no_leak_non_accelerated(tmp_path: Path):
    """
    Recordings in consecutive cl.open() contexts should not have data
    from the other context written to them (non-accelerated mode).
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    rec_sec = 2.0

    dir_a = tmp_path / "ctx1"
    dir_b = tmp_path / "ctx2"
    dir_a.mkdir()
    dir_b.mkdir()

    # Context 1
    with cl.open() as neurons:
        neurons.restart()
        rec_a = neurons.record(
            file_location=str(dir_a),
            stop_after_seconds=rec_sec,
        )
        time.sleep(rec_sec + 0.5)
        rec_a.wait_until_stopped()

    # Context 2
    with cl.open() as neurons:
        neurons.restart()
        rec_b = neurons.record(
            file_location=str(dir_b),
            stop_after_seconds=rec_sec,
        )
        time.sleep(rec_sec + 0.5)
        rec_b.wait_until_stopped()

    files_a = list(dir_a.glob("*.h5"))
    files_b = list(dir_b.glob("*.h5"))
    assert len(files_a) == 1 and len(files_b) == 1

    view_a = _open_recording(files_a[0])
    view_b = _open_recording(files_b[0])
    try:
        _assert_valid_recording_attributes(view_a.attributes)
        _assert_valid_recording_attributes(view_b.attributes)
        _assert_sample_shape_matches_attributes(view_a, exact=False)
        _assert_sample_shape_matches_attributes(view_b, exact=False)

        assert_allclose(view_a.attributes["duration_seconds"], rec_sec, rtol=0.15)
        assert_allclose(view_b.attributes["duration_seconds"], rec_sec, rtol=0.15)

        size_a = files_a[0].stat().st_size
        size_b = files_b[0].stat().st_size
        assert_allclose(size_a, size_b, rtol=0.25)
    finally:
        view_a.close()
        view_b.close()


# ─────────────────────────────────────────────────────────────────────────────
# Stim, spike, and data stream timing correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_stim_timing_in_recording(tmp_path: Path):
    """
    Verify that stim timestamps in a recording are within the recording window
    and appear on the correct channels.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        recording = neurons.record(file_location=str(tmp_path))

        stim_count = 0
        for tick in neurons.loop(ticks_per_second=100, stop_after_seconds=2.0):
            if tick.iteration % 10 == 0:
                neurons.stim(ChannelSet(8), StimDesign(160, -1, 160, 1))
                stim_count += 1

        recording.stop()

    h5_files = list(tmp_path.glob("*.h5"))
    assert len(h5_files) == 1

    view = _open_recording(h5_files[0])
    try:
        _assert_timestamps_relative_to_start(view)

        assert view.stims is not None
        stim_channel_8 = [s for s in view.stims if s["channel"] == 8]

        # We issued stims on every 10th tick for 200 ticks → 20 stims
        assert len(stim_channel_8) == stim_count, \
            f"Expected {stim_count} stims on ch8, got {len(stim_channel_8)}"

        # All stim timestamps should be within the recording window
        duration = view.attributes["duration_frames"]
        for stim in stim_channel_8:
            assert 0 <= stim["timestamp"] <= duration, \
                f"Stim timestamp {stim['timestamp']} outside [0, {duration}]"
    finally:
        view.close()


def test_spike_timing_in_recording(tmp_path: Path):
    """
    Verify that spike timestamps are captured and relative to recording start.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        recording = neurons.record(file_location=str(tmp_path))

        for tick in neurons.loop(ticks_per_second=100, stop_after_seconds=5.0):
            pass

        recording.stop()

    h5_files = list(tmp_path.glob("*.h5"))
    assert len(h5_files) == 1

    view = _open_recording(h5_files[0])
    try:
        _assert_timestamps_relative_to_start(view)

        assert view.spikes is not None
        duration = view.attributes["duration_frames"]

        # The replay data should produce spikes
        if len(view.spikes) > 0:
            # Verify all spikes are within the recording window
            for spike in view.spikes:
                assert 0 <= spike["timestamp"] <= duration
                assert 0 <= spike["channel"] < view.attributes["channel_count"]
                # Spike samples should be the expected length (75 samples)
                assert len(spike["samples"]) == 75
    finally:
        view.close()


def test_data_stream_timing_in_recording(tmp_path: Path):
    """
    Verify data stream events have correct timestamps relative to recording start,
    and that data is correctly serialised/deserialised.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        ds = neurons.create_data_stream(
            name="timing_stream",
            attributes={"purpose": "timing_test"}
        )

        recording = neurons.record(file_location=str(tmp_path))
        rec_start = recording.start_timestamp

        # Issue events at known timestamps during the loop
        event_timestamps = []
        for tick in neurons.loop(ticks_per_second=100, stop_after_seconds=2.0):
            if tick.iteration % 25 == 0:  # Every 250ms
                ts = tick.iteration_timestamp
                ds.append(ts, {"tick": tick.iteration, "ts": ts})
                event_timestamps.append(ts)

        recording.stop()

    h5_files = list(tmp_path.glob("*.h5"))
    assert len(h5_files) == 1

    view = _open_recording(h5_files[0])
    try:
        assert view.data_streams is not None
        assert hasattr(view.data_streams, "timing_stream")

        stream = view.data_streams.timing_stream
        assert stream.attributes["name"] == "timing_stream"
        assert stream.attributes["application"]["purpose"] == "timing_test"

        keys = list(stream.keys())
        assert len(keys) == len(event_timestamps), \
            f"Expected {len(event_timestamps)} events, got {len(keys)}"

        # Keys should be relative to recording start
        for key, abs_ts in zip(keys, event_timestamps):
            expected_relative = abs_ts - rec_start
            assert key == expected_relative, \
                f"Data stream key {key} != expected {expected_relative}"

        # Verify data round-trips correctly
        values = list(stream.values())
        for val, abs_ts in zip(values, event_timestamps):
            assert val["ts"] == abs_ts
    finally:
        view.close()


# ─────────────────────────────────────────────────────────────────────────────
# neurons.read() during recording should NOT cause duplicate data
# ─────────────────────────────────────────────────────────────────────────────

def test_read_during_recording_no_duplicates(tmp_path: Path):
    """
    Calling neurons.read() multiple times during a recording should not
    cause duplicate or overlapping data in the recording file.

    This was the original bug: data forwarded via neurons.read() → recordings
    caused overlapping data when read() was called multiple times.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    duration_sec = 5.0

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        recording = neurons.record(file_location=str(tmp_path))

        # Call neurons.read() many times during the recording loop —
        # this used to cause duplicate frames.
        for tick in neurons.loop(ticks_per_second=100, stop_after_seconds=duration_sec):
            # Explicit read of the same data the loop already reads
            neurons.read(250, tick.analysis.start_timestamp)

        recording.stop()

    h5_files = list(tmp_path.glob("*.h5"))
    assert len(h5_files) == 1

    view = _open_recording(h5_files[0])
    try:
        _assert_valid_recording_attributes(view.attributes)
        _assert_sample_shape_matches_attributes(view)

        expected_frames = int(duration_sec * view.attributes["frames_per_second"])
        actual_frames   = view.attributes["duration_frames"]

        # The number of recorded frames should match the recording duration,
        # NOT be inflated by the extra neurons.read() calls.
        assert_allclose(actual_frames, expected_frames, rtol=0.05), \
            f"Frame count {actual_frames} should be ~{expected_frames}, not inflated by read() calls"

        assert view.attributes["end_timestamp"] == \
            view.attributes["start_timestamp"] + view.attributes["duration_frames"]
    finally:
        view.close()


def test_read_during_recording_no_overlapping_data_content(tmp_path: Path):
    """
    Verify the actual sample content is not duplicated when neurons.read()
    is called during a recording in non-accelerated mode.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    duration_sec = 3.0

    with cl.open() as neurons:
        neurons.restart()

        recording = neurons.record(file_location=str(tmp_path))

        for tick in neurons.loop(ticks_per_second=10, stop_after_seconds=duration_sec):
            neurons.read(250, tick.analysis.start_timestamp)

        recording.stop()

    h5_files = list(tmp_path.glob("*.h5"))
    assert len(h5_files) == 1

    view = _open_recording(h5_files[0])
    try:
        _assert_valid_recording_attributes(view.attributes)
        _assert_sample_shape_matches_attributes(view, exact=False)

        assert view.samples is not None
        frames = view.attributes["duration_frames"]
        expected = int(duration_sec * view.attributes["frames_per_second"])

        # Frames should be close to the expected count, not doubled.
        # Non-accelerated mode has wall-clock jitter; allow up to 25%.
        assert_allclose(frames, expected, rtol=0.25)
        # But must not be 2x (the original duplication bug)
        assert frames < expected * 1.5
    finally:
        view.close()


# ─────────────────────────────────────────────────────────────────────────────
# Stim-specific recording tests
# ─────────────────────────────────────────────────────────────────────────────

def test_stims_recorded_to_correct_channels(tmp_path: Path):
    """
    Verify that stims applied to specific channels appear on those channels
    in the recording, and not on others.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    stim_channels = [8, 9, 16, 17]

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        recording = neurons.record(file_location=str(tmp_path))

        for tick in neurons.loop(ticks_per_second=100, stop_after_seconds=2.0):
            if tick.iteration % 20 == 0:
                neurons.stim(ChannelSet(*stim_channels), StimDesign(160, -1, 160, 1))

        recording.stop()

    h5_files = list(tmp_path.glob("*.h5"))
    assert len(h5_files) == 1

    view = _open_recording(h5_files[0])
    try:
        assert view.stims is not None
        assert len(view.stims) > 0

        recorded_channels = set(int(s["channel"]) for s in view.stims)

        # All recorded stim channels should be in our target set
        for ch in recorded_channels:
            assert ch in stim_channels, f"Unexpected stim on channel {ch}"

        # All target channels should have stims
        for ch in stim_channels:
            assert ch in recorded_channels, f"Missing stims on channel {ch}"
    finally:
        view.close()


def test_stims_across_overlapping_recordings(tmp_path: Path):
    """
    When two recordings overlap, stims issued during the overlap should
    appear in both recordings.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        rec_a = neurons.record(file_location=str(dir_a))

        # Run 1s without stims, then start second recording
        for tick in neurons.loop(ticks_per_second=100, stop_after_seconds=1.0):
            pass

        rec_b = neurons.record(file_location=str(dir_b))

        # Stim on every tick during overlap period
        stim_count = 0
        for tick in neurons.loop(ticks_per_second=100, stop_after_seconds=2.0):
            neurons.stim(ChannelSet(10), StimDesign(160, -1, 160, 1))
            stim_count += 1

        rec_a.stop()
        rec_b.stop()

    view_a = _open_recording(list(dir_a.glob("*.h5"))[0])
    view_b = _open_recording(list(dir_b.glob("*.h5"))[0])
    try:
        assert view_a.stims is not None
        assert view_b.stims is not None

        stims_a_ch10 = [s for s in view_a.stims if s["channel"] == 10]
        stims_b_ch10 = [s for s in view_b.stims if s["channel"] == 10]

        # Both recordings should have captured the stims from the overlap period
        assert len(stims_a_ch10) > 0
        assert len(stims_b_ch10) > 0

        # Recording B started during the stim period so should have all overlap stims.
        # Recording A contains the full period so should have at least as many.
        assert len(stims_a_ch10) >= len(stims_b_ch10)
    finally:
        view_a.close()
        view_b.close()


# ─────────────────────────────────────────────────────────────────────────────
# Recording with stop_after_seconds (auto-stop)
# ─────────────────────────────────────────────────────────────────────────────

def test_recording_auto_stop_accelerated(tmp_path: Path):
    """
    A recording with stop_after_seconds should auto-stop and produce
    a file with the correct duration (accelerated mode).
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    auto_stop_sec = 3.0

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        recording = neurons.record(
            file_location=str(tmp_path),
            stop_after_seconds=auto_stop_sec,
        )
        recording.wait_until_stopped()

    h5_files = list(tmp_path.glob("*.h5"))
    assert len(h5_files) == 1

    view = _open_recording(h5_files[0])
    try:
        _assert_valid_recording_attributes(view.attributes)
        _assert_sample_shape_matches_attributes(view)
        assert_allclose(view.attributes["duration_seconds"], auto_stop_sec, rtol=0.05)
    finally:
        view.close()


def test_recording_auto_stop_non_accelerated(tmp_path: Path):
    """
    A recording with stop_after_seconds should auto-stop in real-time mode.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "0"
    auto_stop_sec = 2.0

    with cl.open() as neurons:
        neurons.restart()

        recording = neurons.record(
            file_location=str(tmp_path),
            stop_after_seconds=auto_stop_sec,
        )
        # Sleep long enough for the recording to finish
        time.sleep(auto_stop_sec + 1.0)
        recording.wait_until_stopped()

    h5_files = list(tmp_path.glob("*.h5"))
    assert len(h5_files) == 1

    view = _open_recording(h5_files[0])
    try:
        _assert_valid_recording_attributes(view.attributes)
        _assert_sample_shape_matches_attributes(view, exact=False)
        assert_allclose(view.attributes["duration_seconds"], auto_stop_sec, rtol=0.15)
    finally:
        view.close()


# ─────────────────────────────────────────────────────────────────────────────
# Recording with include/exclude options
# ─────────────────────────────────────────────────────────────────────────────

def test_recording_exclude_raw_samples(tmp_path: Path):
    """
    Recording with include_raw_samples=False should produce a file without samples.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        recording = neurons.record(
            file_location=str(tmp_path),
            include_raw_samples=False,
            stop_after_seconds=2.0,
        )
        recording.wait_until_stopped()

    view = _open_recording(list(tmp_path.glob("*.h5"))[0])
    try:
        assert view.samples is None or view.samples.shape[0] == 0
    finally:
        view.close()


def test_recording_exclude_spikes(tmp_path: Path):
    """
    Recording with include_spikes=False should produce a file without spikes.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        recording = neurons.record(
            file_location=str(tmp_path),
            include_spikes=False,
            stop_after_seconds=2.0,
        )
        recording.wait_until_stopped()

    view = _open_recording(list(tmp_path.glob("*.h5"))[0])
    try:
        assert view.spikes is None or len(view.spikes) == 0
    finally:
        view.close()


def test_recording_exclude_stims(tmp_path: Path):
    """
    Recording with include_stims=False should not capture stims.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        recording = neurons.record(
            file_location=str(tmp_path),
            include_stims=False,
        )

        for tick in neurons.loop(ticks_per_second=100, stop_after_seconds=2.0):
            neurons.stim(ChannelSet(8), StimDesign(160, -1, 160, 1))

        recording.stop()

    view = _open_recording(list(tmp_path.glob("*.h5"))[0])
    try:
        assert view.stims is None or len(view.stims) == 0
    finally:
        view.close()


def test_recording_exclude_data_stream(tmp_path: Path):
    """
    Recording with exclude_data_streams should omit the named stream.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        ds_included = neurons.create_data_stream(name="included")
        ds_excluded = neurons.create_data_stream(name="excluded")

        recording = neurons.record(
            file_location=str(tmp_path),
            exclude_data_streams=["excluded"],
        )

        ts = neurons.timestamp()
        ds_included.append(ts + 1, {"a": 1})
        ds_excluded.append(ts + 1, {"b": 2})

        recording.stop()

    view = _open_recording(list(tmp_path.glob("*.h5"))[0])
    try:
        assert view.data_streams is not None
        stream_names = list(view.data_streams.keys())
        assert "included" in stream_names
        assert "excluded" not in stream_names
    finally:
        view.close()


# ─────────────────────────────────────────────────────────────────────────────
# Recording application attributes
# ─────────────────────────────────────────────────────────────────────────────

def test_recording_custom_attributes(tmp_path: Path):
    """
    Custom application attributes set via record(), set_attribute(), and
    update_attributes() should all be persisted in the recording file.
    """
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        recording = neurons.record(
            file_location=str(tmp_path),
            attributes={"experiment": "test_1", "version": 1},
        )

        recording.set_attribute("version", 2)
        recording.update_attributes({"added_later": True, "score": 42.5})

        # Use a loop + explicit stop() so that updated attributes are sent
        # via StopCmd to the recording subprocess.
        for tick in neurons.loop(ticks_per_second=100, stop_after_seconds=1.0):
            pass

        recording.stop()

    view = _open_recording(list(tmp_path.glob("*.h5"))[0])
    try:
        app_attrs = view.attributes["application"]
        assert app_attrs["experiment"] == "test_1"
        assert app_attrs["version"] == 2
        assert app_attrs["added_later"] is True
        assert app_attrs["score"] == 42.5
    finally:
        view.close()


# ─────────────────────────────────────────────────────────────────────────────
# Recording file_suffix
# ─────────────────────────────────────────────────────────────────────────────

def test_recording_file_suffix(tmp_path: Path):
    """Recording with a custom file_suffix should use it in the filename."""
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        recording = neurons.record(
            file_location=str(tmp_path),
            file_suffix="my_experiment",
            stop_after_seconds=1.0,
        )
        recording.wait_until_stopped()

    h5_files = list(tmp_path.glob("*.h5"))
    assert len(h5_files) == 1
    assert "my_experiment" in h5_files[0].name


# ─────────────────────────────────────────────────────────────────────────────
# Recording.open() before / after stop
# ─────────────────────────────────────────────────────────────────────────────

def test_recording_open_before_stop_raises(tmp_path: Path):
    """Calling recording.open() before stopping should raise RuntimeError."""
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        recording = neurons.record(
            file_location=str(tmp_path),
            stop_after_seconds=1.0,
        )

        with pytest.raises(RuntimeError):
            recording.open()

        recording.wait_until_stopped()


def test_recording_open_after_stop_returns_view(tmp_path: Path):
    """Calling recording.open() after stopping returns a RecordingView."""
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"

    with cl.open() as neurons:
        neurons._elapsed_frames = 0

        recording = neurons.record(
            file_location=str(tmp_path),
            stop_after_seconds=1.0,
        )
        recording.wait_until_stopped()

        view = recording.open()
        assert isinstance(view, RecordingView)
        _assert_valid_recording_attributes(view.attributes)
        view.close()


# ─────────────────────────────────────────────────────────────────────────────
# RecordingView lifecycle tests
# ─────────────────────────────────────────────────────────────────────────────

def test_recording_view_context_manager_closes_file(tmp_path: Path):
    """RecordingView should close its PyTables file when a context exits."""
    path = tmp_path / "recording_view_context.h5"
    _write_minimal_recording(path)

    with _open_recording(path) as view:
        file = view.file
        assert file.isopen

    assert not file.isopen


def test_recording_view_context_manager_closes_file_after_exception(tmp_path: Path):
    """RecordingView should close its PyTables file and propagate context errors."""
    path = tmp_path / "recording_view_context_exception.h5"
    _write_minimal_recording(path)
    file_holder = {}

    with pytest.raises(ValueError, match="context failure"):
        with _open_recording(path) as view:
            file_holder["file"] = view.file
            raise ValueError("context failure")

    assert not file_holder["file"].isopen


def test_recording_view_opens_multiple_views_simultaneously(tmp_path: Path):
    """Multiple RecordingView instances should be open at the same time."""
    path_a = tmp_path / "recording_view_a.h5"
    path_b = tmp_path / "recording_view_b.h5"
    _write_minimal_recording(path_a)
    _write_minimal_recording(path_b)

    with _open_recording(path_a) as view_a:
        file_a = view_a.file
        with _open_recording(path_b) as view_b:
            file_b = view_b.file

            assert file_a.isopen
            assert file_b.isopen
            assert view_a.attributes["frames_per_second"] == 25_000
            assert view_b.attributes["frames_per_second"] == 25_000

        assert file_a.isopen
        assert not file_b.isopen

    assert not file_a.isopen
    assert not file_b.isopen


def test_recording_view_close_is_idempotent(tmp_path: Path):
    """RecordingView.close() should be safe to call repeatedly."""
    path = tmp_path / "recording_view_idempotent.h5"
    _write_minimal_recording(path)

    view = _open_recording(path)
    file = view.file

    view.close()
    view.close()

    assert not file.isopen


def test_recording_view_closes_file_when_initialisation_fails(
    tmp_path:    Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """RecordingView should close the HDF5 file if validation fails after open."""
    path = tmp_path / "recording_view_initialisation_failure.h5"
    _write_minimal_recording(path)

    original_open_file = tb.open_file
    opened_files       = []

    class _FailingWalkNodesFile:
        """File wrapper that fails after PyTables has opened the real file."""

        def __init__(self, file):
            self._file = file

        @property
        def isopen(self):
            """Whether the wrapped PyTables file is open."""
            return self._file.isopen

        def close(self):
            """Close the wrapped PyTables file."""
            self._file.close()

        def walk_nodes(self, *args, **kwargs):
            """Raise after open to simulate validation failing."""
            raise RuntimeError("walk_nodes failed")

    def _open_file_with_failing_walk_nodes(*args, **kwargs):
        """Open a real HDF5 file, then fail when RecordingView validates it."""
        file = original_open_file(*args, **kwargs)
        opened_files.append(file)
        return _FailingWalkNodesFile(file)

    monkeypatch.setattr(tb, "open_file", _open_file_with_failing_walk_nodes)

    with pytest.raises(RuntimeError, match="walk_nodes failed"):
        _open_recording(path)

    assert opened_files
    assert not opened_files[0].isopen


def test_recording_view_close_clears_state_when_file_close_raises(
    monkeypatch: pytest.MonkeyPatch,
):
    """RecordingView.close() should clear lifecycle state even if file.close() raises."""

    class _FailingCloseFile:
        """File-like object that reports open but raises on close."""

        isopen = True

        def close(self):
            """Raise while closing to simulate an HDF5 teardown failure."""
            raise RuntimeError("close failed")

    # Isolate the class-level registry so this test only closes views created here.
    monkeypatch.setattr(RecordingView, "_all_open_recordings", set())

    view                  = object.__new__(RecordingView)
    view.file             = _FailingCloseFile()
    view._is_open         = True
    RecordingView._all_open_recordings.add(view)

    with pytest.raises(RuntimeError, match="close failed"):
        view.close()

    assert not view._is_open
    assert view not in RecordingView._all_open_recordings


def test_recording_view_global_cleanup_closes_open_views(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The module cleanup helper should close views left open by users."""
    path = tmp_path / "recording_view_cleanup.h5"
    _write_minimal_recording(path)

    # Isolate the class-level registry so this test only closes views created here.
    monkeypatch.setattr(RecordingView, "_all_open_recordings", set())

    view = _open_recording(path)
    file = view.file

    assert file.isopen
    RecordingView._close_all_open()
    assert not file.isopen


def test_recording_view_global_cleanup_closes_multiple_unclosed_views(
    tmp_path:   Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The module cleanup helper should close multiple views left open by users."""

    # Isolate the class-level registry so this test only closes views created here.
    monkeypatch.setattr(RecordingView, "_all_open_recordings", set())

    paths = [
        tmp_path / "recording_view_cleanup_a.h5",
        tmp_path / "recording_view_cleanup_b.h5",
        tmp_path / "recording_view_cleanup_c.h5",
    ]
    for path in paths:
        _write_minimal_recording(path)

    views = [
        _open_recording(path)
        for path in paths
    ]
    files = [
        view.file
        for view in views
    ]

    assert all(file.isopen for file in files)

    RecordingView._close_all_open()

    assert all(not file.isopen for file in files)


# ─────────────────────────────────────────────────────────────────────────────
# Unsafe recording file / RCE regression tests
# ─────────────────────────────────────────────────────────────────────────────

def test_recording_allows_safe_pickled_builtin_and_numpy_attributes(tmp_path: Path):
    """Safe builtins and plain numeric NumPy payloads should still load."""
    path = tmp_path / "safe_pickled_attrs.h5"
    expected_array = np.array([1, 2, 3], dtype=np.int16)

    with tb.open_file(str(path), mode="w") as h5:
        _write_minimal_recording_attributes(h5.root._v_attrs)
        h5.root._v_attrs.safe_builtin = {"hello": "world", "items": [1, 2, 3]}
        h5.root._v_attrs.safe_np_scalar = pickle.dumps(np.int64(7), protocol=0)
        h5.root._v_attrs.safe_np_array = pickle.dumps(expected_array, protocol=0)

    view = _open_recording(path)
    try:
        assert view.attributes["safe_builtin"] == {"hello": "world", "items": [1, 2, 3]}
        assert int(view.attributes["safe_np_scalar"]) == 7
        np.testing.assert_array_equal(view.attributes["safe_np_array"], expected_array)
    finally:
        view.close()


def test_recording_rejects_reduce_gadget_in_root_attribute(tmp_path: Path):
    """A raw pickle REDUCE gadget in a root attribute must be rejected."""
    path = tmp_path / "root_reduce_gadget.h5"
    expected = f"{os.getpid.__module__}.{os.getpid.__name__} not in restricted unpickling whitelist"

    _write_recording_attr(path, pickle.dumps(_GetPidOnLoad(), protocol=0))

    _assert_recording_blocked(path, expected)


def test_recording_rejects_reduce_gadget_in_nested_attribute(tmp_path: Path):
    """RecordingView should scan every node, not just the root group."""
    path = tmp_path / "nested_reduce_gadget.h5"
    expected = f"{os.getpid.__module__}.{os.getpid.__name__} not in restricted unpickling whitelist"

    _write_recording_attr(path, pickle.dumps(_GetPidOnLoad(), protocol=0), node_name="nested")

    _assert_recording_blocked(path, expected)


def test_recording_rejects_extension_opcode_gadget_attribute(tmp_path: Path):
    """EXT opcodes should still resolve through the restricted whitelist."""
    path = tmp_path / "extension_reduce_gadget.h5"
    code = 31337
    module = os.getpid.__module__
    name = os.getpid.__name__
    payload = _make_ext2_reduce_payload(code)

    _write_recording_attr(path, payload)

    copyreg.add_extension(module, name, code)
    try:
        _assert_recording_blocked(path, f"{module}.{name} not in restricted unpickling whitelist")
    finally:
        copyreg.remove_extension(module, name, code)


def test_recording_rejects_numpy_object_array_with_gadget_payload(tmp_path: Path):
    """Object-dtype NumPy arrays must be rejected before any embedded gadget runs."""
    path = tmp_path / "numpy_object_array_gadget.h5"
    payload = pickle.dumps(np.array([_GetPidOnLoad()], dtype=object), protocol=0)

    _write_recording_attr(path, payload)

    _assert_recording_blocked(path, "Blocked NumPy dtype containing Python objects")


def test_recording_rejects_direct_numpy_object_dtype_attribute(tmp_path: Path):
    """Direct numpy.ndarray reconstruction with object dtype must be rejected."""
    path = tmp_path / "direct_object_dtype_array.h5"

    _write_recording_attr(path, pickle.dumps(_DirectObjectDtypeArray(), protocol=0))

    _assert_recording_blocked(path, "Blocked NumPy dtype containing Python objects")


def test_recording_rejects_object_atom_vlarray(tmp_path: Path):
    """PyTables ObjectAtom nodes are an RCE surface and must be refused outright."""
    path = tmp_path / "object_atom_vlarray.h5"

    with tb.open_file(str(path), mode="w") as h5:
        _write_minimal_recording_attributes(h5.root._v_attrs)
        node = h5.create_vlarray("/", "payloads", atom=tb.ObjectAtom())
        node.append(_GetPidOnLoad())

    _assert_recording_blocked(path, "ObjectAtom VLArray")
