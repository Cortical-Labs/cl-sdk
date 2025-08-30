import os
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

import pytest
from pytest_mock import MockerFixture
from inline_snapshot import snapshot

import cl
from cl import ChannelSet, StimDesign
from cl.recording import utcdatestring
from cl.util import RecordingView, AttributesView

def test_recording(mocker: MockerFixture, tmp_path: Path):
    os.environ["CL_MOCK_ACCELERATED_TIME"] = "1"

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
    expected_fname_prefix  = utcdatestring(mock_datetime.now())
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
            assert np.allclose(actual, expected)
        else:
            assert actual == expected