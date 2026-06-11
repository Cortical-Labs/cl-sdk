import os
from pathlib import Path

import pytest

import numpy as np

from pydantic import PositiveInt

# Disable websocket for tests, needs to be set before importing cl
os.environ["CL_SDK_VISUALISATION"] = "0"

from cl.analysis import AnalysisResult, Array2DInt, Array2DFloat

def test_analysis_object(tmp_path: Path):

    class MockAnalysisResult(AnalysisResult):
        result_A:         str
        result_B:         PositiveInt
        result_arr_int:   Array2DInt
        result_arr_float: Array2DFloat

    save_path = tmp_path / "test_result.json"

    result = {
        "metadata": {
            "file_path":          str(save_path.resolve()),
            "channel_count":      64,
            "sampling_frequency": 25_000,
            "duration_frames":    200,
            "duration_seconds":   200 / 25_000
            },
        "result_A":         "test_result",
        "result_B":         42,
        "result_arr_int":   np.random.randint(0, 2, (64, 200)).tolist(),
        "result_arr_float": np.random.rand(64, 200).tolist()
    }

    analysis_result = MockAnalysisResult(**result)
    analysis_result.save(save_path)

    loaded_result = MockAnalysisResult.from_file(save_path)

    assert result == loaded_result.model_dump()

def test_analysis_functions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Tests the entire analysis suite for errors when running them. Does NOT check
    them for accuracy in the calculated result. Tests both analysing the full
    recording as well as when limiting to specific timestamp range.
    """
    from cl import RecordingView
    from tests.conftest import generate_recording

    monkeypatch.setenv("CL_SDK_ACCELERATED_TIME", "1")
    recording_path = generate_recording(tmp_path, duration_sec=30)
    recording      = RecordingView(str(recording_path))

    def _test_analysis_suite(recording: RecordingView):
        BIN_SIZE_SEC = 0.1
        recording.analyse_criticality(
            bin_size_sec         = BIN_SIZE_SEC,
            percentile_threshold = 0.3,
            )
        recording.analyse_dct_features(k=3)
        recording.analyse_firing_stats(bin_size_sec=BIN_SIZE_SEC)
        recording.analyse_functional_connectivity(bin_size_sec=BIN_SIZE_SEC)
        recording.analyse_information_entropy(bin_size_sec=BIN_SIZE_SEC)
        recording.analyse_information_entropy(bin_size_sec=BIN_SIZE_SEC)
        recording.analyse_lempel_ziv_complexity(bin_size_sec=BIN_SIZE_SEC)
        recording.analyse_network_bursts(
            bin_size_sec   = BIN_SIZE_SEC,
            onset_freq_hz  = 2,
            offset_freq_hz = 0.1,
            )
        recording.analyse_spike_triggered_histogram(
            bin_size_sec = BIN_SIZE_SEC,
            start_sec    = 1.0,
            end_sec      = 1.0,
            num_channels = 3
            )

    # Full recording
    _test_analysis_suite(recording)

    # Limit to timestamps
    with pytest.raises(Exception):
        # Min timestamp < 0
        recording.analysis_timestamp_limit(-1, None)
    with pytest.raises(Exception):
        # Max timestamp > recording duration_frames
        recording.analysis_timestamp_limit(None, recording.attributes["duration_frames"] + 1)

    recording.analysis_timestamp_limit(10_000, 100_000)
    _test_analysis_suite(recording)
