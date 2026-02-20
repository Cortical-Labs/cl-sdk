from pathlib import Path

import pytest

import numpy as np

from pydantic import PositiveInt

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