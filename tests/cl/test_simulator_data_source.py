import numpy as np
import pytest

import cl
from cl import sim
from tests.cl.custom_data_source import make_source

SOURCE_METADATA = sim.SimulatorDataSourceMetadata(
    channel_count      = 64,
    frames_per_second  = 25_000,
    uV_per_sample_unit = 0.195,
    start_timestamp    = 0,
)
SOURCE_METADATA_JSON = (
    '{"channel_count":64,"frames_per_second":25000,'
    '"uV_per_sample_unit":0.195,"start_timestamp":0}'
)

def test_custom_simulator_data_source_read_and_analysis(monkeypatch):
    monkeypatch.setenv("CL_SDK_ACCELERATED_TIME", "1")
    sim.set_simulator_data_source(
        "tests.cl.custom_data_source:make_source",
        config   = {"spike_channel": 5, "sample_offset": 11},
        metadata = SOURCE_METADATA,
    )

    try:
        with cl.open() as neurons:
            frames = neurons.read(150, 0)
            assert frames.shape == (150, 64)
            np.testing.assert_array_equal(frames[:, 0], np.arange(150, dtype=np.int16))
            np.testing.assert_array_equal(frames[:, 1], np.full(150, 11, dtype=np.int16))

            analysis = neurons.read(150, 0, analysis=True)
            assert len(analysis.spikes) == 1
            assert analysis.spikes[0].timestamp == 125
            assert analysis.spikes[0].channel == 5
    finally:
        sim.clear_simulator_data_source()

def test_custom_simulator_data_source_accepts_importable_callable(monkeypatch):
    monkeypatch.setenv("CL_SDK_ACCELERATED_TIME", "1")
    sim.set_simulator_data_source(
        make_source,
        config   = {"sample_offset": 7},
        metadata = SOURCE_METADATA,
    )

    try:
        with cl.open() as neurons:
            frames = neurons.read(5, 10)
            np.testing.assert_array_equal(frames[:, 0], np.arange(10, 15, dtype=np.int16))
            np.testing.assert_array_equal(frames[:, 1], np.full(5, 7, dtype=np.int16))
    finally:
        sim.clear_simulator_data_source()

def test_custom_simulator_data_source_env(monkeypatch):
    sim.clear_simulator_data_source()
    monkeypatch.setenv("CL_SDK_ACCELERATED_TIME", "1")
    monkeypatch.setenv("CL_SDK_DATA_SOURCE", "tests.cl.custom_data_source:make_source")
    monkeypatch.setenv("CL_SDK_DATA_SOURCE_CONFIG", '{"sample_offset": 13}')
    monkeypatch.setenv("CL_SDK_DATA_SOURCE_METADATA", SOURCE_METADATA_JSON)

    with cl.open() as neurons:
        frames = neurons.read(4, 20)
        np.testing.assert_array_equal(frames[:, 0], np.arange(20, 24, dtype=np.int16))
        np.testing.assert_array_equal(frames[:, 1], np.full(4, 13, dtype=np.int16))

def test_custom_simulator_data_source_receives_stims(monkeypatch):
    sim.clear_simulator_data_source()
    monkeypatch.setenv("CL_SDK_ACCELERATED_TIME", "1")
    sim.set_simulator_data_source(
        "tests.cl.custom_data_source:make_stim_aware_source",
        metadata=SOURCE_METADATA,
    )

    try:
        with cl.open() as neurons:
            stim_design = cl.StimDesign(80, -0.5, 120, 0.75)
            neurons.stim(6, stim_design, cl.BurstDesign(2, 100))

            frames = neurons.read(300, 0)
            active_indices = np.flatnonzero(frames[:, 3])
            assert active_indices.size > 0
            first_stim = int(active_indices[0])
            second_stim = int(np.flatnonzero(frames[:, 3] == 2)[0])

            assert first_stim < 10
            assert 240 <= second_stim <= 260
            np.testing.assert_array_equal(
                frames[:first_stim, 3],
                np.zeros(first_stim, dtype=np.int16),
            )
            np.testing.assert_array_equal(
                frames[first_stim:second_stim, 3],
                np.ones(second_stim - first_stim, dtype=np.int16),
            )
            np.testing.assert_array_equal(
                frames[second_stim:, 3],
                np.full(300 - second_stim, 2, dtype=np.int16),
            )
            np.testing.assert_array_equal(frames[first_stim:, 4], np.full(300 - first_stim, 6, dtype=np.int16))
            np.testing.assert_array_equal(frames[first_stim:, 5], np.full(300 - first_stim, 2, dtype=np.int16))
            np.testing.assert_array_equal(frames[first_stim:, 6], np.full(300 - first_stim, 200, dtype=np.int16))
            np.testing.assert_array_equal(frames[first_stim:, 7], np.full(300 - first_stim, -500, dtype=np.int16))
            np.testing.assert_array_equal(frames[first_stim:, 8], np.full(300 - first_stim, 750, dtype=np.int16))
    finally:
        sim.clear_simulator_data_source()

def test_live_simulator_data_source_push_adapter(monkeypatch):
    sim.clear_simulator_data_source()
    monkeypatch.setenv("CL_SDK_ACCELERATED_TIME", "0")
    sim.set_simulator_data_source(
        "tests.cl.custom_data_source:make_live_source",
        config={"sample_offset": 17},
    )

    try:
        with cl.open() as neurons:
            frames = neurons.read(20, 0)
            np.testing.assert_array_equal(frames[:, 0], np.arange(20, dtype=np.int16))
            np.testing.assert_array_equal(frames[:, 2], np.full(20, 17, dtype=np.int16))

            analysis = neurons.read(20, 0, analysis=True)
            assert [spike.timestamp for spike in analysis.spikes] == [1, 6, 11, 16]
            assert [spike.channel for spike in analysis.spikes] == [6, 6, 6, 6]
    finally:
        sim.clear_simulator_data_source()

def _report_latency_stats(pytestconfig, latencies_frames: list[int]) -> None:
    if pytestconfig.getoption("verbose", 0) <= 0:
        return
    latencies_ms = [
        frames * 1000 / SOURCE_METADATA.frames_per_second
        for frames in latencies_frames
    ]
    message = (
        "stim_to_spike_latency_ms: "
        f"count={len(latencies_ms)} "
        f"min={min(latencies_ms):.3f} "
        f"avg={sum(latencies_ms) / len(latencies_ms):.3f} "
        f"max={max(latencies_ms):.3f} "
        f"frames={latencies_frames}"
    )
    reporter = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(message)
    else:
        print(message)


def test_realtime_custom_data_source_stim_to_spike_latency_under_5ms(monkeypatch, pytestconfig):
    sim.clear_simulator_data_source()
    monkeypatch.setenv("CL_SDK_ACCELERATED_TIME", "0")
    sim.set_simulator_data_source(
        "tests.cl.custom_data_source:make_reactive_live_source",
        config={"batch_frames": 5, "max_buffer_frames": 64},
    )

    latencies_frames: list[int] = []
    channels = set(range(64)) - {0, 4, 7, 56, 63}  # all valid stim channels

    try:
        with cl.open() as neurons:
            stim_design = cl.StimDesign(80, -0.5, 120, 0.75)
            for channel in channels:
                read_from = neurons.timestamp()
                neurons.stim(channel, stim_design)
                analysis = neurons.read(300, read_from, analysis=True)

                stims = [stim for stim in analysis.stims if stim.channel == channel]
                spikes = [spike for spike in analysis.spikes if spike.channel == channel]
                assert stims, f"No stim returned for channel {channel}"
                assert spikes, f"No reactive spike returned for channel {channel}"

                stim_ts = stims[0].timestamp
                matching_spikes = [
                    spike for spike in spikes if spike.timestamp >= stim_ts
                ]
                assert matching_spikes, f"No post-stim spike returned for channel {channel}"

                spike_ts = matching_spikes[0].timestamp
                latencies_frames.append(spike_ts - stim_ts)
    finally:
        sim.clear_simulator_data_source()

    _report_latency_stats(pytestconfig, latencies_frames)
    max_latency_ms = max(latencies_frames) * 1000 / SOURCE_METADATA.frames_per_second
    assert max_latency_ms < 5.0

def test_live_simulator_data_source_rejects_accelerated_mode(monkeypatch):
    sim.clear_simulator_data_source()
    monkeypatch.setenv("CL_SDK_ACCELERATED_TIME", "1")
    sim.set_simulator_data_source("tests.cl.custom_data_source:make_live_source")

    try:
        with pytest.raises(ValueError, match="does not support accelerated time"):
            with cl.open():
                pass
    finally:
        sim.clear_simulator_data_source()
