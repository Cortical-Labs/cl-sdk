from itertools import combinations

import numpy as np

from ...util import RecordingView
from .. import AnalysisResultSpikeTriggeredHistogram

def _analyse_spike_triggered_histogram(
    recording:                    RecordingView,
    bin_size_sec:                 float,
    start_sec:                    float,
    end_sec:                      float,
    num_channels:                 int,
    min_firing_rate_threshold_hz: float
    ) -> AnalysisResultSpikeTriggeredHistogram:
    """
    See RecordingView.analyse_spike_triggered_histogram()
    """
    from .._metrics._mea_layout import (
        _COMMON_GROUND_CHANNELS,
        _COMMON_REFERENCE_CHANNELS,
        _valid_common_layout
        )
    if not _valid_common_layout(recording):
        raise ValueError("Recording does not conform to common MEA layout.")

    sampling_frequency      = recording._analysis_cache.metadata.sampling_frequency
    duration_secs           = recording.attributes["duration_seconds"]
    channel_count           = recording.attributes["channel_count"]
    excluded_channels       = _COMMON_GROUND_CHANNELS + _COMMON_REFERENCE_CHANNELS
    spike_frames_by_channel = recording._analysis_cache.get_spike_frames_by_channel()

    spike_times_by_channel  = \
        {
            channel : frames / sampling_frequency
            for channel, frames in spike_frames_by_channel.items()
        }
    mean_firing_rate_by_channel_hz = \
        np.array([
            len(frames) / duration_secs
            for frames in spike_frames_by_channel.values()
        ])

    # Identify candidate channels that meet the threshold
    candidate_channel_ids  = []
    candidate_firing_rates = []
    for channel in range(channel_count):
        if channel in excluded_channels:
            continue
        channel_mfr_hz = mean_firing_rate_by_channel_hz[channel]
        if channel_mfr_hz >= min_firing_rate_threshold_hz:
            candidate_channel_ids.append(channel)
            candidate_firing_rates.append(channel_mfr_hz)

    # Init result
    result = AnalysisResultSpikeTriggeredHistogram(
        metadata                     = recording._analysis_cache.metadata,
        bin_size_sec                 = bin_size_sec,
        start_sec                    = start_sec,
        end_sec                      = end_sec,
        num_eligible_channels        = num_channels,
        min_firing_rate_threshold_hz = min_firing_rate_threshold_hz,
        histogram_bins               = np.array([]),
        histograms                   = {}
        )

    result.num_eligible_channels = min(num_channels, len(candidate_channel_ids))
    if result.num_eligible_channels < 1:
        print("No channels to check that meet the firing rate threshold.")
        return result

    # Identify top candidate channels
    top_candidate_channel_idx = np.argsort(candidate_firing_rates)[-num_channels:]
    top_candidate_channels    = np.array(candidate_channel_ids)[top_candidate_channel_idx].tolist()
    print("Selected top channels are:", top_candidate_channels)

    # Build histograms based on pairs of eligible channels
    result.histogram_bins = np.arange(-start_sec, end_sec + bin_size_sec, step=bin_size_sec)
    for ch1, ch2 in combinations(top_candidate_channels, 2):
        for base_channel, target_channel in [(ch1, ch2), (ch2, ch1)]:
            base_spike_times     = spike_times_by_channel[base_channel]
            target_spike_times   = spike_times_by_channel[target_channel]
            relative_spike_times = []

            for spike_time_sec in base_spike_times:
                window_start_sec = spike_time_sec - start_sec
                window_end_sec   = spike_time_sec + end_sec
                triggered_spikes = target_spike_times[(target_spike_times >= window_start_sec) & (target_spike_times <= window_end_sec)]
                relative_spike_times.extend(triggered_spikes - spike_time_sec)

            histogram, _ = np.histogram(relative_spike_times, bins=result.histogram_bins)
            result.histograms[(base_channel, target_channel)] = histogram

    return result
