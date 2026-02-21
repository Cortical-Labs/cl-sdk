import numpy as np

from ...util import RecordingView
from .. import Bursts, AnalysisResultNetworkBursts

def _analyse_network_bursts(
    recording:           RecordingView,
    bin_size_sec:        float,
    onset_freq_hz:       float,
    offset_freq_hz:      float,
    min_active_channels: int | None = None
    ) -> AnalysisResultNetworkBursts:
    """
    See RecordingView.analyse_network_bursts()
    """
    sampling_frequency = recording._analysis_cache.metadata.sampling_frequency
    channel_count      = recording._analysis_cache.metadata.channel_count
    duration_frames    = recording._analysis_cache.metadata.duration_frames
    bin_size_frames    = int(bin_size_sec * sampling_frequency)
    spike_train_array  = recording._analysis_cache.get_spike_train() # (channel_count, duration_frames)

    if min_active_channels is None:
        min_active_channels = channel_count
        assert min_active_channels is not None

    # We use a single firing rate threshold for determining burst onset and offset
    # across all channels
    onset_threshold  = (onset_freq_hz * min_active_channels) / (1 / bin_size_sec)
    offset_threshold = (offset_freq_hz * min_active_channels) / (1 / bin_size_sec)

    # Get a sorted array of spike timestamps from all channels and
    # calculate spike counts and firing rates for each frame bin
    all_spike_frames                = np.sort(spike_train_array.nonzero()[1])
    bin_boundaries_frames           = np.arange(0, duration_frames + bin_size_frames, step=bin_size_frames)
    spike_counts_per_bin, _         = np.histogram(all_spike_frames, bins=bin_boundaries_frames)
    firing_rate_per_channel_and_bin = (spike_counts_per_bin / channel_count) / bin_size_sec

    # Detect Bursts
    bursts = Bursts()
    for frame, spike_count in zip(bin_boundaries_frames[:-1], spike_counts_per_bin):
        if bursts.is_bursting:
            # Exit bursting state if spike count drops below threshold
            if spike_count < offset_threshold:
                bursts.step(frame=frame, is_bursting=False)
        else:
            # Enter bursting state if spike count exceeds threshold
            if spike_count >= onset_threshold:
                bursts.step(frame=frame, is_bursting=True)
    bursts.finalise(end_frame=duration_frames)

    # Analyse bursts
    result = AnalysisResultNetworkBursts(
        metadata                         = recording._analysis_cache.metadata,
        bursts                           = bursts,
        onset_freq_hz                    = onset_freq_hz,
        offset_freq_hz                   = offset_freq_hz,
        network_burst_count              = 0,
        network_burst_durations_sec      = [],
        network_burst_spike_counts       = [],
        total_network_burst_duration_sec = 0.0,
        bin_size_sec                     = bin_size_sec,
        bin_boundaries_frames            = bin_boundaries_frames[:-1],
        firing_rate_per_channel_and_bin  = firing_rate_per_channel_and_bin,
        spike_frames_by_channel          = recording._analysis_cache.get_spike_frames_by_channel()
        )
    if len(bursts) < 1:
        return result

    for burst in bursts:
        burst_duration_sec = burst.get_duration(sampling_frequency=sampling_frequency)
        result.network_burst_durations_sec.append(burst_duration_sec)

        burst_spike_count = np.sum((burst.start_frame <= all_spike_frames) & (all_spike_frames < burst.end_frame))
        result.network_burst_spike_counts.append(int(burst_spike_count))

    result.network_burst_count              = len(bursts)
    result.total_network_burst_duration_sec = sum(result.network_burst_durations_sec)

    return result
