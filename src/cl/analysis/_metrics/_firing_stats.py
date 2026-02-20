import numpy as np

from ...util import RecordingView
from .. import AnalysisResultFiringStats

def _analyse_firing_stats(
    recording:    RecordingView,
    bin_size_sec: float
    ) -> AnalysisResultFiringStats:
    """
    See RecordingView.analyse_firing_stats()
    """
    sampling_frequency      = recording._analysis_cache.metadata.sampling_frequency
    channel_count           = recording._analysis_cache.metadata.channel_count
    bin_size_frames         = int(sampling_frequency * bin_size_sec)
    spike_frames_by_channel = recording._analysis_cache.get_spike_frames_by_channel()
    spike_count_array       = recording._analysis_cache.get_spike_count_per_time_bin(bin_size=bin_size_frames)

    firing_counts      = spike_count_array.toarray()
    firing_rates       = np.zeros_like(firing_counts, dtype=np.float64)

    channel_ISI        = [ [] for _ in range(channel_count) ]
    channel_ISI_mean   = np.full(channel_count, np.nan, dtype=np.float64)
    channel_ISI_var    = np.full(channel_count, np.nan, dtype=np.float64)

    # Loop through available channels in the dict only
    for ch in range(channel_count):
        spike_times = spike_frames_by_channel[ch]
        if spike_times.size == 0:
            continue

        firing_rates[ch] = firing_counts[ch] / bin_size_sec
        isi              = np.diff(spike_times) / sampling_frequency
        channel_ISI[ch]  = isi.tolist()

        if isi.size > 0:
            channel_ISI_mean[ch] = isi.mean()
            channel_ISI_var[ch]  = isi.var()

    # Summary metrics
    mean_firing_per_channel = np.mean(firing_counts, axis=1)
    var_firing_per_channel  = np.var(firing_counts, axis=1)

    culture_mean_firing     = mean_firing_per_channel.mean()
    culture_var_firing      = var_firing_per_channel.mean()
    culture_max_firing      = firing_counts.max()

    culture_ISI_mean        = np.nanmean(channel_ISI_mean)
    culture_ISI_var         = np.nanvar(channel_ISI_mean)

    channel_ISI_mean        = np.nan_to_num(channel_ISI_mean, nan=0.0)
    channel_ISI_var         = np.nan_to_num(channel_ISI_var,  nan=0.0)

    return AnalysisResultFiringStats(
        metadata                  = recording._analysis_cache.metadata,
        bin_size_sec              = bin_size_sec,
        firing_counts             = firing_counts,
        firing_rates              = firing_rates,
        channel_mean_firing_rates = mean_firing_per_channel.tolist(),
        channel_var_firing_rates  = var_firing_per_channel.tolist(),
        culture_mean_firing_rates = culture_mean_firing,
        culture_var_firing_rates  = culture_var_firing,
        culture_max_firing_rates  = culture_max_firing,
        channel_ISI               = channel_ISI,
        channel_ISI_mean          = channel_ISI_mean.tolist(),
        channel_ISI_var           = channel_ISI_var.tolist(),
        culture_ISI_mean          = culture_ISI_mean.item(),
        culture_ISI_var           = culture_ISI_var.item()
        )