import numpy as np
import pandas as pd

from ...util import RecordingView
from .. import AnalysisResultInformationEntropy

def _analyse_information_entropy(
    recording:      RecordingView,
    bin_size_sec:   float,
    fillna:         float | None = 0.0,
    log_base:       float | None = None,
    ) -> AnalysisResultInformationEntropy:
    """
    See RecordingView.analyse_lempel_ziv_complexity()
    """
    sampling_frequency = recording._analysis_cache.metadata.sampling_frequency
    bin_size_frames    = max(1, int(round(bin_size_sec * sampling_frequency)))
    spike_counts_array = recording._analysis_cache.get_spike_count_per_time_bin(
        bin_size                     = bin_size_frames,
        limit_to_max_spike_timestamp = False,
        ).todense()

    channel_count, bin_count = spike_counts_array.shape
    active_channels_per_bin  = np.count_nonzero(spike_counts_array, axis=0)

    if channel_count <= 0:
        raise ValueError("Channel count is 0; cannot compute entropy.")

    # Choose log
    log_fn = np.log if log_base is None else (lambda x: np.log(x) / np.log(log_base))

    # Stable Bernoulli entropy with explicit handling at p=0,1
    p       = active_channels_per_bin / float(channel_count)
    H       = np.zeros_like(p, dtype=float)
    mask    = (p > 0) & (p < 1)
    pm      = p[mask]
    H[mask] = -pm * log_fn(pm) - (1.0 - pm) * log_fn(1.0 - pm)

    if fillna is not None:
        H = pd.Series(H).fillna(fillna).to_numpy(dtype=float)

    return AnalysisResultInformationEntropy(
        metadata                         = recording._analysis_cache.metadata,
        bin_size_sec                     = bin_size_sec,
        information_entropy_per_time_bin = H
        )
