import numpy as np

from ...util import RecordingView
from .. import AnalysisResultDctFeatures, Array1DFloat

def _analyse_dct_features(
    recording:       RecordingView,
    k:               int,
    ) -> AnalysisResultDctFeatures:
    """
    See RecordingView.analyse_dct_features()
    """
    from .._metrics._mea_layout import (
        _COMMON_MEA_LAYOUT,
        _valid_common_layout
        )
    if not _valid_common_layout(recording):
        raise ValueError("Recording does not conform to common MEA layout.")

    common_mea_layout       = np.array(_COMMON_MEA_LAYOUT)
    mea_height, mea_width   = common_mea_layout.shape
    spike_frames_by_channel = recording._analysis_cache.get_spike_frames_by_channel()
    channel_count           = len(spike_frames_by_channel)

    assert mea_height * mea_width == channel_count, \
        f"MEA layout of ({mea_height} x {mea_width}) does not match channel count of {channel_count}"

    # Count total spikes in each channel then convert this to common spatial layout
    spike_counts_per_channel    = np.array([len(spike_frames_by_channel[channel]) for channel in range(channel_count)])
    spike_counts_per_channel_2d = spike_counts_per_channel.reshape(mea_width, mea_height).T # Assumes common MEA layout

    # Calculate DCT features
    dct_height   = _compute_dct_basis(n=mea_height, k=k)
    dct_width    = _compute_dct_basis(n=mea_width,  k=k)
    dct_result   = np.einsum("kl,ik,jl", spike_counts_per_channel_2d, dct_height, dct_width)
    dct_features = \
        {
            f"dct{i:1d}{j:1d}_firing": float(dct_result[i][j])
            for i in range(dct_result.shape[0])
            for j in range(dct_result.shape[1])
        }

    return AnalysisResultDctFeatures(
        metadata                = recording._analysis_cache.metadata,
        mea_layout              = _COMMON_MEA_LAYOUT,
        k                       = k,
        dct_height_coefficients = dct_height,
        dct_width_coefficients  = dct_width,
        dct_features            = dct_features
        )

def _compute_dct_basis(n: int, k: int) -> Array1DFloat:
    """ Compute the first k DCT frequency coefficients of length n. """
    N      = np.arange(n)[np.newaxis, :         ]
    K      = np.arange(k)[:         , np.newaxis]
    V      = np.cos(np.pi / n * (N + 0.5) * K)
    V[0]  *= 1.0 / np.sqrt(2.0)
    V[1:] *= np.sqrt(2.0/n)
    return V