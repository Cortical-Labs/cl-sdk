import numpy as np

from ...util import RecordingView
from .. import AnalysisResultComplexityLempelZiv

def _analyse_lempel_ziv_complexity(
    recording:     RecordingView,
    bin_size_sec:  float,
    min_bin_count: int  = 2,
    normalise:     bool = True,
    use_binary:    bool = True,
    ) -> AnalysisResultComplexityLempelZiv:
    """
    See RecordingView.analyse_lempel_ziv_complexity()
    """
    sampling_frequency = recording._analysis_cache.metadata.sampling_frequency
    bin_size_frames    = max(1, int(round(bin_size_sec * sampling_frequency)))
    spike_count_array  = recording._analysis_cache.get_spike_count_per_time_bin(
        bin_size                     = bin_size_frames,
        limit_to_max_spike_timestamp = False
        ).todense()
    channel_count, bin_count = spike_count_array.shape

    lzc_scores = []
    for i in range(channel_count):
        sequence = spike_count_array[i, :].ravel()

        if bin_count < min_bin_count:
            lzc_scores.append(0.0)
            continue

        # Choose binary vs count-based symbols, k is alphabet size
        if use_binary:
            symbols = (sequence > 0).astype(np.int8)
            k       = 2
        else:
            # Keep counts as integer symbols
            symbols = sequence.astype(int)
            k       = int(len(set(symbols.tolist())))

        # If the alphabet is degenerate, complexity is effectively 0
        if k <= 1:
            lzc_scores.append(0.0)
            continue

        dictionary     = set()
        current_phrase = ()
        for symbol in symbols.tolist():
            phrase_with_symbol = current_phrase + (symbol,)
            if phrase_with_symbol in dictionary:
                current_phrase = phrase_with_symbol
            else:
                dictionary.add(phrase_with_symbol)
                current_phrase = ()

        raw_lzc = float(len(dictionary))

        if normalise:
            if bin_count > 1:
                # normalization using log base k (alphabet size)
                normalized_score = raw_lzc * (np.log(bin_count) / np.log(k)) / bin_count
                lzc_scores.append(float(normalized_score))
            else:
                lzc_scores.append(0.0)
        else:
            lzc_scores.append(raw_lzc)

    return AnalysisResultComplexityLempelZiv(
        metadata               = recording._analysis_cache.metadata,
        bin_size_sec           = bin_size_sec,
        lzc_scores_per_channel = np.array(lzc_scores, dtype=float).reshape(-1, 1)
        )
