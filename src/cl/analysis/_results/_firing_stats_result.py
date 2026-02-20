from .. import Array2DInt, Array2DFloat, AnalysisResult

class AnalysisResultFiringStats(AnalysisResult):
    """ Firing stat analysis for one recording. """

    bin_size_sec:              float
    """ Size of each time bin in seconds which governs time units for value interpretation. """

    firing_counts:             Array2DInt
    """ Array of spike count for each channel and time bin with shape `(channel_count, bin_count)`. """
    firing_rates:              Array2DFloat
    """ Array of spike firing rates in time units for each channel and time bin with shape `(channel_count, bin_count)`. """

    channel_mean_firing_rates: list[float]
    """ Mean firing rate in time units for each channel as a list with length `channel_count`. """
    channel_var_firing_rates:  list[float]
    """ Variance in firing rate in time units for each channel as a list with length `channel_count`. """
    culture_mean_firing_rates: float
    """ Mean firing rate in time units across all channels and time bins. """
    culture_var_firing_rates:  float
    """ Variance in firing rate in time units across all channels and time bins. """
    culture_max_firing_rates:  float
    """ Maximum firing rate in time units across all channels and time bins. """

    channel_ISI:               list[list[float]]
    """ List of inter-spike intervals (ISI) in time units for each channel, where the outer list has length `channel_count`. """
    channel_ISI_mean:          list[float]
    """ Mean inter-spike intervals (ISI) in time units for each channel, with length `channel_count`. """
    channel_ISI_var:           list[float]
    """ Variance in inter-spike intervals (ISI) in time units for each channel, with length `channel_count`. """

    culture_ISI_mean:          float
    """ Mean inter-spike intervals (ISI) in time units for all channels. """
    culture_ISI_var:           float
    """ Variance in inter-spike intervals (ISI) in time units for all channels. """