from .. import Array2DFloat, AnalysisResult

class AnalysisResultComplexityLempelZiv(AnalysisResult):
    """ Lempel-Ziv Complexity (LZC) analysis results for one recording. """

    bin_size_sec: float
    """ Size of each time bin in seconds. """

    lzc_scores_per_channel: Array2DFloat
    """ LZC scores for each channel with shape (channel_count, 1) """
