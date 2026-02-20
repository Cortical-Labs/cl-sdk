from .. import Array1DFloat, AnalysisResult

class AnalysisResultInformationEntropy(AnalysisResult):
    """ Information entropy analysis results for one recording. """

    bin_size_sec: float
    """ Size of each time bin in seconds. """

    information_entropy_per_time_bin: Array1DFloat
    """ Information entropy for each time bin. """