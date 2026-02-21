"""
.. include:: ../../../docs/analysis.md
"""
from ._types import SpecialFloat, Array1DFloat, Array1DInt, Array2DFloat, Array2DInt, BurstsType
from ._bursts import Burst, Bursts
from ._results._base_result import AnalysisMetadata, AnalysisResult
from ._results._bursts_result import AnalysisResultNetworkBursts
from ._results._criticality_result import AnalysisResultCriticality
from ._results._discrete_cosine_transform_result import AnalysisResultDctFeatures
from ._results._firing_stats_result import AnalysisResultFiringStats
from ._results._functional_connectivity_result import AnalysisResultsFunctionalConnectivity
from ._results._information_entropy_result import AnalysisResultInformationEntropy
from ._results._lempel_ziv_complexity import AnalysisResultComplexityLempelZiv
from ._results._spike_triggered_histogram_result import AnalysisResultSpikeTriggeredHistogram

__all__ = [
    "AnalysisMetadata",
    "AnalysisResult",
    "AnalysisResultNetworkBursts",
    "AnalysisResultCriticality",
    "AnalysisResultDctFeatures",
    "AnalysisResultFiringStats",
    "AnalysisResultsFunctionalConnectivity",
    "AnalysisResultInformationEntropy",
    "AnalysisResultComplexityLempelZiv",
    "AnalysisResultSpikeTriggeredHistogram",
    "SpecialFloat",
    "Array1DFloat",
    "Array1DInt",
    "Array1DFloat",
    "Array2DInt",
    "Array2DFloat",
    "Burst",
    "Bursts",
    "BurstsType",
]