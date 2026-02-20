import numpy as np

import networkx as nx

from community import community_louvain

from ...util import RecordingView
from .. import AnalysisResultsFunctionalConnectivity

def _analyse_functional_connectivity(
    recording:             RecordingView,
    bin_size_sec:          float,
    correlation_threshold: float,
    ) -> AnalysisResultsFunctionalConnectivity:
    """
    See RecordingView.analyse_functional_connectivity()
    """
    from .._metrics._mea_layout import _valid_common_layout
    if not _valid_common_layout(recording):
        raise ValueError("Recording does not conform to common MEA layout.")

    assert 0.0 <= correlation_threshold <= 1.0, f"Correlation threshold must be in [0, 1]."

    sampling_frequency       = recording._analysis_cache.metadata.sampling_frequency
    bin_size_frames          = int(bin_size_sec * sampling_frequency)
    spike_count_array        = recording._analysis_cache.get_spike_count_per_time_bin(bin_size_frames).todense()
    channel_count, bin_count = spike_count_array.shape

    # Init result
    result = AnalysisResultsFunctionalConnectivity(
        metadata                   = recording._analysis_cache.metadata,
        bin_size_sec               = bin_size_sec,
        correlation_threshold      = correlation_threshold,
        adjacency_matrix           = np.zeros((channel_count, channel_count), dtype=float),
        total_edge_weights         = 0.0,
        average_edge_weights       = 0.0,
        clustering_coefficient     = 0.0,
        graph_partition            = {},
        modularity_index           = 0.0,
        max_betweenness_centrality = 0.0
        )

    if (channel_count < 2) or (bin_count < 2):
        # Not enough data to define correlations
        return result

    # Functional connectivity with Pearson correlation without self-connections
    adjacency_matrix = np.corrcoef(spike_count_array)
    np.nan_to_num(adjacency_matrix, nan=0.0, copy=False)
    np.fill_diagonal(adjacency_matrix, val=0.0)

    # Apply thresholding
    if correlation_threshold > 0.0:
        adjacency_matrix = np.where(
            np.abs(adjacency_matrix) >= correlation_threshold,
            adjacency_matrix,
            0.0
            )
    result.adjacency_matrix = adjacency_matrix

    # Build weighted FC graph
    graph = nx.from_numpy_array(adjacency_matrix)
    if graph.number_of_edges() < 1:
        # No edges survive thresholding, return metrics as zero
        return result

    # Edge weights
    edge_weights                = np.array([data["weight"] for *_, data in graph.edges(data=True)])
    result.total_edge_weights   = float(edge_weights.sum())
    result.average_edge_weights = float(edge_weights.mean())

    # Clustering coefficient (weighted)
    graph_clustering = nx.clustering(graph, weight="weight")
    assert isinstance(graph_clustering, dict), "Invalid graph clustering result"
    result.clustering_coefficient = float(np.mean(list(graph_clustering.values())))

    # Modularity (community structure)
    result.modularity_index = 0.0
    try:
        # If all nodes are in one community, then mmodularity is effectively 0
        result.graph_partition  = community_louvain.best_partition(graph, weight="weight") # dict {node: community_id}
        if len(set(result.graph_partition.values())) > 1:
            result.modularity_index = float(community_louvain.modularity(result.graph_partition, graph, weight="weight"))
    except:
        ...

    # Betweenness centrality (weighted)
    betweenness_centrality            = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    result.max_betweenness_centrality = float(max(betweenness_centrality.values())) if betweenness_centrality else 0.0

    return result