import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

import networkx as nx

from .. import Array2DFloat, AnalysisResult

class AnalysisResultsFunctionalConnectivity(AnalysisResult):
    """ Functional Connectivity (FC) analysis results for one recording. """

    bin_size_sec: float
    """ Size of each time bin in seconds. """

    correlation_threshold: float
    """ Absolute correlation threshold in [0, 1] for constructing the FC graph. """

    adjacency_matrix: Array2DFloat
    """ Weighted adjacency matrix for the thresholded FC graph, shape (channel_count, channel_count). """

    total_edge_weights: float
    """ Sum of all (non-zero) edge weights in the thresholded FC graph. """

    average_edge_weights: float
    """ Mean edge weight over all existing edges. """

    clustering_coefficient: float
    """ Average weighted clustering coefficient of the FC graph. """

    graph_partition: dict
    """ Graph partition that maximises modularity, based on Louvain community. """

    modularity_index: float
    """ Modularity of the partition found via greedy modularity maximisation. """

    max_betweenness_centrality: float
    """ Maximum weighted betweenness centrality across all nodes. """

    def plot(
        self,
        figsize:    tuple[int, int]     = (6, 6),
        title:      str | None          = None,
        save_path:  str | None          = None,
        ax:         Axes | None         = None
        ):
        """
        Creates a visualisation of functional connectivity where the edges are
        Pearson correlations and nodes are coloured by Louvain community where
        applicable.

        Args:
            figsize:   Size of the plot figure.
            title:     Title for the plot, if not provided, a default will be used.
            save_path: Path to the save the plot instead of showing it.
            ax:        Axes draw the plots. (Defaults to None).
        """
        # Reference results data
        adjacency_marix       = self.adjacency_matrix
        graph_partition       = self.graph_partition
        correlation_threshold = self.correlation_threshold

        # Build weighted FC graph
        graph = nx.from_numpy_array(adjacency_marix)
        if graph.number_of_edges() < 1:
            raise ValueError("Adjacency matrix contains no edges.")

        # Define fixed 8 x 8 grid positions based on common layout
        grid_size = 8
        positions = {}
        for channel in graph.nodes():
            col = channel // grid_size        # column index
            row = channel % grid_size         # row index
            positions[channel] = (col, -row)  # y reversed so top-left is channel 0

        # Normalize edge widths by absolute weight
        edge_weights = np.array([abs(data["weight"]) for *_, data in graph.edges(data=True)])
        edge_widths  = 1.0 + 4.0 * (edge_weights / edge_weights.max()) if edge_weights.max() > 0 else 1.0

        # Color nodes by Louvain community if available
        if len(graph_partition) > 0:
            communities = [graph_partition[channel] for channel in graph.nodes()]
        else:
            communities = "lightblue"

        if ax is None:
            # Create figure and axes if not provided
            fig = plt.figure(figsize=figsize)
            gs  = GridSpec(nrows=1, ncols=1)
            ax  = fig.add_subplot(gs[0, 0])
        else:
            fig = None

        nx.draw_networkx_nodes(
            G          = graph,
            pos        = positions,
            node_size  = 120,
            node_color = communities,
            cmap       = "tab20",
            ax         = ax
            )
        nx.draw_networkx_edges(
            G     = graph,
            pos   = positions,
            width = edge_widths,
            alpha = 0.7,
            ax    = ax
            )
        nx.draw_networkx_labels(
            G         = graph,
            pos       = positions,
            font_size = 6,
            ax        = ax
            )
        ax.axis("off")

        if title is None:
            title = f"Functional Connectivity Graph (thr = {correlation_threshold})"
        ax.set_title(title)

        if save_path is None and fig is not None:
            fig.tight_layout()
            plt.show(block=True)
        else:
            plt.savefig(str(save_path), bbox_inches="tight")

        return ax
