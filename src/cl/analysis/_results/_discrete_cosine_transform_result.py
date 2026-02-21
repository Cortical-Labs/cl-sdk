import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, Colormap

from .. import Array2DFloat, AnalysisResult

class AnalysisResultDctFeatures(AnalysisResult):
    """ Discrete Cosine Transform (DCT) analysis results for one recording. """

    mea_layout: list[list[int]]
    """ Spatial layout of the MEA used to calculate this result. """

    k: int
    """ The frequency index (coefficient index) of the DCT. """

    dct_height_coefficients: Array2DFloat
    """ DCT coefficients along the height dimension. """

    dct_width_coefficients: Array2DFloat
    """ DCT coefficients along the width dimension. """

    dct_features: dict[str, float]
    """
    Dictionary mapping DCT coefficients to their values.
    Keys are formatted as "dct{i}{j}_firing" where i, j are spatial indices for rows / columns respectively.
    """

    def plot(
        self,
        figsize:    tuple[int, int]     = (9, 8),
        title:      str | None          = None,
        save_path:  str | None          = None,
        cmap:       str | Colormap      = "RdBu",
        ) -> Figure:
        """
        Creates a (k x k) plot of DCT coefficients.

        Args:
            figsize:   Size of the plot figure.
            title:     Title for the plot, if not provided, a default will be used.
            save_path: Path to the save the plot instead of showing it.
            cmap:      Colour map override.

        Returns:
            Figure: The figure used to create the plot.
        """
        # Reference results data
        k          = self.k
        dct_height = self.dct_height_coefficients
        dct_width  = self.dct_width_coefficients
        dct_2d     = np.einsum("ik,jl", dct_height, dct_width)
        mea_height = len(self.mea_layout)
        mea_width  = len(self.mea_layout[0])
        assert dct_2d.shape == (k, k, mea_height, mea_width), \
            f"DCT transform of shape {dct_2d.shape} is inconsistent with expected {(k, k, mea_height, mea_width)}."

        # Create figure and axes
        fig  = plt.figure(figsize=figsize)
        gs   = GridSpec(nrows=k, ncols=k + 1, width_ratios=([1] * k) + [0.1]) # Extra one for cbar
        axes = \
            [
                [ fig.add_subplot(gs[i, j]) for j in range(k)]
                for i in range(k)
            ]

        # Draw DCT transform
        scale_max = np.abs(dct_2d).max()
        for i in range(k):
            for j in range(k):
                ax = axes[i][j]
                ax.matshow(dct_2d[i, j], cmap=cmap, vmin=-scale_max, vmax=scale_max)
                ax.set_title(f"DCT basis ({i},{j})")
                ax.xaxis.set_ticks_position("bottom")

        # Add a global color bar
        ax_cbar = fig.add_subplot(gs[:, -1])
        fig.colorbar(
            cax=ax_cbar,
            mappable = cm.ScalarMappable(
                norm = Normalize(vmin=-scale_max, vmax=scale_max),
                cmap = cmap
                )
            )
        ax_cbar.set_ylabel("DCT Coefficient Values")

        if title is None:
            title = f"Discrete Cosine Transform (DCT) with k={k}"
        fig.suptitle(title)

        if save_path is None:
            fig.tight_layout()
            plt.show(block=True)
        else:
            plt.savefig(str(save_path), bbox_inches="tight")

        return fig