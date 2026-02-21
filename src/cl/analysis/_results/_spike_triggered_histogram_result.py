import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from .. import Array1DInt, Array1DFloat, AnalysisResult

class AnalysisResultSpikeTriggeredHistogram(AnalysisResult):
    """ Spike triggered histogram analysis results for one recording. """

    bin_size_sec: float
    """ Bin size in seconds used for the histogram. """

    start_sec: float
    """ Time in seconds to include before the trigger spike. """

    end_sec: float
    """ Time in seconds to include after the trigger spike. """

    num_eligible_channels: int
    """ Number of channels above the minimum firing rate use for analysis. """

    min_firing_rate_threshold_hz: float
    """ Threshold in Hz that defines eligibility of a channel for analysis. """

    histogram_bins: Array1DFloat

    histograms: dict[tuple[int, int], Array1DInt]
    """ Spike triggered histograms for each pair of top eligible channels by mean firing rate. """

    def plot(
        self,
        figsize:    tuple[int, int]   = (8, 2),
        title:      str | None        = None,
        save_path:  str | None        = None,
        axes:       list[Axes] | None = None
        ):
        """
        Creates a plot of all the histograms.

        Args:
            figsize:   Size of the plot for each histogram.
            title:     Title for the plot, if not provided, a default will be used.
            save_path: Path to the save the plot instead of showing it.
            axes:      List of Axes (one for each histogram) to draw the plots. (Defaults to None).

        Returns:
            list[Axes]: List of Axes drawn with the histogram.
        """
        # Reference results data
        histograms     = self.histograms
        bins           = self.histogram_bins
        num_histograms = len(histograms)

        if axes is None:
            # Create figure and axes if not provided
            fig_W = figsize[0]
            fig_H = figsize[1] * num_histograms
            fig   = plt.figure(figsize=(fig_W, fig_H))
            gs    = GridSpec(nrows=num_histograms, ncols=1)
            ax    = None
            axes  = []
            for i in range(num_histograms):
                ax = fig.add_subplot(gs[i, 0], sharey=ax)
                axes.append(ax)
        else:
            fig = None

        for i, ((base_channel, target_channel), histogram) in enumerate(histograms.items()):
            ax = axes[i]
            ax.bar(bins[:-1], histogram, width=np.diff(bins), edgecolor="black", align="edge")
            ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
            ax.set_title(f"Base: Channel {base_channel}, Target: Channel {target_channel}")
            ax.set_xlabel("Time (secs) relative to base spike")
            ax.set_ylabel("Spike Count")
            ax.set_axisbelow(True)
            ax.grid(True)

        if title is None:
            title = f"Spike-triggered Histograms"
        if fig is not None:
            fig.suptitle(title)

        if save_path is None and fig is not None:
            fig.tight_layout()
            plt.show(block=True)
        else:
            plt.savefig(str(save_path), bbox_inches="tight")

        return axes
