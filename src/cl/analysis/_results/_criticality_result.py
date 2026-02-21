from math import isfinite

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from .. import (
    Array1DInt,
    Array1DFloat,
    SpecialFloat,
    AnalysisResult
    )

class AnalysisResultCriticality(AnalysisResult):
    """ Criticality analysis results for one recording. """

    #
    # Configuration
    #

    bin_size_sec: float
    """ Size of each time bin in seconds. """

    percentile_threshold: float
    """ Percentile value used to calculate the activity threshold. """

    activity_threshold: float
    """ Threshold used to identify an avalanche. """

    duration_thresholds: tuple[int, int]
    """ Thresholds for avalanche durations in number of frame bins. """

    random_seed: int
    """ Random seed used for bootstrapping. """

    n_bootstraps: int
    """ Number of bootstraps to perform to estimate beta exponent variability. """

    #
    # Results
    #

    avalanche_spike_counts: Array1DInt
    """ Count of spikes in each avalanche, also known as "size". """

    @property
    def avalanche_sizes(self) -> Array1DInt:
        """ Alias for `avalanche_spike_counts`. """
        return self.avalanche_spike_counts

    avalanche_spike_counts_per_bin: list[Array1DInt]
    """ Spike counts for each time bin in each avalanche, also known as "shape". """

    @property
    def avalanche_shapes(self) -> list[Array1DInt]:
        """ Alias for `avalanche_spike_counts_per_bin`. """
        return self.avalanche_spike_counts_per_bin

    avalanche_durations: Array1DInt
    """ Duration of each avalanche in number of frame bins. """

    avalanche_shape_collapse_error: SpecialFloat
    """ Calculated as the absolute difference between `beta_exponent` and `scaling_relation_exponent_fitted`. """

    inter_avalanche_durations: Array1DInt
    """ Duration of intervals between avalanches in number of frame bins. """

    unique_durations_within_threshold: Array1DInt
    """ Unique durations that is within the threshold. """

    mean_spike_counts_per_bin_by_duration: list[Array1DFloat]
    """ Average spike counts per bin sorted by duration, also known as "profile". """

    @property
    def average_profiles(self) -> list[Array1DFloat]:
        """ Alias for `mean_spike_counts_per_bin_by_duration`. """
        return self.mean_spike_counts_per_bin_by_duration

    beta_exponent: float
    """ Beta exponent of the neural avalanches. """

    beta_exponent_std: SpecialFloat
    """
    Estimated standard deviation of the beta exponent over n_bootstraps.
    Could be NaN if there is not enough profiles to consider (min 2).
    """

    beta_range: Array1DFloat
    """ Range of beta values to scan over. """

    beta_candidates_over_range: Array1DFloat
    """ Candidates for the beta exponent over the beta_range. """

    tau_exponent_spike_counts: SpecialFloat
    """ Tau exponent relating to avalanche spike counts (sizes). """

    alpha_exponent_durations: SpecialFloat
    """ Alpha exponent relating to avalanche durations. """

    ks_min_bound_spike_counts: float
    """ Minimum value (x0) that minimizes the KS statistic for avalanche spike counts (sizes). """

    ks_min_bound_durations: float
    """ Minimum value (x0) that minimizes the KS statistic for avalanche durations (sizes). """

    ks_statistic_spike_counts: SpecialFloat
    """ Kolmogorov-Smirnov statistic for avalanche spike counts (sizes). """

    @property
    def ks_statistic_size(self) -> SpecialFloat:
        """ Alias for `ks_statistic_spike_counts`. """
        return self.ks_statistic_spike_counts

    ks_statistic_duration: SpecialFloat
    """ Kolmogorov-Smirnov statistic for avalanche durations. """

    exclusion_bounds_spike_counts: tuple[float, float]
    """ Exclusion bounds used for calculating ks_statistic_spike_counts. """

    exclusion_bounds_durations: tuple[float, float]
    """ Exclusion bounds used for calculating ks_statistic_durations. """

    scaling_relation_exponent_predicted: SpecialFloat
    """ Predicted scaling relation exponent. """

    scaling_relation_exponent_fitted: SpecialFloat
    """ Fitted scaling relation exponent. """

    scaling_relation_time_values: Array1DInt
    """ Time values used for fitting scaling relation. """

    scaling_relation_fitted_params: Array1DFloat
    """ Parameters obtained from fitting scaling relation. """

    deviation_from_criticality_coefficient: SpecialFloat
    """
    Deviation from criticality coefficient,
    i.e. deviation of predicted and fitted scaling relation exponents.
    """

    branching_ratio: SpecialFloat
    """ Estimated branching ratio. """

    time_lags_k: Array1DFloat
    """ Time lags used for slop estimation (parameter "k"). """

    time_lags_slopes: Array1DFloat
    """ Regression slopes for each time lag. """

    time_lags_fit_parameters: Array1DFloat
    """ Optimal parameters of the exponential fit function. """

    def plot_avalanche_sizes(
        self,
        figsize:    tuple[int, int]     = (5, 4),
        title:      str | None          = None,
        save_path:  str | None          = None,
        ax:         Axes | None         = None
        ) -> Axes:
        """
        Creates a plot of Avalanche sizes and its fitted PDF.

        Args:
            figsize:   Size of the plot figure.
            title:     Title for the plot, if not provided, a default will be used.
            save_path: Path to the save the plot instead of showing it.
            ax:        Axes draw the plots. (Defaults to None).

        Return:
            Axes: Axes on which the plot was drawn.
        """
        # Reference results data
        burst_sizes    = self.avalanche_spike_counts
        burst_min      = self.exclusion_bounds_spike_counts[0]
        burst_max      = self.exclusion_bounds_spike_counts[1]
        burst_exponent = self.tau_exponent_spike_counts
        xmin           = self.ks_min_bound_spike_counts

        if ax is None:
            # Create figure and axes if not provided
            fig = plt.figure(figsize=figsize)
            gs  = GridSpec(nrows=1, ncols=1)
            ax  = fig.add_subplot(gs[0, 0])
        else:
            fig = None

        pdf, bins = np.histogram(burst_sizes, bins=np.arange(0, np.max(burst_sizes) + 2))
        ax.plot(bins[:-1], pdf / np.sum(pdf), "o", label="Data", alpha=0.75)

        x = np.arange(max(1, int(burst_min)), int(burst_max) + 1)
        y = np.size(burst_sizes) / (xmin ** (-burst_exponent)) * x ** (-burst_exponent)
        ax.plot(x, y / np.sum(pdf), label="Fitted PDF", linestyle="--")

        ax.set_xscale("log")
        ax.set_xlabel("Avalanche Size (S)")
        ax.set_yscale("log")
        ax.set_ylabel("PDF for avalanche size")
        ax.legend()

        if title is None:
            title = f"Size Exponent: {float(np.array(burst_exponent).reshape(-1)[0]):.2f}"
        ax.set_title(title)

        if save_path is None and fig is not None:
            fig.tight_layout()
            plt.show(block=True)
        else:
            plt.savefig(str(save_path), bbox_inches="tight")

        return ax

    def plot_avalanche_durations(
        self,
        figsize:    tuple[int, int]     = (5, 4),
        title:      str | None          = None,
        save_path:  str | None          = None,
        ax:         Axes | None         = None
        ) -> Axes:
        """
        Creates a plot of Avalanche durations and its fitted PDF.

        Args:
            figsize:   Size of the plot figure.
            title:     Title for the plot, if not provided, a default will be used.
            save_path: Path to the save the plot instead of showing it.
            ax:        Axes draw the plots. (Defaults to None).

        Return:
            Axes: Axes on which the plot was drawn.
        """
        # Reference results data
        durations         = self.avalanche_durations
        duration_min      = self.exclusion_bounds_durations[0]
        duration_max      = self.exclusion_bounds_durations[1]
        duration_exponent = self.alpha_exponent_durations
        tmin              = self.ks_min_bound_durations

        if ax is None:
            # Create figure and axes if not provided
            fig = plt.figure(figsize=figsize)
            gs  = GridSpec(nrows=1, ncols=1)
            ax  = fig.add_subplot(gs[0, 0])
        else:
            fig = None

        pdf, bins = np.histogram(durations, bins=np.arange(0, np.max(durations) + 2))
        ax.plot(bins[:-1], pdf / np.sum(pdf), "o", label="Data", alpha=0.75)

        x = np.arange(max(1, int(duration_min)), int(duration_max) + 1)
        y = np.size(durations) / (tmin ** (-duration_exponent)) * x ** (-duration_exponent)
        ax.plot(x, y / np.sum(pdf), label="Fitted PDF", linestyle="--")

        ax.set_xscale("log")
        ax.set_xlabel("Avalanche Duration (D)")
        ax.set_yscale("log")
        ax.set_ylabel("PDF for avalanche duration")
        ax.legend()

        if title is None:
            title = f"Duration Exponent: {float(np.array(duration_exponent).reshape(-1)[0]):.2f}"
        ax.set_title(title)

        if save_path is None and fig is not None:
            fig.tight_layout()
            plt.show(block=True)
        else:
            plt.savefig(str(save_path), bbox_inches="tight")

        return ax

    def plot_deviation_from_criticality_coefficient(
        self,
        figsize:    tuple[int, int]     = (5, 4),
        title:      str | None          = None,
        save_path:  str | None          = None,
        ax:         Axes | None         = None
        ) -> Axes | None:
        """
        Creates a plot of Deviation from Criticality Coefficient (DCC).

        Args:
            figsize:   Size of the plot figure.
            title:     Title for the plot, if not provided, a default will be used.
            save_path: Path to the save the plot instead of showing it.
            ax:        Axes draw the plots. (Defaults to None).

        Return:
            Axes: Axes on which the plot was drawn.
        """
        # Reference results data
        dcc = self.deviation_from_criticality_coefficient
        if not isfinite(dcc):
            print("Finite DCC value is required for this plot")
            return ax

        burst_sizes     = self.avalanche_spike_counts
        durations       = self.avalanche_durations
        time_values     = self.scaling_relation_time_values
        fit_params      = self.scaling_relation_fitted_params
        avg_burst_sizes = np.array([np.mean(burst_sizes[durations == t]) for t in time_values])

        if ax is None:
            # Create figure and axes if not provided
            fig = plt.figure(figsize=figsize)
            gs  = GridSpec(nrows=1, ncols=1)
            ax  = fig.add_subplot(gs[0, 0])
        else:
            fig = None

        ax.plot(time_values, avg_burst_sizes, "o", label="Data", alpha=0.75)
        ax.plot(
            time_values, np.exp(fit_params[1]) * time_values ** fit_params[0],
            label="Fit",
            linestyle="--"
            )

        ax.set_xscale("log")
        ax.set_xlabel("D (Duration)")
        ax.set_yscale("log")
        ax.set_ylabel("<S> (Mean Burst Size)")
        ax.legend()

        if title is None:
            title = f"DCC: {dcc:.2f}"
        ax.set_title(title)

        if save_path is None and fig is not None:
            fig.tight_layout()
            plt.show(block=True)
        else:
            plt.savefig(str(save_path), bbox_inches="tight")

        return ax

    def plot_avalanche_shape_collapse_analysis(
        self,
        figsize:    tuple[float, float] = (4.5, 12),
        title:      str | None          = None,
        save_path:  str | None          = None,
        axes:       list[Axes] | None   = None
        ) -> list[Axes]:
        """
        Creates three plots based on avalanche shape collapse analysis, being:
        1. Scaled Variance at Optimal Beta
        2. Collapse Metric vs Scaling Exponent
        3. Scaled Avalanche Profiles (Shape Collapse)

        Args:
            figsize:   Size of the plot figure.
            title:     Title for the plot, if not provided, a default will be used.
            save_path: Path to the save the plot instead of showing it.
            axes:      List of three Axes to draw the plots. (Defaults to None).

        Returns:
            list[Axes]: List of three Axes used to draw the plots.
        """
        # Reference results data
        avg_profiles     = self.mean_spike_counts_per_bin_by_duration
        durations        = self.unique_durations_within_threshold
        beta_exponent    = self.beta_exponent
        beta_range       = self.beta_range
        collapse_metrics = self.beta_candidates_over_range

        if axes is None:
            # Create figure and axes if not provided
            num_axes = 3
            fig      = plt.figure(figsize=figsize)
            gs       = GridSpec(nrows=3, ncols=1)
            axes     = [fig.add_subplot(gs[i, 0]) for i in range(num_axes)]
        else:
            fig = None

        # Plot 1: Scaled Variance at Optimal Beta
        # Plot 2: Collapse Metric vs Scaling Exponent
        variances = []
        for profile, T in zip(avg_profiles, durations):
            t              = (np.arange(1, T + 1)) / T
            center_of_mass = np.sum(t * profile)
            variance       = np.sum(((t - center_of_mass) ** 2) * profile)
            variances.append(variance)
        variances = np.array(variances)

        scaled_variance = variances * ((np.array(durations) ** (2 * beta_exponent)))
        axes[0].plot(durations, scaled_variance, "o-", color="crimson")
        axes[0].set_title(f"Scaled Variance at Optimal $\\beta$ = {beta_exponent:.3f}")
        axes[0].set_xlabel("Avalanche Duration (T)")
        axes[0].set_ylabel("$\\sigma_t^2 \\cdot T^{2\\beta}$")
        axes[0].grid(True, linestyle="--", alpha=0.6)

        axes[1].plot(beta_range, collapse_metrics, color="navy")
        axes[1].axvline(beta_exponent, color="r", linestyle="--", label=f"Optimal $\\beta$ = {beta_exponent:.3f}")
        axes[1].set_title("Collapse Metric vs. Scaling Exponent ($\\beta$)")
        axes[1].set_xlabel("Scaling Exponent ($\\beta$)")
        axes[1].set_ylabel("Collapse Metric (Std Dev)")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.6)

        # Plot 3: Scaled Avalanche Profiles (Shape Collapse)
        ax3 = axes[2]
        ax3.set_title("Scaled Avalanche Profiles (Shape Collapse)")
        ax3.set_xlabel("Scaled Time (t/T)")
        ax3.set_ylabel("Scaled Probability ($P(t/T) \\cdot T^{\\beta}$)")

        colors             = plt.cm.viridis(np.linspace(0, 1, len(avg_profiles)))
        resampled_profiles = []
        common_time_axis   = np.linspace(0, 1, 100)

        for i, (profile, T) in enumerate(zip(avg_profiles, durations)):
            scaled_time = (np.arange(1, T + 1)) / T
            scaled_prob = profile * (T ** beta_exponent)

            # Plot the data without a label
            ax3.plot(scaled_time, scaled_prob, color=colors[i], alpha=0.6)

            interp_time = np.concatenate(([0], scaled_time, [1]))
            interp_prob = np.concatenate(([0], scaled_prob, [0]))
            resampled_profiles.append(np.interp(common_time_axis, interp_time, interp_prob))

        # Manually define the legend entries (handles)
        legend_handles = [Line2D([0], [0], color="grey", linewidth=2, label="Scaled Profiles")]

        if resampled_profiles:
            mean_profile = np.mean(resampled_profiles, axis=0)
            # Plot the fitted line and add its handle
            mean_line, = ax3.plot(
                common_time_axis, mean_profile,
                color     = "red",
                linewidth = 3,
                zorder    = 10,
                label     = "Mean Scaling Function (Fitted)"
                )
            legend_handles.append(mean_line)

        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.legend(handles=legend_handles)

        if title is None:
            title = "Avalanche Shape Collapse Analysis"
        if fig:
            fig.suptitle(title, fontsize=14)

        if save_path is None and fig is not None:
            fig.tight_layout()
            plt.show(block=True)
        else:
            plt.savefig(str(save_path), bbox_inches="tight")

        return axes

    def plot_branching_ratio(
        self,
        figsize:    tuple[int, int]     = (5, 4),
        title:      str | None          = None,
        save_path:  str | None          = None,
        ax:         Axes | None         = None
        ) -> Axes:
        ...
        """
        Creates a plot of Deviation from Criticality Coefficient (DCC).

        Args:
            figsize:   Size of the plot figure.
            title:     Title for the plot, if not provided, a default will be used.
            save_path: Path to the save the plot instead of showing it.
            ax:        Axes draw the plots. (Defaults to None).

        Return:
            Axes: Axes on which the plot was drawn.
        """
        from .._metrics._criticality import _exponential_function
        # Reference results data
        branching_ratio  = self.branching_ratio
        time_lags_k      = self.time_lags_k
        time_lags_slopes = self.time_lags_slopes
        fit_params       = self.time_lags_fit_parameters

        if ax is None:
            # Create figure and axes if not provided
            fig = plt.figure(figsize=figsize)
            gs  = GridSpec(nrows=1, ncols=1)
            ax  = fig.add_subplot(gs[0, 0])
        else:
            fig = None

        # Plot the raw data points with a subtle, non-distracting style
        ax.plot(
            time_lags_k,
            time_lags_slopes,
            marker    = "o",
            linestyle = "none",
            color     = "steelblue",
            alpha     = 0.6,
            label     = "Data"
            )

        # Plot the fitted line with a distinct color
        branching_ratio_text = f"MR Estimation (BR = {branching_ratio:.4f})"
        ax.plot(
            time_lags_k,
            _exponential_function(time_lags_k, *fit_params),
            linestyle = "-",
            linewidth = 2,
            color     = "orangered",
            label     = branching_ratio_text
            )

        ax.set_xlabel(r"Time lag $k$")
        ax.set_ylabel(r"Autocorrelation $r_k$")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Add a legend
        ax.legend(fontsize=11, frameon=True, loc="best")

        if title is None:
            title = "Autocorrelation with Mean-Response Estimation"
        ax.set_title(title, pad=15)

        if save_path is None and fig is not None:
            fig.tight_layout()
            plt.show(block=True)
        else:
            plt.savefig(str(save_path), bbox_inches="tight")

        return ax