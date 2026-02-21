from pathlib import Path

from .. import (
    Array1DInt,
    Array1DFloat,
    BurstsType,
    AnalysisResult
    )

class AnalysisResultNetworkBursts(AnalysisResult):
    """ Burst analysis results for one recording. """

    bursts: BurstsType

    onset_freq_hz: float
    """ Firing rate threshold to enter a burst. """

    offset_freq_hz: float
    """ Firing rate threshold to exit a burst. """

    network_burst_count: int
    """ The total number of network bursts detected. """

    network_burst_durations_sec: list[float]
    """ A list of the durations (in seconds) of each burst. """

    network_burst_spike_counts: list[int]
    """ A list of the number of spikes in each burst. """

    total_network_burst_duration_sec: float
    """ The sum of all network burst durations. """

    bin_size_sec: float
    """ Size of each time bin in seconds. """

    bin_boundaries_frames: Array1DInt
    """ Boundaries of the time bins in frames shape (time_bin_count,). """

    firing_rate_per_channel_and_bin: Array1DFloat
    """ Specifies firing rate over time shape (time_bin_count,). """

    spike_frames_by_channel: dict[int, Array1DInt]
    """ Frames that a spike has occurred (values) for each channel (keys). """

    def plot(
        self,
        figsize:                  tuple[int, int]            = (12, 8),
        title:                    str | None                 = None,
        save_path:                str | None                 = None,
        limit_to_time_range_secs: tuple[float, float] | None = None,
        ):
        """
        Creates a visualisation of the spikes and bursts in this recording, containing:
        - Upper: a line plot of the spike rate per channel and time bin, and
        - Lower: a raster plot of spikes with bursts as highlighted regions.

        Args:
            figsize:                  Size of the plot figure.
            title:                    Title for the plot, if not provided, a default will be used.
            save_path:                Path to the save the plot instead of showing it.
            limit_to_time_range_secs: If provided, limit the time axis as a tuple of `(start_time_secs, end_time_secs)`.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from cl.analysis._plots import (
            _plot_firing_rate_bins,
            _plot_spikes_and_stims_raster,
            _plot_bursts
            )

        sampling_frequency              = self.metadata.sampling_frequency
        bin_boundaries_frames           = self.bin_boundaries_frames
        spike_frames_by_channel         = self.spike_frames_by_channel
        firing_rate_per_channel_and_bin = self.firing_rate_per_channel_and_bin
        bursts                          = self.bursts

        fig = plt.figure(figsize=figsize)
        gs  = GridSpec(nrows=2, ncols=1, height_ratios=[1, 3])

        # x-axis labels
        bin_boundaries_sec = bin_boundaries_frames / sampling_frequency

        # Plot firing rate
        ax_rate = fig.add_subplot(gs[0, 0])
        _plot_firing_rate_bins(
            x  = bin_boundaries_sec,
            y  = firing_rate_per_channel_and_bin,
            ax = ax_rate
            )
        ax_rate.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Raster plot of stims with bursts highlighted
        ax_raster = fig.add_subplot(gs[1, 0], sharex=ax_rate)
        spike_times_per_channel: list[Array1DFloat] = [
            spike_frames / sampling_frequency
            for spike_frames in spike_frames_by_channel.values()
            ]
        _plot_spikes_and_stims_raster(
            spike_times_per_channel = spike_times_per_channel,
            ax                      = ax_raster
            )
        _plot_bursts(
            bursts             = bursts,
            sampling_frequency = sampling_frequency,
            ax                 = ax_raster
            )

        if limit_to_time_range_secs is not None:
            ax_rate.set_xlim(*limit_to_time_range_secs)

        if title is None:
            title = (
                "Network Firing Rate and Detected Bursts\n"
                f"{self.metadata.file_path}"
                )

        fig.suptitle(title)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
        else:
            plt.show(block=True)
        plt.close()