import numpy as np

from matplotlib.axes import Axes

from . import Array1DInt, Array1DFloat, Bursts

def _plot_firing_rate_bins(
    x:  Array1DInt | Array1DFloat,
    y:  Array1DFloat,
    ax: Axes,
    ):
    """
    Plots firing rates over time as a line plot.

    Args:
        x:  Time axis for each time bin in seconds shape (time_bins).
        y:  Firing rates 2D array of shape (channel_count, time_bins).
        ax: Axes to draw this plot on.
    """
    ax.plot(x, y, color="black", lw=1.5)
    ax.set_ylabel("Avg Firing Rate (Hz/Ch)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

def _plot_spikes_and_stims_raster(
    spike_times_per_channel:  list[Array1DFloat],
    ax:                       Axes,
    stim_times_per_channel:   list[Array1DFloat] | None  = None,
    limit_to_time_range_secs: tuple[float, float] | None = None,
    limit_to_channels:        list[int] | None           = None,
    ):
    """
    Plots per channel spikes and stims over time as a raster plot. Spikes will
    be drawn in black and stims will be drawn in red.

    Args:
        spike_times_per_channel:  List of 1D Array of spike times (in sec) for each channel.
        stim_times_per_channel:   List of 1D Array of stim times (in sec) for each channel.
        limit_to_time_range_secs: If provided, limit the time axis as a tuple of `(start_time_secs, end_time_secs)`.
        limit_to_channels:        If provided, limit the number of channels if provided.
        ax:                       Axes to draw this plot on.
    """
    if limit_to_channels is not None:
        spike_times_per_channel = \
            [
                spike_times if channel in limit_to_channels else np.array([])
                for channel, spike_times in enumerate(spike_times_per_channel)
            ]
    ax.eventplot(
        positions  = spike_times_per_channel,
        color      = "black",
        linewidths = 0.5,
        alpha      = 0.7
        )
    if stim_times_per_channel is not None:
        if limit_to_channels is not None:
            stim_times_per_channel = \
                [
                    stim_times if channel in limit_to_channels else np.array([])
                    for channel, stim_times in enumerate(stim_times_per_channel)
                ]
        ax.eventplot(
            positions  = stim_times_per_channel,
            color      = "red",
            linewidths = 1,
            linelengths = 2.2,
            alpha      = 1,
        )

        for channel, stim_times in enumerate(stim_times_per_channel):
            if len(stim_times) > 0:
                ax.scatter(
                    stim_times,
                    np.full_like(stim_times, channel + 0.4),
                    facecolors = "white",
                    edgecolors = "red",
                    linewidths = 1.5,
                    s          = 30,
                    marker     = "o",
                    zorder     = 4
                )
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Channel", fontsize=18)

    if limit_to_time_range_secs is not None:
        ax.set_xlim(*limit_to_time_range_secs)

    # Use a octal scale to be more representative of MEA layout a small padding for aesthetics
    ax.set_ylim(-1, len(spike_times_per_channel) + 1)
    ax.set_yticks(range(0, len(spike_times_per_channel) + 1, 8))

    ax.tick_params(axis="both", which="major", labelsize=16)


    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

def _plot_bursts(
    bursts:             Bursts,
    sampling_frequency: int | float,
    ax:                 Axes
    ):
    """
    Plots a burst as highlighted region on a given axis.

    Args:
        bursts:             Bursts to plot.
        sampling_frequency: Sampling frequency to convert frames to secs.
        ax:                 Axes to draw this plot on.
    """
    for burst in bursts:
        ax.axvspan(
            xmin  = burst.start_frame / sampling_frequency,
            xmax  = burst.end_frame / sampling_frequency,
            color = "red",
            alpha = 0.3
            )
