from __future__ import annotations
from collections.abc import Generator, Mapping
from typing import Any, TypedDict, cast

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

import tables
from tables.file import File
from tables.table import Table
from tables.attributeset import AttributeSet
from tables.group import Group
from tables.earray import EArray

from pathlib import Path

from . import (
    from_msgpacked, binary_search, binary_search_range,
    builtins_only_unpickling, BlockedUnpicklingError
    )

from ..analysis import (
    Array1DFloat,
    AnalysisResultFiringStats,
    AnalysisResultNetworkBursts,
    AnalysisResultCriticality,
    AnalysisResultInformationEntropy,
    AnalysisResultComplexityLempelZiv,
    AnalysisResultDctFeatures,
    AnalysisResultSpikeTriggeredHistogram,
    AnalysisResultsFunctionalConnectivity
    )

from ..import Spike

class RecordingView:
    """
    Recording files are standard HDF5 files and can be opened with any
    HDF5 viewer or library. A `RecordingView` provides a more convenient
    way to access the data, providing ready access for:
    - `attributes`,
    - `samples`,
    - `spikes`,
    - `stims`,
    - `data_streams`, and
    - A range of analysis functions (coming soon).

    Full access to the underlying HDF5 file (via the PyTables library)
    is provided through the `file` attribute. This allows access to the full range
    of PyTables functionality if needed.

    For example:

    ```python
    from cl import RecordingView

    file_path = "/path/to/recording.h5"
    recording = RecordingView(file_path)
    # Do something ...
    recording.close()
    ```
    """
    file: File
    """ The underlying PyTables file. """

    attributes: AttributesDict
    """
    The file / root level attributes accessible as a dictionary.

    | Attribute          | Type             | Description                                                                                             |
    |--------------------|------------------|---------------------------------------------------------------------------------------------------------|
    | application        | `dict[str, Any]` | Application attributes as a user provided dict from the attributes parameter when creating a recording. |
    | hostname           | `str`            | Hostname of the CL1 system, managed through the CL1 dashboard.                                          |
    | project_id         | `str`            | Unique identifier for the undergoing project, managed through the CL1 dashboard.                        |
    | cell_batch_id      | `str`            | Unique identifier of the cell batch, managed through the CL1 dashboard.                                 |
    | created_localtime  | `str`            | When the recording is created in ISO format in the local timezone.                                      |
    | created_utc        | `str`            | When the recording is created in ISO format in UTC timezone.                                            |
    | ended_localtime    | `str`            | When the recording ended in ISO format in the local timezone.                                           |
    | ended_utc          | `str`            | When the recording ended in ISO format in UTC timezone.                                                 |
    | git_hash           | `str`            | Metadata relating to the software version.                                                              |
    | git_branch         | `str`            | Metadata relating to the software version.                                                              |
    | git_tags           | `str`            | Metadata relating to the software version.                                                              |
    | git_status         | `str`            | Metadata relating to the software version.                                                              |
    | channel_count      | `int`            | Number of channels.                                                                                     |
    | sampling_frequency | `int`            | Sampling frequency in Hz.                                                                               |
    | frames_per_second  | `int`            | Number of frames per second, same as sampling frequency.                                                |
    | uV_per_sample_unit | `float`          | Multiply the recording sample values by this constant to obtain sample values as microvolts (uV).       |
    | duration_frames    | `int`            | Duration of this recording in frames.                                                                   |
    | duration_seconds   | `float`          | Duration of this recording in seconds.                                                                  |
    | start_timestamp    | `int`            | Timestamp of the first frame.                                                                           |
    | end_timestamp      | `int`            | Timestamp of the last frame.                                                                            |
    | file_format        | `dict`           | See below.                                                                                              |

    The `file_format` attribute contains information relating to the format of the recording as a `dict`.
    This contains two attributes being `version` and `stim_and_spike_timestamps_relative_to_start`.
    The latter, when `True`, indicates that the timestamps included for stims and spikes are relative to the
    `start_timestamp` of the recording.

    For example:

    ```python
    recording     = RecordingView(file_path)
    channel_count = recording.attributes["channel_count"]
    ```
    """

    samples: EArray | None = None
    """
    Recorded raw samples (frames) as an `int16` array with shape `(duration_frames, channel_count)`.
    Sample values can be converted to microvolts (uV) by multiplying with the `uV_per_sample_unit` attribute.
    This is `None` if a recording is created with `Neurons.record(include_raw_samples=False)`.

    For example:

    ```python
    recording = RecordingView(file_path)

    # Get a slice of the first 1000 samples from channel 8, 9, 10 and convert to uV
    samples    = recording.samples[:1000, 8:11] # shape (1000, 3), dtype int16
    samples_uV = samples * recording.attributes["uV_per_sample_unit"] # shape (1000, 3), dtype float64
    ```
    """

    spikes: Table | None = None
    """
    Recorded detected spikes as a `Table` with columns `channel`, `timestamp` and `samples`.
    This is `None` if a recording is created with `Neurons.record(include_spikes=False)`.

    Spike timestamps are relative to the start of the recording by default and can be checked
    with the attribute `stim_and_spike_timestamps_relative_to_start`.

    For example:

    ```python
    recording = RecordingView(file_path)

    # Get a count of total spikes
    spike_count = len(recording.spikes)

    # Iterate through all spikes in the recording
    for spike in recording.spikes:
        # spike["channel"],  equivalent to `Spike.channel`
        # spike["timestamp], equivalent to `Spike.timestamp`
        # spike["samples],   equivalent to `Spike.samples`

    # Get spikes from channels 8, 9, occurring within the the first 1000 frames
    spike_indices = recording.spikes.get_where_list("((channel==8) | (channel==9)) & ((timestamp >= 0) & (timestamp < 1000))")
    spikes        = recording.spikes[spike_indices]
    ```
    """

    stims: Table | None = None
    """
    Recorded stimulation events as a `Table` with columns `channel` and `timestamp`.
    This is `None` if a recording is created with `Neurons.record(include_stims=False)`.

    Stim timestamps are relative to the start of the recording by default and can be checked
    with the attribute `stim_and_spike_timestamps_relative_to_start`.

    For example:

    ```python
    recording = RecordingView(file_path)

    # Get a count of total stims
    stim_count = len(recording.stims)

    # Iterate through all stims in the recording
    for stim in recording.stims:
        # stim["channel"],  equivalent to `Stim.channel`
        # stim["timestamp], equivalent to `Stim.timestamp`

    # Get stims from channels 8, 9, occurring within the the first 1000 frames
    stim_indices = recording.stims.get_where_list("((channel==8) | (channel==9)) & ((timestamp >= 0) & (timestamp < 1000))")
    stims        = recording.stims[stim_indices]
    ```
    """

    data_streams: DataStreamCollection | None = None
    """
    Recorded data streams with a dictionary like interface.
    This is `None` if a recording is created with `Neurons.record(include_data_streams=False)`.

    Available data streams can be accessed as follows:

    ```python
    recording = RecordingView(file_path)

    # Print a list of available data streams
    print(recording.data_streams)

    # Get available data stream names as a list
    data_stream_names = list(recording.data_streams.keys())
    ```

    Data within each named data stream can be accessed like a dictionary, where the
    keys are timestamps and values contain data, for example:

    ```python
    my_data_stream = recording.data_streams["my_data_stream"]
    for timestamp, data in my_data_stream.items():
        print(timestamp, data)
    ```

    It is also possible to get data for using timestamps within a data stream:

    ```python
    my_data_stream = recording.data_streams["my_data_stream"]

    # Single timestamp
    data = my_data_stream[timestamp]

    # Range of timestamps
    data_list = my_data_stream[start_timestamp : end_timestamp]
    ```
    """
    def __init__(self, file_path: str):
        """
        Constructor for RecordingView.

        Args:
            file_path: Path to the recording (`.h5`) file to be opened.
        """
        try:
            with builtins_only_unpickling():
                self.file = tables.open_file(file_path, mode='r')

                # Helper function to check for object types in dtypes, which could indicate potentially unsafe pickled data
                def dtype_has_object(self, dtype):
                    """ Recursively check if a dtype contains any object types. """
                    if dtype is None:
                        return False
                    if dtype.kind == "O":
                        return True
                    if dtype.fields:
                        for _name, (subdt, _off) in dtype.fields.items():
                            if dtype_has_object(self, subdt):
                                return True
                    return False

                for node in self.file.walk_nodes("/", classname=None):
                    # Opening the file seems to be enough to trigger loading of attributes,
                    # but just in case, we will attempt to access each attribute in the file
                    # to ensure that all potentially unsafe pickle data is detected.
                    for name in node._v_attrs._f_list():
                        _ = node._v_attrs[name]

                    # Ban VLArray storing Python objects
                    if isinstance(node, tables.VLArray) and isinstance(node.atom, tables.ObjectAtom):
                        raise BlockedUnpicklingError(f"ObjectAtom VLArray: {node._v_pathname}")

                    # Ban any ObjectAtom (covers other containers that use atoms)
                    if hasattr(node, "atom") and isinstance(getattr(node, "atom", None), tables.ObjectAtom):
                        raise BlockedUnpicklingError(f"ObjectAtom node: {node._v_pathname}")

                    # Ban object dtype arrays (Array/CArray/EArray/Table readouts)
                    # Many nodes expose .dtype; some expose .description/.coltypes etc.
                    dtype = getattr(node, "dtype", None)
                    if dtype is not None and dtype_has_object(self, dtype):
                        raise BlockedUnpicklingError(f"Object dtype in node: {node._v_pathname} dtype={dtype}")

                    # Tables: also check column dtypes explicitly
                    if isinstance(node, tables.Table):
                        dtype = node.dtype
                        if dtype_has_object(self, dtype):
                            raise BlockedUnpicklingError(f"Object column(s) in table: {node._v_pathname} dtype={dtype}")
        except BlockedUnpicklingError as e:
            try:
                self.file.close()
            except Exception:
                pass
            e.add_note(f"Refusing to open recording file {file_path}, as it contains potentially unsafe data (security risk).")
            raise

        self.attributes = cast(AttributesDict, AttributesView(self.file.root._v_attrs))

        if "samples" in self.file.root:
            self.samples = self.file.root.samples
        if "spikes" in self.file.root:
            self.spikes = self.file.root.spikes
        if "stims" in self.file.root:
            self.stims = self.file.root.stims
        if "data_stream" in self.file.root:
            self.data_streams = DataStreamCollection(self.file.root.data_stream)

        # Data cache used for analysis
        from cl.analysis._data_cache import _AnalysisDataCache
        self._analysis_cache = _AnalysisDataCache(file_path, self)

    def close(self):
        """ Close the underlying PyTables file. """
        self.file.close()

    def __repr__(self):
        return (
            f"RecordingView of file: {str(Path(self.file.filename).resolve())}"
            "\n    file:         Direct access to the underlying PyTables object"
            "\n    attributes:   A view of the recording attributes"
            "\n    spikes:       Access spikes stored in the recording"
            "\n    stims:        Access stims stored in the recording"
            "\n    samples:      Access raw frames of samples stored in the recording"
            "\n    data_streams: A collection of recorded data streams"
            )

    def __del__(self):
        self.close()

    #
    # Analysis functions
    #

    def analyse_firing_stats(
        self,
        bin_size_sec: float = 1.0
        ) -> AnalysisResultFiringStats:


        """ Compute firing statistics efficiently for a neural recording by binning spike activity into fixed-width time bins..

        Args:
            bin_size_sec: Size of each time bin in seconds used to aggregate spike counts.
                          When this is set to `1.0` second, the results can be interpreted as Hertz.

        Returns:
            `cl.analysis.AnalysisResultFiringStats`:
            Returns per-channel and population-level firing statistics computed from binned spike activity, including firing counts, firing rates,
            inter-spike interval (ISI) distributions, and summary statistics across channels. The result provides both channel-wise measures
            and aggregated culture-level metrics for baseline activity characterisation.

        """
        from cl.analysis._metrics import _analyse_firing_stats
        return _analyse_firing_stats(
            recording    = self,
            bin_size_sec = bin_size_sec
            )

    def analyse_network_bursts(
        self,
        bin_size_sec:        float,
        onset_freq_hz:       float,
        offset_freq_hz:      float,
        min_active_channels: int | None = None
        ) -> AnalysisResultNetworkBursts:
        """
        Detects network-level bursts from spike data using a spike rate thresholding method.
        This function identifies periods of high-frequency, synchronised spiking activity
        across multiple channels, classifying them as network bursts.

        Args:
            bin_size_sec:        Size of each time bin in seconds.
            onset_freq_hz:       Per channel spike rate in Hz to mark a burst onset.
            offset_freq_hz:      Per channel spike rate in Hz to mark a burst offset.
            min_active_channels: Scalar constant to apply to the onset / offset
                                 frequencies.

        Returns:
            `cl.analysis.AnalysisResultNetworkBursts`:
              Returns detected network-level burst events based on population spike-rate thresholding, including burst count,
              durations, and spike counts, along with the underlying binned firing rates used for detection. The result also
              stores burst timing information and per-bin activity.
        """
        from cl.analysis._metrics import _analyse_network_bursts
        return _analyse_network_bursts(
            recording           = self,
            bin_size_sec        = bin_size_sec,
            onset_freq_hz       = onset_freq_hz,
            offset_freq_hz      = offset_freq_hz,
            min_active_channels = min_active_channels
            )

    def analyse_criticality(
        self,
        bin_size_sec:              float,
        percentile_threshold:      float,
        max_lags_branching_ratio:  int             = 40,
        duration_thresholds:       tuple[int, int] = (2, 5),
        min_spike_count_threshold: int             = 10,
        n_bootstraps:              int             = 100,
        random_seed:               int             = 42
        ) -> AnalysisResultCriticality:
        """
        Detects **neuronal avalanches** and computes criticality-related metrics such as avalanche size distributions,
        duration distributions, power-law statistics, deviation from criticality coefficient (DCC), shape collapse error,
        and branching ratio.

        To find avalanches:
        1. Compute the total network activity by summing spike counts across channels for each time bin.
        2. Define a threshold based on the provided percentile.
        3. Identify avalanches where network activity exceed the threshold in consecutive time bins.
        4. Calculates spike counts for each avalanche as well as durations in number of time bins.

        Args:
            bin_size_sec:               Size of each time bin in seconds.
            percentile_threshold:       A percentile value (0 to 1) used to calculate a threshold for detecting bursts.
                                        If percentile > 0, the threshold is determined as the percentile of the summed network activity.
                                        If percentile == 0, the threshold is set to 0.
            max_lags_branching_ratio:   Maximum number of time lags to consider for slope estimation.
            duration_thresholds:        Thresholds (min, max) for avalanche durations (number of frame_bins).
                                        Recommend values between (3-6).
            min_spike_count_threshold:  Minimum threshold for spike counts in avalanches for calculating the size exponent.
                                        Recommend values between (8-20).
            n_boostraps:                Number of random resampling iterations used to estimate the variability of the beta exponent.


        Returns:
            `cl.analysis.AnalysisResultCriticality`:
              Returns a comprehensive set of avalanche and criticality summaries derived from thresholded network activity,
              including avalanche sizes and durations, inter-avalanche intervals, power-law fit exponents and KS statistics,
              scaling-relation / shape-collapse measures (e.g., DCC and collapse error), and a branching-ratio estimate.
              All intermediate arrays used to compute these metrics (e.g., per-avalanche binned spike profiles and fitted
              parameter traces) are included in the result object for inspection and downstream re-analysis.

        """
        from cl.analysis._metrics import _analyse_criticality
        return _analyse_criticality(
            recording                 = self,
            bin_size_sec              = bin_size_sec,
            percentile_threshold      = percentile_threshold,
            duration_thresholds       = duration_thresholds,
            max_lags_branching_ratio  = max_lags_branching_ratio,
            min_spike_count_threshold = min_spike_count_threshold,
            n_bootstraps              = n_bootstraps,
            random_seed               = random_seed
            )

    def analyse_information_entropy(
        self,
        bin_size_sec: float,
        fillna:       float | None = 0.0,
        log_base:     float | None = None
        ) -> AnalysisResultInformationEntropy:
        """
        Computes per-bin Bernoulli entropy of the fraction of channels that have >=1 spike in the bin.

        Args:
            bin_size_sec: Size of each time bin in seconds.
            fillna:       Value to fill for `NaN` (defaults to 0.).
            log_base:     Entropy units where None uses natural log (nats), 2 uses log2 (bits)

        Returns:
            `cl.analysis.AnalysisResultInformationEntropy`:
              Returns per-bin Bernoulli entropy computed from the fraction of active channels in each time bin,
              providing a population-level measure of activity variability over time. Entropy can be expressed in nats
              or bits.
        """
        from cl.analysis._metrics import _analyse_information_entropy
        return _analyse_information_entropy(
            recording    = self,
            bin_size_sec = bin_size_sec,
            fillna       = fillna,
            log_base     = log_base
            )

    def analyse_lempel_ziv_complexity(
        self,
        bin_size_sec:  float,
        min_bin_count: int  = 2,
        normalise:     bool = True,
        use_binary:    bool = True
        ) -> AnalysisResultComplexityLempelZiv:
        """
        Computes **Lempel–Ziv complexity (LZ78)** for each channel’s binned spike activity, measuring temporal complexity
        and structure.

        For each channel, a symbol sequence is formed per time bin:
        - use_binary=True: binary sequence (spike present/absent per bin).
        - use_binary=False: integer spike-count sequence.

        Complexity is the number of phrases added by an LZ78 dictionary-building procedure.
        If normalise=True, scores are length-normalised using
            c_norm = c(n) * log_k(n) / n,
        where n is the number of bins and k is the alphabet size (k=2 for binary).

        Args:
            bin_size_sec:  Time-bin size in seconds.
            min_bin_count: Minimum number of bins required; below this threshold complexity is set to zero.
            normalise:     Whether to return the length-normalised score.
            use_binary:    Whether to binarise spike counts before computing complexity or use raw spike counts.

        Returns:
            `cl.analysis.AnalysisResultComplexityLempelZiv`:
              Returns per-channel Lempel–Ziv (LZ78) complexity scores computed from binned spike activity,
              using either binary or count-based symbol sequences. Scores can be returned raw or length-normalised based
              on the sequence length and alphabet size, providing a measure of temporal complexity for each channel.
        """

        from cl.analysis._metrics import _analyse_lempel_ziv_complexity
        return _analyse_lempel_ziv_complexity(
            recording     = self,
            bin_size_sec  = bin_size_sec,
            min_bin_count = min_bin_count,
            normalise     = normalise,
            use_binary    = use_binary
            )

    def analyse_dct_features(
        self,
        k: int
        ) -> AnalysisResultDctFeatures:
        """
        Calculates the Discrete Cosine Transform (DCT) features based on channel spike counts.

        Args:
            k: The frequency index (coefficient index) of the DCT.

        Returns:
            `cl.analysis.AnalysisResultDctFeatures`:
              Returns spatial Discrete Cosine Transform (DCT) features computed from per-channel spike counts arranged
              in the common MEA layout, capturing low-frequency spatial structure in neural activity. The result includes
              the DCT basis coefficients for both MEA dimensions and a dictionary of DCT feature values indexed by spatial
              frequency components.
        """
        from cl.analysis._metrics import _analyse_dct_features
        return _analyse_dct_features(
            recording = self,
            k         = k
            )

    def analyse_spike_triggered_histogram(
        self,
        bin_size_sec:                 float,
        start_sec:                    float,
        end_sec:                      float,
        num_channels:                 int,
        min_firing_rate_threshold_hz: float = 0.1
        ) -> AnalysisResultSpikeTriggeredHistogram:
        """
        Generates spike-triggered histograms using the most active channels as triggers,
        quantifying population responses around trigger spikes.

        Args:
            bin_size_sec:                 Bin size in seconds for the histogram.
            start_sec:                    Time in seconds to include before the trigger spike.
            end_sec:                      Time in seconds to include after the trigger spike.
            num_channels:                 How many of the most active channels to use as triggers.
            min_firing_rate_threshold_hz: Threshold (in Hz) for minimum firing rate.
                                          Only channels with firing rates above this value are
                                          considered.

        Returns:
            `cl.analysis.AnalysisResultSpikeTriggeredHistogram`:
              Contains spike-triggered histograms computed for ordered pairs of the most active channels,
              where each histogram represents the timing of target-channel spikes relative to trigger-channel
              spikes within a fixed temporal window. The result includes the common time bins and a dictionary mapping
              channel pairs to their corresponding histograms.
        """
        from cl.analysis._metrics import _analyse_spike_triggered_histogram
        return _analyse_spike_triggered_histogram(
            recording                    = self,
            bin_size_sec                 = bin_size_sec,
            start_sec                    = start_sec,
            end_sec                      = end_sec,
            num_channels                 = num_channels,
            min_firing_rate_threshold_hz = min_firing_rate_threshold_hz
            )

    def analyse_functional_connectivity(
        self,
        bin_size_sec         : float,
        correlation_threshold: float = 0.6,
        ) -> AnalysisResultsFunctionalConnectivity:
        """
        Compute functional connectivity (based on Pearson correlation) and
        summary graph metrics from spike data.

        Args:
            bin_size_sec:           Size of each time bin in seconds.
            correlation_threshold:  Absolute correlation threshold in [0, 1]. Only connections where
                                    Pearson correlation >= correlation_threshold are kept as graph edges.
                                    Use 0.0 to keep the full weighted correlation matrix.

        Returns:
            `cl.analysis.AnalysisResultsFunctionalConnectivity`:
              Returns a weighted functional connectivity matrix computed using Pearson correlation between binned channel
              spike counts, along with basic graph-level metrics including total and average edge weights, clustering
              coefficient, Louvain community structure, modularity index, and maximum betweenness centrality. These network
              metrics are provided as baseline summaries using default parameters; users requiring fine-tuned or
              domain-specific network analysis are encouraged to directly use the returned connectivity matrix to
              recompute graph metrics with custom methods and parameter settings.
        """
        from cl.analysis._metrics import _analyse_functional_connectivity
        return _analyse_functional_connectivity(
            recording             = self,
            bin_size_sec          = bin_size_sec,
            correlation_threshold = correlation_threshold
            )

    #
    # Visualisation functions
    #

    @staticmethod
    def plot_spike(
        spike:      tuple[int, int, np.ndarray[tuple[int], np.dtype[np.float64]]] | Spike,
        figsize:    tuple[int, int]            = (6, 2),
        title:      str | None                 = None,
        save_path:  str | None                 = None,
        ax:         Axes | None                = None,
        ):
        """
        Creates a plot of a single spike.

        Args:
            spike:     Either a Spike object or a tuple of (timestamp, channel, samples).
            figsize:   Size of the plot figure.
            title:     Title for the plot, if not provided, a default will be used.
            save_path: Path to the save the plot instead of showing it.
            ax:        Axes draw the plots. (Defaults to None).

        Return:
            Axes: Axes on which the plot was drawn.

        For example:

        ```python
        from cl import RecordingView
        recording = RecordingView(file_path)

        # Plot the largest N spikes
        N = 5
        largest_spikes = sorted(
            recording.spikes[:],
            key     = lambda spike: max(spike["samples"]) - min(spike["samples"]),
            reverse = True
            )[:N]

        for spike in largest_spikes:
            recording.plot_spike(spike)
        ```
        """
        if isinstance(spike, np.void) or isinstance(spike, tuple):
            timestamp, channel, samples = spike
        else:
            timestamp = spike.timestamp
            channel   = spike.channel
            samples   = spike.samples

        def uV_formatter(value, pos) -> str:
            """ A function to be passed to matplotlib.ticker.FuncFormatter to format value as microvolts. """
            return f"{value} µV"

        if ax is None:
            # Create figure and axes if not provided
            fig = plt.figure(figsize=figsize)
            gs  = GridSpec(nrows=1, ncols=1)
            ax  = fig.add_subplot(gs[0, 0])
        else:
            fig = None

        ax.plot(samples)
        ax.axvline(x=25, color="gray", linestyle=":", linewidth=1)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(uV_formatter))

        if title is None:
            title = f"Spike on Channel {channel} at Timestamp {timestamp}"
        ax.set_xlabel(title)

        if save_path is None and fig is not None:
            fig.tight_layout()
            plt.show(block=True)
        else:
            plt.savefig(str(save_path), bbox_inches="tight")

        return ax

    def plot_spikes_and_stims(
        self,
        figsize:                  tuple[int, int]            = (12, 8),
        title:                    str | None                 = None,
        save_path:                str | None                 = None,
        limit_to_time_range_secs: tuple[float, float] | None = None,
        limit_to_channels:        list[int] | None           = None,
        ):
        """
        Creates a raster plot of spikes and stims in the recording.

        Args:
            figsize:                  Size of the plot figure.
            title:                    Title for the plot, if not provided, a default will be used.
            save_path:                Path to the save the plot instead of showing it.
            limit_to_time_range_secs: If provided, limit the time axis as a tuple of `(start_time_secs, end_time_secs)`.
            limit_to_channels:        If provided, limit the number of channels if provided.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from ..analysis._plots import _plot_spikes_and_stims_raster

        sampling_frequency       = self.attributes["sampling_frequency"]
        duration_seconds         = self.attributes["duration_seconds"]

        spike_frames_per_channel = self._analysis_cache.get_spike_frames_by_channel()
        spike_times_per_channel: list[Array1DFloat] = [
            spike_frames / sampling_frequency
            for spike_frames in spike_frames_per_channel.values()
            ]

        stim_times_per_channel: list[Array1DFloat] | None = None
        if self.stims is not None:
            stim_frames_per_channel = self._analysis_cache.get_stim_frames_by_channel()
            stim_times_per_channel = [
                stim_frames / sampling_frequency
                for stim_frames in stim_frames_per_channel.values()
                ]

        fig = plt.figure(figsize=figsize)
        gs  = GridSpec(nrows=1, ncols=1)
        ax  = fig.add_subplot(gs[0, 0])

        _plot_spikes_and_stims_raster(
            spike_times_per_channel  = spike_times_per_channel,
            stim_times_per_channel   = stim_times_per_channel,
            ax                       = ax,
            limit_to_time_range_secs = limit_to_time_range_secs,
            limit_to_channels        = limit_to_channels
            )
        if limit_to_time_range_secs is None:
            ax.set_xlim(0, duration_seconds)

        if title is None:
            title = (
                "Spikes (black) and Stims (red) per Channel over Time\n"
                f"{self._analysis_cache.metadata.file_path}"
                )

        fig.suptitle(title)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
        else:
            plt.show(block=True)
        plt.close()

class AttributesView(Mapping):
    def __init__(self, h5_attributes):
        super().__setattr__('_h5_attributes', h5_attributes)

    def keys(self):
        return self._h5_attributes._v_attrnamesuser

    def items(self):
        return { k: getattr(self, k) for k in self._h5_attributes._v_attrnamesuser }.items()

    def values(self):
        return [ getattr(self, k) for k in self._h5_attributes._v_attrnamesuser ]

    def __getattr__(self, name):
        # Pass through for all attributes/methods of the wrapped object
        value = getattr(self._h5_attributes, name)

        # Convert numpy scalar types back to native Python types
        if isinstance(value, np.generic):
            return value.item()

        return value

    def __setattr__(self, name, value):
        # Pass through for setting attributes on the wrapped object
        setattr(self._h5_attributes, name, value)

    def __delattr__(self, name):
        # Pass through for deleting attributes on the wrapped object
        delattr(self._h5_attributes, name)

    def __str__(self):
        return "AttributesView:\n" + repr(self)

    def __repr__(self):
        s = "{\n"
        for key in self._h5_attributes._v_attrnamesuser:
            s += f"    {repr(key)}: {repr(getattr(self, key))}\n"
        s += "}"
        return s

    # Explicitly handle common special methods
    def __len__(self):
        return len(self._h5_attributes._v_attrnamesuser)

    def __getitem__(self, key):
        if not key in self._h5_attributes._v_attrnamesuser:
            raise KeyError(key)

        value = self._h5_attributes[key]

        # Convert numpy scalar types back to native Python types
        if isinstance(value, np.generic):
            return value.item()

        return value

    def __iter__(self):
        return iter(self.items())

    def __contains__(self, item):
        return item in self._h5_attributes._v_attrnamesuser

class AttributesDict(TypedDict):
    """
    Describes attributes that is typically contained within a recording and can be
    accessed with AttributesView, for static type hints.

    Keep these docs consistent with RecordingView.attributes.
    """

    application: dict[str, Any]
    """
    Application attributes as a user provided dict from
    the attributes parameter in the Recording constructor.
    """

    hostname: str
    """ Hostname of the CL1 system, managed through the CL1 dashboard. """

    project_id: str
    """ Unique identifier for the undergoing project, managed through the CL1 dashboard. """

    cell_batch_id: str
    """ Unique identifier of the cell batch, managed through the CL1 dashboard. """

    created_localtime : str
    """ When the recording is created in ISO format in the local timezone. """

    created_utc : str
    """ When the recording is created in ISO format in UTC timezone. """

    ended_localtime: str
    """ When the recording ended in ISO format in the local timezone. """

    ended_utc: str
    """ When the recording ended in ISO format in UTC timezone. """

    git_hash: str
    """ Metadata relating to the software version. """

    git_branch: str
    """ Metadata relating to the software version. """

    git_tags: str
    """ Metadata relating to the software version. """

    git_status: str
    """ Metadata relating to the software version. """

    channel_count: int
    """ Number of channels. """

    sampling_frequency: int
    """ Sampling frequency in Hz. """

    frames_per_second: int
    """ Number of frames per second, same as sampling frequency. """

    uV_per_sample_unit: float
    """ Multiply the recording sample values by this constant to obtain sample values as microvolts (uV). """

    duration_frames: int
    """ Duration of this recording in frames. """

    duration_seconds: float
    """ Duration of this recording in seconds. """

    start_timestamp: int
    """ Timestamp of the first frame. """

    end_timestamp: int
    """ Timestamp of the last frame. """

    file_format: dict
    """
    Information relating to the format of the recording
    as a dict. This contains two attributes being
    "version" and "stim_and_spike_timestamps_relative_to_start".
    The latter, when True, indicates that the timestamps
    included for stims and spikes are relative to the
    start_timestamp of the recording.
    """

AttributesDict.__module__ = "RecordingView" # Hide messy module path from docs

class DataStreamCollection:
    """ Interface for accessing a collection of DataStreams. """

    def __init__(self, data_streams: Group):
        self.data_streams = data_streams

    def keys(self) -> Generator[str, None, None]:
        return self.data_streams._v_children.keys()

    def items(self) -> Generator[tuple[str, DataStreamView], None, None]:
        for data_stream_name in self.keys():
            yield data_stream_name, DataStreamView(self.data_streams[data_stream_name])

    def values(self) -> Generator[DataStreamView, None, None]:
        for data_stream_name in self.keys():
            yield DataStreamView(self.data_streams[data_stream_name])

    def __repr__(self):
        data_streams = "".join(f"\n    {data_stream_name}" for data_stream_name in self.keys())
        return f"Data Streams:{data_streams}"

    def __iter__(self) -> Generator[str, None, None]:
        for data_stream_name in self.keys():
            yield data_stream_name

    def __getitem__(self, key):
        return DataStreamView(self.data_streams[key])

    def __getattr__(self, name):
        return DataStreamView(self.data_streams[name])

    def __len__(self):
        return len(self.data_streams._v_children.keys())

    def __contains__(self, key):
        return key in self.keys()

class DataStreamView:
    """
    Provides a read-only interface to data stream entries.

    DataStreamView is designed to allow iteration over data stream entries
    without loading the entire data stream into memory.
    """

    @staticmethod
    def data_for_entry(data, entry: Group):
        return from_msgpacked(data[entry["start_index"]:entry["end_index"]])

    def __init__(self, data_stream):
        self.index      = data_stream.index
        self.data       = data_stream.data
        self.attributes = AttributesView(data_stream._v_attrs)
        self._len       = len(data_stream.index)

    class DataStreamKeysView:
        def __init__(self, index):
            self.index = index

        def __iter__(self) -> Generator[int, None, None]:
            for entry in self.index:
                yield entry["timestamp"]

        def __len__(self) -> int:
            return len(self.index)

        def __repr__(self) -> str:
            return str(list(self))

    class DataStreamValuesView:
        def __init__(self, index, data) -> None:
            self.index  = index
            self.data   = data

        def __iter__(self) -> Generator[Any, None, None]:
            for entry in self.index:
                yield DataStreamView.data_for_entry(self.data, entry)

        def __len__(self):
            return len(self.index)

        def __repr__(self) -> str:
            return str(list(self))

    class DataStreamItemsView:
        def __init__(self, index, data) -> None:
            self.index  = index
            self.data   = data

        def __iter__(self) -> Generator[tuple[int, Any], None, None]:
            for entry in self.index:
                ts      = entry["timestamp"]
                data    = DataStreamView.data_for_entry(self.data, entry)
                yield ts, data

        def __len__(self):
            return len(self.index)

        def __repr__(self) -> str:
            return str(list(self))

    def keys(self) -> DataStreamKeysView:
        return self.DataStreamKeysView(self.index)

    def values(self) -> DataStreamValuesView:
        return self.DataStreamValuesView(self.index, self.data)

    def items(self) -> DataStreamItemsView:
        return self.DataStreamItemsView(self.index, self.data)

    def __len__(self):
        return self._len

    def __iter__(self):
        return iter(self.items())

    def __getitem__(self, key):
        """
        Get data for either a specific timestamp or for a range of timestamps.

        Single timestamp:
            data = data_stream[timestamp]

        Range of timestamps:
            items_view = data_stream[start_timestamp:end_timestamp]

        Raises KeyError if a specific timestamp is passed and it is not found.
        """
        if isinstance(key, slice):
            return self.values_for_range(key.start, key.stop)
        elif isinstance(key, int):
            entry_index = binary_search(self.index, key, lambda x: x["timestamp"])
            if entry_index is None:
                raise KeyError(key)

            return DataStreamView.data_for_entry(self.data, self.index[entry_index])
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def keys_for_range(self, start_timestamp, end_timestamp):
        """ Get all keys (timestamps) from start_timestamp up to but not including end_timestamp. """
        range_start, range_end = binary_search_range(self.index, start_timestamp, end_timestamp, lambda x: x["timestamp"])
        return self.DataStreamKeysView(self.index[range_start:range_end])

    def values_for_range(self, start_timestamp, end_timestamp):
        """ Get all values from start_timestamp up to but not including end_timestamp. """
        range_start, range_end = binary_search_range(self.index, start_timestamp, end_timestamp, lambda x: x["timestamp"])
        return self.DataStreamValuesView(self.index[range_start:range_end], self.data)

    def items_for_range(self, start_timestamp, end_timestamp):
        """ Get all items from start_timestamp up to but not including end_timestamp. """
        range_start, range_end = binary_search_range(self.index, start_timestamp, end_timestamp, lambda x: x["timestamp"])
        return self.DataStreamItemsView(self.index[range_start:range_end], self.data)

DataStreamCollection.__module__ = "RecordingView" # Hide messy module path from docs
