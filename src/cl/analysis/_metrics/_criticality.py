"""
Private functions that calculate metrics for neural avalanches and criticality.
DO NOT use these directly, instead access via RecordingView.
"""
from typing import TypedDict
from collections import defaultdict
from math import isfinite

import numpy as np
import scipy

from ...util import RecordingView
from .. import AnalysisResultCriticality, Array1DInt, Array1DFloat

def _analyse_criticality(
    recording:                 RecordingView,
    bin_size_sec:              float,
    percentile_threshold:      float,
    max_lags_branching_ratio:  int             = 40,
    duration_thresholds:       tuple[int, int] = (2, 5),
    min_spike_count_threshold: int             = 10,
    n_bootstraps:              int             = 100,
    random_seed:               int             = 42
    ) -> AnalysisResultCriticality:
    """
    See RecordingView.analyse_neural_avalanche()
    """
    from .._metrics._mea_layout import (
        _COMMON_GROUND_CHANNELS,
        _COMMON_REFERENCE_CHANNELS,
        _valid_common_layout
        )
    if not _valid_common_layout(recording):
        raise ValueError("Recording does not conform to common MEA layout.")

    sampling_frequency = recording._analysis_cache.metadata.sampling_frequency
    bin_size_frames    = int(bin_size_sec * sampling_frequency)
    excluded_channels  = _COMMON_GROUND_CHANNELS + _COMMON_REFERENCE_CHANNELS
    spike_count_array  = recording._analysis_cache.get_spike_count_per_time_bin(
        bin_size_frames,
        excluded_channels            = excluded_channels,
        limit_to_max_spike_timestamp = True
        ) # (channel_count, bin_count)

    #
    # Find avalanches
    #

    # Calculate network activity and define threshold based on specified percentile
    # Network activity is the total spike counts for each time bin (sum across channels)
    assert 0 <= percentile_threshold <= 1, f"Percentile {percentile_threshold} is out of range [0, 1]"
    network_activity   = spike_count_array.sum(axis=0)
    activity_threshold = np.quantile(network_activity, q=percentile_threshold).item()

    # Avalanches are defined where network_activity > threshold, otherwise they are known as "intervals"
    avalanche_mask         = np.where((network_activity > activity_threshold), 1, 0)

    # Pad both ends of the avalanche mask to capture bins at the start and end of activity sequence
    avalanche_mask         = np.pad(avalanche_mask, pad_width=(1, 1)) # [0, *avalanche_mask, 0]
    avalanche_start_bins   = np.where(np.diff(avalanche_mask) ==  1)[0]
    avalanche_end_bins     = np.where(np.diff(avalanche_mask) == -1)[0]

    # Get spike counts as sum of network activity between each start and end of each avalanche
    avalanche_spike_counts         = []
    avalanche_spike_counts_per_bin = []
    for start_bin, end_bin in zip(avalanche_start_bins, avalanche_end_bins):
        avalanche_activity = network_activity[start_bin : end_bin]
        avalanche_spike_counts.append(avalanche_activity.sum())
        avalanche_spike_counts_per_bin.append(avalanche_activity.astype(int))
    avalanche_spike_counts = np.array(avalanche_spike_counts, dtype=int)

    # Calculate durations of each avalanche and between avalanches
    avalanche_durations       = (avalanche_end_bins - avalanche_start_bins)
    inter_avalanche_durations = (avalanche_start_bins[1:] - avalanche_end_bins[:-1])

    #
    # Analyse avalanches
    #
    # sizes    aka. avalanche_spike_counts
    # shapes   aka. avalanche_spike_counts_per_bin
    # profiles aka. mean_spike_counts_per_bin_by_duration
    #

    # Average shapes
    min_duration, max_duration = duration_thresholds
    shapes_by_duration         = defaultdict(list)
    for shape, size, duration in \
        zip(avalanche_spike_counts_per_bin, avalanche_spike_counts, avalanche_durations):
        if not(min_duration <= duration <= max_duration):
            continue
        if not (size > 0):
            continue
        normalized_shape = shape / size
        shapes_by_duration[duration].append(normalized_shape)

    # Profiles
    mean_spike_counts_per_bin_by_duration = []
    unique_durations_within_threshold     = np.array(sorted(shapes_by_duration.keys()))
    for duration in unique_durations_within_threshold:
        if not (duration in shapes_by_duration):
            continue
        average_spike_counts_per_bin = np.mean(np.array(shapes_by_duration[duration]), axis=0)
        mean_spike_counts_per_bin_by_duration.append(average_spike_counts_per_bin)

    # Calculate beta exponent
    beta_range           = np.arange(0, 1.001, step=0.001)  # [0, 1] in steps of 0.001
    beta_exponent_result = _calc_beta_exponent(
        profiles   = mean_spike_counts_per_bin_by_duration,
        durations  = unique_durations_within_threshold,
        beta_range = beta_range
        )
    beta_exponent              = beta_exponent_result["beta_candidate"]
    beta_candidates_over_range = beta_exponent_result["beta_candidates_over_range"]

    # Estimate variance of beta exponent via bootstrapping (randomly choosing profiles)
    bootstrapped_beta_exponents = []
    profile_count               = len(mean_spike_counts_per_bin_by_duration)
    profile_indices             = np.arange(profile_count)
    rng                         = np.random.RandomState(random_seed)
    for _ in range(n_bootstraps):
        if profile_count < 2:
            # Too few profiles to perform bootstrapping calculations
            break
        bootstrap_indices    = rng.choice(profile_indices, size=profile_count, replace=True)
        beta_exponent_result = _calc_beta_exponent(
            profiles   = [mean_spike_counts_per_bin_by_duration[i] for i in bootstrap_indices],
            durations  = np.array([unique_durations_within_threshold[i] for i in bootstrap_indices]),
            beta_range = beta_range
            )
        bootstrapped_beta_exponents.append(beta_exponent_result["beta_candidate"])

    beta_exponent_std = float("nan")
    if len(bootstrapped_beta_exponents) > 0:
        beta_exponent_std = float(np.std(bootstrapped_beta_exponents))

    # Power-law analysis on avalanche_spike_counts (sizes)
    max_spike_count_threshold = avalanche_spike_counts.max() ** 0.8
    filtered_spike_counts     = avalanche_spike_counts[avalanche_spike_counts < max_spike_count_threshold]
    min_spike_count_bound     = min_spike_count_threshold
    max_spike_count_bound     = max_spike_count_threshold
    tau_exponent_spike_counts = float("nan")
    ks_stat_spike_counts      = float("inf")

    try:
        exclusion_bounds = _find_exclusion_bounds(
            data            = filtered_spike_counts,
            power_law_limit = min_spike_count_threshold
            )
        min_spike_count_bound = exclusion_bounds["min_bound"]
        max_spike_count_bound = exclusion_bounds["max_bound"]
        ks_stat_spike_counts  = exclusion_bounds["ks_statistic"]
    except ValueError:
        ...

    candidate_spike_count_indices = np.where(
        (avalanche_spike_counts >= min_spike_count_bound) &
        (avalanche_spike_counts <= max_spike_count_bound)
        )[0]
    spike_counts_power_laws = _fit_power_law(
        data  = avalanche_spike_counts[candidate_spike_count_indices],
        limit = min_spike_count_bound
        )
    tau_exponent_spike_counts = spike_counts_power_laws["power_law_exponent"]
    ks_min_bound_spike_counts = spike_counts_power_laws["min_bound"]

    # Power-law analysis on avalanche_durations
    max_duration_threshold   = avalanche_durations.max() ** 0.8
    filtered_durations       = avalanche_durations[avalanche_durations < max_duration_threshold]
    min_durations_bound      = min_duration
    max_durations_bound      = max_duration_threshold
    alpha_exponent_durations = float("nan")
    ks_stat_duration         = float("inf")

    try:
        exclusion_bounds = _find_exclusion_bounds(
            data            = filtered_durations,
            power_law_limit = min_durations_bound
            )
        min_durations_bound = exclusion_bounds["min_bound"]
        max_durations_bound = exclusion_bounds["max_bound"]
        ks_stat_duration    = exclusion_bounds["ks_statistic"]
    except ValueError:
        ...

    candidate_durations_indices = np.where(
        (avalanche_durations >= min_durations_bound) &
        (avalanche_durations <= max_durations_bound)
        )[0]
    durations_power_laws = _fit_power_law(
        data  = avalanche_durations[candidate_durations_indices],
        limit = min_durations_bound
        )
    alpha_exponent_durations = durations_power_laws["power_law_exponent"]
    ks_min_bound_durations   = durations_power_laws["min_bound"]

    # Scaling relation
    scaling_relation_exponent_predicted    = float("nan")
    scaling_relation_exponent_fitted       = float("nan")
    deviation_from_criticality_coefficient = float("nan")
    avalanche_shape_collapse_error         = float("nan")
    time_values                            = np.array([])
    scaling_relation_fit_params            = np.array([])
    if isfinite(tau_exponent_spike_counts) & isfinite(alpha_exponent_durations):
        time_values = np.arange(1, np.max(avalanche_durations) + 1)
        valid_times = (time_values > min_durations_bound) & (time_values < min_durations_bound + 60)
        time_values = time_values[valid_times]
        average_spike_counts = np.array([
            np.mean(avalanche_spike_counts[avalanche_durations == t])
            for t in time_values
            ])
        valid_average_spike_counts = ~np.isnan(average_spike_counts)
        time_values                = time_values[valid_average_spike_counts]
        average_spike_counts       = average_spike_counts[valid_average_spike_counts]

        if len(time_values) >= 2 and np.all(average_spike_counts > 0):
            scaling_relation_fit_params            = np.polyfit(np.log(time_values), np.log(average_spike_counts), 1)
            scaling_relation_exponent_fitted       = float(scaling_relation_fit_params[0])
            scaling_relation_exponent_predicted    = \
                (alpha_exponent_durations - 1) / (tau_exponent_spike_counts - 1)
            deviation_from_criticality_coefficient = \
                float(abs(scaling_relation_exponent_predicted - scaling_relation_exponent_fitted))
            avalanche_shape_collapse_error = abs(beta_exponent - scaling_relation_exponent_fitted)

    #
    # Branching Ratio
    #
    # activities aka. channels_with_spike_per_bin
    #

    channels_with_spikes_per_bin = np.count_nonzero(spike_count_array.todense(), axis=0) # (bin_count,)
    branching_ratio              = float("nan")
    time_lags_k                  = np.array([])
    time_lags_slopes             = np.array([])
    time_lags_fit_parameters     = np.array([])
    try:
        branching_ratio_result = _estimate_branching_ratio(
            activities_over_time = [channels_with_spikes_per_bin],
            max_lags_k           = max_lags_branching_ratio
            )
        branching_ratio          = branching_ratio_result["branching_ratio"]
        time_lags_k              = branching_ratio_result["time_lags_k"]
        time_lags_slopes         = branching_ratio_result["slopes"]
        time_lags_fit_parameters = branching_ratio_result["optimal_fit_parameters"]
    except:
        ...

    return AnalysisResultCriticality(
        metadata                               = recording._analysis_cache.metadata,
        bin_size_sec                           = bin_size_sec,
        percentile_threshold                   = percentile_threshold,
        activity_threshold                     = activity_threshold,
        duration_thresholds                    = duration_thresholds,
        random_seed                            = random_seed,
        n_bootstraps                           = n_bootstraps,
        avalanche_spike_counts                 = avalanche_spike_counts,
        avalanche_spike_counts_per_bin         = avalanche_spike_counts_per_bin,
        avalanche_durations                    = avalanche_durations,
        avalanche_shape_collapse_error         = avalanche_shape_collapse_error,
        unique_durations_within_threshold      = unique_durations_within_threshold,
        inter_avalanche_durations              = inter_avalanche_durations,
        mean_spike_counts_per_bin_by_duration  = mean_spike_counts_per_bin_by_duration,
        beta_exponent                          = beta_exponent,
        beta_exponent_std                      = beta_exponent_std,
        beta_range                             = beta_range,
        beta_candidates_over_range             = beta_candidates_over_range,
        tau_exponent_spike_counts              = tau_exponent_spike_counts,
        alpha_exponent_durations               = alpha_exponent_durations,
        ks_min_bound_spike_counts              = ks_min_bound_spike_counts,
        ks_min_bound_durations                 = ks_min_bound_durations,
        ks_statistic_spike_counts              = ks_stat_spike_counts,
        ks_statistic_duration                  = ks_stat_duration,
        exclusion_bounds_spike_counts          = (min_spike_count_bound, max_spike_count_bound),
        exclusion_bounds_durations             = (min_durations_bound,   max_durations_bound),
        scaling_relation_exponent_predicted    = scaling_relation_exponent_predicted,
        scaling_relation_exponent_fitted       = scaling_relation_exponent_fitted,
        scaling_relation_time_values           = time_values,
        scaling_relation_fitted_params         = scaling_relation_fit_params,
        deviation_from_criticality_coefficient = deviation_from_criticality_coefficient,
        branching_ratio                        = branching_ratio,
        time_lags_k                            = time_lags_k,
        time_lags_slopes                       = time_lags_slopes,
        time_lags_fit_parameters               = time_lags_fit_parameters
        )

class _ResultBetaExponent(TypedDict):
    """ Results for calculating the beta exponent in avalanche-shape collapse. """

    beta_candidate: float
    """ Candidate value for the beta exponent. """

    beta_candidates_over_range: Array1DFloat
    """ Beta values calculated over the required range. """

def _calc_beta_exponent(
    profiles:   list[Array1DInt],
    durations:  Array1DInt,
    beta_range: Array1DFloat
    ) -> _ResultBetaExponent:
    """
    Helper function to find the beta (scaling) exponent in avalanche-shape collapse.

    Args:
        profiles:   Mean spike counts per time bin sorted by duration.
        durations:  Durations (in frame bins) corresponding to profiles.
        beta_range: Range of beta values to consider.

    Returns:
        _ResultBetaExponent
    """
    variances = []
    for profile, duration in zip(profiles, durations):
        t = (np.arange(1, duration + 1)) / duration
        center_of_mass = np.sum(t * profile)
        variance       = np.sum(((t - center_of_mass)**2) * profile)
        variances.append(variance)
    variances = np.array(variances)

    log_durations              = np.log(durations)
    log_variances              = np.log(variances)
    min_beta_error             = np.inf
    beta_candidate             = 0
    beta_candidates_over_range = []
    for beta in beta_range:
        y          = log_variances + 2 * beta * log_durations
        beta_std   = np.std(y)
        beta_candidates_over_range.append(beta_std)
        if beta_std < min_beta_error:
            min_beta_error = beta_std
            beta_candidate = float(beta)

    return _ResultBetaExponent(
        beta_candidate             = beta_candidate,
        beta_candidates_over_range = np.array(beta_candidates_over_range)
        )

class _ResultPowerLaw(TypedDict):
    """ Result for fitting power-law distribution. """

    power_law_exponent: float
    """ Power-law exponent from data fitting. """

    min_bound: int
    """ Minimum value (x0) that minimizes the KS statistic. """

    ks_statistic: float
    """ Kolmogorov-Smirnov (KS) statistic for the best fit. """

    log_likelihood: float
    """ Log-likelihood value for the best fit. """

def _fit_power_law(
    data:  Array1DInt | Array1DFloat,
    limit: int
    ) -> _ResultPowerLaw:
    """
    Fits a power-law distribution to the burst data using the Kolmogorov-Smirnov (KS) test
    and maximum likelihood estimation (MLE).

    This function iterates over possible minimum values (`x0`) for the power-law distribution,
    computes the KS statistic for each fit, and selects the `x0` that minimizes the KS statistic.

    Steps:
    1. Iterate over a range of potential `x0` values.
    2. For each `x0`, fit the power-law exponent (`p_exponent`) using maximum likelihood estimation (MLE).
    3. Compute the log-likelihood and KS statistic for the fit.
    4. Return the best-fitting parameters that minimize the KS statistic.

    Args:
        data:  Array containing burst sizes or durations to fit the power-law distribution.
        limit: Upper limit for the range of potential minimum values (x0) for the power-law fitting.

    Returns:
        _ResultPowerLaw
    """
    ks_stats        = []  # KS statistics for each x0
    p_exponents     = []  # p_exponent values for each x0
    log_likelihoods = []  # log-likelihood values for each x0

    # Loop over potential minimum values (x0) for the power-law fit
    for x0 in np.arange(1, limit + 1):

        # Filter data based on current x0
        filtered_data = data[data >= x0]
        xmax          = np.max(data)
        indices       = np.where((x0 <= filtered_data) & (filtered_data <= xmax))[0]
        n             = np.size(indices)
        unique_values = np.unique(filtered_data[indices])

        # Define the log-likelihood function for power-law fitting
        log_likelihood_func = lambda p_exponent: (
            p_exponent * np.sum(np.log(filtered_data[indices]))
            - n * np.log(1 / np.sum(np.power(unique_values, -p_exponent)))
            )

        # Optimize the log-likelihood to find the best p_exponent
        p_exponent_estimate  = scipy.optimize.fmin(func=log_likelihood_func, x0=2.3, disp=False)  # Start search from 2.3, returns array shape (1,)
        log_likelihood_value = -log_likelihood_func(p_exponent_estimate)                          # Value of the log-likelihood at the minimum
        log_likelihoods.append(log_likelihood_value)

        # Compute the cumulative distribution function (CDF) of the burst data
        data_cdf        = np.cumsum(np.histogram(filtered_data, bins=np.arange(x0, xmax + 2))[0] / n)
        scaling_factor  = 1 / np.sum(np.power(unique_values, -p_exponent_estimate))
        theoretical_cdf = np.cumsum(scaling_factor * np.power(np.arange(x0, xmax + 1), -p_exponent_estimate))

        # Compute the Kolmogorov-Smirnov statistic for the fit
        ks_stat_current = np.max(np.abs(data_cdf - theoretical_cdf))
        ks_stats.append(ks_stat_current)

        # Store the p_exponent value
        p_exponents.append(float(p_exponent_estimate[0]))

    # Find the x0 that minimizes the KS statistic
    best_x0_indices = np.where(ks_stats == np.min(ks_stats))[0]
    best_x0_index   = int(best_x0_indices[0])
    p_exponent      = p_exponents[best_x0_index]
    log_likelihood  = log_likelihoods[best_x0_index]
    ks_stat         = float(np.min(ks_stats))
    xmin            = best_x0_index + 1 # Adjust index to account for range starting from 1

    return _ResultPowerLaw(
        power_law_exponent = p_exponent,
        min_bound          = xmin,
        ks_statistic       = ks_stat,
        log_likelihood     = log_likelihood
        )

class _ResultKSExclusionBounds(TypedDict):
    """ Result for exclusion bound analysis using Kolmogorov-Smirnov (KS) test. """

    min_bound: int
    """ Minimum identified boundary. """

    max_bound: float
    """ Maximum identified boundary. """

    power_law_exponent: float
    """ Power-law exponent from data fitting. """

    ks_statistic: float
    """ Final Kolmogorov-Smirnov (KS) statistic. """

def _find_exclusion_bounds(
    data:            Array1DInt | Array1DFloat,
    power_law_limit: int,
    ) -> _ResultKSExclusionBounds:
    """
    Helper function to determine the lower and upper boundaries for avalanche data using the
    Kolmogorov-Smirnov (KS) test to compare with a power-law distribution.

    This function iteratively calculates the KS statistic to find the range of avalanche sizes or
    durations (burst_min and burst_max) that best fit a power-law distribution. It adjusts the
    upper boundary of the data and stops when the KS statistic converges or reaches the threshold.

    Args:
        data:            Array containing avalanche spike counts (sizes) or durations.
        power_law_limit: Limit to apply when fitting power-law distribution to the data.

    Returns:
        _ResultKSExclusionBounds
    """
    # Initialize variables for the KS test
    ks_stat                          = 1
    ks_delta                         = 1
    min_bound                        = 1
    max_bound                        = float(np.max(data))
    ks_threshold                     = np.min([1 / np.sqrt(np.size(data[data > min_bound])), 0.1])
    power_law_exponent: float | None = None

    while ks_stat > ks_threshold and ks_delta > 0.0005:
        # Fit power-law distribution to the data
        power_law_result = _fit_power_law(
            data        = data,
            limit = power_law_limit
            )
        power_law_exponent = power_law_result["power_law_exponent"]
        min_bound          = power_law_result["min_bound"]

        # Calculate the cumulative distribution function (CDF) of the data
        filtered_data = data[data >= min_bound]
        filtered_size = np.size(filtered_data)
        empirical_cdf = np.cumsum(np.histogram(filtered_data, bins=np.arange(min_bound, max_bound + 2))[0] / filtered_size)

        # Calculate the perfect power-law CDF for comparison
        unique_bursts   = np.unique(data[(min_bound <= data) & (data <= max_bound)])
        scaling_factor  = 1 / np.sum(np.power(unique_bursts, -power_law_exponent))
        theoretical_cdf = np.cumsum(scaling_factor * np.power(np.arange(min_bound, max_bound + 1), -power_law_exponent))

        # Compute KS statistic and its change
        previous_ks_stat = ks_stat
        ks_stat          = float(np.max(np.abs(empirical_cdf - theoretical_cdf)))
        ks_delta         = np.abs(previous_ks_stat - ks_stat)

        # Update data by lowering the upper boundary
        data = data[data < max_bound]

    assert power_law_exponent is not None, "Insufficient data to determine power-law exponent."
    return _ResultKSExclusionBounds(
        min_bound          = min_bound,
        max_bound          = max_bound,
        power_law_exponent = power_law_exponent,
        ks_statistic       = ks_stat
        )

class _ResultTimeLaggedSlopes(TypedDict):
    """ Results containing linear regression slopes for time-lagged activity data. """
    time_lags_k: Array1DFloat
    """ Time lags used for slop estimation (parameter "k"). """

    slopes: Array1DFloat
    """ Regression slopes for each time lag. """

    intercepts: Array1DFloat
    """ Intercept values for each regression. """

    correlation_coefficients: Array1DFloat
    """ Correlation coefficients (r-values). """

    p_values: Array1DFloat
    """ p-values for testing the null hypothesis that the slope is zero. """

    std_errors: Array1DFloat
    """ Standard errors of the regression slopes. """

    data_length: int
    """ Length of the time series data used for regression. """

    mean_activity: float
    """ Mean activity value across all time series. """

    # TODO: x_points, y_points -- for scatterplot

def _compute_time_lagged_slopes(
    activities_over_time: list[Array1DInt],
    max_lag_k:            int,
    min_points_per_lag:   int = 10
    ) -> _ResultTimeLaggedSlopes:
    """
    Compute linear regression slopes for time-lagged activity data, where there is sufficient
    data points as defined by min_points_per_lag.

    Args:
        activities_over_time: List of 1D array of activity counts over time, each with shape (time_bins,).
        max_lags_k:           Maximum number of time lags ("k") to consider for slope estimation.
        min_points_per_lag:   Minimum number of activity data points required for each time lag (defaults to 10).

    Returns:
        _ResultTimeLaggedSlopes
    """
    time_lags_k              = np.arange(1, max_lag_k)  # Time lag values
    valid_time_lags_k        = []                       # Only store lags that have sufficient data
    slopes                   = []                       # Regression slopes
    intercepts               = []                       # Regression intercepts
    correlation_coefficients = []                       # Correlation coefficients (r-values)
    p_values                 = []                       # p-values for significance of slope
    std_errors               = []                       # Standard errors of slopes
    data_length              = 0
    mean_activity            = float("nan")

    # Loop through each lag value
    for i, lag in enumerate(time_lags_k):
        # Preallocate arrays for x and y values
        total_points = sum(len(activity) - lag for activity in activities_over_time)

        # Check if total points are too few to perform regression
        if total_points < min_points_per_lag:
            raise ValueError(f"lag {lag} results in less than {min_points_per_lag} data points")

        valid_time_lags_k.append(lag)  # Keep track of valid lags

        x_values = np.empty(total_points)
        y_values = np.empty(total_points)

        start_idx = 0

        # Populate x and y arrays with time-lagged activity pairs
        for activity in activities_over_time:
            num_points = len(activity) - lag
            x_values[start_idx:start_idx + num_points] = activity[:-lag]
            y_values[start_idx:start_idx + num_points] = activity[lag:]
            start_idx += num_points

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_values, y_values)
        slopes.append(slope)
        intercepts.append(intercept)
        correlation_coefficients.append(r_value)
        p_values.append(p_value)
        std_errors.append(std_err)

        # Store data length and mean activity for first lag
        if i == 0:
            data_length   = len(x_values) + 1
            mean_activity = float(x_values.mean())

    return _ResultTimeLaggedSlopes(
        time_lags_k              = np.array(valid_time_lags_k),
        slopes                   = np.array(slopes),
        intercepts               = np.array(intercepts),
        correlation_coefficients = np.array(correlation_coefficients),
        p_values                 = np.array(p_values),
        std_errors               = np.array(std_errors),
        data_length              = data_length,
        mean_activity            = mean_activity
        )

class _ResultBranchingRatio(_ResultTimeLaggedSlopes):
    """ Result containing estimated branching ratios. """
    branching_ratio: float
    """ Estimated branching ratio. """

    optimal_fit_parameters: Array1DFloat
    """ Optimal parameters of the exponential fit function. """

    # TODO: xs, xy -- scatterplot x and y values


_exponential_function = lambda x, a, b: np.abs(a) * np.abs(b) ** x
""" a * b^x """

def _estimate_branching_ratio(
    activities_over_time: list[Array1DInt],
    max_lags_k:           int = 40,
    min_points_per_lag:   int = 10
    ) -> _ResultBranchingRatio:
    """
    Estimates the branching ratio and related metrics from time series data using MR estimation.

    Args:
        activities_per_channel: List of 1D array of activity counts over time, each with shape (time_bins,).
        max_lags_k:             Maximum number of time lags to consider for slop estimation (defaults to 40).
        min_points_per_lag:     Minimum number of activity data points required for each time lag (defaults to 10).

    Returns:
        _ResultBranchingRatio
    """
    # Calculate time lagged slopes
    slope_results = _compute_time_lagged_slopes(
        activities_over_time = activities_over_time,
        max_lag_k            = max_lags_k,
        min_points_per_lag   = min_points_per_lag
        )

    time_lags  = slope_results["time_lags_k"]
    slopes     = slope_results["slopes"]
    std_errors = slope_results["std_errors"]

    # Fit exponential model
    fit_function  = _exponential_function
    initial_guess = [slopes[0], 1.0]                            # Initial guess for the fit parameters
    optimal_params, covariance = scipy.optimize.curve_fit(
        fit_function,
        time_lags,
        slopes,
        p0     = initial_guess,
        maxfev = 50_000,
        sigma  = std_errors * np.linspace(1, 10, len(std_errors)),
        )

    return _ResultBranchingRatio(
        branching_ratio        = float(optimal_params[1]),
        optimal_fit_parameters = optimal_params,
        **slope_results
        )