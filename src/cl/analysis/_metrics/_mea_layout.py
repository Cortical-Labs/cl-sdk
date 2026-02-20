import numpy as np

from ...util import RecordingView

_COMMON_MEA_LAYOUT: list[list[int]] = \
    [
        [ 0,  8, 16, 24, 32, 40, 48, 56],
        [ 1,  9, 17, 25, 33, 41, 49, 57],
        [ 2, 10, 18, 26, 34, 42, 50, 58],
        [ 3, 11, 19, 27, 35, 43, 51, 59],
        [ 4, 12, 20, 28, 36, 44, 52, 60],
        [ 5, 13, 21, 29, 37, 45, 53, 61],
        [ 6, 14, 22, 30, 38, 46, 54, 62],
        [ 7, 15, 23, 31, 39, 47, 55, 63]
    ]
""" Describes channel indices spatially in the common MEA layout. """

_COMMON_GROUND_CHANNELS: list[int] = [0, 7, 56, 63]
""" Ground channels in the common MEA layout. """

_COMMON_REFERENCE_CHANNELS: list[int] = [4]
""" Reference channels in the common MEA layout. """

def _valid_common_layout(recording: RecordingView) -> bool:
    """ Performs a check that indicates whether a recording conforms to the common MEA layout expected for analysis. """
    # Maximum data limits on ground channels
    duration_frames           = recording.attributes["duration_frames"]
    max_ground_nonzero_frames = 0.1 * duration_frames # 10%
    max_ground_spikes         = 0

    # Do a check for mostly zero samples and/or spikes in ground channels,
    # noting that samples and spikes may be optionally disabled in a recording
    if recording.samples is not None:
        valid_samples = \
            np.count_nonzero(recording.samples[:, _COMMON_GROUND_CHANNELS]) <= max_ground_nonzero_frames
        if valid_samples:
            return True
    if recording.spikes is not None:
        channel_query = " | ".join([f"(channel=={ch})" for ch in _COMMON_GROUND_CHANNELS])
        valid_spikes  = \
            len(recording.spikes.get_where_list(channel_query)) <= max_ground_spikes
        if valid_spikes:
            return True
    return False