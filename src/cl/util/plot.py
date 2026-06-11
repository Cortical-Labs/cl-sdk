import numpy as np
from numpy import ndarray

from .. import Spike
from .util import deprecated

def uV_formatter(value, _) -> str:
    """ A function to be passed to matplotlib.ticker.FuncFormatter to format value as microvolts. """
    return f"{value} µV"

@deprecated("RecordingView.plot_spike()")
def plot_spike(spike: tuple[int, int, ndarray] | Spike):
    """ Creates a plot of a single spike.

    Args:
        spike: Either a Spike object or a tuple of (timestamp, channel, samples).
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if isinstance(spike, np.void) or isinstance(spike, tuple):
        timestamp, channel, samples = spike
    else:
        timestamp = spike.timestamp
        channel   = spike.channel
        samples   = spike.samples

    _, ax = plt.subplots(figsize=(6, 2))
    ax.axvline(x=25, color="gray", linestyle=":", linewidth=1)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(uV_formatter))
    ax.set_xlabel(f"Spike on Channel {channel} at Timestamp {timestamp}")
    ax.plot(samples)
    plt.show()
