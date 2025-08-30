import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
from numpy import ndarray

from cl import Spike

def uV_formatter(value, pos) -> str:
    """ A function to be passed to matplotlib.ticker.FuncFormatter to format value as microvolts. """
    return f"{value} µV"

def plot_spike(spike: tuple[int, int, ndarray] | Spike):
    """ Creates a plot of a single spike.

    Args:
        spike: Either a Spike object or a tuple of (timestamp, channel, samples).
    """
    if isinstance(spike, np.void) or isinstance(spike, tuple):
        timestamp, channel, samples = spike
    else:
        timestamp = spike.timestamp
        channel   = spike.channel
        samples   = spike.samples

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axvline(x=25, color="gray", linestyle=":", linewidth=1)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(uV_formatter))
    ax.set_xlabel(f"Spike on Channel {channel} at Timestamp {timestamp}")
    ax.plot(samples)
    plt.show()