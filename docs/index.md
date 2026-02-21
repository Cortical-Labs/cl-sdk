# Introduction

The Cortical Labs API (CL API) is a Python library that allows interaction with complex Biological Neural Networks (BNNs) via a customised hardware platform called the [CL1](https://corticallabs.com).

CL API provides transparent control for all key tasks performed with the CL1 including: recording, stimulation and real-time closed-loop algorithms with micro-second latency. Read more about the technical specifications of the CL API and the contract-based design philosophy in our [whitepaper](https://doi.org/10.48550/arXiv.2602.11632).

## Installation

The CL API comes pre-installed with every CL1 device.

Those without access to a CL1 can experiment with the CL API via the [CL SDK Simulator](#cl-sdk-simulator).

The CL API and Simulator are intentionally designed to be drop-in replacements for one another. Any code developed against the Simulator can be executed on a physical CL1 system with minimal or no modification.

To use the **Simulator** on your local device, simply install the `pip` package:

```shell
pip install cl-sdk
```

## Quick start

For an interactive walk-through of the CL API, check out the [Jupyter API develop guide](https://github.com/Cortical-Labs/cl-api-doc/tree/main), which can be run both on the CL1 or on your local device with the Simulator.

To begin, simply import the `cl` module and open a `Neuron` connection as follows. Note that `cl.open()` is the preferred way to interface with the CL1 and the `cl.Neurons` object should *not* be used in isolation.

```python
import cl

with cl.open() as neurons:
  # Your code here
  ...
```

# Core API Examples

## Recording

Users can make recordings via the `Neurons.record()` interface, which gives a simple way to record spikes, stimulation events, custom [data streams](#data-streams) produced by applications, and a complete record of raw electrode samples in **HDF5** format. The application simply requests for a recording to start, and by default all spikes, stims and frames of samples will be added to the recording file until it is stopped.

```python
import cl
import time

with cl.open() as neurons:
    recording = neurons.record()

    # Recording has started, place your application code here
    time.sleep(3)

    recording.stop()
```

Once a recording is made, a HDF5 file will be generated in the `/data/recordings/` directory of the CL1 system. In the Simulator, the HDF5 will be saved to the current working directory (i.e. `./`). You can also pass in `file_location` as an argument to change the directory.

## Detecting and Reacting to Spikes

The `Neurons.loop()` method provides a low latency time-driven way to interact with BNNs, up to maximum iteration frequency of 25 kHz.

Within each loop iteration, the system exposes all spikes detected since the previous iteration via the `tick.analysis.spikes` list, enabling closed-loop algorithms to react to neural activity at timescales comparable to the underlying sampling and stimulation hardware, without requiring users to manage low-level buffering, polling, or synchronisation.

```python
import cl

with cl.open() as neurons:
    # Loop 1000 times per second for 5 seconds
    for tick in neurons.loop(ticks_per_second=1000, stop_after_seconds=5):
        # Loop through each detected spike object
        for spike in tick.analysis.spikes:
            # Print out the spike object
            print(spike
```

Expected output:

```text
Spike(timestamp=2707168587, channel=52)
Spike(timestamp=2707168645, channel=58)
Spike(timestamp=2707168737, channel=32)
Spike(timestamp=2707168855, channel=45)
Spike(timestamp=2707168987, channel=43)
...
Spike(timestamp=2707291908, channel=58)
Spike(timestamp=2707292040, channel=21)
Spike(timestamp=2707292056, channel=60)
Spike(timestamp=2707292259, channel=52)
Spike(timestamp=2707292898, channel=62)
```

A `Spike` object is created for each spike detected by the system, and these are placed in a list at `LoopTick.analysis`.`spikes`. Spike objects expose the following properties:

| Property    | Data                                                                 |
|:----------- |:-------------------------------------------------------------------- |
| `channel`   | Which channel the spike was detected on                              |
| `timestamp` | Timestamp of the sample that triggered the detection of the spike    |
| `samples`   | NumPy array of 75 floating point µV sample values around `timestamp` |

Note that `samples` provides 1 ms (25 samples) before, and 2 ms (50 samples) from the moment of detection, with the values mean-centered at the time of detection. As it is common for a loop body to process a detected spike within 2 ms of it occurring, the samples data is loaded only on request. Be aware that accessing `Spike.samples` before 2 ms has passed will block until the requested data is available, i.e., your code could wait for up to 2 ms

The `Neurons.loop()` also exposes `tick.analysis.stims`, which contains a list of `Stim` objects
that serve as a record of stimulation that began during the previous tick. The following 25 kHz loop
stimulates in response to each detected spike, then collates and later prints detected spikes and stims:

```python
import cl
with cl.open() as neurons:
    # Loop at 25 kHz for 25000 ticks, respond to spikes
    # with a stim on the same channel, and collect all
    # detected spikes and stims.
    spikes = []
    stims  = []
    stim_design = cl.StimDesign(160, -1, 160, 1)
    for tick in neurons.loop(ticks_per_second=25000, stop_after_ticks=25000):
        # Respond to each detected spike with a stim on the same channel.
        for spike in tick.analysis.spikes:
            neurons.stim(spike.channel, stim_design)
            # Collect the spikes and stims from the previous tick
            spikes.extend(tick.analysis.spikes)
            stims.extend(tick.analysis.stims)

for spike_or_stim in sorted(spikes + stims, key=lambda x: x.timestamp):
    print(spike_or_stim)
```

Expected output:

```text
Spike(timestamp=1195060728, channel=58)
Stim(timestamp=1195060733, channel=58)
Spike(timestamp=1195061224, channel=5)
Stim(timestamp=1195061229, channel=5)
Spike(timestamp=1195061710, channel=54)
Stim(timestamp=1195061715, channel=54)
...
Spike(timestamp=1195084744, channel=42)
Stim(timestamp=1195084749, channel=42)
Spike(timestamp=1195085149, channel=54)
Stim(timestamp=1195085154, channel=54)
Spike(timestamp=1195085546, channel=36)
Stim(timestamp=1195085550, channel=36)
```

## Jitter Detection and Recovery

For closed-loop neurocomputing experiments, it is essential that analysis and response logic execute within predictable and well-defined time windows. When using `Neurons.loop()`, executing code within the loop body that exceeds the available time budget will raise a `TimeoutError`. This behaviour enforces the temporal guarantees of the API by preventing silent accumulation of jitter.

```python
import cl

from time import time_ns

with cl.open() as neurons:
    for tick in neurons.loop(ticks_per_second=25000, stop_after_ticks=25000):
        # Consume approximately 35 μs, exceeding the 25 kHz per-iteration budget
        wait_until_ns = time_ns() + 35_000
        while time_ns() < wait_until_ns:
            pass

print('Done')
```

Expected output:

```text
---------------------------------------------------------------------------
TimeoutError                              Traceback (most recent call last)
Cell In[64], line 6
      3 from time import time_ns
      5 with cl.open() as neurons:
----> 6     for tick in neurons.loop(ticks_per_second=25000, stop_after_ticks=25000):
      7         # Consume approximately 35 µs, exceeding the 25 kHz per-iteration budget
      8         wait_until_ns = time_ns() + 35_000
      9         while time_ns() < wait_until_ns:

TimeoutError: Loop fell behind by 1 frame (40 µs) when entering the 8th
iteration. Jitter tolerance is currently set to 0 frames. Ideally - optimise
the worst-case performance of your loop body. You may also adjust the jitter
tolerance via jitter_tolerance_frames=1, or ignore jitter entirely via
ignore_jitter=True.
```

In the [Simulator](#simulator), a `TimeoutError` due to jitter will not be raised. Instead, users will see the following message:

```text
Warning: Jitter detection is currently not supported in cl-sdk. This may lead to unexpected
loop timing behaviour if your loop body takes a long time to execute.
```

In some closed-loop experiments, it may be desirable to temporarily relax enforcement of per-iteration execution deadlines. The `Loop.recover_from_jitter()` method allows a user to opt-in to tolerate a single instance of lapse in loop body execution without triggering a `TimeoutError`. See the method definition for usage examples.


## Data Streams

Data streams allow client applications to publish named streams of arbitrary structured data which are added to recordings and are available for live visualisation. A data stream can be created using `Neurons.create_data_stream()`. Attributes can also be defined to further describe the data stream.

```python
import cl
import numpy
with cl.open() as neurons:
    # Create a named data stream - by default, it will be added to any active or future recordings.
    data_stream = \
        neurons.create_data_stream(
            name       = 'example_data_stream',
            attributes = { 'score': 0, 'another_attribute': [0, 1, 2, 3] }
            )
    # Start a recording
    recording = neurons.record(stop_after_seconds=1)
```

Each entry in a single data stream is is required to have a unique, always ascending timestamp. The entry data itself can be a Python type such as an `dict` , `tuple` , `list` , `int` , `float` , or a NumPy `NDArray`.

Data stream entries are useful for storing changes to data over time. For example you might store the *(x, y)* position of a ball as it moves within a Pong simulation.

After a data stream has been created, data of the appropriate type can be appended as follows:

```python
import cl
import numpy
with cl.open() as neurons:
    # Create a named data stream - by default, it will be added to any active or future recordings.
    data_stream = \
        neurons.create_data_stream(
            name       = 'example_data_stream',
            attributes = { 'score': 0, 'another_attribute': [0, 1, 2, 3] }
            )
    # Start a recording
    recording = neurons.record(stop_after_seconds=1)
    timestamp = neurons.timestamp() # Get current timestamp

    # Add some data stream entries with unique, ascending timestamps:
    data_stream.append(timestamp + 0, { 'arbitrary': 'data' })
    data_stream.append(timestamp + 1, ['of', 'arbitrary', 'size'])
    data_stream.append(timestamp + 2, 'and type.')
    data_stream.append(
        timestamp + 3,
        numpy.array([2**64 - 1, 2**64 - 2, 2**64 - 3], dtype=numpy.uint64)
        )

    # Update a single attribute
    data_stream.set_attribute('score', 1)

    # Update multiple attributes at once
    data_stream.update_attributes({ 'score': 2, 'new_attribute': 9.9 })

    recording.wait_until_stopped() # Wait until 1 second which was requested by the neurons.record
```

## Loading Recordings

An HDF5 file can be loaded and inspected with `RecordingView` as follows:

```python
from cl import RecordingView

# Load a recording file from CL1 using RecordingView
# Files are timestampted in a `YYYY_MM_DD_HH_MM_SS` format
recording = RecordingView("/data/recordings/2025_10_29_11_53_42_my_application.h5")

# Access core datasets
print(recording.samples)        # Raw voltage samples array with shape (duration_frames × channel_count)
print(recording.spikes)         # Detected spike events with metadata
print(recording.stims)          # Stimulus timestamps and channels
print(recording.attributes)     # Global recording metadata
print(recording.data_streams)   # User defined data streams (e.g., gamestate)

# Example: Count total spikes
num_spikes = len(recording.spikes)
print(f"Detected spikes: {num_spikes}")

# Close file when done
recording.close()
```

## Loading Data Streams

The same can be done for a data stream object inside the HDF5 file. The following example demonstrates how to load and analyze the data stream `'example_data_stream'` saved [previously](#data-streams):

```python
from cl import RecordingView

recording = RecordingView("/data/recordings/2025_10_29_11_53_42_my_application.h5")
stream    = recording.data_streams["example_data_stream"]

# Loop through and extract each timestamp and item:
for timestamp, data in stream.items():
    print('\n', timestamp, data)
```

This would output:

```console
2664 {'arbitrary': 'data'}
2665 ['of', 'arbitrary', 'size']
2666 and type.
2667 [18446744073709551615 18446744073709551614 18446744073709551613]
```

## Stimulating

The CL1 system uses current-based stimulations (stims) that can be easily controlled by the CL API. Users have control over which channels to stim (via `ChannelSet`), parameters that defines the stim pulse (via `StimDesign`) and/or whether to perform a single stim versus a burst of stims.

### ChannelSet

A `ChannelSet` object stores a set of channels for stimulation. Each of the chosen channels will receive the same input. It takes in one or more channels as integers, lists or tuples. For example:

```python
from cl import ChannelSet

# Select channels 8, 9 and 10
ChannelSet(8, 9, 10)

# Pass in a list to select same channels
example_set = [8, 9, 10]
ChannelSet(example_set)
```

### StimDesign

A `StimDesign` object stores the parameters of a mono-, bi-, or tri-phasic stim by specifying 2, 4 or 6 pairs of arguments respectively.

Each pair of arguments is made of Pulse Width duration in microseconds ( μs), and Current in microamperes (μA). Each pair of arguments reflects one segment of the pulse. Duration should be in multiples of 20 μs, as the maximum CL1 stim rate is 50 kHz.

It is recommended to use a negative leading edge for charge balancing purposes. For example:

```python
from cl import StimDesign

# Monophasic stim with current of -1.0 uA, pulse width of 160 us.
StimDesign(160, -1.0)

# Biphasic stim with current of 1.0 uA, pulse width of 160 us and negative leading edge.
StimDesign(160, -1.0, 160, 1.0)

# Triphasic stim with current of 1.0 uA, pulse width of 160 us and negative leading edge.
StimDesign(160, -1.0, 160, 1.0, 160, -1.0)
```

### BurstDesign

A `BurstDesign` object stores the parameters of a stimulation burst, if multiple stimulations with the same parameters are desired in a row.

The number of desired bursts and the frequency of stimulation within the bursts are required. For example:

```python
from cl import BurstDesign

# A burst containing 10 stims operating at 150 Hz:
BurstDesign(10, 150)
```

When these are combined, for example, to deliver a single biphasic stim with current of 2.0 μA, pulse width of 200 μs and negative leading edge on channels 20, 42, 51 and 60 it looks like this:

```python
import cl
from cl import ChannelSet, StimDesign, BurstDesign
with cl.open() as neurons:
    # Deliver a single stim
    channel_set = ChannelSet(20, 42, 51, 60)
    stim_design = StimDesign(200, -2.0, 200, 2.0)
    neurons.stim(channel_set, stim_design)

    # Deliver the same stim as a burst of 20 at 40 Hz
    burst_design = BurstDesign(20, 40)
    neurons.stim(channel_set, stim_design, burst_design)
```

## Stimulation Plans

It is possible to combine different types of `ChannelSet` , `StimDesign` and `BurstDesign` to create a stimulation plan via the CL API function `Neurons.create_stim_plan()`.
Multiple sets of channels, stimulation and bursts can be added to a stim plan. For example:

```python
import cl
from cl import ChannelSet, StimDesign, BurstDesign

# First set of channels and stims
channel_set1 = ChannelSet(20, 42, 51, 60)
stim_design1 = StimDesign(200, -2.0, 200, 2.0)

# Second set of channels and stims
channel_set2 = ChannelSet(2, 6, 12, 18)
stim_design2 = StimDesign(100, -1.5, 100, 1.5)

# Share a common burst design for both stims
burst_design = BurstDesign(20, 40)

with cl.open() as neurons:
    stim_plan = neurons.create_stim_plan() # Create Stim Plan
    stim_plan.stim(channel_set1(...),stim_design1(...),burst_design(...)) # Add first set to plan
    stim_plan.stim(channel_set2(...),stim_design2(...),burst_design(...)) # Add second set to plan
    stim_plan.run()
```

In the above example, since both `ChannelSet` variables have no overlap in their channels, the stimulation will occur simultaneously on all channels designated both in `channel_set1` and in `channel_set2` . Each `StimDesign` will **only** occur for the corresponding `ChannelSet`.

This means that channels 2 , 6, 12 and 18 will be stimulated with biphasic pulses at 1.5 μA with a pulse width of 100 μs, while channels 20, 42, 51 and 60 will be stimulated with biphasic pulses at 2 μA with a pulse width of 200 μs, **at the same time**.

Since only one type of `BurstDesign` was defined, both sets of channels will burst 20 times at 40 Hz.

# CL SDK Simulator

The [CL SDK Simulator](https://github.com/Cortical-Labs/cl-sdk) enables local application development and debugging by maintaining 1:1 interface parity with the on-device API. Data is [simulated](#simulation-from-a-recording) from either randomly generated or replayed from recordings and can run either in wall-clock or [accelerated modes](#speed-of-simulation).

## Simulation from a recording

Spikes and samples is simulated by replaying recordings as set by the `CL_SDK_REPLAY_PATH` environment variable in an `.env` file. If this is omitted, a temporary recording with randomly generated samples and spikes will be used that is based on a Poisson distribution and the following optional environment variables:

- `CL_SDK_SAMPLE_MEAN`: Mean samples value (default 170). This value will be in microvolts when multiplied by the constant "uV_per_sample_unit" in the recording attributes;
- `CL_SDK_SPIKE_PERCENTILE`: Percentile threshold for sample values, above which will correspond to a spike (default 99.995);
- `CL_SDK_DURATION_SEC`: Duration of the temporary recording (default 60); and
- `CL_SDK_RANDOM_SEED`: Random seed (defaults to Unix time).

The starting position of the replay recording will be randomised every time `cl.open()` is called. This can be overriden by setting `CL_SDK_REPLAY_START_OFFSET`, where a value of `0` indicates the first frame of the recording.

## Speed of simulation

The Simulator can operate in two timing modes:

- Based on wall clock time (default), or
- Accelerated time.

Accelerated time mode can be enabled by setting `CL_SDK_ACCELERATED_TIME=1` environment variable in the `.env` file. When enabled, passage of time will be decouple from the system wall clock time, enabling accelerated testing of applications.

## Visualisations

Basic visualisation are supported by the Simulator. To enable a local WebSocket server for visualization and testing, set the environment variable `CL_SDK_WEBSOCKET=1`. The server will start on port 1025 (configurable via `CL_SDK_WEBSOCKET_PORT`).

# Citing CL API

If you find the CL API useful in your research or application and wish to cite it, please use the following BibTex entry:

```bibtex
@software{cl_api_2026,
    author  = {
                David Hogan and Andrew Doherty and Boon Kien Khoo and
                Johnson Zhou and Richard Salib and James Stewart and
                Kiaran Lawson and Alon Loeffler and Brett J. Kagan
              },
    title   = {CL API: Real-Time Closed-Loop Interactions with Biological Neural Networks},
    version = {1.0},
    doi     = {10.48550/arXiv.2602.11632},
    year    = {2026}
}
```

---