# CL SDK

This package provides an implementation of the CL API to assist with local development of applications that can run on a Cortical Labs CL1 system.

Please refer to [docs.corticallabs.com](https://docs.corticallabs.com) for the latest documentation.

## Prerequisites

This SDK requires Python 3.12 or later.

## Installation

Use of a venv is recommended:
```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip3 install cl-sdk
```

### On a CL1 System

A custom version of this SDK is included on CL1 systems, that interfaces with the hardware. For applications, if `cl-sdk` is listed as a dependency in the `requirements.txt` file, it will automatically consolidate with the system, regardless of the version of `cl-sdk` specified (although versions `<0.30.0` are not supported). If using a custom virtual environment on the CL1 without system site-packages, `cl-sdk` can be installed, but after this, the `cl-api` package is required to be installed from the local file included on the CL1 at `/opt/cl-api-whl/cl_api-*.whl` (where `*` corresponds to the version number). This will allow the application to properly interface with the CL1 hardware.

## Cortical Labs Developer Guide

This SDK is capable of running most of the Jupyter notebooks in our developer guide. Install cl-sdk as above, then:

```bash
$ git clone https://github.com/Cortical-Labs/cl-api-doc.git
```

From here you can open and run the `*.ipynb` notebooks directly in Visual Studio Code, or by installing and running Jupyter Lab:

```bash
$ pip3 install jupyterlab
$ jupyter lab cl-api-doc
```

### Development

For working on the simulator itself:

```bash
$ pip3 install -e .
```

### Running Tests

```bash
$ pip3 install -e '.[test]'
$ pytest
```

### Building Documentation

```bash
$ pip3 install -e '.[test]'
$ python3 -m docs.make
```

Serve the built docs to view in a browser:

```bash
$ python3 -m http.server -d docs/html
```

## User Options

Several user options can be set by defining environment variables in a `.env` file of your project directory.

### Simulation from a recording

Spikes and samples are simulated by replaying recordings as set by the `CL_SDK_REPLAY_PATH` environment variable in the `.env` file. If this is omitted, randomly generated samples and spikes are produced on the fly that are based on a Poisson distribution and the following optional environment variables:
- `CL_SDK_SAMPLE_MEAN`: Mean samples value (default 170). This value will be in microvolts when multiplied by the constant "uV_per_sample_unit" in the recording attributes;
- `CL_SDK_SPIKE_PERCENTILE`: Percentile threshold for sample values, above which will correspond to a spike (default 99.995);
- `CL_SDK_RANDOM_SEED`: Random seed (default 42).
- `CL_SDK_SPIKE_VISIBILITY`: Whether to amplify the sample value of detected spikes to increase their visibility in the sample data (default 0, i.e., no amplification).

The starting position of the replay recording will be randomised every time `cl.open()` is called. This can be overriden by setting `CL_SDK_REPLAY_START_OFFSET`, where a value of `0` indicates the first frame of the recording.

The randomly generated data is computed deterministically from the parameters above, so no recording file is written and the same data is reproduced on every run with the same parameters.

### Custom simulator data source

The simulator can read from a custom data source instead of a recording. Define an importable `SimulatorDataSource` factory and set it before `cl.open()`:

```python
import cl

cl.sim.set_simulator_data_source(
    "my_project.sources:create_source",
    config={"seed": 1},
)

with cl.open() as neurons:
    frames = neurons.read(250)
```

The factory is imported by the data producer subprocess, so it must be available as a module path, not only as an object in the current Python process. The source should return `DataSourceBatch` objects from `read(from_timestamp, frame_count)`. If `frames` is omitted, the simulator writes zero-valued frames while still publishing any returned `DataSourceSpike` values. Override `on_stim(stim)` or `on_stims(stims)` to receive committed stimulation events, including channel, timestamp, and `StimDesign` phase data. Pass static `metadata=` when registering to avoid constructing the source in the parent process.

For push-style live simulators, subclass `LiveSimulatorDataSource` and implement `start(sink)`. The source can run its own thread/process and call `sink.emit_batch(...)`; the SDK adapter buffers that live stream and serves sequential producer reads. Live sources default to realtime-only (`supports_accelerated=False`).

Environment variable equivalent:

```bash
CL_SDK_DATA_SOURCE=my_project.sources:create_source
CL_SDK_DATA_SOURCE_CONFIG='{"seed": 1}'
CL_SDK_DATA_SOURCE_METADATA='{"channel_count":64,"frames_per_second":25000}'
```

### Speed of simulation

The simulator can operate in two timing modes:
- Based on wall clock time (default), or
- Accelerated time.

Accelerated time mode can be enabled by setting `CL_SDK_ACCELERATED_TIME=1` environment variable in the `.env` file. When enabled, passage of time will be decouple from the system wall clock time, enabling accelerated testing of applications.

### Visualisation service

An included WebSocket server can be used to stream simulated data. By default, the server is enabled. It can be disabled by setting `CL_SDK_VISUALISATION=0` environment variable in the `.env` file. The server will automatically find an available port starting from 1025. The server will be hosted on `localhost` by default, but this can be changed by setting the `CL_SDK_VISUALISATION_HOST` environment variable. The WebSocket server is not compatible with accelerated time mode, and will be automatically disabled if accelerated time mode is enabled.
