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

Spikes and samples are simulated by replaying recordings as set by the `CL_SDK_REPLAY_PATH` environment variable in the `.env` file. If this is omitted, a temporary recording with randomly generated samples and spikes will be used that is based on a Poisson distribution and the following optional environment variables:
- `CL_SDK_SAMPLE_MEAN`: Mean samples value (default 170). This value will be in microvolts when multiplied by the constant "uV_per_sample_unit" in the recording attributes;
- `CL_SDK_SPIKE_PERCENTILE`: Percentile threshold for sample values, above which will correspond to a spike (default 99.995);
- `CL_SDK_DURATION_SEC`: Duration of the temporary recording (default 60); and
- `CL_SDK_RANDOM_SEED`: Random seed (defaults to Unix time).

The starting position of the replay recording will be randomised every time `cl.open()` is called. This can be overriden by setting `CL_SDK_REPLAY_START_OFFSET`, where a value of `0` indicates the first frame of the recording.

### Speed of simulation

The simulator can operate in two timing modes:
- Based on wall clock time (default), or
- Accelerated time.

Accelerated time mode can be enabled by setting `CL_SDK_ACCELERATED_TIME=1` environment variable in the `.env` file. When enabled, passage of time will be decouple from the system wall clock time, enabling accelerated testing of applications.

### WebSocket server

An included webSocket server can be used to stream simulated data. By default, the server is disabled. It can be enabled by setting `CL_SDK_WEBSOCKET=1` environment variable in the `.env` file. The port used by the server can be set using the `CL_SDK_WEBSOCKET_PORT` environment variable (default 1025). The server will be hosted on `localhost` by default, but this can be changed by setting the `CL_SDK_WEBSOCKET_HOST` environment variable.