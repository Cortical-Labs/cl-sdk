# Cortical Labs Mock API

*** EARLY ACCESS RELEASE ***

This package provides a mock cl API to assist with local development of applications that can run on a Cortical Labs CL1 system.

The following key features are implemented:

- Simulation of spikes and waveforms
- Stimulation (`neurons.stim(...)`, `neurons.create_stim_plan()`)
- Time based code execution (`neurons.loop(...)`)
- Recording (`neurons.record(...)`)

Planned for future versions:

- Real-time visualisation of live data

## Installation

Use of a venv is recommended:
```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip3 install cl-sdk
```

## Cortical Labs Developer Guide

The mock API is capable of running most of the Jupyter notebooks in our developer guide. Install cl-sdk as above, then:

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

## Needs Fixing

### Loop timing

Currently, loop ticks will read the requisite number of frames. When using wall clock mode, this will block for the actual duration of the tick, which may cause a jitter error to be thrown as it leaves very little margin for executing the user's code for each tick.

## User Options

Several user options can be set by defining environment variables in a `.env` file of your project directory.

### Simulation from a recording

The Mock API simulates spikes and samples by replaying recordings as set by the `CL_MOCK_REPLAY_PATH` environment variable in the `.env` file. If this is omitted, a temporary recording with randomly generated samples and spikes will be used that is based on a Poisson distribution and the following optional environment variables:
- `CL_MOCK_SAMPLE_MEAN`: Mean samples value (default 170). This value will be in microvolts when multiplied by the constant "uV_per_sample_unit" in the recording attributes;
- `CL_MOCK_SPIKE_PERCENTILE`: Percentile threshold for sample values, above which will correspond to a spike (default 99.995);
- `CL_MOCK_DURATION_SEC`: Duration of the temporary recording (default 60); and
- `CL_MOCK_RANDOM_SEED`: Random seed (defaults to Unix time).

### Speed of simulation

The Mock API can operate in two timing modes:
- Based on wall clock time (default), or
- Accelerated time.

Accelerated time mode can be enabled by setting `CL_MOCK_ACCELERATED_TIME=1` environment variable in the `.env` file. When enabled, passage of time will be decouple from the system wall clock time, enabling accelerated testing of applications.
