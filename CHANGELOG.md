# Changelog

## [1.0.0]

### Added

- Added the `cl.sim` module for pluggable simulator data sources, including custom pull and live data sources, simulator metadata/config hooks, stim callbacks, and `CL_SDK_DATA_SOURCE*` environment configuration.
- Added `cl.is_simulator()` to check whether code is running against the simulator backend.
- Added `analysis=True` support to `Neurons.read()` and `Neurons.read_async()` so reads can return a `DetectionResult` for the requested window.
- Added `start_timestamp` and `stop_timestamp` to `DetectionResult`; the older `timestamp` attribute remains as a deprecated alias for `start_timestamp`.
- Added context manager support and idempotent cleanup to `RecordingView`.
- Added `RecordingView.analysis_timestamp_limit(...)` to constrain analysis methods to a timestamp range.
- Added `BaseApplicationConfig.config_version`, `BaseApplication.migrate_config()`, and migration helpers in `cl.app.util`.
- Added `OutputType.MARKDOWN` for application output summaries.
- Added the `cl.error` module with named exceptions for control, transaction, recording, WebSocket, and unsafe-recording failures.
- Added `CL_SDK_SPIKE_VISIBILITY` to make synthetic spikes more visible in random simulator samples.

### Changed

- Simulator random data is now generated on demand instead of being pre-written to a temporary finite HDF5 file. Unset `CL_SDK_RANDOM_SEED` values now generate and log a seed, while explicit seeds keep random simulator output reproducible.
- Simulator recordings now use independent background recording processes and read cursors, allowing overlapping recordings and concurrent `neurons.read()` calls without consuming or duplicating each other's data.
- Simulator recordings now honour `from_seconds_ago`, `from_frames_ago`, and `from_timestamp` when the simulator buffer has enough history.
- Recording playback now streams samples, spikes, stim events, and data streams through the visualiser, and includes interactive pause, seek, restart, and speed controls.
- Visualisation configuration was renamed from WebSocket terminology to `CL_SDK_VISUALISATION` and `CL_SDK_VISUALISATION_HOST`. The legacy `CL_SDK_WEBSOCKET` and `CL_SDK_WEBSOCKET_HOST` variables are still read for compatibility, with a deprecation warning when the legacy enable flag is used without the new flag.
- The visualisation service is enabled by default, manages its own available port, and disables itself automatically under accelerated simulator time.
- `ChannelSet` now supports empty sets, iterable construction, iteration, non-mutating set operators, and explicit in-place set operators.
- `BurstDesign` now rounds burst intervals to the nearest 20 microsecond frame boundary.
- `StimFrequencyHz` is now limited to a maximum of 200 Hz.
- `StimDesign` and `StimPulseComponentModel` now enforce the 3 nC per-phase/component charge limit using absolute charge, so negative phases are validated consistently.
- Stimulation scheduling for sync, `interrupt_then_stim`, and multi-channel bursts now avoids rewriting already-published bursts and bounds long-running history growth.
- Analysis methods now respect timestamp limits and handle empty or sparse recordings more robustly in criticality and functional connectivity calculations.
- `RecordingView` now rejects unsafe pickled or object-dtype HDF5 contents with `UnsafeOperationError`.
- `cl.app.pack` now follows symlinks with cycle protection.
- General performance and stability improvements.

### Removed

- Removed `CL_SDK_DURATION_SEC`; the default simulator source is no longer a finite pre-generated recording.
- Removed `CL_SDK_WEBSOCKET_PORT`; the visualisation service now manages its own port automatically.
- Removed the previously planned `OutputType.BASE64` and `OutputType.HTML` enum options for application run summaries.
