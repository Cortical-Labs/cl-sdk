"""
## Cortical Labs Recording Playback Tool

This module provides a command-line tool for playing back HDF5 recording files
with visualization support. It streams data from the recording file to a
WebSocket server at real-time rate, allowing you to visualize past experiments
using the same visualiser as the original application.

### Usage

```shell
python -m cl.playback <recording_file.h5> [app_directory]
```

### Arguments

- **recording_file**: Path to the HDF5 recording file to play back.
- **app_directory** (optional): Path to the application directory or zip file containing
  the visualiser. If a directory is provided and contains a `web/` folder with
  `vis.html` and `vis.mjs`, the app's custom visualiser will be served. If a zip file
  is provided, it must contain a single top-level directory with a `web/` folder inside.

### CLI Controls

Once the playback is running, the following keyboard controls are available:

| Key                              | Action                    |
|----------------------------------|---------------------------|
| SPACE                            | Toggle pause/play         |
| ← / →                            | Skip ±5 seconds           |
| Shift (Ctrl on Windows) + ← / →  | Skip ±1 minute            |
| g                                | Go to specific timestamp  |
| r                                | Restart from beginning    |
| h / ?                            | Show help                 |
| q                                | Quit                      |

### Example

```shell
# Play back a recording with the default visualiser
python -m cl.playback my_experiment.h5

# Play back with an application's custom visualiser
python -m cl.playback my_experiment.h5 /path/to/my-app

# Use a different port
python -m cl.playback my_experiment.h5 --port 8080
```

### Features

- Real-time playback of samples, spikes, stims, and data streams
- Starts in paused state for inspection
- Interactive seek and skip controls
- Supports custom application visualisers
- WebSocket server for live data streaming to browser

### Notes

- This module is a command-line tool only and cannot be imported directly.
- This module currently is only available in the SDK package, and not on the device.
- Datastream attributes only hold their final value at the end of the recording, so playback cannot accurately reflect attribute
  changes over time.
"""

__all__: list[str] = []  # Nothing to export
