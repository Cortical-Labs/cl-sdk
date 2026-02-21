"""
# Cortical Labs API: Applications Module

## Overview

This module provides the foundation for creating applications that can be
installed and run on Cortical Labs devices. It defines base classes, configuration
models, and data structures that all applications must use to integrate with the
device's application runtime and management system.

## Submodule: model

The `model` submodule provides shared types and models used across applications,
including stimulation parameters, channel definitions, and common data structures.
See `cl.app.model` documentation for details.

## Submodules: init and pack

`init` and `pack` are command-line tools for generating application boilerplate
and packaging applications into distributable ZIP files, respectively. They are
not intended for direct import. See `cl.app.init` and `cl.app.pack` documentation
for usage details.

## Submodules: run

`run` is a submodule for testing applications locally. This module presently does not exist on the device, as the application
can simply be installed and run directly.

## Quick Start

Follow these steps to create a new application:

1. Open a terminal and navigate to where you want to create your application.
2. With the Cortical Labs SDK installed (optionally in a virtual environment), run: `python -m cl.app.init [application_name]`
    - Replace `[application_name]` with your desired application name, or leave it blank to use the current directory as the base.
3. This will create a new folder with the minimum required application structure and pre-filled files.
4. Modify the generated files to implement your application's functionality.
5. To package your application for installation, run: `python -m cl.app.pack [application_folder]`
    - Replace `[application_folder]` with the path to your application folder, or leave it blank to use the current directory.
6. This will create a ZIP file containing your application, ready for installation on the device.

## Application Structure

Each application needs to be contained within a dedicated folder. The name of
the folder should be unique, and only consist of alphanumeric characters, hyphens,
underscores and periods (e.g. `my-application_01`). The `cl-` prefix is reserved for
system applications.

Within this folder, three key files are required to define the application:
- `info.json`: This JSON file contains metadata about the application, containing the following fields:
    - `name`: The name of the application.
    - `version`: The version of the application.
    - `description`: A brief description of what the application does. This will be displayed in the application launcher once the application is installed.
    - `author`: The author of the application.
    - `config_version`: The configuration version of the application. This is used to manage configuration changes between different versions of the application.
- `src/__init__.py`: This Python file is loaded as the main entry point of each application. Further information on application structure is provided below.
- `default.json`: This JSON file contains the default configuration settings for the application. Further information on application configuration is provided below.

### Source Code Structure and Format

All source code for the application should be contained within the `src` folder. This folder
can contain any number of subfolders and files. Local imports within the application should
use relative imports to ensure compatibility with the application loader.

The `__init__.py` file serves as the main entry point for the application. This file should
define an instance of the application object that inherits from the `cl.app.BaseApplication` class. This class provides the necessary interface for the application to interact with the application framework.

### Example Usage

In `src/my_app.py`:
```python
from cl.app import BaseApplication, BaseApplicationConfig, RunSummary, OutputType

class MyAppConfig(BaseApplicationConfig):
    parameter1: str
    parameter2: int = 100

class MyApp(BaseApplication[MyAppConfig]):
    def run(self, config: MyAppConfig, output_directory: str) -> RunSummary:
        # Application logic here
        # Save results to output_directory

        return RunSummary(
            type    = OutputType.TEXT,
            content = f"Completed with parameter1={config.parameter1}"
        )

    @staticmethod
    def config_class() -> type[MyAppConfig]:
        return MyAppConfig
```

In `src/__init__.py`:
```python
from .my_app import MyApp

application = MyApp()
```

In `info.json`:
```json
{
    "name": "My Application",
    "version": "1.0.0",
    "description": "An example application for Cortical Labs devices.",
    "author": "Your Name",
    "config_version": 1
}
```

In `default.json`:
```json
{
    "parameter1": "default_value",
    "parameter2": 100
}
```

### Optional Files

In addition to the required files, the following optional files maybe in included in the application folder for enhanced functionality:

```plaintext
<application_name>/
├── src/
│   ├── __init__.py         # Required: Application instance
│   └── ...                 # Optional: Supporting Python files
├── info.json               # Required: Application metadata
├── requirements.txt        # Optional: External dependencies
├── web/                    # Optional: Web visualisation
│   ├── vis.html
│   ├── vis.css
│   └── vis.mjs
└── presets/                # Optional: Preset configurations
    └── <config_name>.json
```

- `requirements.txt`: A text file listing any external Python dependencies required by the application (one per line, optionally with version specifiers).
- `web/`: A directory containing files for a web-based visualisation interface for the application. See below for more details.
- `presets/`: A directory containing additional preset configuration JSON files for the application. Each file should be named `<config_name>.json`, where `<config_name>` is a descriptive name for the configuration.

#### Web Visualisation

To add a web-based visualisation to your application, create a `web` folder within your application directory. This folder should contain the following files:
- `vis.html`: The HTML file that defines the structure of the visualisation interface.
- `vis.mjs`: A JavaScript ES6 module that contains the logic for rendering the visualisation.
- `vis.css`: (Optional) A CSS file for styling the visualisation interface.

##### HTML Structure

The `vis.html` file should not be a complete document, and only include the necessary elements for the visualisation.

##### JavaScript Module

The `vis.mjs` file needs to contain two definitions:
- `dataStreams`: An array object of strings defining the data streams that the visualisation requires. These should correspond to the data streams provided by the application during runtime. The system additionally provides `cl_spikes` and `cl_stims` data streams by default, that hold spike and stimulation event data respectively.
- `createVisualiser(uniqueId, div)`: A function that is responsible for handling the visualisation rendering, returning a collection of functions for managing the visualisation lifecycle.
    - The function takes two parameters:
        - `uniqueId`: A unique identifier string for the visualiser instance.
        - `div`: The HTML `div` element that will contain the visualisation.
    - The function should return an object containing the following functions and properties:
        - `reset()`: A function that resets the visualisation to its initial state.
        - `process(dataStreamName, timestamp, data)`: A function that processes incoming data for the visualisation.
            - `dataStreamName`: The name of the data stream from which the data originates.
            - `timestamp`: The timestamp of the incoming data.
            - `data`: The actual data to be processed.
        - `draw(browserTimestampMs, dataStreamTimestamp)`: A function that handles the rendering of the visualisation.
            - `browserTimestampMs`: The current browser timestamp in milliseconds.
            - `dataStreamTimestamp`: The timestamp of the latest data stream.
        - `attributesReset(dataStreamName, initialAttributes)`: A function that handles resetting attributes for a specific data stream.
            - `dataStreamName`: The name of the data stream.
            - `initialAttributes`: The initial attributes to be set.
        - `attributesUpdated(dataStreamName, updatedAttributes)`: A function that handles updating attributes for a specific data stream.
            - `dataStreamName`: The name of the data stream.
            - `updatedAttributes`: The updated attributes to be applied.
            - `bufferMs`: An optional property that defines the buffer duration in milliseconds for the visualisation. If not provided, it defaults to `1000 / 60` (approximately 16.67 ms).

###### CSS Styling

The optional `vis.css` file can be used to style the visualisation interface. This file should contain standard CSS rules targeting the elements defined in the `vis.html` file.

##### Example

A simple example for visualising MEA activity rendered inside a HTML canvas is shown below:

In `web/vis.html`:
```html
<div class="visualiser-container">
    <div class="canvas-container electrode-canvas-container">
        <canvas id="electrodeCanvas" class="electrode-canvas"></canvas>
    </div>
</div>
```

In `web/vis.mjs`:
```javascript
const dataStreams = ["cl_spikes"];

function createVisualiser(uniqueId, div) {

    const layout =
        [[0,  8, 16, 24, 32, 40, 48, 56],
         [1,  9, 17, 25, 33, 41, 49, 57],
         [2, 10, 18, 26, 34, 42, 50, 58],
         [3, 11, 19, 27, 35, 43, 51, 59],
         [4, 12, 20, 28, 36, 44, 52, 60],
         [5, 13, 21, 29, 37, 45, 53, 61],
         [6, 14, 22, 30, 38, 46, 54, 62],
         [7, 15, 23, 31, 39, 47, 55, 63]];
    const layoutMap = new Map();
    for (let row = 0; row < layout.length; row++) {
        for (let col = 0; col < layout[row].length; col++) {
            const electrode = layout[row][col];
            const idx = row * layout[row].length + col;
            layoutMap.set(electrode, { row, col, idx });
        }
    }

    const electrodeCanvas = div.querySelector('#electrodeCanvas');
    const electrodeContext = electrodeCanvas.getContext('2d', { alpha: true });

    // Colour definitions for spike activity gradient from low (rgb(240,230,140)) to high (rgb(255,152,79))
    const spikeColors = []
    for (let i = 0; i <= 5; i++) {
        const ratio = i / 5;
        const r = Math.round((1 - ratio) * 240 + ratio * 255);
        const g = Math.round((1 - ratio) * 230 + ratio * 152);
        const b = Math.round((1 - ratio) * 140 + ratio * 79);
        spikeColors.push(`rgb(${r}, ${g}, ${b})`);
    }

    let spikes        = new Array(64).fill(0);
    let spikeDecay    = new Array(64).fill(0);

    let drawTime      = 0
    let dpr           = window.devicePixelRatio || 1;
    let lastDpr       = 0;

    const decayFrames = 8;

    function reset() {
        spikes.fill(0);
        spikeDecay.fill(0);

        electrodeContext.clearRect(0, 0, electrodeCanvas.width, electrodeCanvas.height);

        dpr = window.devicePixelRatio || 1;
        const dprChanged = dpr !== lastDpr;
        lastDpr = dpr;

        drawElectrodeVisualiser(dprChanged);
    }

    function process(dataStreamName, timestamp, data) {
        if (dataStreamName == 'cl_spikes') {
            spikes[data.channel]++;
            spikeDecay[data.channel] = decayFrames;
        }
    }

    function draw(browserTimestampMs, dataStreamTimestamp) {
        dpr = window.devicePixelRatio || 1;
        const dprChanged = dpr !== lastDpr;
        lastDpr = dpr;

        // Clear the canvas
        electrodeContext.clearRect(0, 0, electrodeCanvas.width, electrodeCanvas.height);

        // Render the latest data
        drawElectrodeVisualiser(dprChanged);
        updateStoredData(browserTimestampMs);
    }

    function drawElectrodeVisualiser(dprChanged) {
        const electrodeContainer = electrodeCanvas.parentElement;
        const electrodeCanvasSize = electrodeContainer.clientWidth;
        const electrodeScale = electrodeCanvas.width / electrodeCanvasSize;

        // Resize electrode canvas if needed
        if (dprChanged || electrodeCanvas.width !== Math.round(electrodeCanvasSize * dpr) || electrodeCanvas.height !== Math.round(electrodeCanvasSize * dpr)) {
            electrodeCanvas.width = Math.round(electrodeCanvasSize * dpr);
            electrodeCanvas.height = Math.round(electrodeCanvasSize * dpr);
        }

        const hiddenElectrodes = [0, 7, 56, 63];
        const padding = 10;
        const gridWidth = electrodeCanvasSize - 2 * padding;
        const gridHeight = electrodeCanvasSize - 2 * padding;

        const cellWidth = gridWidth / 8;
        const cellHeight = gridHeight / 8;

        // Draw rounded rect background for electrode canvas
        electrodeContext.fillStyle = 'rgba(191, 191, 191, 0.2)';
        electrodeContext.beginPath();
        electrodeContext.roundRect(0, 0, electrodeCanvas.width, electrodeCanvas.height, [Math.min(cellWidth, cellHeight) * 1.2]);
        electrodeContext.fill();

        for (let i = 0; i < 64; i++) {
            let radius = Math.min(cellWidth, cellHeight) / 2 * 0.8;
            var row_index = layoutMap.get(i);
            if (!row_index) continue;

            let color = '#000000';
            let alpha = 0.25;

            if (hiddenElectrodes.includes(i)) {
                continue; // Skip corner electrodes
            }

            const centerX = padding + row_index['col'] * cellWidth + cellWidth / 2;
            const centerY = padding + row_index['row'] * cellHeight + cellHeight / 2;

            electrodeContext.globalAlpha = alpha;
            electrodeContext.fillStyle = color;
            electrodeContext.beginPath();
            electrodeContext.arc(centerX * electrodeScale, centerY * electrodeScale, radius * electrodeScale, 0, 2 * Math.PI);
            electrodeContext.fill();

            // Draw spike activity separately
            if (spikeDecay[i] > 0) {
                // Use a gradient for spikes: if the spike count is 1, use the low color, otherwise interpolate between low and high colors where the max spike count is 5.
                const spikeColorIndex = Math.min(spikes[i], 5);
                color = spikeColors[spikeColorIndex];
                alpha = spikeDecay[i] / decayFrames;

                electrodeContext.globalAlpha = alpha;
                electrodeContext.fillStyle = color;
                electrodeContext.beginPath();
                electrodeContext.arc(centerX * electrodeScale, centerY * electrodeScale, radius * electrodeScale, 0, 2 * Math.PI);
                electrodeContext.fill();
            }
        }

        electrodeContext.globalAlpha = 1.0;
    }

    function updateStoredData(browserTimestampMs) {
        // Exponential decay for spikes
        for (let i = 0; i < 64; i++) {
            if (spikeDecay[i] > 0) {
                spikeDecay[i]--;
                if (spikeDecay[i] === 0) {
                    // Reset spike count when decay is finished
                    spikes[i] = 0;
                }
            }
        }
    }

    // Call reset once to initialize the visualiser
    reset();

    return {
        // Settings
        bufferMs: (1000 / 60) * 10, // Optional, default is 1000 / 60

        // Functions
        reset,
        process,
        draw
    };
}
```

In `web/vis.css`:
```css
.visualiser-container {
    width: 460px;
    margin: 0 auto;
}

.canvas-container {
    margin: 0;
    background: transparent;
}

.electrode-canvas-container {
    aspect-ratio: 1 / 1;
}

canvas {
    width: 100%;
    height: 100%;
    display: block;
    background: transparent;
}
```
"""

from . import init, model, pack, run
from .base import BaseApplication, BaseApplicationConfig, OutputType, RunSummary

__all__ = (
    "BaseApplicationConfig",
    "OutputType",
    "RunSummary",
    "BaseApplication",
    "model",
    "init",
    "pack",
    "run"
)
