"""
# Cortical Labs API: Applications Models and Types

## Overview

This module defines reusable types and Pydantic models used across applications
for configuration and data representation. These types provide:

- **Annotated constraints**: Type aliases with built-in validation to ensure
    parameter values stay within safe operational ranges (e.g., stimulation
    amplitudes between 0 (exclusive) to 3 (inclusive) µA, pulse widths in 20 µs steps).

- **Custom UI rendering**: Special formatting hints for the web-based configuration
    editor, providing enhanced input controls like channel pickers and duration
    entries.

- **Frozen models**: Immutable Pydantic models with strict validation that prevent
    accidental modification and ensure configuration integrity throughout application
    execution.

## Key Categories

### Base Models

- `FrozenBaseModel`: Recommended base class for all configuration models,
    providing immutability and strict field validation.

### Stimulation Types

Constrained types for defining electrical stimulation parameters within hardware
limits:

- `StimAmplitudeMicroAmps`: range (0.0, 3.0] µA in absolute terms
- `StimPulseWidthMicroSeconds`: range (0, 10000] µs in 20 µs steps
- `StimFrequencyHz`: range (0.0, 200.0] Hz
- `StimmableChannel`: Valid channel numbers (0-63, excluding 0, 4, 7, 56, 63)
- `ChannelList`: Lists of channels with custom picker UI

**Important**: Total charge delivered per stim pulse should not exceed 3 nC.

### Composite Models

- `StimPulseComponentModel`: Single-phase pulse definition with amplitude and width
- `StimDesignModel`: Multi-component pulse designs with automatic validation
- `StimFrequencyRangeHzModel`: Min/max frequency ranges with validation
- `SizeIntModel`: 2D integer dimensions for layouts and grids

### General Types

- `DurationSeconds`: A non-negative integer formatted as separate hours, minutes, and seconds entries in the config UI

These types streamline application development by handling validation, serialization,
and providing a consistent interface to the configuration editor.
"""

from .model import (
    STIMMABLE_CHANNELS,
    ChannelList,
    DurationSeconds,
    FrozenBaseModel,
    SizeIntModel,
    StimAmplitudeMicroAmps,
    StimDesignModel,
    StimFrequencyHz,
    StimFrequencyRangeHzModel,
    StimmableChannel,
    StimPulseComponentModel,
    StimPulseWidthMicroSeconds,
)

# Reassign __module__ for pdoc to show correct import paths
FrozenBaseModel.__module__           = "cl.app.model"
SizeIntModel.__module__              = "cl.app.model"
StimDesignModel.__module__           = "cl.app.model"
StimFrequencyRangeHzModel.__module__ = "cl.app.model"
StimPulseComponentModel.__module__   = "cl.app.model"

__all__ = [
    "STIMMABLE_CHANNELS",
    "StimmableChannel",
    "ChannelList",
    "StimAmplitudeMicroAmps",
    "StimFrequencyHz",
    "StimPulseWidthMicroSeconds",
    "DurationSeconds",
    "FrozenBaseModel",
    "SizeIntModel",
    "StimPulseComponentModel",
    "StimDesignModel",
    "StimFrequencyRangeHzModel",
]
