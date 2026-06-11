"""
## Application Utilities

This submodule contains utility functions for the application module.
"""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from .model import StimDesignModel, StimPulseComponentModel

def run_migrations(
    config_dict: dict[str, Any],
    migrations : dict[int, Callable[[dict[str, Any]], None]],
) -> dict[str, Any] | None:
    """
    Apply a sequence of versioned migration functions to a config dictionary.

    Each key in `migrations` is a target config version, and the value is a
    function that mutates the config dict **in place** to bring it up to that
    version. Migrations are applied in ascending version order, and
    `config_version` is updated automatically after each step.

    Returns the migrated config dict, or `None` if no migration was needed.
    """
    config_version = config_dict.get("config_version", 0)
    applicable = sorted(v for v in migrations if v > config_version)

    if not applicable:
        return None

    migrated = deepcopy(config_dict)
    for target_version in applicable:
        migrations[target_version](migrated)
        migrated["config_version"] = target_version

    return migrated

def migrate_biphasic_to_stim_design(
    section        : dict[str, Any],
    current_key    : str = "current_ua",
    phase_width_key: str = "phase_width_us",
    stim_design_key: str = "stim_design",
) -> None:
    """
    In-place migration of legacy biphasic stimulation parameters to a
    `StimDesignModel`-compatible dict.

    Pops `current_key` and `phase_width_key` from *section* and replaces
    them with a `stim_design_key` entry containing a symmetric biphasic pulse
    with equivalent amplitude and phase width.
    """
    current_ua     = section.pop(current_key)
    phase_width_us = section.pop(phase_width_key)

    biphasic_stim_design = StimDesignModel(
        components = [
            StimPulseComponentModel(
                signed_amplitude_ua = -current_ua,
                pulse_width_us      = phase_width_us,
            ),
            StimPulseComponentModel(
                signed_amplitude_ua = current_ua,
                pulse_width_us      = phase_width_us,
            ),
        ],
    )

    section[stim_design_key] = biphasic_stim_design.model_dump()
