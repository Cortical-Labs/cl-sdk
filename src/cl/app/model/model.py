from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationInfo, field_validator
from pydantic_core import PydanticCustomError

from ... import StimDesign

#
# Annotated and constrained config type definitions
#

class FrozenBaseModel(BaseModel):
    """
    A Pydantic BaseModel with frozen (immutable) instances, and extra fields forbidden.

    This is the recommended base class for all configuration models and sub-models to be used to define
    application configurations, for maximum compatibility with the applications configuration web UI.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

type STIMMABLE_CHANNELS = Literal[1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62]
"""A literal type representing all stimulatable channel numbers (0-63, excluding non-stimulatable channels: 0, 4, 7, 56, 63)."""

type StimAmplitudeMicroAmps = Annotated[
    float,
    Field(
        gt            = 0.0,
        le            = 3.0,
        allow_inf_nan = False,
        title         = "Stimulation Amplitude (µA)",
        description   = "Stimulation amplitude in microamperes.",
    ),
]
"""
A valid stimulation amplitude, defined as a float greater than 0.0 and less than or equal to 3.0 microamperes.
1.0 microamperes is a commonly used default value.
"""

type StimPulseWidthMicroSeconds = Annotated[
    int,
    Field(
        gt          = 0,
        le          = 10000,
        multiple_of = 20,
        title       = "Stimulation Pulse Width (µs)",
        description = "Stimulation pulse width in microseconds.",
    ),
]
"""
A valid stimulation pulse width, defined as an integer greater than 0 and less than or equal to 10000 microseconds,
in steps of 20 microseconds. 160 microseconds is a commonly used default value.
"""

type StimFrequencyHz = Annotated[
    float,
    Field(
        gt            = 0.0,
        le            = 200.0,
        allow_inf_nan = False,
        title         = "Stimulation Frequency (Hz)",
        description   = "Frequency of repeated stimulation pulses in a burst in Hertz.",
    ),
]
"""
A valid stimulation frequency, defined as a float greater than 0.0 and less than 10000.0 Hz.
"""

type StimmableChannel = Annotated[
    STIMMABLE_CHANNELS,
    Field(
        title       = "Stimmable Channel",
        description = "Channel number (0-63, excluding non-stimulatable channels: [0, 4, 7, 56, 63]).",
    ),
]
"""
A stimulatable channel number, defined as an integer between 0 and 63, excluding the invalid channels: 0, 4, 7, 56, and 63.
An annotated version of `STIMMABLE_CHANNELS` with a name and description for use in Pydantic models.
"""

type ChannelList = Annotated[
    list[StimmableChannel],
    Field(
        title             = "Stimmable Channel List",
        description       = "List of stimulatable channel numbers (0-63, excluding non-stimulatable channels: [0, 4, 7, 56, 63]).",
        json_schema_extra = {"format": "channel_list", "uniqueItems": True},
        min_length        = 1,
        max_length        = 59,
    ),
]
"""
A valid list of channel numbers, defined as a list containing between 1 and 59 valid channel numbers. This will be rendered with a custom interface in the config UI.
"""

type DurationSeconds = Annotated[
    int,
    Field(
        title             = "Duration",
        json_schema_extra = {"format": "duration"},
        gt                = 0,
    )
]
"""
A positive integer representing a duration in seconds. This will be rendered with a custom HH:MM:SS format in the config UI.
"""

class StimPulseComponentModel(FrozenBaseModel):
    """A single component of a stimulation pulse, either a negative or positive leading-edge phase."""

    model_config = ConfigDict(title="Stimulation Pulse Component", extra="forbid", frozen=True, json_schema_extra={"format": "grid"})

    pulse_width_us: Annotated[
        StimPulseWidthMicroSeconds,
        Field(
            title       = "Pulse Width (µs)",
            description = "The width of the pulse component in microseconds.",
        ),
    ]
    """Pulse width of the component in microseconds."""

    signed_amplitude_ua: Annotated[
        StimAmplitudeMicroAmps,
        Field(
            title       = "Signed Amplitude (µA)",
            description = "The signed amplitude of the pulse component in microamperes. Must not be zero.",
            ge          = -3.0,
            le          = 3.0,
        ),
    ]
    """Signed amplitude of the pulse component in microamperes."""

    @field_validator("signed_amplitude_ua", mode="after")
    @classmethod
    def validate_non_zero_amplitude(cls, value: float) -> float:
        """@private"""
        if value == 0.0:
            raise PydanticCustomError(
                "zero_amplitude",
                "Stimulation pulse component amplitude must be non-zero.",
            )
        return value

class StimDesignModel(FrozenBaseModel):
    """Configuration for a single stimulation design."""

    model_config = ConfigDict(title="Stimulation Design", extra="forbid", frozen=True)

    components: Annotated[
        list[StimPulseComponentModel],
        Field(
            title             = "Stimulation Pulse Components",
            description       = "List of stimulation pulse components (consisting from 1 to 3 components).",
            json_schema_extra = {"format": "stim_pulse_components", "uniqueItems": False},
            min_length        = 1,
            max_length        = 3,
        ),
    ]
    """List of stimulation pulse components that make up the stimulation design. Must contain between 1 and 3 components."""

    _stim_design: StimDesign | None = PrivateAttr(default=None)
    """Cached StimDesign instance created from the components."""

    @field_validator("components", mode="after")
    @classmethod
    def validate_pulse_signs(cls, value: list[StimPulseComponentModel]) -> list[StimPulseComponentModel]:
        """@private"""
        if len(value) > 1:
            # Ensure that the pulse components alternate in sign (negative/positive)
            for i in range(1, len(value)):
                if (value[i].signed_amplitude_ua > 0) == (value[i - 1].signed_amplitude_ua > 0):
                    raise PydanticCustomError(
                        "invalid_pulse_signs",
                        "Stimulation pulse components must alternate in sign (negative/positive).",
                    )
        return value

    def to_stim_design(self) -> StimDesign:
        """Converts the stimulation design configuration model to a `StimDesign` instance that can be used directly for stimulation."""

        if self._stim_design is None:
            stim_design_components = []
            for component in self.components:
                stim_design_components.extend((component.pulse_width_us, component.signed_amplitude_ua))

            self._stim_design = StimDesign(*stim_design_components)

        return self._stim_design

class StimFrequencyRangeHzModel(FrozenBaseModel):
    """A valid stimulation frequency range in Hz."""

    model_config = ConfigDict(title="Stimulation Frequency Range (Hz)", extra="forbid", frozen=True)

    min: Annotated[StimFrequencyHz, Field(title="Minimum", description=None)]
    """Minimum value of the range, must be smaller than the maximum."""

    max: Annotated[StimFrequencyHz, Field(title="Maximum", description=None)]
    """Maximum value of the range, must be greater than the minimum."""

    @property
    def span(self) -> StimFrequencyHz:
        """Returns the span of the range, i.e. the difference between max and min."""
        return self.max - self.min

    @field_validator("max", mode="after")
    @classmethod
    def validate_max(cls, value: StimFrequencyHz, info: ValidationInfo) -> StimFrequencyHz:
        """@private"""
        if "min" not in info.data:
            return value

        if value < info.data["min"]:
            raise PydanticCustomError(
                "max_less_than_min",
                "The maximum value {value} must be greater than or equal to the minimum value {min}",
                {"value": value, "min": info.data["min"]},
            )
        return value

class SizeIntModel(FrozenBaseModel):
    """Base class for positive 2D integer size configurations."""

    width: Annotated[int, Field(gt=0)]
    """Width of the element, must be greater than 0."""

    height: Annotated[int, Field(gt=0)]
    """Height of the element, must be greater than 0."""
