from __future__ import annotations
from pathlib import Path
import json

from pydantic import BaseModel, TypeAdapter

class AnalysisMetadata(BaseModel):
    """ Contains metadata relating to the recording for analysis results. """

    file_path: str
    """ Path to the recording file. """

    channel_count: int
    """ Number of channels in the recording. """

    sampling_frequency: int | float
    """ Sampling frequency in Hz. """

    duration_frames: int
    """ Total frame count. """

    duration_seconds: float
    """ Total recording duration in seconds. """

class AnalysisResult(BaseModel):
    """ Base analysis class for one recording. """

    metadata: AnalysisMetadata
    """ Metadata relating to the recording for analysis results. """

    @classmethod
    def from_file(cls, fpath: Path | str) -> AnalysisResult:
        """ Loads an instance of supported AnalysisResult from a save file.

        Args:
            fpath: path to the file to load.
        """
        if isinstance(fpath, str):
            fpath = Path(fpath)

        def with_int_key(dct: dict) -> dict:
            """ Parses potentially integer dict keys. """
            return {
                int(k) if isinstance(k, str) and k.lstrip('-').isdigit() else k : v
                for k, v in dct.items()
                }

        with open(fpath, "r") as f:
            data = json.load(f, object_hook=with_int_key)

        return TypeAdapter(cls).validate_python(data, extra="forbid")

    def save(self, fpath: Path | str):
        """ Saves the AnalysisResult a json file.

        Args:
            fpath: save path.
        """
        if isinstance(fpath, str):
            fpath = Path(fpath)

        data = self.model_dump()
        with open(fpath, "w") as f:
            json.dump(data, f, indent=4)