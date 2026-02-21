from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Burst:
    """ Represents a single burst with start and end times in frames. """
    start_frame: int
    end_frame  : int

    def get_duration(self, sampling_frequency: int | float) -> float:
        """ Calculates the burst duration in seconds. """
        return float((self.end_frame - self.start_frame) / sampling_frequency)

@dataclass
class Bursts:
    """ A collection of Burst objects. """
    data:                       list[Burst] = field(default_factory=list)
    _current_burst_start_frame: int | None  = None

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _add_new_burst(self, end_frame: int):
        assert self._current_burst_start_frame is not None
        self.data.append(Burst(
            start_frame = int(self._current_burst_start_frame),
            end_frame   = int(end_frame)
        ))
        self._current_burst_start_frame = None

    @property
    def is_bursting(self) -> bool:
        return not self._current_burst_start_frame is None

    def step(self, frame: int, is_bursting: bool):
        """
        Updates the burst list based on the current frame and bursting state.
        This method is called iteratively to open or close burst events.
        """
        if is_bursting and not self.is_bursting:
            # A new burst has started
            self._current_burst_start_frame = frame
        elif not is_bursting and self.is_bursting:
            # A burst has just ended
            assert self._current_burst_start_frame is not None
            self._add_new_burst(end_frame = frame)

    def finalise(self, end_frame: int):
        """
        Ensures any ongoing burst at the end of the recording is properly closed.
        """
        if self.is_bursting:
            assert self._current_burst_start_frame is not None
            self._add_new_burst(end_frame = end_frame)

    @staticmethod
    def _serialise(bursts: Bursts) -> list[dict]:
        """ Serialisation for saving Bursts with Pydantic. """
        from dataclasses import asdict
        return [ asdict(burst) for burst in bursts ]

    @staticmethod
    def _validate(obj: list[dict]) -> Bursts:
        """ Validation for loading from JSON using Pydantic. """
        if isinstance(obj, Bursts):
            return obj
        return Bursts([
            Burst(
                start_frame = obj["start_frame"],
                end_frame   = obj["end_frame"]
                )
            for obj in obj
            ])