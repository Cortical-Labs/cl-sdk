from typing import Any, cast
from collections import OrderedDict

from cl.util import to_msgpacked

class DataStream():

    def __init__(
        self,
        neurons,
        name:       str,
        attributes: dict[str, Any] | None = None
        ):
        super().__init__()
        from cl import Neurons
        neurons = cast(Neurons, neurons)
        self.name = name

        # Container to hold the data for this datastream
        self._data = OrderedDict()

        # Store attributes
        self.attributes = attributes if isinstance(attributes, dict) else {}

        # Register this datastream in Neurons so that it can be saved in a recording
        neurons._data_streams[name] = self

    def append(self, timestamp: int, data: Any):
        """ Append a new data point to the stream. """
        self._data[timestamp] = to_msgpacked(data)

    def set_attribute(self, key: str, value: Any):
        """ Set an attribute on the data stream. """
        self.update_attributes({ key: value })

    def update_attributes(self, attributes: dict[str, Any]):
        """ Update multiple attributes on the data stream. """
        self.attributes.update(attributes)