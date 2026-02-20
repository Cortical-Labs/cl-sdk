import contextlib
from typing import Any, cast

from ._data_buffer import DataStreamEventRecord
from .util import to_msgpacked

class DataStream:
    """
    Manages a named stream of (timestamp, serialised_data) for recordings and visualisation.
    This is created using `Neurons.create_data_stream()`. Do not create instances
    of `DataStream` directly.

    See `RecordingView.data_streams` for how to use data streams saved in a recording.
    """

    name: str
    """ Name of this data stream. """

    def __init__(
        self,
        neurons,
        name:       str,
        attributes: dict[str, Any] | None = None
        ):
        """
        Constructor for `DataStream`.

        @private -- hide from docs
        """
        super().__init__()
        from . import Neurons
        neurons = cast("Neurons", neurons)
        self.name = name
        self._neurons = neurons  # Keep reference for accessing shared buffer

        # Track most recent timestamp to avoid O(n) max() lookup on every append
        self._most_recent_ts: int | None = None

        # Store attributes
        self._attributes = attributes if isinstance(attributes, dict) else {}

        # Register this datastream in Neurons so that it can be saved in a recording
        neurons._data_streams[name] = self

        # Send initial attributes to WebSocket subprocess
        if self._attributes:
            self._broadcast_attributes_reset()

    def append(self, timestamp: int, data: Any):
        """
        Append a new data point to the stream.

        Args:
            timestamp: Timestamp that marks this data point.
            data:      Any type of serialisable data.

        Constraints:
        - New `data` must have `timestamp` greater than existing in the data stream, otherwise
          a `RuntimeError` will be raised.
        """
        if self._most_recent_ts is not None and timestamp <= self._most_recent_ts:
            raise RuntimeError(f"New data stream data must have a newer timestamp than the most recent data. ({timestamp} <= {self._most_recent_ts})")
        msgpacked_data: bytes = to_msgpacked(data)  # type: ignore
        self._most_recent_ts = timestamp

        # Push to all active recordings
        for recording in self._neurons._recordings:
            recording._write_data_stream_event(self.name, timestamp, msgpacked_data)

        # Also write to shared buffer for WebSocket broadcasting
        self._write_to_shared_buffer(timestamp, msgpacked_data)

    def set_attribute(self, key: str, value: Any):
        """
        Set a single attribute on the data stream. The attribute refers to the
        attribute dictionary passed to `Neurons.create_data_stream(attributes)`.

        Args:
            key:   Attribute key.
            value: Attribute value.
        """
        self.update_attributes({ key: value })

    def update_attributes(self, attributes: dict[str, Any]):
        """
        Update multiple attributes on the data stream. The attribute refers to the
        attribute dictionary passed to `Neurons.create_data_stream(attributes)`.

        Args:
            attributes: `dict` containing attribute keys and values to be updated.
        """
        self._attributes.update(attributes)
        # Broadcast attribute update via WebSocket
        self._broadcast_attributes_updated(attributes)

    def _write_to_shared_buffer(self, timestamp: int, msgpacked_data: bytes):
        """Write data stream event to shared buffer for WebSocket broadcasting."""
        # Check if producer is running and has a buffer
        if self._neurons._data_producer is not None:
            buffer = self._neurons._data_producer.buffer
            if buffer is not None:
                record = DataStreamEventRecord(
                    timestamp   = timestamp,
                    stream_name = self.name,
                    data        = msgpacked_data
                )
                with contextlib.suppress(Exception):
                    buffer.write_datastream_event(record)

    def _broadcast_attributes_updated(self, updated_attributes: dict[str, Any]):
        """Broadcast attribute update to WebSocket clients."""
        # Check if WebSocket server is running
        if self._neurons._websocket_server is not None:
            with contextlib.suppress(Exception):
                self._neurons._websocket_server.send_attribute_update(
                    self.name,
                    updated_attributes
                )

    def _broadcast_attributes_reset(self):
        """Broadcast full attribute set to WebSocket clients."""
        # Check if WebSocket server is running
        if self._neurons._websocket_server is not None:
            with contextlib.suppress(Exception):
                # Send via subprocess queue
                self._neurons._websocket_server.send_attribute_reset(
                    self.name,
                    self._attributes
                )
