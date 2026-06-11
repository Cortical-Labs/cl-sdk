import contextlib
import logging
from typing import Any, cast

from ._sim._data_buffer import DataStreamEventIndexRecord, DataStreamEventRecord
from .util import to_msgpacked

_logger = logging.getLogger("cl.data_stream")

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
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > DataStreamEventIndexRecord.MAX_STREAM_NAME_LENGTH:
            raise ValueError(
                f"DataStream name encoded length must be <= "
                f"{DataStreamEventIndexRecord.MAX_STREAM_NAME_LENGTH} bytes, got {len(name_bytes)}"
            )
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

        # Write to shared buffer for WebSocket broadcasting
        self._write_to_shared_buffer(timestamp, msgpacked_data)

        # Forward to active recordings via command queue
        for recording in self._neurons._recordings:
            recording._write_data_stream_event(self.name, timestamp, msgpacked_data)

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
                try:
                    buffer.write_datastream_event(record)
                except Exception:
                    _logger.debug(
                        "Failed to write data stream %r event at timestamp %s to shared buffer",
                        self.name,
                        timestamp,
                        exc_info=True,
                    )

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
