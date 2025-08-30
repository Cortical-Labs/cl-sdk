from __future__ import annotations
from collections.abc import Generator
from typing import Any

import tables
from tables.file import File
from tables.table import Table
from tables.attributeset import AttributeSet
from tables.group import Group

from numpy import ndarray

from pathlib import Path

from . import from_msgpacked, binary_search, binary_search_range

class AttributesView:
    """ Convenience class for accessing recording attributes. """

    def __init__(self, h5_attributes: AttributeSet):
        super().__setattr__("_h5_attributes", h5_attributes)

    def __getattr__(self, name):
        """ Pass through for all attributes/methods of the wrapped object. """
        return getattr(self._h5_attributes, name)

    def __setattr__(self, name, value):
        """ Pass through for setting attributes on the wrapped object. """
        setattr(self._h5_attributes, name, value)

    def __delattr__(self, name):
        """ Pass through for deleting attributes on the wrapped object. """
        delattr(self._h5_attributes, name)

    def __str__(self):
        """ Override __str__ to use __repr__ as the HD5F str() result isn't helpful. """
        return repr(self._h5_attributes)

    def __repr__(self):
        return repr(self._h5_attributes)

    #
    # Explicitly handle common special methods
    #

    def __len__(self):
        return len(self._h5_attributes)

    def __getitem__(self, key):
        return self._h5_attributes[key]

    def __setitem__(self, key, value):
        self._h5_attributes[key] = value

    def __delitem__(self, key):
        del self._h5_attributes[key]

    def __iter__(self):
        return iter(self._h5_attributes)

    def __contains__(self, item):
        return item in self._h5_attributes

class DataStreamCollection:
    """ Interface for accessing a collection of DataStreams. """

    def __init__(self, data_streams: Group):
        self.data_streams = data_streams

    def keys(self) -> Generator[str, None, None]:
        return self.data_streams._v_children.keys()

    def items(self) -> Generator[tuple[str, DataStreamView], None, None]:
        for data_stream_name in self.keys():
            yield data_stream_name, DataStreamView(self.data_streams[data_stream_name])

    def values(self) -> Generator[DataStreamView, None, None]:
        for data_stream_name in self.keys():
            yield DataStreamView(self.data_streams[data_stream_name])

    def __repr__(self):
        data_streams = "".join(f"\n    {data_stream_name}" for data_stream_name in self.keys())
        return f"Data Streams:{data_streams}"

    def __iter__(self) -> Generator[str, None, None]:
        for data_stream_name in self.keys():
            yield data_stream_name

    def __getitem__(self, key):
        return DataStreamView(self.data_streams[key])

    def __getattr__(self, name):
        return DataStreamView(self.data_streams[name])

    def __len__(self):
        return len(self.data_streams._v_children.keys())

    def __contains__(self, key):
        return key in self.keys()

class DataStreamView:
    """
    Provides a read-only interface to data stream entries.

    DataStreamView is designed to allow iteration over data stream entries
    without loading the entire data stream into memory.
    """

    @staticmethod
    def data_for_entry(data, entry: Group):
        return from_msgpacked(data[entry["start_index"]:entry["end_index"]])

    def __init__(self, data_stream):
        self.index      = data_stream.index
        self.data       = data_stream.data
        self.attributes = AttributesView(data_stream._v_attrs)
        self._len       = len(data_stream.index)

    class DataStreamKeysView:
        def __init__(self, index):
            self.index = index

        def __iter__(self) -> Generator[int, None, None]:
            for entry in self.index:
                yield entry["timestamp"]

        def __len__(self) -> int:
            return len(self.index)

        def __repr__(self) -> str:
            return str(list(self))

    class DataStreamValuesView:
        def __init__(self, index, data) -> None:
            self.index  = index
            self.data   = data

        def __iter__(self) -> Generator[Any, None, None]:
            for entry in self.index:
                yield DataStreamView.data_for_entry(self.data, entry)

        def __len__(self):
            return len(self.index)

        def __repr__(self) -> str:
            return str(list(self))

    class DataStreamItemsView:
        def __init__(self, index, data) -> None:
            self.index  = index
            self.data   = data

        def __iter__(self) -> Generator[tuple[int, Any], None, None]:
            for entry in self.index:
                ts      = entry["timestamp"]
                data    = DataStreamView.data_for_entry(self.data, entry)
                yield ts, data

        def __len__(self):
            return len(self.index)

        def __repr__(self) -> str:
            return str(list(self))

    def keys(self) -> DataStreamKeysView:
        return self.DataStreamKeysView(self.index)

    def values(self) -> DataStreamValuesView:
        return self.DataStreamValuesView(self.index, self.data)

    def items(self) -> DataStreamItemsView:
        return self.DataStreamItemsView(self.index, self.data)

    def __len__(self):
        return self._len

    def __iter__(self):
        return iter(self.items())

    def __getitem__(self, key):
        """
        Get data for either a specific timestamp or for a range of timestamps.
        Single timestamp:
            data = data_stream[timestamp]
        Range of timestamps:
            items_view = data_stream[start_timestamp:end_timestamp]

        Raises KeyError if a specific timestamp is passed and it is not found.
        """
        if isinstance(key, slice):
            return self.values_for_range(key.start, key.stop)
        elif isinstance(key, int):
            entry_index = binary_search(self.index, key, lambda x: x["timestamp"])
            if entry_index is None:
                raise KeyError(key)

            return DataStreamView.data_for_entry(self.data, self.index[entry_index])
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def keys_for_range(self, start_timestamp, end_timestamp):
        """ Get all keys (timestamps) from start_timestamp up to but not including end_timestamp. """
        range_start, range_end = binary_search_range(self.index, start_timestamp, end_timestamp, lambda x: x["timestamp"])
        return self.DataStreamKeysView(self.index[range_start:range_end])

    def values_for_range(self, start_timestamp, end_timestamp):
        """ Get all values from start_timestamp up to but not including end_timestamp. """
        range_start, range_end = binary_search_range(self.index, start_timestamp, end_timestamp, lambda x: x["timestamp"])
        return self.DataStreamValuesView(self.index[range_start:range_end], self.data)

    def items_for_range(self, start_timestamp, end_timestamp):
        """ Get all items from start_timestamp up to but not including end_timestamp. """
        range_start, range_end = binary_search_range(self.index, start_timestamp, end_timestamp, lambda x: x["timestamp"])
        return self.DataStreamItemsView(self.index[range_start:range_end], self.data)

class RecordingView:
    """
    Convenience class for accessing recording data.

    Full access to the underlying PyTables file is provided through the
    `file` attribute. This allows access to the full range of PyTables
    functionality if needed.

    Attributes:
        file        : The underlying PyTables file.
        samples     : Recorded raw samples.
        spikes      : Recorded deteceted spikes.
        stims       : Recorded stimulation events.
        attributes  : The file / root level attributes.
        data_streams: Recorded data streams.
    """
    file: File
    """ The underlying PyTables file. """

    attributes: AttributesView
    """ The file / root level attributes. """

    samples: ndarray | None = None
    """ Recorded raw samples. """

    spikes: Table | None = None
    """ Recorded deteceted spikes. """

    stims: Table | None = None
    """ Recorded stimulation events. """

    data_streams: DataStreamCollection | None = None
    """ Recorded data streams. """

    def __init__(self, file_path: str):
        self.file       = tables.open_file(file_path)
        self.attributes = AttributesView(self.file.root._v_attrs)

        if "samples" in self.file.root:
            self.samples = self.file.root.samples
        if "spikes" in self.file.root:
            self.spikes = self.file.root.spikes
        if "stims" in self.file.root:
            self.stims = self.file.root.stims
        if "data_stream" in self.file.root:
            self.data_streams = DataStreamCollection(self.file.root.data_stream)

    def close(self):
        self.file.close()

    def __repr__(self):
        return (
            f"RecordingView of file: {str(Path(self.file.filename).resolve())}"
            "\n    file:         Direct access to the underlying PyTables object"
            "\n    attributes:   A view of the recording attributes"
            "\n    spikes:       Access spikes stored in the recording"
            "\n    stims:        Access stims stored in the recording"
            "\n    samples:      Access raw frames of samples stored in the recording"
            "\n    data_streams: A colletion of recorded data streams"
            )

    def __del__(self):
        self.close()