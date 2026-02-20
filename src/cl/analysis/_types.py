from __future__ import annotations
from typing import Annotated, TypedDict
from pathlib import Path

import numpy as np
from numpy import ndarray

from pydantic import PlainSerializer, PlainValidator

from ._bursts import Bursts

#
# Array types
#

def _ndarray_to_list(array: Array2DInt | Array2DFloat) -> list:
    """ Convert a NumPy array to list. """
    return array.tolist()

def _list_to_1darray_int(lst: list) -> Array2DInt:
    """ Convert a list to a 1D integer NumPy array. """
    array = np.array(lst, dtype=int)
    assert len(array.shape) == 1, f"Array expected to be 1D but have shape {array.shape}"
    return array

def _list_to_1darray_float(lst: list) -> Array2DFloat:
    """ Convert a list to a 1D integer NumPy array. """
    array = np.array(lst, dtype=float)
    assert len(array.shape) == 1, f"Array expected to be 1D but have shape {array.shape}"
    return array

def _list_to_2darray_int(lst: list) -> Array2DInt:
    """ Convert a list to a 2D integer NumPy array. """
    array = np.array(lst, dtype=int)
    assert len(array.shape) == 2, f"Array expected to be 2D but have shape {array.shape}"
    return array

def _list_to_2darray_float(lst: list) -> Array2DFloat:
    """ Convert a list to a 2D integer NumPy array. """
    array = np.array(lst, dtype=float)
    assert len(array.shape) == 2, f"Array expected to be 2D but have shape {array.shape}"
    return array

type Array1DInt = Annotated[
    ndarray[tuple[int], np.dtype[np.int_]],
    PlainValidator(_list_to_1darray_int),
    PlainSerializer(_ndarray_to_list, return_type=list, when_used="unless-none")
    ]
""" A 1D integer NumPy array type that serialises to a list. """

type Array1DFloat = Annotated[
    ndarray[tuple[int], np.dtype[np.floating]],
    PlainValidator(_list_to_1darray_float),
    PlainSerializer(_ndarray_to_list, return_type=list, when_used="unless-none")
    ]
""" A 1D float NumPy array type that serialises to a list. """

type Array2DInt = Annotated[
    ndarray[tuple[int, int], np.dtype[np.int_]],
    PlainValidator(_list_to_2darray_int),
    PlainSerializer(_ndarray_to_list, return_type=list, when_used="unless-none")
    ]
""" A 2D integer NumPy array type that serialises to a list. """

type Array2DFloat = Annotated[
    ndarray[tuple[int, int], np.dtype[np.floating]],
    PlainValidator(_list_to_2darray_float),
    PlainSerializer(_ndarray_to_list, return_type=list, when_used="unless-none")
    ]
""" A 2D float NumPy array type that serialises to a list. """

def _nan_inf_to_float(value: str | float) -> float:
    """ Converts nan or inf strings into special floats. """
    if isinstance(value, str):
        assert value in ["nan", "inf"], f"Expected either 'nan' or 'inf'"
        return float(value)
    else:
        return value

def _nan_inf_to_string(value: float) -> str | float:
    """ Coverts special floats (nan, inf) to string. """
    from math import isnan, isinf
    if isnan(value):
        return "nan"
    elif isinf(value):
        return "inf"
    else:
        return value

type SpecialFloat = Annotated[
    float,
    PlainValidator(_nan_inf_to_float),
    PlainSerializer(_nan_inf_to_string, return_type=float | str, when_used="unless-none")
    ]
""" Float type that support serialisation of special floats like nan and inf. """

#
# Burst types
#

BurstsType = Annotated[
    Bursts,
    PlainValidator(Bursts._validate),
    PlainSerializer(Bursts._serialise, return_type=list[dict], when_used="unless-none")
    ]
""" Handles serialisation of `Bursts` objects when saving results. """
