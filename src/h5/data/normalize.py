# The content from this file has been copied from loompy repository.
#
# Copyright (c) 2016, Linnarsson Lab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# flake8: noqa

import html
import logging

import numpy as np

log = logging.getLogger(__name__)


def normalize_attr_strings(a):
    """
    Take an np.ndarray of all kinds of string-like elements, and return an array of ascii (np.string_) objects

    Args:
        a: value to normalize

    Raises:
        ValueError: on unsupported input type

    Returns:
        normalized value
    """
    if np.issubdtype(a.dtype, np.object_):
        if np.all(
            [(type(x) is str or type(x) is np.str_ or type(x) is np.unicode_) for x in a.flatten()]
        ):
            return np.vectorize(lambda x: x.encode("ascii", "xmlcharrefreplace"))(a)
        elif np.all([type(x) is np.string_ for x in a]) or np.all(
            [type(x) is np.bytes_ for x in a]
        ):
            return a.astype("string_")
        else:
            log.debug(
                f"Attribute contains mixed object types ({np.unique([str(type(x)) for x in a])}); casting all to string"
            )
            return np.array([str(x) for x in a.flatten()], dtype="string_").reshape(a.shape)
    elif np.issubdtype(a.dtype, np.string_) or np.issubdtype(a.dtype, np.object_):
        return a
    elif np.issubdtype(a.dtype, np.str_) or np.issubdtype(a.dtype, np.unicode_):
        return np.vectorize(lambda x: x.encode("ascii", "xmlcharrefreplace"))(a)
    else:
        raise ValueError("String values must be object, ascii or unicode.")


def normalize_attr_array(a):
    """
    Take all kinds of array-like inputs and normalize to a one-dimensional np.ndarray

    Args:
        a: value to normalize

    Raises:
        ValueError: on unsupported input type

    Returns:
        normalized value
    """
    if type(a) is np.ndarray:
        return a
    elif type(a) is np.matrix:
        if a.shape[0] == 1:
            return np.array(a)[0, :]
        elif a.shape[1] == 1:
            return np.array(a)[:, 0]
        else:
            raise ValueError("Attribute values must be 1-dimensional.")
    elif type(a) is list or type(a) is tuple:
        return np.array(a)
    else:
        raise ValueError(
            "Argument must be a list, tuple, numpy matrix, numpy ndarray or sparse matrix."
        )


def normalize_attr_values(a) -> np.ndarray:
    """
    Take all kinds of input values and validate/normalize them.

    Args:
        a: List, tuple, np.matrix, np.ndarray or sparse matrix
            Elements can be strings, numbers or bools

    Returns:
        a_normalized: An np.ndarray with elements conforming to one of the valid Loom attribute types

    Remarks:
        This method should be used to prepare the values to be stored in the HDF5 file. You should not
        return the values to the caller; for that, use materialize_attr_values()
    """
    scalar = False
    if np.isscalar(a):
        a = np.array([a])
        scalar = True
    arr = normalize_attr_array(a)
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):
        pass  # We allow all these types
    elif np.issubdtype(arr.dtype, np.character) or np.issubdtype(arr.dtype, np.object_):
        arr = normalize_attr_strings(arr)
    elif np.issubdtype(arr.dtype, np.bool_):
        arr = arr.astype("ubyte")
    if scalar:
        return arr[0]
    else:
        return arr


def materialize_attr_values(a: np.ndarray) -> np.ndarray:
    scalar = False
    if np.isscalar(a):
        scalar = True
        a = np.array([a])
    if np.issubdtype(a.dtype, np.string_) or np.issubdtype(a.dtype, np.object_):
        if hasattr(a, "decode"):
            temp = np.array([x.decode("ascii", "ignore") for x in a])
        else:
            temp = a
        # Then unescape XML entities and convert to unicode
        result = np.array([html.unescape(x) for x in temp.astype(str)], dtype=object)
    elif np.issubdtype(a.dtype, np.str_) or np.issubdtype(a.dtype, np.unicode_):
        result = np.array(a.astype(str), dtype=object)
    else:
        result = a
    if scalar:
        return result[0]
    else:
        return result


def decode_value(value, filt: slice = None):
    """Decode value read from hdf5 file

    Args:
        value: value to decode
        filt: the slice of the value to load

    Returns:
        decoded value
    """
    if filt is None:
        value = np.array(value)
    else:
        value = np.array(value[filt])

    if not value.shape:
        value = value[()]
    value = materialize_attr_values(value)
    return value
