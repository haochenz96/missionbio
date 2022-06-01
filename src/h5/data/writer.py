"""
Writer for MissionBio hdf5 files
"""
import logging
from datetime import date
from typing import Any, Union

import h5py
import numpy as np

from missionbio.h5 import __version__
from missionbio.h5.constants import (
    ASSAYS,
    COL_ATTRS,
    DATE_CREATED,
    LAYERS,
    METADATA,
    RAW_COUNTS,
    ROW_ATTRS,
    SDK_VERSION,
)
from missionbio.h5.data import Assay
from missionbio.h5.data.normalize import normalize_attr_values
from missionbio.h5.data.validation import check_assay

log = logging.getLogger(__name__)

SUPPORTED_MODES = ("r+", "w-", "w", "a")


class H5Writer:
    """Writer for MissionBio hdf5 files

    Can be used as a context manager:
    ```
    with H5Writer("output.hdf5") as f:
        f.write(dna)
        f.write(cnv)
    ```
    """

    def __init__(self, filename: Union[str, h5py.File], *, mode="w-"):
        """Construct a writer for a new MissionBio hdf5 file

        Args:
            filename: path to the hdf5 file to write to, or an open h5py.File handle
            mode: mode to open the h5 file with
                r+  Read/write, file must exist
                w-  Create file, raise error if file exists (default)
                w   Create file, truncate if exists
                a   Read/write if exists, create otherwise


        Raises:
            ValueError: on invalid mode
        """
        if isinstance(filename, h5py.File):
            self.__file = filename
            mode = self.__file.mode
        else:
            if mode not in SUPPORTED_MODES:
                raise ValueError(f"Invalid mode; must be one of {SUPPORTED_MODES}")
            self.__file = h5py.File(filename, mode=mode)

        if METADATA not in self.__file:
            self.add_file_metadata()

    def write(self, assay: Assay):
        """Write assay to a MissionBio hdf5 file.

        Args:
            assay: assay to export

        Raises:
            ValueError: if file is closed
        """
        if self.__file is None:
            raise ValueError("Cannot write to a closed file")

        check_assay(assay)

        group = self.__file.require_group(ASSAYS).create_group(assay.name)
        self.__write_assay(group, assay)

    def write_raw_counts(self, assay):
        """Write assay to all_barcodes group in the file.

        Args:
            assay: assay to export

        Raises:
            ValueError: if file is closed
        """
        if self.__file is None:
            raise ValueError("Cannot write to a closed file")

        check_assay(assay)

        group = self.__file.require_group(RAW_COUNTS).create_group(assay.name)
        self.__write_assay(group, assay)

    def add_file_metadata(self):
        """Write file metadata to file"""
        metadata = self.__file.require_group(METADATA)
        self.__write_value(metadata, DATE_CREATED, date.today().strftime("%Y-%m-%d"))
        self.__write_value(metadata, SDK_VERSION, __version__)

    def __write_assay(self, parent_group, assay):
        groups = [
            (LAYERS, assay.layers),
            (ROW_ATTRS, assay.row_attrs),
            (COL_ATTRS, assay.col_attrs),
            (METADATA, assay.metadata),
        ]
        for group_name, data in groups:
            group = parent_group.create_group(group_name)
            for key, value in data.items():
                self.__write_value(group, key, value)

    def close(self):
        """Close underlying file"""
        self.__file.close()
        self.__file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__file.close()
        self.__file = None
        return False

    def __write_value(self, group: h5py.Group, name: str, value: np.ndarray):
        """Write value to given group

        Value is normalized, then __write_scalar or __write_array
        is invoked based on the type of the value

        Args:
            group: group to write the array to
            name: name of array
            value: value to write

        Raises:
            ValueError: if value cannot be normalized
        """
        try:
            normalized = normalize_attr_values(value)
        except Exception as ex:
            raise ValueError(f'Could normalize {type(value)}(key "{name}")') from ex

        if np.isscalar(normalized) or normalized.dtype == np.object_:
            group[name] = normalized
        else:
            self.__write_array(group, name, normalized)

    def __write_array(self, group: h5py.Group, name: str, data: np.array):
        """Write array to given group

        Array is stored using 64x64 chunks and compressed

        Args:
            group: group to write the array to
            name: name of array
            data: data to write
        """
        # make sure chunk size is not bigger than actual matrix size
        if hasattr(data, "shape") and len(data.shape) > 1:
            chunks = (min(64, data.shape[0]), min(64, data.shape[1]))
        else:
            chunks = True
        group.create_dataset(
            name, data=data, chunks=chunks, compression="gzip", shuffle=False, compression_opts=2
        )

    def append_group_attr(self, analyte: str, group_name: str, key: str, value: Any):
        """Add new attribute to given group in existing h5 file

        Args:
            analyte: assay type
            group_name: name of group
            key: attribute name
            value: attribute value

        Raises:
            ValueError: if group does not exist in file
        """
        group_path = "/".join(["", ASSAYS, analyte, group_name])
        if group_path not in self.__file:
            raise ValueError("{} missing".format(group_path))
        group = self.__file[group_path]
        if key in group.keys():
            key_path = group_path + "/" + key
            data = self.__file[key_path]
            normalized = normalize_attr_values(value)
            data[...] = normalized
        else:
            self.__write_value(group, key, value)
