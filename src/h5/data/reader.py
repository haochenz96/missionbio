"""
Reader for MissionBio hdf5 files
"""
import logging
from contextlib import ExitStack
from typing import List, Set, Union

import h5py
import numpy as np

from h5.constants import (
    ASSAYS,
    COL_ATTRS,
    DATE_CREATED,
    FILTERED_KEY,
    ID,
    LAYERS,
    METADATA,
    RAW_COUNTS,
    ROW_ATTRS,
    SAMPLE,
    SDK_VERSION,
)
from h5.data import Assay
from h5.data.normalize import decode_value
from h5.data.validation import check_file

log = logging.getLogger(__name__)


class H5Reader:
    """Reader for MissionBio hdf5 files

    Can be used as a context manager:
    ```
    with H5Reader("output.hdf5") as r:
        print(r.assays())
        print(r.samples())
        assay = r.read("dna")
    ```
    """

    def __init__(self, filename: Union[str, h5py.File]):
        """Construct a reader for a MissionBio hdf5 file

        Args:
            filename: path to the hdf5 file to read from, or an open h5py.File handle
        """
        with ExitStack() as stack:
            if isinstance(filename, h5py.File):
                self.__file = filename
            else:
                self.__file = stack.enter_context(h5py.File(filename, "r"))
            self.filename = self.__file.filename

            check_file(self.__file)
            self.metadata = self.__read_metadata()

            # File is valid, will be closed when self.close is called.
            stack.pop_all()

    def assays(self) -> List[str]:
        """Return a list of all assay names from the file

        Returns:
            list of assay keys
        """
        return list(self.__file[ASSAYS].keys())

    def raw_counts(self) -> List[str]:
        """Return a list of all raw counts names from the file

        Returns:
            list of raw counts keys
        """
        if RAW_COUNTS not in self.__file:
            # raw counts are optional
            return []

        return list(self.__file[RAW_COUNTS].keys())

    def samples(self) -> Set[str]:
        """Return a set of sample names present in all samples in this file

        Returns:
            Set[str]
        """
        samples = None
        for assay in self.__file[ASSAYS].values():
            if SAMPLE in assay[METADATA]:
                assay_samples = decode_value(assay[METADATA][SAMPLE])
                if isinstance(assay_samples, str):
                    assay_samples = {assay_samples}
                else:
                    assay_samples = set(np.array(assay_samples).flatten())
            else:
                assay_samples = set()

            if samples is None:
                samples = assay_samples
            else:
                samples &= assay_samples

        return samples

    def read(self, assay_name: str, apply_filter: bool = False, whitelist: list = None) -> Assay:
        """Load assay from file

        Args:
            assay_name: name of the assay to load
            apply_filter: whether only the filtered columns are to be loaded
            whitelist: the column attributes to load even if filtered.

        Raises:
            ValueError: If assay does not exist or 'filtered'
                        is not in the column attributes and
                        apply_filter is given or an element
                        from the whitelist is not found in
                        the columns

        Returns:
            loaded assay
        """
        if assay_name not in self.__file[ASSAYS]:
            raise ValueError(f'Assay "{assay_name}" does not exist')
        assay_group = self.__file[ASSAYS][assay_name]

        filt_list, id_list = None, None
        if apply_filter:
            if FILTERED_KEY not in assay_group[COL_ATTRS]:
                raise ValueError(f"{FILTERED_KEY} not found in the column attributes")
            filt_list = assay_group[COL_ATTRS][FILTERED_KEY][()]
            filt_list = np.where(filt_list == 0)[0]

        if whitelist is not None:
            id_list = assay_group[COL_ATTRS][ID][()]
            id_list = np.where(np.isin(id_list, whitelist))[0]

        if filt_list is None and id_list is None:
            col_filter = slice(None)
        else:
            col_filter = np.array([])
            if filt_list is not None:
                col_filter = np.union1d(col_filter, filt_list)
            if id_list is not None:
                col_filter = np.union1d(col_filter, id_list)

        return self.__read_assay(assay_group, assay_name, col_filter)

    def read_raw_counts(self, assay_name: str) -> Assay:
        """Read all_barcodes assay from file

        Args:
            assay_name: name of the assay to load

        Raises:
            ValueError: If assay does not exist

        Returns:
            loaded assay
        """
        if RAW_COUNTS not in self.__file or assay_name not in self.__file[RAW_COUNTS]:
            raise ValueError(f'Raw counts for "{assay_name}" do not exist')
        raw_counts_group = self.__file[RAW_COUNTS][assay_name]

        return self.__read_assay(raw_counts_group, assay_name)

    def __read_assay(
        self, assay_group: h5py.Group, assay_name: str, col_filter: list = slice(None)
    ):
        assay = Assay.create(assay_name)
        groups = [
            (LAYERS, assay.add_layer, (slice(None), col_filter)),
            (ROW_ATTRS, assay.add_row_attr, None),
            (COL_ATTRS, assay.add_col_attr, col_filter),
            (METADATA, assay.add_metadata, None),
        ]
        for group_name, add, col_fil in groups:
            for key, value in assay_group[group_name].items():
                add(key, decode_value(value, col_fil))
        return assay

    def __read_metadata(self):
        metadata = self.__file[METADATA]
        return {key: decode_value(metadata[key]) for key in {DATE_CREATED, SDK_VERSION}}

    def close(self):
        """Close underlying file"""
        self.__file.close()
        self.__file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
