"""
In-memory representation of an assay
"""
import re
from typing import Any, Dict, Set

import numpy as np

from h5.constants import BARCODE, SAMPLE
from h5.exceptions import UserError

__all__ = ["Assay"]


VALID_BARCODE = re.compile("[ACGT]+")


class Assay:
    """An in-memory representation of a missionbio data assay"""

    def __init__(
        self,
        *,
        name: str,
        metadata: Dict[str, Any],
        layers: Dict[str, Any],
        row_attrs: Dict[str, Any],
        col_attrs: Dict[str, Any],
    ):
        """Create a new data assay

        Args:
            name: assay name
            metadata: assay metadata
            layers: data layers
            row_attrs: row metadata
            col_attrs: column metadata
        """
        self.name = name
        self.shape = None
        self.metadata = {}
        self.layers = {}
        self.row_attrs = {}
        self.col_attrs = {}

        # metadata is added first since it can be updated in add_row_attrs
        for name, value in metadata.items():
            self.add_metadata(name, value)

        # layers are added before row/col attrs as they define the assay shape
        for name, value in layers.items():
            self.add_layer(name, value)

        for name, value in row_attrs.items():
            self.add_row_attr(name, value)

        for name, value in col_attrs.items():
            self.add_col_attr(name, value)

    def add_layer(self, name: str, array: np.ndarray):
        """Add layer to the assay

        All layers need to have the same dimension

        Args:
            name: name of the attribute
            array: array with layer values

        Raises:
            ValueError: if array has different shape than existing layers
        """
        if self.shape is None:
            self.shape = array.shape

        if array.shape != self.shape:
            raise ValueError(
                f"All layers should have the same shape. {array.shape} != {self.shape}"
            )

        self.layers[name] = array

    def add_row_attr(self, name: str, value: np.ndarray):
        """Add row attribute to the assay

        Verifies that the value has expected number of elements

        Args:
            name: name of the attribute
            value: array with a value for each row in the assay

        Raises:
            ValueError: if value does not have the same number of elements
                        as number of rows in the assay
        """
        if value.shape[0] != self.shape[0]:
            raise ValueError(
                f"Row annotations should have {self.shape[0]} elements "
                f"(got array with {value.shape})"
            )
        self.row_attrs[name] = value

        if name == SAMPLE:
            _, idx = np.unique(value, return_index=True)
            unique_by_appearance = value[np.sort(idx)]
            self.add_metadata(SAMPLE, unique_by_appearance[:, np.newaxis])

    def add_col_attr(self, name: str, value: np.ndarray):
        """Add column attribute to the assay

        Verifies that the value has expected number of elements

        Args:
            name: name of the attribute
            value: array with a value for each column in the assay

        Raises:
            ValueError: if value does not have the same number of elements
                        as number of rows in the assay
        """
        if value.shape[0] != self.shape[1]:
            raise ValueError(
                f"Columns attributes should have {self.shape[1]} elements "
                f"(got array with {value.shape})"
            )
        self.col_attrs[name] = value

    def add_metadata(self, name: str, value: Any):
        """Add metadata to the assay

        Args:
            name: metadata name
            value: metadata value
        """
        self.metadata[name] = value

    def samples(self) -> Set[str]:
        """Return set of all samples in this assay

        Returns:
            set of sample names
        """
        samples = self.metadata[SAMPLE]
        if isinstance(samples, str):
            samples = [samples]
        elif isinstance(samples, np.ndarray):
            samples = samples.flatten()
        return set(samples)

    @classmethod
    def create(cls, assay_name: str) -> "Assay":
        """Create a new empty data assay

        Args:
            assay_name: assay name

        Returns:
            new assay
        """
        return cls(name=assay_name, metadata={}, layers={}, row_attrs={}, col_attrs={})

    def select_columns(self, selection: np.ndarray):
        """Filter/sort assay inplace, keeping columns in selection

        Args:
            selection: array with selection. Used for indexing layers/col_attrs
        """
        for key, value in self.layers.items():
            value = np.array(value)
            self.layers[key] = value[:, selection]
            self.shape = self.layers[key].shape
        for key, value in self.col_attrs.items():
            value = np.array(value)
            self.col_attrs[key] = value[selection]

    def select_rows(self, selection: np.ndarray):
        """Filter/sort assay inplace, keeping rows in selection

        Args:
            selection: array with selection. Used for indexing layers/row_attrs
        """
        for key, value in self.layers.items():
            self.layers[key] = value[selection, :]
            self.shape = self.layers[key].shape
        for key, value in self.row_attrs.items():
            self.row_attrs[key] = value[selection]

    def rename_sample(self, old_name: str, new_name: str):
        """Rename sample to a new_name

        Args:
            old_name: name of the sample to rename
            new_name: new sample name

        Raises:
            ValueError: when old_sample does not exist or new_sample already exists
        """
        samples = self.samples()
        if old_name not in samples:
            raise ValueError(f"Sample {old_name} does not exist in {self}")
        if new_name in samples:
            raise ValueError(f"Sample {new_name} already exists in {self}")

        if SAMPLE in self.row_attrs:
            row_sample = self.row_attrs[SAMPLE]
            row_sample[row_sample == old_name] = new_name

        self.metadata[SAMPLE][self.metadata[SAMPLE] == old_name] = new_name

    def normalize_barcodes(self):
        """Remove all but [^ACGT] from barcodes

        This is needed since some pipelines add tube suffix to the barcodes
        """
        normalized_barcodes = [VALID_BARCODE.search(bc).group(0) for bc in self.row_attrs[BARCODE]]

        self.__ensure_no_duplicates(normalized_barcodes)
        self.row_attrs[BARCODE][:] = normalized_barcodes

    def __ensure_no_duplicates(self, normalized_barcodes):
        if SAMPLE in self.row_attrs:
            normalized_barcodes = list(zip(self.row_attrs[SAMPLE], normalized_barcodes))

        if len(normalized_barcodes) != len(set(normalized_barcodes)):
            raise UserError(f"Normalized barcodes in {self} are not unique")

    def __str__(self):
        for _, layer in self.layers.items():
            n_rows, n_columns = layer.shape
            break
        else:
            n_rows, n_columns = "no", "no"

        return (
            f"{self.name} assay with {len(self.layers)} "
            f"layer{'s' if len(self.layers) != 1 else ''}, "
            f"{n_rows} rows and {n_columns} columns"
        )
