import logging
from typing import Any, Mapping, Union

import h5py

from h5.constants import (
    ASSAYS,
    BARCODE,
    COL_ATTRS,
    DATE_CREATED,
    ID,
    LAYERS,
    METADATA,
    ROW_ATTRS,
    SAMPLE,
    SDK_VERSION,
)
from h5.data import Assay
from h5.data.normalize import decode_value
from h5.exceptions import ValidationError

log = logging.getLogger(__name__)


def check_file(file: Union[h5py.File, Mapping[str, Any]]):
    """Check if file contains all required groups

    Args:
        file: h5 file or a nested dictionary with same structure to check

    Raises:
        ValidationError: if file structure is invalid
    """
    for key in (METADATA,):
        if key not in file:
            raise ValidationError(f'Invalid file (file does not contain group "{key}")')

    metadata = file[METADATA]
    for key in (DATE_CREATED, SDK_VERSION):
        if key not in metadata:
            raise ValidationError(
                f'Invalid file (file does not contain dataset "{METADATA}/{key}")'
            )

    if ASSAYS in file:
        for assay_name in file[ASSAYS].keys():
            try:
                check_assay(file[ASSAYS][assay_name])
            except ValidationError as err:
                raise ValidationError(f"Invalid file ({err})") from err
    else:
        log.warning("File does not contain any assays")


def check_assay(assay: Union[h5py.Group, Assay]):
    """Validate assay structure

    Args:
        assay: h5py.Group or a nested dictionary with same structure to check

    Raises:
        ValidationError: if assay structure is not correct
    """
    if isinstance(assay, Assay):
        assay = AssayWrapper(assay)

    for key in (LAYERS, ROW_ATTRS, COL_ATTRS, METADATA):
        if key not in assay:
            raise ValidationError(f'{assay.name} does not contains group "{key}"')

    if SAMPLE not in assay[METADATA]:
        raise ValidationError(f'{assay.name} does not contain "{METADATA}/{SAMPLE}"')

    samples = decode_value(assay[METADATA][SAMPLE])
    if isinstance(samples, str):
        raise ValidationError(f'"{assay.name}/{METADATA}/{SAMPLE}" is not a list')
    if len(samples.flatten()) != len(set(list(samples.flatten()))):
        raise ValidationError(f"{assay.name} contains duplicate samples")

    for ra in (SAMPLE, BARCODE):
        if ra not in assay[ROW_ATTRS]:
            raise ValidationError(f'{assay.name} does not contain "{ROW_ATTRS}/{ra}"')

    for ca in (ID,):
        if ca not in assay[COL_ATTRS]:
            raise ValidationError(f'{assay.name} does not contain "{COL_ATTRS}/{ca}"')


class AssayWrapper:
    """Wrapper for accessing Assay object as a h5 Group"""

    def __init__(self, assay):
        self.assay = assay

    @property
    def name(self):
        return self.assay.name

    def __iter__(self):
        return iter([COL_ATTRS, LAYERS, METADATA, ROW_ATTRS])

    def __getitem__(self, item):
        if item == COL_ATTRS:
            return self.assay.col_attrs
        elif item == LAYERS:
            return self.assay.layers
        elif item == METADATA:
            return self.assay.metadata
        elif item == ROW_ATTRS:
            return self.assay.row_attrs
        else:
            raise KeyError(item)
