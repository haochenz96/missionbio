import logging
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from typing import List, Tuple, Union

import h5py
import numpy as np

from missionbio.h5.constants import BARCODE, SAMPLE
from missionbio.h5.data import Assay, H5Reader, H5Writer
from missionbio.h5.exceptions import UserError, ValidationError
from missionbio.h5.utils import find_common_keys, is_sorted

log = logging.getLogger(__name__)


def merge_assays(*assays: Assay, inplace: bool = False) -> Tuple[Assay]:
    """Merges assays by keeping the rows with the same barcodes

    Args:
        *assays: assays to merge
        inplace: when True, input assays are modified inplace

    Raises:
        UserError: if assays cannot be merged

    Returns:
        merged assays
    """
    if not inplace:
        assays = deepcopy(assays)

    duplicates = [
        key for key, count in Counter(assay.name for assay in assays).items() if count > 1
    ]
    if duplicates:
        raise UserError(f"Cannot merge multiple assays of same type: {duplicates}")

    # warn when samples do not match
    find_common_keys(
        (assay.samples() for assay in assays),
        warning="Some samples are missing in {name} ({missing_keys})",
        names=assays,
    )

    for assay in assays:
        if BARCODE not in assay.row_attrs:
            raise UserError(f"Cannot merge assay {assay.name}, it does not contain barcodes")
        assay.normalize_barcodes()
        ids = _get_row_ids(assay)
        if not is_sorted(ids):
            log.warning(f"Barcodes in {assay} are not ordered. Rows will be sorted")
            order = sorted(range(len(ids)), key=ids.__getitem__)
            assay.select_rows(order)

    common_barcodes = find_common_keys((_get_row_ids(assay) for assay in assays))
    for assay in assays:
        ids = _get_row_ids(assay)
        keep_rows = np.array([id_ in common_barcodes for id_ in ids])
        assay.select_rows(keep_rows)
        log.info(f"Keeping {len(common_barcodes)}/{len(ids)} barcodes in {assay.name} assay")

    # sanity check, all assays should have the same barcodes after merging
    barcodes = None
    samples = None
    for assay in assays:
        if barcodes is None:
            barcodes = assay.row_attrs[BARCODE]
        else:
            assert len(assay.row_attrs[BARCODE]) == len(barcodes)
            assert (assay.row_attrs[BARCODE] == barcodes).all()

        if samples is None:
            samples = assay.row_attrs.get(SAMPLE)
        else:
            assert (assay.row_attrs[SAMPLE] == samples).all()
    return assays


def _get_row_ids(assay: Assay) -> List[Tuple[str, str]]:
    barcodes = assay.row_attrs[BARCODE]
    if SAMPLE in assay.row_attrs:
        samples = assay.row_attrs[SAMPLE]
    else:
        samples = np.full((len(barcodes),), assay.metadata[SAMPLE])
    return list(zip(samples, barcodes))


def merge_assay_files(
    inputs: List[Union[str, h5py.File]], output: Union[str, h5py.File], force: bool
):
    """Merge all assays from multiple input files

    Args:
        inputs: files to merge
        output: output  file
        force: Rename samples to the name from the first file.
            Requires that all files contain a single sample

    Returns:
        True, if merging was successful, False otherwise
    """
    with ExitStack() as stack:
        readers: List[H5Reader] = []
        for fn in inputs:
            try:
                readers.append(stack.enter_context(H5Reader(fn)))
            except ValidationError as err:
                log.error(f"Could not load {fn}. {err}")
                return False

        if force:
            for reader in readers:
                if len(reader.samples()) != 1:
                    log.error(
                        f"Force flag can only be used when all files contain"
                        f" a single sample. {reader.filename} contains"
                        f" {len(reader.samples())} samples"
                    )
                    return False
        else:
            common_samples = find_common_keys(
                (reader.samples() for reader in readers),
                warning="Some samples are missing in one or more "
                "assays in {name} ({missing_keys}). "
                "Only samples that  are present in all input "
                "files will be present in merged file",
                names=inputs,
            )
            if not common_samples:
                log.error(
                    "There are no samples shared between all assays. "
                    "Select fewer assays, unify sample names using "
                    "`tapestri h5 rename-sample` command or use "
                    "--force flag"
                )
                return False

        def get_first_sample_name(assay):
            for sample in assay.samples():
                return sample

        assays = []
        raw_counts = []
        sample_name = None
        for reader in readers:
            for assay_name in reader.assays():
                log.info(f"Reading {assay_name} assay from {reader.filename}")
                assay = reader.read(assay_name)

                if force:
                    name = get_first_sample_name(assay)
                    if sample_name is None:
                        sample_name = name
                        log.info(f"Found sample {name}, using it for all assays")
                    else:
                        if name != sample_name:
                            log.info(f"Renaming sample {name} to {sample_name}")
                            assay.rename_sample(name, sample_name)
                        else:
                            log.info(f"Keeping sample {name}")

                assays.append(assay)

            for assay_name in reader.raw_counts():
                raw_counts.append(reader.read_raw_counts(assay_name))

        log.info("Merging assays")
        try:
            merged = merge_assays(*assays, inplace=True)
        except UserError as exc:
            log.error(str(exc))
            return False
        except Exception as exc:
            log.exception(exc)
            return False

        for assay in merged:
            if assay.shape[0] == 0:
                log.error("Incompatible assays (no common barcodes)")
                return False

        writer: H5Writer = stack.enter_context(H5Writer(output, mode="w"))
        for assay in merged:
            log.info(f"Writing {assay} to {output}")
            writer.write(assay)

        log.info("Done")

        log.info("Copying all_barcodes assays to merged file")
        for reader in readers:
            for name in reader.raw_counts():
                writer.write_raw_counts(reader.read_raw_counts(name))

        return True
