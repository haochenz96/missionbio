import logging
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np

from missionbio.h5.constants import CHROM, DNA_ASSAY, DP, GQ, ID, NGT, POS, SAMPLE
from missionbio.h5.data import Assay
from missionbio.h5.utils import find_common_keys, is_sorted

log = logging.getLogger(__name__)


def merge_samples(*assays: Assay, inplace: bool = False) -> Assay:
    """Merges multiple assays of the same kind by concatenating rows
    and imputing missing columns

    When inplace is set to True, layer data will be removed from input
    assays while it is being added to the merged assay. This way memory
    consumption is kept low.

    Args:
        assays: assays to merge
        inplace: if True, missing columns will be added to the input assays

    Returns:
        merged assay
    """
    if not inplace:
        assays = deepcopy(assays)

    assay_name = check_assay_name(assays)

    for assay in assays:
        ensure_sample_row_attrs(assay)
    col_attrs, matching_ids = merge_col_attrs(assays)
    row_attrs = merge_row_attrs(assays)
    metadata = merge_metadata(assays)
    layers = merge_layers(assays, col_attrs, matching_ids)

    merged_assay = Assay.create(assay_name)

    for key, data in layers.items():
        merged_assay.add_layer(key, data)

    for key, data in col_attrs.items():
        merged_assay.add_col_attr(key, data)

    for key, data in row_attrs.items():
        merged_assay.add_row_attr(key, data)

    for key, data in metadata.items():
        merged_assay.add_metadata(key, data)

    return merged_assay


def check_assay_name(assays: List[Assay]) -> str:
    """Get assay name of all assays.

    Args:
        assays: assays to check

    Raises:
        ValueError: if all assays do not share the same name

    Returns:
        name of the assay
    """
    assay_name = None
    for assay in assays:
        if assay_name is None:
            assay_name = assay.name
        elif assay.name != assay_name:
            raise ValueError(
                f"Cannot merge samples from different assays. {assay.name} != {assay_name}"
            )
    return assay_name


def ensure_sample_row_attrs(assay: Assay):
    """Ensure that assay contains SAMPLE row annotation

    If assay contains a single sample, row annotation is created.
    If it contains multiple samples, an exception is raised.

    Args:
        assay: assay to check

    Raises:
        ValueError: if multi-sample assay does not contain sample_name row attr
    """
    if SAMPLE in assay.row_attrs:
        return

    if len(assay.samples()) > 1:
        raise ValueError(
            f"Assay {assay.name} contains multiple samples,"
            f"but does not define {SAMPLE} row_attr"
        )

    assay.row_attrs[SAMPLE] = np.full((assay.shape[0],), assay.metadata[SAMPLE])


def merge_col_attrs(assays: List[Assay]) -> Tuple[Dict[str, np.array], List[np.array]]:
    """Get a merged dict of column attributes common to all assays

    Args:
        assays: list of assays to merge

    Raises:
        ValueError: if assays does not contain column ids

    Returns:
        merged attributes and their matching ids
    """
    all_ids = set()
    for assay in assays:
        if ID not in assay.col_attrs:
            raise ValueError(f"Cannot merge {assay}, it does not contain column ids")

        ids = assay.col_attrs[ID]
        if not is_sorted(ids):
            log.warning(f"Features in {assay} are not ordered. Columns will be sorted")
            order = np.argsort(ids)
            assay.select_columns(order)

        all_ids.update(ids)

    all_ids = np.array(sorted(all_ids))

    common_attrs = find_common_keys(
        (assay.col_attrs for assay in assays),
        warning="Some col_attrs are missing in {name} ({missing_keys})",
        names=[assay.metadata[SAMPLE] for assay in assays],
    )

    merged_attrs = {}
    matching_ids = []
    for attr in common_attrs:
        dtypes = [assay.col_attrs[attr].dtype for assay in assays]
        dtype = np.find_common_type(dtypes, [])
        if dtype is None:
            raise ValueError(f"Could not merge col_attrs {attr}. Unable to unify dtypes ({dtypes})")
        merged_attrs[attr] = np.empty(all_ids.shape, dtype)
        for assay in assays:
            data = assay.col_attrs[attr]

            ca = set(assay.col_attrs[ID])
            matching = np.array([id_ in ca for id_ in all_ids])
            merged_attrs[attr][matching] = data
            matching_ids.append(matching)

    for assay, matching in zip(assays, matching_ids):
        log.info(
            f"{matching.sum()}/{len(all_ids)} columns found in "
            f"{assay.metadata.get(SAMPLE, 'unknown')}"
        )

    return merged_attrs, matching_ids


def merge_row_attrs(assays: List[Assay]) -> Dict[str, Any]:
    """Get a merged dict with row attributes common to all assays

    Args:
        assays: list of assays to merge

    Returns:
        dict with merged row attributes
    """

    common_attrs = find_common_keys(
        (assay.row_attrs for assay in assays),
        warning="Some row_attrs are missing in {name} ({missing_keys})",
        names=[assay.metadata[SAMPLE] for assay in assays],
    )

    merged_attrs = {}
    for attr in common_attrs:
        merged_attrs[attr] = np.hstack([np.array(assay.row_attrs[attr]).T for assay in assays]).T
    return merged_attrs


def merge_metadata(assays: List[Assay]) -> Dict[str, Any]:
    """Get a merged dict with metadata common to all assays

    Args:
        assays: list of assays to merge

    Returns:
        merged metadata
    """
    common_metadata = find_common_keys(
        (assay.metadata for assay in assays),
        warning="Some metadata entries are missing in {name} ({missing_keys})",
        names=[assay.metadata[SAMPLE] for assay in assays],
    )

    merged_metadata = {}
    for attr in common_metadata:
        merged_metadata[attr] = np.row_stack([assay.metadata[attr] for assay in assays])
    return merged_metadata


def merge_layers(
    assays: List[Assay], col_attrs: Dict[str, Any], matching_ids: List[np.array]
) -> Dict[str, np.array]:
    """Get a merged dict with layers common to all assays

    Args:
        assays: assays to merge
        col_attrs: merged column attributes
        matching_ids: list of bool arrays with matching column ids for each sample

    Returns:
        merged layers
    """
    common_layers = find_common_keys(
        (assay.layers for assay in assays),
        warning="Some layers are missing in {name} ({missing_keys})",
        names=[assay.metadata[SAMPLE] for assay in assays],
    )

    imputers = {assay: Imputer.create(assay, col_attrs) for assay in assays}

    merged_layers = {}
    for layer in common_layers:
        log.info(f"Merging data for layer {layer}")
        merged_data = []
        for assay, matching in zip(assays, matching_ids):
            layer_data = assay.layers[layer]
            expanded_layer = np.zeros(
                (layer_data.shape[0], len(col_attrs[ID])), dtype=layer_data.dtype
            )

            expanded_layer[:, matching] = layer_data

            missing_ids = np.where(~matching)[0]
            for id_ in missing_ids:
                expanded_layer[:, id_] = imputers[assay].impute(layer, col_attrs[ID][id_])
            del assay.layers[layer]

            merged_data.append(expanded_layer)
        merged_layers[layer] = np.row_stack(merged_data)
    return merged_layers


class Imputer:
    @classmethod
    def create(cls, assay, *args):
        if assay.name == DNA_ASSAY:
            return DnaImputer(assay, *args)
        else:
            return DefaultImputer(assay, *args)


class DefaultImputer(Imputer):
    def __init__(self, assay, *args):
        self.assay = assay

    def impute(self, layer, ids):
        return 0


class DnaImputer(Imputer):
    def __init__(self, assay, merged_col_attrs):
        self.assay = assay
        self.id_metadata = {
            merged_col_attrs[ID][i]: {attr: merged_col_attrs[attr][i] for attr in merged_col_attrs}
            for i, _ in enumerate(merged_col_attrs[ID])
        }
        self.regions = {}

        for i, (id_, chr, pos) in enumerate(
            zip(assay.col_attrs[ID], assay.col_attrs[CHROM], assay.col_attrs[POS])
        ):
            self.regions.setdefault((chr, pos), [i, i])
            self.regions[(chr, pos)][1] = i + 1

    def impute(self, layer, id_):
        md = self.id_metadata[id_]
        chr, pos = md[CHROM], md[POS]

        if (chr, pos) in self.regions:
            first, last = self.regions[(chr, pos)]
            source_data = self.assay.layers[layer][:, first:last]
        else:
            source_data = None

        if source_data is None:
            return np.nan

        if layer == DP:
            return source_data.mean(axis=1)
        elif layer == GQ:
            return source_data.mean(axis=1)
        elif layer == NGT:
            # if any of other variants at same pos is non-ref, return
            # 3, otherwise return 0
            return (source_data.sum(axis=1) > 0) * 3
        else:
            return 0
