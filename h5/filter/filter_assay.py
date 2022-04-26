import logging
from typing import Iterable, NamedTuple, Tuple

import numpy as np
from h5py import Dataset

from missionbio.h5.constants import AF, DP, GQ, NGT
from missionbio.h5.data import Assay

log = logging.getLogger(__name__)


class FilteredAssay(NamedTuple):
    """
    Internal representation to store variant filtering output
    """

    passing_variants: np.ndarray
    passing_cells: np.ndarray
    keep_mask: np.ndarray

    def __str__(self):
        n_passing_variants = self.passing_variants.sum()
        n_passing_variants_pct = n_passing_variants * 100 / self.passing_variants.size
        n_passing_cells = self.passing_cells.sum()
        n_passing_cells_pct = n_passing_cells * 100 / self.passing_cells.size
        return (
            f"n_passing_variants = {n_passing_variants} ({n_passing_variants_pct:.2f} %), "
            f"n_passing_cells = {n_passing_cells} ({n_passing_cells_pct:.2f} %)"
        )


class FilterConfig(NamedTuple):
    """
    Class representation for filtering thresholds
    """

    gq_cutoff: int
    dp_cutoff: float
    af_cutoff: float
    missing_cells_cutoff: float
    missing_variants_cutoff: float
    mutated_cells_cutoff: float


# Thresholds for default filtering
DefaultFilter = FilterConfig(
    gq_cutoff=30,
    dp_cutoff=10,
    af_cutoff=20,
    missing_cells_cutoff=50,
    missing_variants_cutoff=50,
    mutated_cells_cutoff=1,
)


def mutated_in_x_cells(assay: Assay, mutated_threshold: float = 1, batch_size=256,) -> np.ndarray:
    """
    Identifies variants mutated in x% cells,
    specified with mutated_thresholds

    Args:
        assay: Assay object
        mutated_threshold: Mutation threshold which variant must pass to be loaded.
            Values are in percentages (1% default). If whitelist_only
            is set to True, this value is ignored.
        batch_size: Number of variants which should be loaded at the same time
            then scanning NGT layer

    Returns:
        Variants mask mutated in x% cells
    """

    n_cells, n_variants = assay.shape

    index = np.zeros((n_variants,), dtype=np.bool)
    for part, ngt in _scan(assay.layers[NGT], batch_size=batch_size):
        part_mask = False
        has_mutation = ((ngt == 1) | (ngt == 2)).sum(axis=0)
        mutated_mask = (has_mutation / n_cells) > (mutated_threshold / 100)

        part_mask = part_mask | mutated_mask
        index[part] = part_mask

    return index


def _scan(layer: Dataset, batch_size=256) -> Iterable[Tuple[slice, np.ndarray]]:
    """
    Scans layer.

    Args:
        layer: Dataset to be sacnned
        batch_size: Number of columns to be loaded at the same time

    Yields:
        tuples of batch slice and loaded data in numpy array
    """
    _, columns = layer.shape
    batches = (
        slice(i * batch_size, (i + 1) * batch_size) for i in range(columns // batch_size + 1)
    )

    for part in batches:
        yield part, layer[:, part]


def filter_dna(assay: Assay, config: FilterConfig, mutated_variants: np.ndarray) -> FilteredAssay:
    """Identify cells and variants passing variant quality thresholds

    Args:
        assay: DNA assay object
        config: object containing filtering thresholds
        mutated_variants: mask for mutated variants

    Returns:
        FilteredAssay
    """

    # mutated NGT
    ngt = assay.layers[NGT][:, mutated_variants]
    shape = ngt.shape
    ngt = ngt.ravel()
    ngt_mutated = (ngt == 1) | (ngt == 2)

    log.debug("Applying NGT filters")
    # ngt_filter
    ngt_filter = compute_ngt_filter(assay, config, mutated_variants)
    ngt_filter = ngt_filter.reshape(shape)

    # filter variants passing mc filter
    kept_variants = mc_filter(ngt_filter, shape, config.missing_cells_cutoff)

    # filter cells passing mv filter
    kept_cells = mv_filter(ngt_filter, kept_variants, config.missing_variants_cutoff)

    # Filter cells/variants with mm filter, to retain mutated data
    fa = mm_filter(
        ngt_filter,
        ngt_mutated,
        mutated_variants,
        kept_cells,
        kept_variants,
        shape,
        config.mutated_cells_cutoff,
    )
    log.info(f"Filtered Assay: {fa}")
    return fa


def mc_filter(ngt_filter: np.ndarray, shape: tuple, missing_cells_cutoff: int) -> np.ndarray:
    """Variants missing from fraction of cells greater than cutoff are removed

    Args:
        ngt_filter: mutations passing quality thresholds
        shape: ngt shape
        missing_cells_cutoff: threshold for missing cells

    Returns:
        bool array marking variants passing filter
    """
    log.debug("Applying MC filter")
    cells, _ = shape
    removed = ngt_filter
    mc_values = removed.sum(axis=0) / cells * 100
    kept_variants = mc_values >= missing_cells_cutoff
    return kept_variants


def mv_filter(
    ngt_filter: np.ndarray, kept_variants: np.ndarray, missing_variants_cutoff: int
) -> np.ndarray:
    """Cell having fraction of mutations less than cutoff are removed

    Args:
        ngt_filter: mutations passing quality thresholds
        kept_variants: variants passing mc filter
        missing_variants_cutoff: threshold for missing variants

    Returns:
        bool array marking cells passing filter
    """
    log.debug("Applying MV filter")
    removed = ngt_filter[:, kept_variants]
    _, variants = removed.shape
    mv_values = removed.sum(axis=1) / (variants or 1) * 100
    kept_cells = mv_values >= missing_variants_cutoff
    return kept_cells


def mm_filter(
    ngt_filter: np.ndarray,
    ngt_mutated: np.ndarray,
    mutated_variants: np.ndarray,
    kept_cells: np.ndarray,
    kept_variants: np.ndarray,
    shape: tuple,
    mutated_cells_cutoff: int,
) -> FilteredAssay:
    """Variants with mutation rate per cell less than cutoff are removed

    Args:
        ngt_filter: mutations passing quality thresholds
        ngt_mutated: mutated ngt mask
        mutated_variants: mutated variants mask
        kept_cells: cells passing mv filter
        kept_variants: variants passing mc filter
        shape: ngt shape
        mutated_cells_cutoff: threshold for mutated cells

    Returns:
        filter specs
    """
    log.debug("Applying MM filter")
    ngt_mutated &= ngt_filter.flatten()
    ngt_mutated = ngt_mutated.reshape(shape)
    ngt_mutated = ngt_mutated[kept_cells, :][:, kept_variants]
    cells, _ = ngt_mutated.shape
    mm_values = ngt_mutated.sum(axis=0) / (cells or 1) * 100
    kept_variants[kept_variants] = mm_values >= mutated_cells_cutoff
    passing_variants = mutated_variants.copy()
    passing_variants[np.where(passing_variants)] = kept_variants

    return FilteredAssay(
        passing_variants=passing_variants, passing_cells=kept_cells, keep_mask=ngt_filter
    )


def compute_ngt_filter(assay: Assay, config: FilterConfig, columns: np.ndarray) -> np.ndarray:
    """Find NGT passing variant quality thresholds

    Args:
        assay: Assay object
        config: filtering thresholds
        columns: mask for mutated variants

    Returns:
        mask for NGT values based on variant qualities
    """
    dp = assay.layers[DP][:, columns].ravel()
    af = assay.layers[AF][:, columns].ravel()

    gq = assay.layers[GQ][:, columns].ravel()

    ngt = assay.layers[NGT][:, columns].ravel()

    return (
        (ngt < 3)
        & (gq >= config.gq_cutoff)
        & (dp >= config.dp_cutoff)
        & ((af >= config.af_cutoff) | (ngt == 0))
    )
