import logging
from math import ceil
from typing import NamedTuple, Optional

import numpy as np

from missionbio.h5.constants import (
    COL_ATTRS,
    DNA_ASSAY,
    FILTER_MASK,
    FILTERED_KEY,
    LAYERS,
    METADATA,
    N_PASSING_CELLS,
    N_PASSING_VARIANTS,
    N_PASSING_VARIANTS_PER_CELL,
    N_VARIANTS_PER_CELL,
    NGT,
    ROW_ATTRS,
)
from missionbio.h5.data import Assay, H5Reader, H5Writer
from missionbio.h5.filter import FilterConfig, filter_dna, mutated_in_x_cells

log = logging.getLogger(__name__)


class FilteredMetrics(NamedTuple):
    """
    Class representation to store variant filtering metrics
    """

    PASSING_CELLS: int
    PASSING_VARIANTS: int
    N_VARIANTS_PER_CELL: int
    N_PASSING_VARIANTS_PER_CELL: int
    FILTER_MASK: np.ndarray
    CONFIG: FilterConfig


def add_variant_stats(filepath: str, config: FilterConfig):
    """Filter variants and calculate metrics

    Args:
        filepath: path to single sample h5 file
        config: Variants quality thresholds config

    Raises:
        NotImplementedError: when file contains more than one sample
    """
    filtered_metrics = None
    with H5Reader(filepath) as reader:
        n_samples = len(reader.samples())
        if n_samples > 1:
            raise NotImplementedError("Filtering from multisample data is not supported")
        dna = reader.read(DNA_ASSAY)
        filtered_metrics = collect(dna, config)

    # override existing filtering data
    with H5Writer(filepath, mode="r+") as writer:
        add_metrics(writer, filtered_metrics)


def collect(dna: Assay, config: FilterConfig) -> FilteredMetrics:
    """Filter variants and calculate metrics

    Args:
        dna: DNA Assay object
        config: Variants quality thresholds

    Returns:
        Filtering metrics
    """
    log.info(
        "Filtering data mutated in >{} percent cells \
        ".format(
            config.mutated_cells_cutoff
        )
    )
    mutated_variants = mutated_in_x_cells(dna, config.mutated_cells_cutoff)
    filtered_dna = filter_dna(dna, config, mutated_variants)
    n_variants_per_cell = n_variant_per_cell(dna)

    passing_cells = filtered_dna.passing_cells
    passing_variants = filtered_dna.passing_variants
    keep_mask = filtered_dna.keep_mask
    shape = dna.shape

    filter_mask = np.zeros(shape=shape, dtype=np.bool)
    filter_mask[:, mutated_variants] = ~keep_mask

    n_passing_variants_per_cell = n_variant_per_cell(dna, passing_cells, passing_variants)

    return FilteredMetrics(
        passing_cells,
        passing_variants,
        n_variants_per_cell,
        n_passing_variants_per_cell,
        filter_mask,
        config,
    )


def add_metrics(writer: H5Writer, filtered_metrics: FilteredMetrics):
    """Add metrics to Multiomics file

    Args:
        writer: writer for h5 file
        filtered_metrics: filtered data and metrics
    """
    log.info("Adding filtering metrics to multiomics file")
    passing_cells = filtered_metrics.PASSING_CELLS
    passing_variants = filtered_metrics.PASSING_VARIANTS
    filter_mask = filtered_metrics.FILTER_MASK

    # filtered mask for low quality reads, can be used to mark ngt as missing
    writer.append_group_attr(DNA_ASSAY, LAYERS, FILTER_MASK, filter_mask)
    writer.append_group_attr(DNA_ASSAY, ROW_ATTRS, FILTERED_KEY, ~passing_cells)
    writer.append_group_attr(DNA_ASSAY, COL_ATTRS, FILTERED_KEY, ~passing_variants)

    # add filtering stats
    writer.append_group_attr(DNA_ASSAY, METADATA, N_PASSING_VARIANTS, passing_variants.sum())
    writer.append_group_attr(DNA_ASSAY, METADATA, N_PASSING_CELLS, passing_cells.sum())
    writer.append_group_attr(
        DNA_ASSAY, METADATA, N_VARIANTS_PER_CELL, filtered_metrics.N_VARIANTS_PER_CELL
    )
    writer.append_group_attr(
        DNA_ASSAY,
        METADATA,
        N_PASSING_VARIANTS_PER_CELL,
        filtered_metrics.N_PASSING_VARIANTS_PER_CELL,
    )

    # Write filtering thresholds
    for k, v in filtered_metrics.CONFIG._asdict().items():
        writer.append_group_attr(DNA_ASSAY, METADATA, k, v)

    log.info("Finished writing metrics")


def n_variant_per_cell(
    assay: Assay, cells: Optional[np.ndarray] = None, variants: Optional[np.ndarray] = None
):
    """Compute median of mutations per cell for a given assay

    Args:
        assay: DNA Assay object
        cells: passing cells (or None)
        variants: passing variants (or None)

    Returns:
        median number of mutated variants per cell
    """

    if cells is not None and variants is not None:
        ngt = assay.layers[NGT][cells, :][:, variants]
    else:
        ngt = assay.layers[NGT]
    has_mutation = ((ngt == 1) | (ngt == 2)).sum(axis=1)
    return ceil(np.median(has_mutation))
