import logging
from typing import Optional

import numpy as np
import pandas as pd

from h5.constants import BARCODE, DEFAULT_SAMPLE, ID, PROTEIN_ASSAY, SAMPLE
from h5.data import Assay

log = logging.getLogger(__name__)


def create_protein_assay(counts_tsv: str, metadata: Optional[dict] = None) -> Assay:
    """Create protein assay from read count tsv file

    counts_tsv should be a "sparse" file with the following columns:
    barcode
        cell barcode
    antibody
        description of the antibody
    raw
        number of reads for combination of barcode and antibody

    Args:
        counts_tsv: path to the read_counts file
        metadata: optional assay metadata

    Returns:
        protein assay
    """
    if metadata is None:
        metadata = {}

    assay = Assay.create(PROTEIN_ASSAY)

    log.info("Reading read counts file")
    counts = pd.read_csv(
        counts_tsv, sep="\t", header=0, index_col=False, names=["barcode", "antibody", "raw"]
    ).pivot_table(index="barcode", columns="antibody", values=["raw"], fill_value=0)

    log.info("Creating layers")
    assay.add_layer("read_counts", counts["raw"].values)

    log.info("Adding cell attributes")
    assay.add_row_attr(BARCODE, counts.index.values)
    log.info("Adding antibody attributes")
    assay.add_col_attr(ID, counts.columns.levels[1].values)

    log.info("Adding metadata")
    for name, value in metadata.items():
        assay.add_metadata(name, value)

    sample = metadata.get(SAMPLE, DEFAULT_SAMPLE)
    assay.add_row_attr(SAMPLE, np.array([sample] * assay.shape[0]))

    return assay
