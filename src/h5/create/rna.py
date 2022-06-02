from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from h5.constants import BARCODE, DEFAULT_SAMPLE, ID, RNA_ASSAY, SAMPLE
from h5.data import Assay


def create_rna_assay(counts_tsv: str, *, metadata: Optional[Dict[str, Any]] = None) -> Assay:
    """Create rna assay from read count tsv file

    The first column in the tsv file should contain barcodes,
    the first row should contain amplicon names

    Args:
        counts_tsv: path to the read_counts file
        metadata: optional assay metadata

    Returns:
        rna assay
    """
    if metadata is None:
        metadata = {}

    assay = Assay.create(RNA_ASSAY)

    counts = pd.read_csv(counts_tsv, sep="\t", header=0, index_col=0)
    counts.sort_index(inplace=True)

    assay.add_layer("read_counts", counts.values)

    assay.add_row_attr(BARCODE, counts.index.values)
    assay.add_col_attr(ID, counts.columns.values)

    for name, value in metadata.items():
        assay.add_metadata(name, value)

    sample = metadata.get(SAMPLE, DEFAULT_SAMPLE)
    assay.add_row_attr(SAMPLE, np.array([sample] * assay.shape[0]))

    return assay
