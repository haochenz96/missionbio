import json
import os
from contextlib import contextmanager
from os.path import dirname, join
from tempfile import NamedTemporaryFile
from typing import Optional

import numpy as np

from h5.data import Assay

TEST_DIR = dirname(__file__)

TEST_VCF = join(TEST_DIR, "500k.vcf.gz")
TEST_METADATA = join(TEST_DIR, "metadata.json")
TEST_DNA_COUNTS = join(TEST_DIR, "dna.counts.tsv")
TEST_PROTEIN_COUNTS = join(TEST_DIR, "protein.counts.tsv")
TEST_RNA_COUNTS = join(TEST_DIR, "rna.counts.tsv")

# datasets for multisample merging
TEST_MERGE_A = join(TEST_DIR, "merge.a.h5")
TEST_MERGE_B = join(TEST_DIR, "merge.b.h5")
TEST_MERGE_C = join(TEST_DIR, "merge.c.h5")
TEST_FILTER_DNA = join(TEST_DIR, "filter.dna.h5")


def sample_metadata() -> dict:
    """Read and return a sample metadata object

    Returns:
        dict with sample metadata
    """
    with open(TEST_METADATA) as f:
        return json.load(f)


@contextmanager
def get_temp_writable_path(suffix: Optional[str] = None, file_exists: bool = False):
    """Return a path to a temporary file

    File is removed when context manager exists.

    Args:
        suffix: when specified, filename will have given suffix
        file_exists: when True, path points to an existing file
                     when False, path points to a non-existing file

    Yields:
        path to the temp file
    """
    tmp_file = NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_file.close()
    if not file_exists:
        os.remove(tmp_file.name)

    try:
        yield tmp_file.name
    finally:
        try:
            os.remove(tmp_file.name)
        except OSError:
            # file has already been removed
            pass


def dummy_assay(assay_name, size=3):
    return Assay(
        name=assay_name,
        layers={"data": np.zeros((size, size))},
        row_attrs={
            "barcode": np.array(list("CTGA"))[:size],
            "sample_name": np.array(["sample"] * size),
        },
        col_attrs={"id": np.arange(size)},
        metadata={"sample_name": np.array(["sample"])},
    )
