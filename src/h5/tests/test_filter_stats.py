from unittest import TestCase

import h5py

from h5.constants import (
    ASSAYS,
    COL_ATTRS,
    DNA_ASSAY,
    FILTER_MASK,
    LAYERS,
    METADATA,
    ROW_ATTRS,
)
from h5.data import H5Reader, H5Writer
from h5.filter import DefaultFilter, stats
from h5.tests.base import TEST_FILTER_DNA, get_temp_writable_path

FILTERED_KEY = "filtered"


class FilterStatsTests(TestCase):
    def test_collect(self):

        with H5Reader(TEST_FILTER_DNA) as reader:
            dna = reader.read(DNA_ASSAY)
            filtered_metrics = stats.collect(dna, DefaultFilter)
        self.assertEqual(filtered_metrics.PASSING_CELLS.sum(), 10)
        self.assertEqual(filtered_metrics.PASSING_VARIANTS.sum(), 2)
        self.assertEqual(filtered_metrics.N_VARIANTS_PER_CELL, 11)
        self.assertEqual(filtered_metrics.N_PASSING_VARIANTS_PER_CELL, 2)

    def test_add_metrics(self):
        with H5Reader(TEST_FILTER_DNA) as reader:
            dna = reader.read(DNA_ASSAY)
            filtered_metrics = stats.collect(dna, DefaultFilter)

        with get_temp_writable_path() as filename:
            with H5Writer(filename) as writer:
                writer.write(dna)
                stats.add_metrics(writer, filtered_metrics)

            with h5py.File(filename, "r") as f:
                self.assertIn(FILTER_MASK, f[ASSAYS][DNA_ASSAY][LAYERS])
                self.assertIn(FILTERED_KEY, f[ASSAYS][DNA_ASSAY][ROW_ATTRS])
                self.assertIn(FILTERED_KEY, f[ASSAYS][DNA_ASSAY][COL_ATTRS])

                for key in DefaultFilter._asdict().keys():
                    self.assertIn(key, f[ASSAYS][DNA_ASSAY][METADATA])
