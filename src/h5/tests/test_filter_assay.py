from unittest import TestCase

import numpy as np

from h5.constants import DNA_ASSAY
from h5.data import H5Reader
from h5.filter import DefaultFilter, filter_assay, filter_dna, mutated_in_x_cells
from h5.tests.base import TEST_FILTER_DNA

CONFIG = filter_assay.FilterConfig(0, 0, 0, 80, 90, 1)

TEST_NGT_FILTER = np.array(
    [
        [True, True, False, False, True],
        [True, True, True, False, True],
        [True, True, True, True, True],
        [True, False, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, False],
    ]
)

TEST_KEPT_CELLS = np.array([False, True, True, False, True, False])
TEST_KEPT_VARIANTS = np.array([True, True, True, False, True])


class FilterAssayTests(TestCase):
    def test_mutated_in_x_cells(self):
        with H5Reader(TEST_FILTER_DNA) as reader:
            dna = reader.read(DNA_ASSAY)
            mutated = mutated_in_x_cells(dna, DefaultFilter.mutated_cells_cutoff)
            self.assertEqual(mutated.sum(), 11)

    def test_filter_dna(self):
        with H5Reader(TEST_FILTER_DNA) as reader:
            dna = reader.read(DNA_ASSAY)
            mutated = mutated_in_x_cells(dna, DefaultFilter.mutated_cells_cutoff)
            filtered_dna = filter_dna(dna, DefaultFilter, mutated)
            self.assertEqual(filtered_dna.passing_cells.sum(), 10)
            self.assertEqual(filtered_dna.passing_variants.sum(), 2)

    def test_compute_ngt_gilter(self):
        with H5Reader(TEST_FILTER_DNA) as reader:
            # custom config
            config = filter_assay.FilterConfig(50, 0, 0, 0, 0, 1)
            dna = reader.read(DNA_ASSAY)
            mutated_variants = mutated_in_x_cells(dna, config.mutated_cells_cutoff)
            self.assertTrue(mutated_variants.all())  # all variants mutated

            ngt_filter = filter_assay.compute_ngt_filter(dna, config, mutated_variants)

            gq_filter = dna.layers["GQ"].ravel() > config.gq_cutoff

            self.assertTrue((ngt_filter == gq_filter).all())
            self.assertEqual(ngt_filter.sum(), 100)

    def test_mc_filter(self):
        kept_variants = filter_assay.mc_filter(
            TEST_NGT_FILTER, TEST_NGT_FILTER.shape, CONFIG.missing_cells_cutoff
        )

        self.assertEqual(kept_variants.sum(), 4)
        self.assertTrue((TEST_KEPT_VARIANTS == kept_variants).all())

    def test_mv_filter(self):
        kept_cells = filter_assay.mv_filter(
            TEST_NGT_FILTER, TEST_KEPT_VARIANTS, CONFIG.missing_variants_cutoff
        )

        self.assertEqual(kept_cells.sum(), 3)
        self.assertTrue((TEST_KEPT_CELLS == kept_cells).all())

    def test_mm_filter(self):
        MUTATED = np.ones(TEST_NGT_FILTER.shape[1], dtype=np.bool)
        NGT_MUTATED = np.ones(TEST_NGT_FILTER.shape, dtype=np.bool).ravel()
        fa = filter_assay.mm_filter(
            TEST_NGT_FILTER,
            NGT_MUTATED,
            MUTATED,
            TEST_KEPT_CELLS,
            TEST_KEPT_VARIANTS,
            TEST_NGT_FILTER.shape,
            CONFIG.mutated_cells_cutoff,
        )

        self.assertEqual(fa.passing_variants.sum(), 4)
        self.assertEqual(fa.passing_cells.sum(), 3)

        self.assertTrue((TEST_KEPT_VARIANTS == fa.passing_variants).all())
        self.assertTrue((TEST_KEPT_CELLS == fa.passing_cells).all())
