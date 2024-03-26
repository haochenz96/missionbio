from unittest.case import TestCase

import numpy as np

from h5.constants import BARCODE, DNA_ASSAY, ID, NGT, SAMPLE
from h5.data import Assay, H5Reader
from h5.merge import merge_samples
from h5.tests.base import TEST_MERGE_A, TEST_MERGE_B, TEST_MERGE_C, dummy_assay


class MergeSamplesTests(TestCase):
    def test_error_on_different_assays(self):
        with self.assertRaises(ValueError):
            merge_samples(Assay.create("test1"), Assay.create("test2"))

    def test_error_on_missing_ids(self):
        test = dummy_assay("test")
        del test.col_attrs[ID]
        with self.assertRaises(ValueError):
            merge_samples(test, test)

    def test_error_on_invalid_multi_sample_assay(self):
        test = Assay.create("test")
        test.add_metadata(SAMPLE, ["a", "b"])
        test.add_layer("data", np.zeros((2, 1)))
        test.add_col_attr(ID, np.array(["a"]))

        with self.assertRaises(ValueError):
            merge_samples(test, test)

    def test_trivial_merge(self):
        a1 = dummy_assay("test")
        a2 = dummy_assay("test")
        merged = merge_samples(a1, a2)

        for layer in a1.layers:
            self.assertIn(layer, merged.layers)
        for attr in a1.row_attrs:
            self.assertIn(attr, merged.row_attrs)
        for attr in a1.col_attrs:
            self.assertIn(attr, merged.col_attrs)
        for attr in a1.metadata:
            self.assertIn(attr, merged.metadata)

    def test_2d_row_merge(self):
        a1 = dummy_assay("test")
        a2 = dummy_assay("test", size=4)

        a1.add_row_attr("2dattr", np.zeros((3, 2)))
        a2.add_row_attr("2dattr", np.zeros((4, 2)))

        merge_samples(a1, a2)

    def test_shuffled_columns(self):
        a1 = dummy_assay("test")
        a2 = dummy_assay("test")
        a2.select_columns(np.array([1, 0, 2]))

        merge_samples(a1, a2, inplace=True)

    def test_merge_dna(self):
        a = H5Reader(TEST_MERGE_A).read(DNA_ASSAY)
        b = H5Reader(TEST_MERGE_B).read(DNA_ASSAY)
        c = H5Reader(TEST_MERGE_C).read(DNA_ASSAY)

        # Sanity check, sample a contains 2 variants
        # first 2 cells are ref
        # 3rd cell has het in second variant
        # 4th cell has NA for both variants
        self.assertEqual(a.shape[1], 2)
        np.testing.assert_array_equal(a.layers[NGT][0], [0, 0])
        np.testing.assert_array_equal(a.layers[NGT][1], [0, 0])
        np.testing.assert_array_equal(a.layers[NGT][2], [0, 1])
        np.testing.assert_array_equal(a.layers[NGT][3], [3, 3])

        merged = merge_samples(a, b, c, inplace=True)

        # after merging we should have 4 variants.
        # The new variants should have the following values:
        # - first 2 cells are ref (since all values at this position are ref)
        # - 3rd and 4th cells are NA (since both contain at least one value
        #                             that is not ref)
        self.assertEqual(merged.shape[1], 4)
        np.testing.assert_array_equal(merged.layers[NGT][0], [0, 0, 0, 0])
        np.testing.assert_array_equal(merged.layers[NGT][1], [0, 0, 0, 0])
        np.testing.assert_array_equal(merged.layers[NGT][2], [0, 1, 3, 3])
        np.testing.assert_array_equal(merged.layers[NGT][3], [3, 3, 3, 3])

        # merged assay should have all cells from all input assays
        self.assertEqual(merged.shape[0], a.shape[0] + b.shape[0] + c.shape[0])

    @staticmethod
    def __test_assays():
        assays = [Assay.create("test1"), Assay.create("test1")]
        for assay in assays:
            assay.add_layer("data", np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))
            assay.add_row_attr(BARCODE, np.array(list("123")))
            assay.add_col_attr(ID, np.array(list("123")))
            assay.add_metadata("test", "value")

        return assays
