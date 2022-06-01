from contextlib import ExitStack
from typing import Tuple
from unittest import TestCase
from unittest.mock import patch

import h5py
import numpy as np

from missionbio.h5.constants import BARCODE, ID, SAMPLE
from missionbio.h5.data import Assay, H5Reader, H5Writer
from missionbio.h5.merge import merge_assay_files, merge_assays
from missionbio.h5.tests.base import dummy_assay


class MergeAssaysTests(TestCase):
    def test_error_on_duplicates(self):
        with self.assertRaises(ValueError):
            merge_assays(dummy_assay("test"), dummy_assay("test"))

    def test_error_on_missing_barcodes(self):
        test = dummy_assay("test")
        del test.row_attrs[BARCODE]
        with self.assertRaises(ValueError):
            merge_assays(test, dummy_assay("test2"))

    def test_trivial_merge(self):
        merge_assays(*self.__test_assays())

    def test_unordered_barcodes(self):
        a1, a2 = self.__test_assays()
        a2.row_attrs[BARCODE] = np.array(list("CAG"))

        b1, b2 = merge_assays(a1, a2)
        np.testing.assert_array_equal(b1.row_attrs[BARCODE], b2.row_attrs[BARCODE])
        # check that layers are sorted as well
        np.testing.assert_array_equal(b2.layers["data"], [[2, 2, 2], [1, 1, 1], [3, 3, 3]])
        # check that original assay is unmodified
        np.testing.assert_array_equal(a2.layers["data"], [[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    def test_unordered_barcodes_inplace(self):
        a1, a2 = self.__test_assays()
        a2.row_attrs[BARCODE] = np.array(list("CAG"))

        b1, b2 = merge_assays(a1, a2, inplace=True)
        self.assertIs(a1, b1)
        self.assertIs(a2, b2)

        np.testing.assert_array_equal(b1.row_attrs[BARCODE], b2.row_attrs[BARCODE])
        # check that layers are sorted as well
        np.testing.assert_array_equal(b2.layers["data"], [[2, 2, 2], [1, 1, 1], [3, 3, 3]])

    def test_multi_sample_data(self):
        a1, a2 = self.__multisample_test_assays()

        b1, b2 = merge_assays(a1, a2, inplace=True)
        self.assertIs(a1, b1)
        self.assertIs(a2, b2)

        np.testing.assert_array_equal(b1.row_attrs[BARCODE], b2.row_attrs[BARCODE])
        np.testing.assert_array_equal(b1.layers["data"], [[4, 4, 4], [5, 5, 5], [6, 6, 6]])
        np.testing.assert_array_equal(b2.layers["data"], [[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    def test_duplicate_barcodes(self):
        a1, a2 = self.__multisample_test_assays()
        a1.row_attrs[BARCODE][:] = "G"

        with self.assertRaises(ValueError):
            merge_assays(a1, a2)

    def test_copies_all_barcodes(self):
        with ExitStack() as stack:
            a, b = self.__test_assays()

            file_a = stack.enter_context(
                h5py.File("input1.h5", mode="w", driver="core", backing_store=False)
            )
            file_b = stack.enter_context(
                h5py.File("input2.h5", mode="w", driver="core", backing_store=False)
            )

            for file, assay in zip([file_a, file_b], [a, b]):
                w = H5Writer(file)
                w.write(assay)
                w.write_raw_counts(assay)

            merged = stack.enter_context(
                h5py.File("output.h5", mode="w", driver="core", backing_store=False)
            )

            with patch.object(merged, "close"):
                merge_assay_files([file_a, file_b], merged, True)
            w = H5Writer(merged)

            r = H5Reader(merged)
            self.assertEqual(2, len(r.raw_counts()))

    @staticmethod
    def __test_assays() -> Tuple[Assay, Assay]:
        assays = (Assay.create("test1"), Assay.create("test2"))
        for assay in assays:
            assay.add_metadata(SAMPLE, np.array(["A"]))
            assay.add_layer("data", np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))
            assay.add_row_attr(BARCODE, np.array(list("ACG")))
            assay.add_row_attr(SAMPLE, np.array(["A"] * 3))
            assay.add_col_attr(ID, np.array([1, 2, 3]))
        return assays

    @staticmethod
    def __multisample_test_assays() -> Tuple[Assay, Assay]:
        assays = (Assay.create("test1"), Assay.create("test2"))
        for assay in assays:
            assay.add_layer(
                "data", np.array([[3, 3, 3], [2, 2, 2], [1, 1, 1], [4, 4, 4], [5, 5, 5], [6, 6, 6]])
            )
            assay.add_row_attr(BARCODE, np.array(list("GCAACG")))
            assay.add_row_attr(SAMPLE, np.array(list("AAABBB")))
        assays[1].add_row_attr(SAMPLE, np.array(list("BBBCCC")))
        return assays
