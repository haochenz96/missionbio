from unittest import TestCase

import numpy as np

from h5.constants import BARCODE, ID, SAMPLE
from h5.data import Assay
from h5.data.validation import check_assay
from h5.tests.base import dummy_assay


class AssayTests(TestCase):
    def test_str_on_empty_assay(self):
        assay = Assay.create("test")
        str(assay)

    def test_init(self):
        assay = Assay(
            name="test",
            layers={"data": np.zeros((3, 3))},
            row_attrs={"barcode": np.array(list("CTG")), "sample_name": np.array(["sample"] * 3)},
            col_attrs={"id": np.array([1, 2, 3])},
            metadata={"sample_name": np.array(["sample"])},
        )

        self.assertEqual(assay.shape, (3, 3))
        self.assertIn(BARCODE, assay.row_attrs)
        self.assertIn(SAMPLE, assay.row_attrs)
        self.assertIn(ID, assay.col_attrs)
        self.assertIn(SAMPLE, assay.metadata)
        check_assay(assay)

    def test_rename_sample(self):
        assay = dummy_assay("test")
        self.assertEqual(assay.samples(), {"sample"})

        assay.rename_sample("sample", "b")
        self.assertEqual(assay.samples(), {"b"})
        check_assay(assay)

        # cannot rename missing sample
        with self.assertRaises(ValueError):
            assay.rename_sample("c", "d")

    def test_rename_sample_multiple(self):
        assay = Assay.create("test")
        assay.add_metadata(SAMPLE, np.array(["a", "b"]))

        assay.rename_sample("a", "c")
        self.assertEqual(assay.samples(), {"b", "c"})

        # cannot merge samples
        with self.assertRaises(ValueError):
            assay.rename_sample("c", "b")

    def test_add_row_attr(self):
        assay = dummy_assay("test")
        assay.add_row_attr(SAMPLE, assay.row_attrs[SAMPLE])
        self.assertIsInstance(assay.metadata[SAMPLE], np.ndarray)

    def test_sample_names_not_lexicographically_ordered(self):
        assay = Assay(
            name="test",
            layers={"data": np.zeros((3, 3))},
            row_attrs={
                "barcode": np.array(list("CTG")),
                "sample_name": np.array(["sample-2"] * 2 + ["sample-1"]),
            },
            col_attrs={"id": np.array([1, 2, 3])},
            metadata={},
        )

        self.assertEqual(["sample-2", "sample-1"], list(assay.metadata[SAMPLE]))
        check_assay(assay)
