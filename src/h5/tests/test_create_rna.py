from unittest import TestCase

import numpy as np

from h5.create import create_rna_assay
from h5.tests.base import TEST_RNA_COUNTS


class CreateRnaAssayTests(TestCase):
    def test_create(self):
        rna = create_rna_assay(TEST_RNA_COUNTS)

        np.testing.assert_almost_equal(
            rna.layers["read_counts"],
            [[42.0, 2.0, 0.0, 7.0], [0.0, 17.0, 1.0, 1.0], [7.0, 0.0, 0.0, 15.0]],
        )
