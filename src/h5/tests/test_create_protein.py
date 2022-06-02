from unittest import TestCase

import numpy as np

from h5.create import create_protein_assay
from h5.tests.base import TEST_PROTEIN_COUNTS


class CreateProteinAssayTests(TestCase):
    def test_create(self):
        protein = create_protein_assay(TEST_PROTEIN_COUNTS)

        np.testing.assert_almost_equal(
            protein.layers["read_counts"],
            [[42.0, 2.0, 0.0, 7.0], [0.0, 17.0, 1.0, 1.0], [7.0, 0.0, 0.0, 15.0]],
        )
