from unittest import TestCase

import numpy as np

from missionbio.h5.create import create_cnv_assay
from missionbio.h5.tests.base import TEST_DNA_COUNTS


class CreateCnvAssayTests(TestCase):
    def test_create(self):
        cnv = create_cnv_assay(TEST_DNA_COUNTS)

        np.testing.assert_almost_equal(
            cnv.layers["read_counts"],
            [[42.0, 2.0, 0.0, 7.0], [0.0, 17.0, 1.0, 1.0], [7.0, 0.0, 0.0, 15.0]],
        )
