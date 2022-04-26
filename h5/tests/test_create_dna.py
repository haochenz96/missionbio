import json
import pickle
from unittest import TestCase

import allel
import numpy as np

from missionbio.h5.constants import AF
from missionbio.h5.create.dna import VCFFile, create_dna_assay
from missionbio.h5.tests.base import TEST_METADATA, TEST_VCF


class CreateDnaAssayTests(TestCase):
    def test_metadata(self):
        with open(TEST_METADATA) as f:
            metadata = json.load(f)
        assay = create_dna_assay(TEST_VCF, metadata=metadata)

        for key, value in metadata.items():
            self.assertIn(key, assay.metadata)
            self.assertEqual(value, assay.metadata[key])

    def test_af(self):
        assay = create_dna_assay(TEST_VCF)
        af = assay.layers[AF]

        np.testing.assert_almost_equal(
            np.array(
                [
                    [0.0, 0.0, 0.49, 0.0, 0.0],
                    [0.0, 0.0, 0.49, 0.0, 0.0],
                    [0.0, 1.06, 0.71, 1.06, 0.0],
                    [0.0, 1.06, 0.71, 1.06, 0.0],
                    [1.38, 0.0, 0.69, 0.34, 20.0],
                ]
            ),
            af[:5, :5],
            decimal=2,
        )


class VCFFileTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.VCF = pickle.dumps(VCFFile(TEST_VCF))

    def setUp(self):
        self.file = pickle.loads(self.VCF)  # type: VCFFile

    def test_layer_do_not_contain_missing_values(self):
        self.assert_no_missing_values(VCFFile.AD)
        self.assert_no_missing_values(VCFFile.DP)
        self.assert_no_missing_values(VCFFile.GQ)
        self.assert_no_missing_values(VCFFile.GT)

    def test_layer_keep_missing_values(self):
        gt = self.file.layer(VCFFile.GT, remove_missing=False)
        self.assertTrue((gt == -1).any(), "no -1, despite remove_missing_values=False")

    def test_good_quality_variants_come_first(self):
        q = allel.read_vcf(TEST_VCF, fields=[VCFFile.QUAL])[VCFFile.QUAL]

        # no threshold should return original order
        self.file.set_quality_threshold(None)
        data = self.file.layer(VCFFile.QUAL)
        np.testing.assert_almost_equal(data, q)

        # test different thresholds
        for threshold in [1000, 100, 10]:
            self.file.set_quality_threshold(threshold)

            passing = q[q >= threshold]
            data = self.file.layer(VCFFile.QUAL)[: len(passing)]
            np.testing.assert_almost_equal(data, passing)

    def assert_no_missing_values(self, layer_name, missing_value=-1):
        data = self.file.layer(layer_name, remove_missing=True)
        self.assertFalse(
            (data == missing_value).any(),
            "Layer {layer_name} contains missing values ({missing_value})",
        )
