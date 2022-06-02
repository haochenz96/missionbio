from unittest import TestCase

import numpy as np

from h5.data.normalize import normalize_attr_strings, normalize_attr_values


class NormalizeAttrStringsTests(TestCase):
    def test_multi_dimensional_string_arrays(self):
        a = np.array([["a"], ["b"]], dtype=object)

        b = normalize_attr_strings(a)
        np.testing.assert_array_equal(b, [[b"a"], [b"b"]])

    def test_multi_dimensional_string_array(self):
        a = np.array([["NA"], [0.094]], dtype=object)

        b = normalize_attr_values(a)
        print(b)
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(b, [[b"NA"], [b"0.094"]])
