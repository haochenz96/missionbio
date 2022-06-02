from unittest import TestCase

from h5.utils import find_common_keys


class FindCommonKeysTests(TestCase):
    def test_on_dicts(self):
        a = {"a": None, "b": None}
        b = {"a": 0, "b": 0, "c": 0}

        common_keys = find_common_keys([a, b])
        self.assertEqual(common_keys, {"a", "b"})

    def test_on_lists(self):
        a = ["c", "a"]
        b = ["d", "b", "a"]

        common_keys = find_common_keys([a, b])
        self.assertEqual(common_keys, {"a"})

    def test_on_generator(self):
        a = ["c", "a"]
        b = ["d", "b", "a"]

        common_keys = find_common_keys(iter([a, b]))
        self.assertEqual(common_keys, {"a"})
