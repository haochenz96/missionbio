import os
import unittest


def suite(loader=None, pattern="test*.py"):
    test_dir = os.path.dirname(__file__)
    if loader is None:
        loader = unittest.TestLoader()
    if pattern is None:
        pattern = "test*.py"
    all_tests = [loader.discover(test_dir, pattern, test_dir)]
    return unittest.TestSuite(all_tests)


def load_tests(loader, tests, pattern):
    return suite(loader, pattern)
