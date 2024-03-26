from contextlib import ExitStack, contextmanager
from unittest import TestCase
from unittest.mock import Mock, patch

import h5py
import numpy as np
from h5py import File

from h5.constants import (
    ASSAYS,
    BARCODE,
    COL_ATTRS,
    DATE_CREATED,
    FILTERED_KEY,
    ID,
    LAYERS,
    METADATA,
    RAW_COUNTS,
    ROW_ATTRS,
    SAMPLE,
    SDK_VERSION,
)
from h5.data import Assay, H5Reader, H5Writer
from h5.data.normalize import normalize_attr_values
from h5.exceptions import ValidationError
from h5.tests.base import get_temp_writable_path


class MBReaderTests(TestCase):
    def test_open_invalid_file(self):
        invalid_files = [
            {},  # no assays group
            {ASSAYS: {"test": {}}},  # no layers
            {ASSAYS: {"test": {LAYERS: {}}}},  # no ra
            {ASSAYS: {"test": {LAYERS: {}, ROW_ATTRS: {}}}},  # no ca
            {ASSAYS: {"test": {LAYERS: {}, ROW_ATTRS: {}, COL_ATTRS: {}}}},  # no metadata
        ]

        for content in invalid_files:
            with self.create_tmp_h5_file(content) as f:
                with self.assertRaises(ValueError):
                    H5Reader(f)

    def test_open_valid_empty_file(self):
        valid_file = {
            METADATA: {DATE_CREATED: "<date_created>", SDK_VERSION: "<sdk_version>"},
        }
        with self.create_tmp_h5_file(valid_file) as f:
            H5Reader(f).close()

    def test_read_invalid_assay(self):
        invalid_files = [
            {
                ASSAYS: {
                    "test": {
                        LAYERS: {
                            "test": np.zeros((3, 3)),
                            "test2": np.zeros((4, 4)),  # incompatible layers
                        },
                        ROW_ATTRS: {},
                        COL_ATTRS: {},
                        METADATA: {},
                    }
                },
                METADATA: {DATE_CREATED: "<date_created>", SDK_VERSION: "<sdk_version>"},
            },
            {
                ASSAYS: {
                    "test": {
                        LAYERS: {"test": np.zeros((3, 3))},
                        ROW_ATTRS: {"test": np.zeros((4,))},  # wrong number of annotations
                        COL_ATTRS: {},
                        METADATA: {},
                    }
                },
                METADATA: {DATE_CREATED: "<date_created>", SDK_VERSION: "<sdk_version>"},
            },
            {
                ASSAYS: {
                    "test": {
                        LAYERS: {"test": np.zeros((3, 3))},
                        ROW_ATTRS: {},
                        COL_ATTRS: {"test": np.zeros((4,))},  # wrong number of annotations
                        METADATA: {},
                    }
                },
                METADATA: {DATE_CREATED: "<date_created>", SDK_VERSION: "<sdk_version>"},
            },
        ]
        for content in invalid_files:
            with self.create_tmp_h5_file(content) as filename:
                with self.assertRaises(ValueError):
                    with H5Reader(filename) as reader:
                        reader.read("test")

    def test_read_valid_assay(self):
        valid_file = self.__dummy_file("test")
        with self.create_tmp_h5_file(valid_file) as filename:
            with H5Reader(filename) as reader:
                reader.read("test")

    def test_read_missing(self):
        f = self.__dummy_file("test")
        with self.create_tmp_h5_file(f) as filename:
            with H5Reader(filename) as reader:
                with self.assertRaises(ValueError):
                    reader.read("data")

                with self.assertRaises(ValueError):
                    reader.read_raw_counts("data")

    def test_read_samples(self):
        a = Assay.create("test")
        a.add_layer("data", np.zeros((2, 1)))
        a.add_col_attr(ID, np.array([1]))
        a.add_row_attr(SAMPLE, np.array(["a", "b"]))
        a.add_row_attr(BARCODE, np.array(list("CT")))
        b = Assay.create("test2")
        b.add_layer("data", np.zeros((2, 1)))
        b.add_col_attr(ID, np.array([1]))
        b.add_row_attr(SAMPLE, np.array(["b", "c"]))
        b.add_row_attr(BARCODE, np.array(list("TG")))

        with get_temp_writable_path(suffix="h5") as filename:
            # single sample
            with H5Writer(filename, mode="w") as writer:
                writer.write(a)
            with H5Reader(filename) as reader:
                self.assertEqual(reader.samples(), {"a", "b"})

            # two incompatible samples
            with H5Writer(filename, mode="w") as writer:
                writer.write(a)
                writer.write(b)

            with H5Reader(filename) as reader:
                self.assertEqual(reader.samples(), {"b"})

    def test_read_valid_raw_counts(self):
        valid_file = self.__dummy_file("test")
        with self.create_tmp_h5_file(valid_file) as filename:
            with H5Reader(filename) as reader:
                self.assertSequenceEqual(["test"], reader.raw_counts())
                reader.read_raw_counts("test")

    def test_duplicate_sample_name(self):
        valid_file = self.__dummy_file("test")
        valid_file[ASSAYS]["test"][METADATA][SAMPLE] = np.array(list("aab"))

        with self.create_tmp_h5_file(valid_file) as filename:
            with self.assertRaises(ValueError):
                H5Reader(filename)

    def test_no_barcodes(self):
        valid_file = self.__dummy_file("test")
        del valid_file[ASSAYS]["test"][ROW_ATTRS][BARCODE]

        with self.create_tmp_h5_file(valid_file) as filename:
            with self.assertRaises(ValueError):
                H5Reader(filename)

    def test_no_ids(self):
        valid_file = self.__dummy_file("test")
        del valid_file[ASSAYS]["test"][COL_ATTRS][ID]

        with self.create_tmp_h5_file(valid_file) as filename:
            with self.assertRaises(ValueError):
                H5Reader(filename)

    def test_reading_filtered_data(self):
        valid_file = self.__dummy_file("test")
        valid_file[ASSAYS]["test"][COL_ATTRS][FILTERED_KEY] = np.array([1, 0, 0])

        with self.create_tmp_h5_file(valid_file) as filename:
            with H5Reader(filename) as reader:
                assay = reader.read("test", apply_filter=True)

        self.assertEqual(set(assay.col_attrs[ID]), {2, 3})

    def test_reading_whitelist_data(self):
        valid_file = self.__dummy_file("test")

        with self.create_tmp_h5_file(valid_file) as filename:
            with H5Reader(filename) as reader:
                assay = reader.read("test", whitelist=[1, 2])

        self.assertEqual(set(assay.col_attrs[ID]), {1, 2})

    def test_reading_whitelist_and_filtered_data(self):
        valid_file = self.__dummy_file("test")
        valid_file[ASSAYS]["test"][COL_ATTRS][FILTERED_KEY] = np.array([1, 0, 1])

        with self.create_tmp_h5_file(valid_file) as filename:
            with H5Reader(filename) as reader:
                assay = reader.read("test", apply_filter=True, whitelist=[3])

        self.assertEqual(set(assay.col_attrs[ID]), {2, 3})

    def test_closes_file_on_validation_error(self):
        invalid_file = self.__dummy_file("test")
        del invalid_file[ASSAYS]["test"][ROW_ATTRS]["sample_name"]
        with ExitStack() as stack:
            filename = stack.enter_context(get_temp_writable_path(suffix=".h5"))
            with File(filename, "w") as f:
                copy_data(invalid_file, f)

            class MockFile(File):
                mock_close = Mock(name="h5py.File.close")

                def close(self):
                    self.mock_close()
                    super().close()

            stack.enter_context(patch("h5py.File", MockFile))

            with self.assertRaises(ValidationError):
                H5Reader(filename)

            MockFile.mock_close.assert_called()

    def test_does_not_close_external_file_on_validation_error(self):
        invalid_file = self.__dummy_file("test")
        del invalid_file[ASSAYS]["test"][ROW_ATTRS]["sample_name"]
        with ExitStack() as stack:
            file = stack.enter_context(self.create_tmp_h5_file(invalid_file))
            file.close = Mock(name="h5py.File.close")

            with self.assertRaises(ValidationError):
                H5Reader(file)

            file.close.assert_not_called()

    @contextmanager
    def create_tmp_h5_file(self, content: dict):
        """Create h5 "file" with content

        Args:
            content: hierarchical content of the file

        Yields:
            h5p.File handle
        """

        with File("test.h5", "w", driver="core", backing_store=False) as f:
            copy_data(content, f)
            yield f

    def __dummy_file(self, assay_name):
        assays = {
            assay_name: {
                LAYERS: {"data": np.zeros((3, 3))},
                ROW_ATTRS: {
                    "barcode": np.array(list("CTG")),
                    "sample_name": np.array(["sample"] * 3),
                },
                COL_ATTRS: {"id": np.array([1, 2, 3])},
                METADATA: {"sample_name": np.array(["sample"])},
            }
        }
        return {
            RAW_COUNTS: assays,
            ASSAYS: assays,
            METADATA: {DATE_CREATED: "<date_created>", SDK_VERSION: "<sdk_version>"},
        }


def copy_data(data: dict, group: h5py.Group):
    """Copy data form dict to the h5py group.

    If a value in data dict is a dictionary, a new subgrup is created and data is
    copied recursively.

    Args:
        data: data to copy
        group: output h5py group
    """
    for key, value in data.items():
        if isinstance(value, dict):
            g = group.create_group(key)
            copy_data(value, g)
        else:
            group.create_dataset(key, data=normalize_attr_values(value))
