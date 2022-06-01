from unittest import TestCase
from unittest.mock import MagicMock as Mock

import h5py
import numpy as np

from missionbio.h5.constants import (
    ASSAYS,
    BARCODE,
    COL_ATTRS,
    DATE_CREATED,
    ID,
    LAYERS,
    METADATA,
    RAW_COUNTS,
    ROW_ATTRS,
    SAMPLE,
    SDK_VERSION,
)
from missionbio.h5.data import Assay
from missionbio.h5.data.writer import H5Writer
from missionbio.h5.tests.base import dummy_assay, get_temp_writable_path, sample_metadata


class MBWriterTests(TestCase):
    def test_store_assay_metadata(self):
        assay = dummy_assay("test")
        metadata = sample_metadata()
        for key, value in metadata.items():
            assay.add_metadata(key, value)

        with get_temp_writable_path() as filename:
            with H5Writer(filename) as writer:
                writer.write(assay)

            with h5py.File(filename, "r") as f:
                for key in metadata:
                    self.assertIn(key, f[ASSAYS][assay.name][METADATA])

    def test_store_file_metadata(self):
        with get_temp_writable_path() as filename:
            with H5Writer(filename):
                pass

            with h5py.File(filename, "r") as f:
                self.assertIn(SDK_VERSION, f[METADATA])
                self.assertIn(DATE_CREATED, f[METADATA])

    def test_store_layers_and_attrs(self):
        layers = {"A": np.array([[1, 2, 3], [4, 5, 6]]), "B": np.array([[3, 4, 5], [6, 7, 8]])}
        col_attrs = {ID: np.array(["A", "B", "C"])}
        row_attrs = {BARCODE: np.array(["X", "Y"]), SAMPLE: np.array(["A", "A"])}
        metadata = {SAMPLE: np.array(["A"])}

        assay = Assay(
            name="test", metadata=metadata, layers=layers, row_attrs=row_attrs, col_attrs=col_attrs
        )

        with get_temp_writable_path() as filename:
            with H5Writer(filename) as w:
                w.write(assay)

            with h5py.File(filename, "r") as f:
                for key in layers:
                    self.assertIn(key, f[ASSAYS][assay.name][LAYERS])

                for key in row_attrs:
                    self.assertIn(key, f[ASSAYS][assay.name][ROW_ATTRS])

                for key in col_attrs:
                    self.assertIn(key, f[ASSAYS][assay.name][COL_ATTRS])

    def test_closes_h5py_file(self):
        mock_file = Mock(spec=h5py.File)
        w = H5Writer(mock_file)
        w.write(dummy_assay("test"))
        mock_file.close.assert_not_called()

        w.write(dummy_assay("test2"))
        mock_file.close.assert_not_called()

        w.close()
        mock_file.close.assert_called()

    def test_with_closes_h5py_file(self):
        mock_file = Mock(spec=h5py.File)
        with H5Writer(mock_file) as w:
            w.write(dummy_assay("test"))
            mock_file.close.assert_not_called()
        mock_file.close.assert_called()

    def test_abort_on_existing_file(self):
        # should raise OSError if file already exists
        with get_temp_writable_path(file_exists=True) as filename:
            with self.assertRaises(OSError):
                H5Writer(filename, mode="w-")

    def test_write_raises_if_file_is_closed(self):
        with get_temp_writable_path() as filename:
            w = H5Writer(filename)
            w.close()

            with self.assertRaises(ValueError):
                w.write(dummy_assay("test"))

            with self.assertRaises(ValueError):
                w.write_raw_counts(dummy_assay("test"))

    def test_write_raw_counts(self):
        layers = {"A": np.array([[1, 2, 3], [4, 5, 6]]), "B": np.array([[3, 4, 5], [6, 7, 8]])}
        col_attrs = {ID: np.array(["A", "B", "C"])}
        row_attrs = {BARCODE: np.array(["X", "Y"]), SAMPLE: np.array(["A", "A"])}
        metadata = {SAMPLE: np.array(["A"])}

        assay = Assay(
            name="test", metadata=metadata, layers=layers, row_attrs=row_attrs, col_attrs=col_attrs
        )

        with get_temp_writable_path() as filename:
            with H5Writer(filename) as w:
                w.write_raw_counts(assay)

            with h5py.File(filename, "r") as f:
                for key in layers:
                    self.assertIn(key, f[RAW_COUNTS][assay.name][LAYERS])

                for key in row_attrs:
                    self.assertIn(key, f[RAW_COUNTS][assay.name][ROW_ATTRS])

                for key in col_attrs:
                    self.assertIn(key, f[RAW_COUNTS][assay.name][COL_ATTRS])

    def test_invalid_file_mode(self):
        # should raise ValueError if provided invalid mode
        with get_temp_writable_path() as filename:
            with self.assertRaises(ValueError):
                H5Writer(filename, mode="r")
