import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from wifi_csi_quality.readers import read_csi_bench_h5, read_xrf55_mat


class ReaderTests(unittest.TestCase):
    def test_csi_bench_h5_is_time_first(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "sample.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("CSI_amps", data=np.ones((56, 500, 1), np.float32))
            sample = read_csi_bench_h5(path)
        self.assertEqual(sample.values.shape, (500, 56, 1))
        self.assertEqual(sample.timestamps.shape, (500,))
        self.assertAlmostEqual(sample.timestamps[-1], 4.99)

    def test_xrf55_mat_is_time_subcarrier_link(self) -> None:
        dtype = [
            ("timestamp_low", "O"),
            ("rssi_a", "O"),
            ("rssi_b", "O"),
            ("rssi_c", "O"),
            ("csi", "O"),
        ]
        packets = np.empty((2, 1), dtype=dtype)
        for index in range(2):
            packets[index, 0]["timestamp_low"] = 1000 + index * 5000
            packets[index, 0]["rssi_a"] = 30
            packets[index, 0]["rssi_b"] = 31
            packets[index, 0]["rssi_c"] = 32
            packets[index, 0]["csi"] = np.ones((3, 30), np.complex64) * (index + 1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "sample.mat"
            savemat(path, {"data": packets})
            sample = read_xrf55_mat(path)
        self.assertEqual(sample.values.shape, (2, 30, 3))
        self.assertAlmostEqual(sample.timestamps[1], 0.005)


if __name__ == "__main__":
    unittest.main()

