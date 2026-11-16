"""CPU binary PLY contract used by the current train_scene row binding."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from v44_native_inputs import ply


class NativeSourceContractTests(unittest.TestCase):
    def test_native_xyz_and_model_times_are_exact_float32_rows(self):
        points = np.array([[1234.12345, 7., -2.], [987.654321, -3., 4.]])
        times = np.array([-.73, 2.19])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"points.ply"; ply(path, points, times); content = path.read_bytes()
        header, data = content.split(b"end_header\n", 1)
        self.assertIn(b"format binary_little_endian 1.0", header)
        self.assertIn(b"element vertex 2", header)
        dtype = [(key, "<f4") for key in ("x", "y", "z", "nx", "ny", "nz")]
        dtype += [(key, "u1") for key in ("red", "green", "blue")]+[("time", "<f4")]
        rows = np.frombuffer(data, dtype=dtype)
        np.testing.assert_array_equal(np.column_stack([rows[k] for k in ("x", "y", "z")]), points.astype(np.float32))
        for key in ("nx", "ny", "nz", "time"):
            np.testing.assert_array_equal(rows[key], times.astype(np.float32))


if __name__ == "__main__": unittest.main()
