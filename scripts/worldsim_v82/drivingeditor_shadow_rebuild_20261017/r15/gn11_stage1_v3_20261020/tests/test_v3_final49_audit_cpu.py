import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
spec = importlib.util.spec_from_file_location('audit_final49', ROOT/'tools/audit_v3_final49.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditCPU(unittest.TestCase):
    def test_weighted_missing_evidence_is_not_zero_error(self):
        rows = [dict(n=2, error=.3), dict(n=0), dict(n=6, error=.1)]
        self.assertAlmostEqual(audit.weighted(rows, 'error', 'n'), .15)
        self.assertIsNone(audit.weighted([dict(n=0)], 'missing', 'n'))
        with self.assertRaises(KeyError): audit.weighted([dict(n=1)], 'missing', 'n')

    def test_floating_assets_and_json_reject_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'maps.npz'
            np.savez_compressed(path, alpha=np.zeros((1, 2, 2)), mask=np.ones(2, bool))
            self.assertEqual(set(audit.check_npz(path)), {'alpha', 'mask'})
            np.savez_compressed(path, alpha=np.array([float('inf')]))
            with self.assertRaises(ValueError): audit.check_npz(path)
        with self.assertRaises(ValueError): audit.finite(dict(value=[float('nan')]))


if __name__ == '__main__': unittest.main()
