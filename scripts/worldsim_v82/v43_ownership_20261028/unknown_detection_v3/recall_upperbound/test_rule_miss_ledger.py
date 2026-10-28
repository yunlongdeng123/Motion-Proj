"""CPU测试：可多标签漏选、三值缺测与无背景veto。"""
import unittest
from pathlib import Path
import tempfile
import json

import numpy as np

from rule_miss_ledger import (DEFAULT_CONFIG, candidate_variants, dense_weights,
                              evaluate, row_gates, stage_audit, tri_and, tri_or)


class LedgerTests(unittest.TestCase):
    def test_missing_is_not_false_or_true(self):
        a = np.array([-1, -1, 0, 1], np.int8)
        b = np.array([0, 1, 1, 1], np.int8)
        np.testing.assert_array_equal(tri_and(a, b), [0, -1, 0, 1])
        np.testing.assert_array_equal(tri_or(a, b), [-1, 1, 1, 1])

    def test_shadow_missing_does_not_override_known_positive_body(self):
        n = 2
        data = {"body": np.array([.2, 0]), "photo": np.zeros(n), "strict": np.full(n, np.nan),
                "strict_upper": np.zeros(n), "total": np.ones(n), "history": np.array([2, 0]),
                "core": np.full(n, np.nan), "core_upper": np.array([.2, 0]),
                "boundary": np.zeros(n), "boundary_upper": np.zeros(n)}
        g = row_gates(data, DEFAULT_CONFIG)
        np.testing.assert_array_equal(evaluate(g), [1, 0])

    def test_multilabel_post_removal_audit_and_background_record_only(self):
        initial = np.array([[.2, .2, .1, 0, .8, 1], [.001, .001, .001, 0, .999, 1],
                            [0, .1, 0, 0, .9, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1]], np.float32)
        after = initial.copy()
        after[0] = 0
        after[3] = [.05, .05, .05, 0, .95, 1]
        after[4] = [0, .1, 0, .1, .9, 1]
        selected = np.array([1, 0, 0, 0, 0], bool)
        removed = selected.copy()
        data = {"body": initial[:, 2], "photo": initial[:, 3], "strict": np.zeros(5), "strict_upper": np.zeros(5),
                "total": initial[:, 5], "history": np.array([2, 0, 0, 0, 0]),
                "core": np.zeros(5), "core_upper": np.zeros(5), "boundary": np.zeros(5), "boundary_upper": np.zeros(5)}
        gates = row_gates(data, DEFAULT_CONFIG)
        variants, _, _, _ = candidate_variants(gates, data, initial, selected, 1e-4, "photo")
        stage = {"path": Path("mechanical-post-removal.npz"), "receipt": {"regions": {"broad_domain": {"negative_environment_pixels": 4}}}}
        args = (stage, initial, after, removed, selected, np.ones(5, bool), gates, variants)
        bits1, report1 = stage_audit(*args, np.zeros(5, bool), 1e-4, None)
        bits2, report2 = stage_audit(*args, np.ones(5, bool), 1e-4, None)
        np.testing.assert_array_equal(bits1, bits2)
        self.assertEqual(int(bits1[4]) & 5, 5)  # 范围漏选和新显露可同时成立。
        self.assertEqual(report1["inside_but_gated"], 1)
        self.assertEqual(report1["newly_visible_vs_initial"], 2)
        self.assertFalse(report1["background_veto"])
        self.assertFalse(report1["labels_for_training"])
        self.assertNotEqual(report1["background_counterevidence_overlap_record_only"], report2["background_counterevidence_overlap_record_only"])
        for mask in variants.values():
            self.assertTrue(np.all(mask[selected]))

    def test_global_removed_ids_and_sparse_zero_fill(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "contributors.npz"
            np.savez(path, ids=np.array([2]), weights=np.array([[0, .2, 0, .2, 0, .2]], np.float32), removed_ids=np.array([0, 1]))
            w, removed = dense_weights(path, 4)
            self.assertEqual(w.shape, (4, 6))
            np.testing.assert_array_equal(removed, [1, 1, 0, 0])
            self.assertEqual(float(w[3].sum()), 0)


if __name__ == "__main__":
    unittest.main()
