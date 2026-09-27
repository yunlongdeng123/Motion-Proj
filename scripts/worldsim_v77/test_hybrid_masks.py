import unittest
import numpy as np
from hybrid_masks import mask_contract, compose_background


class MaskBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.delete = np.zeros((15, 20), bool)
        self.delete[5:10, 5:10] = True
        self.protect = np.zeros_like(self.delete)
        self.protect[:, 11:14] = True
        self.observed = np.zeros_like(self.delete)
        self.observed[6:8, 6:8] = True

    def test_neighbor_protection_survives_large_context(self):
        m = mask_contract(self.delete, self.protect, self.observed, 6)
        self.assertGreater(m['generate'].sum(), self.delete.sum())
        self.assertFalse((m['generate'] & self.protect).any())
        self.assertTrue(np.all(m['generate'][self.delete]))

    def test_evidence_partition_and_pixel_writeback(self):
        m = mask_contract(self.delete, self.protect, self.observed, 6)
        self.assertFalse((m['observed'] & m['residual_generate']).any())
        np.testing.assert_array_equal(m['observed'] | m['residual_generate'], m['generate'])
        original = np.full((15, 20, 3), 15, np.uint8)
        evidence = np.full_like(original, 50)
        generated = np.full_like(original, 200)
        out = compose_background(original, evidence, generated, m)
        np.testing.assert_array_equal(out[self.protect | ~m['generate']], original[self.protect | ~m['generate']])
        np.testing.assert_array_equal(out[m['observed']], evidence[m['observed']])
        np.testing.assert_array_equal(out[m['residual_generate']], generated[m['residual_generate']])

    def test_ambiguous_instance_overlap_rejected(self):
        self.protect[7, 7] = True
        with self.assertRaisesRegex(ValueError, '冲突'):
            mask_contract(self.delete, self.protect, self.observed, 3)

    def test_soft_confidence_cannot_be_silently_cast(self):
        with self.assertRaisesRegex(ValueError, 'confidence'):
            mask_contract(self.delete, self.protect, self.observed.astype(float)*0.001, 3)


if __name__ == '__main__':
    unittest.main()
