"""CPU-only tests for the all-qualified data-role adapter."""
import copy
import unittest
import numpy as np
from scipy import ndimage as ndi
from build_inputs import union_frame, canonical_snapshot


def fixture():
    h, w = 40, 48
    yy, xx = np.indices((h, w))
    target = {'K_vehicle': np.zeros((h, w), bool)}
    split = {'K': target['K_vehicle'].copy(), 'fixed_alpha': np.ones((h, w), np.float32)}
    for name in ('A', 'C', 'S'):
        target['T_' + name] = np.stack([xx / w, yy / h, xx * 0 + .3]).astype(np.float32)
        valid = (yy >= 2) & (yy < h - 2) & (xx >= 2) & (xx < w - 2)
        split[name + '_seen'] = valid & (xx < 24)
        split[name + '_held'] = valid & (xx >= 24)
        split[name + '_seen_context'] = split[name + '_seen'].copy()
        split[name + '_held_context'] = split[name + '_held'].copy()
        split[name + '_unmapped'] = ~valid
    return target, split


class TestBuildInputs(unittest.TestCase):
    def test_complete_mapped_union_unchanged_teacher(self):
        target, split = fixture()
        original = copy.deepcopy(split)
        frame = union_frame(target, split)
        for name in ('A', 'C', 'S'):
            np.testing.assert_array_equal(frame['scores'][name], ~split[name + '_unmapped'])
            np.testing.assert_array_equal(frame['targets'][name], target['T_' + name])
        for key in split:
            np.testing.assert_array_equal(original[key], split[key])

    def test_union_before_filter_restores_cross_split_support(self):
        target, split = fixture()
        frame = union_frame(target, split)
        def core(mask):
            return ndi.minimum_filter(mask.astype(np.uint8), size=7, mode='constant', cval=0) > 0
        union_core = core(frame['contexts']['S'])
        isolated = core(split['S_seen_context']) | core(split['S_held_context'])
        self.assertTrue(union_core[20, 24])
        self.assertFalse(isolated[20, 24])
        self.assertGreater(union_core.sum(), isolated.sum())

    def test_unmapped_or_K_cannot_enter_union(self):
        target, split = fixture()
        split['C_seen'][0, 0] = True
        with self.assertRaisesRegex(ValueError, 'unmapped'):
            union_frame(target, split)
        target, split = fixture()
        target['K_vehicle'][20, 20] = split['K'][20, 20] = True
        with self.assertRaisesRegex(ValueError, 'protected'):
            union_frame(target, split)

    def test_unstable_context_rejected_not_silently_pruned(self):
        target, split = fixture()
        split['fixed_alpha'][20, 20] = .8
        with self.assertRaisesRegex(ValueError, 'unstable'):
            union_frame(target, split)

    def test_overlap_rejected_and_json_tuple_fields_restored(self):
        target, split = fixture()
        split['A_held'][20, 20] = True
        with self.assertRaisesRegex(ValueError, 'overlap'):
            union_frame(target, split)
        saved = dict(spec=dict(scales=[1., 2., 4.], source_weights=[['A', 1.], ['C', .5], ['S', 1.]]))
        restored = canonical_snapshot(saved)
        self.assertEqual(restored['spec']['scales'], (1., 2., 4.))
        self.assertEqual(restored['spec']['source_weights'], (('A', 1.), ('C', .5), ('S', 1.)))


if __name__ == '__main__':
    unittest.main()
