"""候选漏选和目标保持冲突的 CPU 回归；无需模型、Torch 或 CUDA。"""
from copy import deepcopy
from io import BytesIO
import unittest

import numpy as np

from v44_handoff_policy import (
    eligible_static_contributors,
    handoff_domains,
    rebind_other_vehicles,
    target_mask_from_labels,
)


def view(shape=(2, 4), **changes):
    fields = dict(
        fit=np.zeros(shape, np.float32), keep=np.ones(shape, np.float32),
        inside=np.zeros(shape, bool), other=np.zeros(shape, bool),
        marking=np.zeros(shape, bool),
    )
    fields.update(changes)
    return fields


class TargetIdentityTests(unittest.TestCase):
    def test_multiple_explicit_ids_and_relabeling_preserve_target_identity(self):
        labels = np.array([[0, 37, 91, 5], [91, 5, 0, 37]], np.uint16)
        targets = target_mask_from_labels(labels, [37, 91])
        relabeled = np.zeros_like(labels, dtype=np.int64)
        for old, new in [(37, 400017), (91, 9), (5, 712)]:
            relabeled[labels == old] = new
        np.testing.assert_array_equal(targets, target_mask_from_labels(relabeled, {9, 400017}))
        np.testing.assert_array_equal(targets, [[0, 1, 1, 0], [1, 0, 0, 1]])

    def test_absent_and_empty_targets_do_not_invent_a_region(self):
        labels = np.array([[0, 23, 0], [17, 23, 17]], np.int32)
        for ids in [[], (), [19999]]:
            with self.subTest(ids=ids):
                target = target_mask_from_labels(labels, ids)
                self.assertEqual(target.shape, labels.shape)
                self.assertFalse(target.any())
                source = view(labels.shape, fit=np.full(labels.shape, .25, np.float32))
                result = handoff_domains(source, target)
                np.testing.assert_array_equal(result['keep'], source['keep'])
                np.testing.assert_array_equal(result['candidate_support'], True)
                self.assertFalse(result['unknown_target'].any())

    def test_arbitrary_positive_image_shapes(self):
        for shape in [(1, 1), (1, 9), (7, 1), (3, 11), (17, 5)]:
            with self.subTest(shape=shape):
                labels = np.zeros(shape, np.int64)
                labels.flat[-1] = 80713
                target = target_mask_from_labels(labels, [80713, 42])
                result = handoff_domains(view(shape), target)
                self.assertEqual(int(result['candidate_support'].sum()), 1)
                self.assertEqual(int(result['unknown_target'].sum()), 1)
                self.assertEqual(result['left'].shape, shape)

    def test_identity_inputs_reject_casts_ambiguity_and_invalid_shapes(self):
        labels = np.array([[0, 17]], np.int64)
        bad_ids = [True, 17, '17', [True], [17.0], [0], [-1], [2**63], [17, 17], [None]]
        for ids in bad_ids:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                target_mask_from_labels(labels, ids)
        bad_labels = [labels.astype(float), labels.astype(bool), [[-1, 17]],
                      np.array([[2**63]], np.uint64), [0, 17], np.zeros((0, 2), int)]
        for bad in bad_labels:
            with self.subTest(labels=bad), self.assertRaises(ValueError):
                target_mask_from_labels(bad, [17])


class OtherVehicleRebindingTests(unittest.TestCase):
    def test_complete_delete_set_releases_second_target_from_old_other(self):
        labels = np.array([[37, 91, 5, 0, 0], [0, 91, 13, 37, 0]], np.uint16)
        old_other = np.array([[0, 1, 0, 1, 0], [0, 1, 0, 0, 1]], bool)
        other = rebind_other_vehicles(old_other, labels, [37, 91])
        np.testing.assert_array_equal(other, [[0, 0, 1, 1, 0], [0, 0, 1, 0, 1]])
        target = target_mask_from_labels(labels, [37, 91])
        result = handoff_domains(view(labels.shape, other=other), target)
        self.assertTrue(result['candidate_support'][target].all())
        self.assertTrue(result['unknown_target'][target].all())
        self.assertFalse(result['target_protected_conflict'].any())
        self.assertFalse(result['keep'][target].any())
        self.assertFalse(result['left'][target].any())
        self.assertEqual(int(result['candidate_support'].sum()), 4)
        np.testing.assert_array_equal(old_other, [[0, 1, 0, 1, 0], [0, 1, 0, 0, 1]])

    def test_multiple_deleted_ids_can_be_reordered_and_relabelled(self):
        labels = np.array([[0, 16, 701], [8, 16, 0]], np.int64)
        other = np.array([[1, 0, 1], [0, 1, 0]], np.uint8)
        baseline = rebind_other_vehicles(other, labels, [16, 701])
        relabeled = labels.copy()
        for old, new in [(16, 99993), (701, 27), (8, 19003)]:
            relabeled[labels == old] = new
        np.testing.assert_array_equal(baseline, rebind_other_vehicles(other, relabeled, [27, 99993]))

    def test_absent_target_keeps_unlabelled_other_and_all_nontarget_labels(self):
        labels = np.array([[0, 7, 0, 19, 0]], np.int32)
        old_other = np.array([[1., 0., 0., 0., 1.]])
        for ids in [[], [17001]]:
            with self.subTest(ids=ids):
                np.testing.assert_array_equal(rebind_other_vehicles(old_other, labels, ids), [[1, 1, 0, 1, 1]])

    def test_other_rebinding_rejects_bad_labels_masks_and_delete_ids(self):
        labels = np.array([[0, 7]], np.int64)
        for other, supplied_labels, ids in [
            ([[0, .5]], labels, [7]), ([[0, np.nan]], labels, [7]),
            ([[0]], labels, [7]), ([[0, 0]], labels.astype(float), [7]),
            ([[0, 0]], labels, [True]), ([[0, 0]], labels, [7.]),
            ([[0, 0]], labels, [7, 7]), ([[0, 0]], labels, 7),
        ]:
            with self.subTest(other=other, labels=supplied_labels, ids=ids), self.assertRaises(ValueError):
                rebind_other_vehicles(other, supplied_labels, ids)


class DomainRegressionTests(unittest.TestCase):
    def test_target_outside_chart_is_candidate_without_becoming_photo_truth(self):
        source = view()
        source['fit'][0, 0] = .25
        source['inside'][0, 0] = True
        target = np.zeros((2, 4), bool)
        target[1, 3] = True
        result = handoff_domains(source, target)
        self.assertTrue(result['candidate_support'][0, 0])
        self.assertTrue(result['candidate_support'][1, 3])
        self.assertEqual(int(result['candidate_support'].sum()), 2)
        self.assertTrue(result['unknown_target'][1, 3])
        self.assertEqual(result['keep'][1, 3], 0)
        self.assertFalse(result['left'][1, 3])
        np.testing.assert_array_equal(result['fit'], source['fit'])
        self.assertEqual(result['fit'][1, 3], 0)
        self.assertTrue(result['left'][1, 2])
        self.assertEqual(result['keep'][1, 2], 1)

    def test_unknown_is_distinct_from_existing_fit_at_fixed_threshold(self):
        source = view((1, 6), fit=np.array([[0, .01, .05, .051, .25, 1.]], np.float64))
        target = np.ones((1, 6), bool)
        result = handoff_domains(source, target)
        np.testing.assert_array_equal(result['unknown_target'], [[1, 1, 1, 0, 0, 0]])
        self.assertTrue(result['candidate_support'].all())
        self.assertFalse(result['keep'].any())
        self.assertFalse(result['left'].any())
        np.testing.assert_array_equal(result['fit'], source['fit'])

    def test_other_and_marking_win_target_conflicts_and_are_auditable(self):
        source = view((1, 5), fit=np.ones((1, 5), np.float32),
                      keep=np.array([[.4, .8, .6, 1, .3]], np.float32),
                      other=np.array([[1, 0, 1, 0, 0]], bool),
                      marking=np.array([[0, 1, 1, 0, 0]], bool))
        target = np.array([[1, 1, 1, 1, 0]], bool)
        result = handoff_domains(source, target)
        np.testing.assert_array_equal(result['protected'], [[1, 1, 1, 0, 0]])
        np.testing.assert_array_equal(result['target_protected_conflict'], [[1, 1, 1, 0, 0]])
        np.testing.assert_array_equal(result['keep'], np.array([[1, 1, 1, 0, .3]], np.float32))
        np.testing.assert_array_equal(result['candidate_support'], [[0, 0, 0, 1, 1]])
        np.testing.assert_array_equal(result['left'], [[0, 1, 0, 0, 1]])
        self.assertFalse(result['unknown_target'].any())

    def test_protected_unknown_target_is_conflict_not_unknown_background(self):
        source = view((1, 3), marking=np.array([[1, 0, 0]], bool),
                      other=np.array([[0, 1, 0]], bool))
        result = handoff_domains(source, np.ones((1, 3), bool))
        np.testing.assert_array_equal(result['unknown_target'], [[0, 0, 1]])
        np.testing.assert_array_equal(result['target_protected_conflict'], [[1, 1, 0]])
        np.testing.assert_array_equal(result['keep'], [[1, 1, 0]])
        np.testing.assert_array_equal(result['candidate_support'], [[0, 0, 1]])

    def test_binary_numeric_masks_and_boolean_weights_are_supported(self):
        source = view((1, 3), fit=np.array([[0, 1, 0]], bool),
                      keep=np.array([[1, 1, 1]], bool),
                      inside=np.array([[0., 1., 0.]], np.float32),
                      marking=np.array([[0, 0, 1]], np.uint8))
        result = handoff_domains(source, np.array([[1., 0., 1.]]))
        np.testing.assert_array_equal(result['candidate_support'], [[1, 1, 0]])
        np.testing.assert_array_equal(result['keep'], [[0, 1, 1]])

    def test_mapping_npz_and_noncontiguous_inputs_are_pure(self):
        source = view((4, 6))
        source['fit'][0, 0] = .25
        source = {name: value[::-1, ::2] for name, value in source.items()}
        target = np.ones((4, 3), bool)
        before = deepcopy(source)
        result = handoff_domains(source, target)
        for name in source:
            np.testing.assert_array_equal(source[name], before[name])
        self.assertTrue(target.all())
        result['fit'][0, 0] = 1
        result['keep'][0, 0] = 1
        np.testing.assert_array_equal(source['fit'], before['fit'])
        np.testing.assert_array_equal(source['keep'], before['keep'])
        buffer = BytesIO()
        np.savez(buffer, **source)
        buffer.seek(0)
        with np.load(buffer) as archive:
            actual = handoff_domains(archive, target)
        expected = handoff_domains(source, target)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])

    def test_malformed_view_masks_and_weights_are_rejected(self):
        source = view()
        for name in ('inside', 'other', 'marking'):
            for bad in [np.full((2, 4), .2), np.full((2, 4), np.nan),
                        np.full((2, 4), 2), np.ones((1, 4), bool),
                        np.full((2, 4), '1'), np.ones((2, 4), complex)]:
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    handoff_domains({**source, name: bad}, np.zeros((2, 4), bool))
        for name in ('fit', 'keep'):
            for bad in [np.full((2, 4), -.01), np.full((2, 4), 1.01),
                        np.full((2, 4), np.inf), np.full((2, 4), np.nan),
                        np.ones((2, 4), complex), np.ones((2, 3)),
                        np.ma.array(np.ones((2, 4)), mask=False)]:
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    handoff_domains({**source, name: bad}, np.zeros((2, 4), bool))
        for bad in [None, [], {'fit': source['fit']}]:
            with self.subTest(view=bad), self.assertRaises(ValueError):
                handoff_domains(bad, np.zeros((2, 4), bool))
        for bad in [np.ones((2, 4), float) * .2, np.full((2, 4), np.nan),
                    np.ones((2, 4, 1), bool), np.zeros((0, 4), bool)]:
            with self.subTest(target=bad), self.assertRaises(ValueError):
                handoff_domains(source, bad)


class FixedSelectorTests(unittest.TestCase):
    def test_max_scores_use_original_threshold_guard_and_fixed_owner(self):
        # 不同 BUILD 视角的贡献合并后一次选择；无需 QUERY 帧或场景 ID。
        candidate_by_view = np.array([[0, .0001, .2, .1, .4, .2],
                                      [.3, 0, .1, .2, .1, .1]], np.float64)
        guard_by_view = np.array([[0, 0, 4., 0, 0, 4.000001],
                                  [0, 0, 3., 0, 0, 0]], np.float64)
        fixed_owner = np.array([0, -1, -1, 907, 0, 0], np.int64)
        selected = eligible_static_contributors(candidate_by_view.max(0), guard_by_view.max(0), fixed_owner)
        np.testing.assert_array_equal(selected, [1, 0, 1, 0, 1, 0])
        np.testing.assert_array_equal(fixed_owner, [0, -1, -1, 907, 0, 0])

    def test_empty_static_model_and_float32_threshold(self):
        selected = eligible_static_contributors(np.array([], np.float32), np.array([], np.float32),
                                                np.array([], np.int64))
        self.assertEqual(selected.shape, (0,))
        self.assertEqual(selected.dtype, bool)
        score = np.array([1e-4, .1, .1], np.float32)
        guard = np.array([0, 2, np.nextafter(np.float32(2), np.float32(3))], np.float32)
        np.testing.assert_array_equal(eligible_static_contributors(score, guard, [0, 0, 0]), [0, 1, 0])

    def test_invalid_scores_owner_or_vector_shapes_are_rejected(self):
        for score, guard, owner in [
            ([np.nan], [0], [0]), ([np.inf], [0], [0]), ([-1], [0], [0]),
            ([1], [-1], [0]), ([1], [np.inf], [0]), ([True], [0], [0]),
            ([1], [True], [0]), ([1], [0], [False]), ([1], [0], [0.]),
            ([1], [0], np.array([2**63], np.uint64)),
            ([1, 2], [0], [0, 0]), ([1], [0], [0, 0]),
            ([[1]], [[0]], [[0]]), ([1 + 0j], [0], [0]),
        ]:
            with self.subTest(score=score, guard=guard, owner=owner), self.assertRaises(ValueError):
                eligible_static_contributors(score, guard, owner)


if __name__ == '__main__':
    unittest.main()
