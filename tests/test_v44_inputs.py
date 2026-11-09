"""CPU 故障回归；用 unittest 运行，无 Torch/GPU/pytest 依赖。"""
import json
import unittest
import numpy as np
from v44_inputs import (IdentityRegistry, SceneInputs, LabelMasks, make_label_masks,
                        candidate_observation, preserve_supported_observations,
                        validate_component_groups, extrapolate_short_gap)


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.registry = IdentityRegistry({'raw-track/z': 81, 'raw-track/a': 7})

    def scene(self, **updates):
        args = dict(images=np.zeros((3, 5, 7, 3), np.uint8),
                    camera_to_world=np.repeat(np.eye(4)[None], 3, axis=0),
                    timestamps_s=np.array([421.0, 421.07, 421.23]),
                    frame_ids=('source-F', 'source-K', 'source-T'),
                    raw_track_ids=tuple(self.registry.track_to_actor), registry=self.registry)
        args.update(updates)
        return SceneInputs(**args)

    def test_default_is_all_frames_with_actual_timestamps(self):
        s = self.scene()
        self.assertEqual(s.indices(), (0, 1, 2))
        self.assertEqual(s.indices(('source-F', 'source-T')), (0, 2))
        np.testing.assert_array_equal(s.timestamps_s, [421.0, 421.07, 421.23])
        with self.assertRaises(ValueError): s.indices(('source-T', 'source-F'))

    def test_mismatched_camera_time_and_frame_lengths_rejected(self):
        cases = [dict(camera_to_world=np.repeat(np.eye(4)[None], 2, axis=0)),
                 dict(timestamps_s=[1., 2.]), dict(frame_ids=('a', 'b')),
                 dict(frame_ids=('a', 'a', 'b')), dict(timestamps_s=[1., 1., 2.]),
                 dict(timestamps_s=[1., 3., 2.]), dict(timestamps_s=[1., np.nan, 3.]), dict(frame_ids='abc')]
        for case in cases:
            with self.subTest(case=list(case)), self.assertRaises(ValueError): self.scene(**case)

    def test_camera_scale_reflection_and_wrong_bottom_row_rejected(self):
        for index, value in [((0, 0, 0), 1.1), ((1, 1, 1), -1.), ((2, 3, 0), .01)]:
            cameras = np.repeat(np.eye(4)[None], 3, axis=0); cameras[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError): self.scene(camera_to_world=cameras)

    def test_identity_manifest_roundtrip_and_corruption(self):
        value = json.loads(json.dumps(self.registry.manifest()))
        restored = IdentityRegistry.from_manifest(value)
        self.assertEqual(restored.track_to_actor['raw-track/z'], 81)
        self.assertEqual(restored.actor_to_track[7], 'raw-track/a')
        value['actor_to_track']['7'] = 'a display alias'
        with self.assertRaises(ValueError): IdentityRegistry.from_manifest(value)
        bad_version = self.registry.manifest(); bad_version['schema_version'] = True
        with self.assertRaises(ValueError): IdentityRegistry.from_manifest(bad_version)
        for bad in [{'x': 3, 'y': 3}, {'x': 0}, {'x': True}, {'x': -1}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError): IdentityRegistry(bad)

    def test_track_renaming_and_order_do_not_change_geometry_or_owner(self):
        renamed = IdentityRegistry({'different UUID a': 7, 'different UUID z': 81})
        original = self.scene()
        other = self.scene(registry=renamed, raw_track_ids=tuple(renamed.track_to_actor))
        np.testing.assert_array_equal(original.camera_to_world, other.camera_to_world)
        np.testing.assert_array_equal(original.images, other.images)
        self.assertEqual(original.indices(), other.indices())
        fg = np.array([[1, 1], [0, 1]], bool)
        labels = [[self.registry.track_to_actor['raw-track/z'], 0], [0, self.registry.track_to_actor['raw-track/a']]]
        renamed_labels = [[renamed.track_to_actor['different UUID z'], 0], [0, renamed.track_to_actor['different UUID a']]]
        first = make_label_masks(fg, np.array(labels), self.registry)
        second = make_label_masks(fg, np.array(renamed_labels), renamed)
        np.testing.assert_array_equal(first.owner, second.owner)
        np.testing.assert_array_equal(first.binary_dynamic_mask, second.binary_dynamic_mask)

    def test_unknown_identity_is_foreground_not_background(self):
        fg = np.array([[1, 1, 0], [0, 1, 0]], bool)
        masks = make_label_masks(fg, np.array([[81, 0, 0], [0, -1, 0]], np.int64), self.registry)
        np.testing.assert_array_equal(masks.owner, [[81, -1, 0], [0, -1, 0]])
        np.testing.assert_array_equal(masks.unknown_identity, [[0, 1, 0], [0, 1, 0]])
        exported = masks.export()
        self.assertEqual(exported['binary_dynamic_mask'][0, 1], 255)
        self.assertEqual(exported['unknown_identity'][0, 1], 255)
        self.assertEqual(exported['owner'][0, 1], -1)
        unknown_only = make_label_masks(fg)
        self.assertTrue((unknown_only.owner[fg] == -1).all())

    def test_scene_without_known_tracks_can_keep_unknown_foreground(self):
        scene = self.scene(registry=IdentityRegistry({}), raw_track_ids=())
        self.assertEqual(scene.indices(), (0, 1, 2))
        masks = make_label_masks(np.ones((2, 2), bool), registry=scene.registry)
        self.assertTrue(masks.binary_dynamic_mask.all())
        self.assertTrue(masks.unknown_identity.all())

    def test_bad_owner_contract_is_rejected(self):
        fg = np.array([[1, 0]], bool)
        with self.assertRaises(ValueError): make_label_masks(fg, np.array([[9, 0]]), self.registry)
        with self.assertRaises(ValueError): make_label_masks(fg, np.array([[81, 7]]), self.registry)
        with self.assertRaises(ValueError): make_label_masks(fg, np.array([[81, 0]], np.uint16), self.registry)
        with self.assertRaises(ValueError): LabelMasks(np.array([[0, 0]]), fg)

    def test_no_lidar_positive_does_not_erase_candidate(self):
        candidate = np.array([[0, 1, 1], [0, 1, 0]], np.uint8) * 255
        kept, meta = candidate_observation(candidate)
        np.testing.assert_array_equal(kept, candidate > 0)
        self.assertEqual(meta['support_status'], 'unresolved')
        kept2, no_hit = candidate_observation(candidate, [[20., 20.]])
        np.testing.assert_array_equal(kept2, kept)
        self.assertEqual(no_hit['positive_on_mask'], 0)
        hit, yes = candidate_observation(candidate, [[1., 0.]])
        np.testing.assert_array_equal(hit, kept)
        self.assertEqual(yes['support_status'], 'measured')
        with self.assertRaises(ValueError): candidate_observation(candidate, [[np.nan, 0.]])

    def test_preservation_uses_only_external_support_and_keeps_current_identity(self):
        prior = make_label_masks(np.ones((2, 3), bool), np.array([[81, 81, 81], [81, -1, 7]]), self.registry)
        current = make_label_masks(np.array([[0, 1, 1], [0, 0, 0]], bool),
                                   np.array([[0, 7, -1], [0, 0, 0]]), self.registry)
        support = np.array([[1, 1, 1], [0, 1, 0]], bool)
        result = preserve_supported_observations(current, prior, support, self.registry)
        np.testing.assert_array_equal(result.owner, [[81, 7, -1], [0, -1, 0]])
        self.assertTrue(result.foreground[1, 1])
        self.assertEqual(result.owner[0, 1], 7)  # 不能把前景遮挡者改成旧后车
        with self.assertRaises(ValueError): preserve_supported_observations(current, prior, np.ones((3, 2)), self.registry)

    def test_component_groups_require_explicit_evidence_and_preserve_pose_policy(self):
        group = dict(parent_track='raw-track/z', member_tracks=['raw-track/z', 'raw-track/a'],
                     evidence={'source': 'reviewed images and track motion', 'reference': 'audit.json'})
        result = validate_component_groups([group], self.registry)
        self.assertEqual(result[0]['pose_policy'], 'retain_member_transforms')
        self.assertEqual(result[0]['member_tracks'], ('raw-track/z', 'raw-track/a'))
        with self.assertRaises(ValueError): validate_component_groups([{**group, 'evidence': ''}], self.registry)
        with self.assertRaises(ValueError): validate_component_groups([{**group, 'evidence': '   '}], self.registry)
        with self.assertRaises(ValueError): validate_component_groups([group, group], self.registry)

    def test_short_extrapolation_uses_irregular_seconds_and_reports_inferred(self):
        times = np.array([1000., 1000.05, 1000.2])
        positions = np.column_stack([2 * (times - 1000), -(times - 1000), np.ones(3)])
        values, meta = extrapolate_short_gap(times, positions, [1000.35])
        np.testing.assert_allclose(values, [[.7, -.35, 1]], atol=1e-10)
        self.assertEqual(meta[0]['source'], 'inferred_linear_extrapolation')
        self.assertIs(meta[0]['is_ground_truth'], False)
        self.assertAlmostEqual(meta[0]['horizon_s'], .15)
        self.assertLess(meta[0]['fit_residual_rms'], 1e-12)
        self.assertEqual(meta[0]['support_timestamps_s'], times.tolist())
        backward, _ = extrapolate_short_gap(times, positions, [999.9])
        np.testing.assert_allclose(backward, [[-.2, .1, 1]], atol=1e-10)

    def test_gap_limit_fit_residual_and_bad_time_rejection(self):
        t = [0., .1, .2]; noisy = [[0., 0.], [.12, 0.], [.2, 0.]]
        _, meta = extrapolate_short_gap(t, noisy, [.5], max_gap_s=.3)
        self.assertGreater(meta[0]['fit_residual_rms'], 0)
        with self.assertRaises(ValueError): extrapolate_short_gap(t, noisy, [.5001], max_gap_s=.3)
        with self.assertRaises(ValueError): extrapolate_short_gap(t, noisy, [.15])
        with self.assertRaises(ValueError): extrapolate_short_gap([0., 0., .2], noisy, [.3])
        with self.assertRaises(ValueError): extrapolate_short_gap(t, np.repeat(np.eye(4)[None], 3, axis=0), [.3])


if __name__ == '__main__':
    unittest.main()
