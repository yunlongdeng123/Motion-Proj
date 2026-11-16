"""纯数组公共流水线的真实故障回归，不读模型或调用 GPU。"""
import unittest
import numpy as np
from v44_inputs import IdentityRegistry
from v44_visibility import Evidence, visibility_evidence
from v44_temporal import AcceptedObservation, TemporalCandidate, select_track
from v44_prepare import FrameMaskEvidence, prepare_frame_masks


class FramePreparationTests(unittest.TestCase):
    def setUp(self):
        self.shape = (4, 5); self.t = 17.36
        self.registry = IdentityRegistry({'raw-track-A': 81, 'other/raw/UUID': 7})

    def mask(self, *pixels):
        m = np.zeros(self.shape, bool)
        for y, x in pixels: m[y, x] = True
        return m

    def prepare(self, evidence, **kwargs):
        return prepare_frame_masks(evidence, self.registry, timestamp_s=self.t, shape=self.shape, **kwargs)

    def accepted(self, mask, **kwargs):
        return FrameMaskEvidence(candidate=mask, candidate_accepted=True,
                                 candidate_provenance='independent source-image QA', **kwargs)

    def depth(self, value):
        return dict(measured_depth=np.full(self.shape, value, float),
                    measured_depth_valid=np.ones(self.shape, bool), depth_provenance='actual same-frame LiDAR')

    def test_truck_without_positive_keeps_candidate_pending_without_inventing_vehicle_class(self):
        truck = self.mask((1, 2), (1, 3), (2, 3))
        labels, trace = self.prepare({81: FrameMaskEvidence(candidate=truck,
                                       candidate_provenance='actual SAM silhouette; not yet identity-reviewed')})
        np.testing.assert_array_equal(trace['semantic_unresolved_regions'], truck)
        self.assertFalse(labels.foreground.any())
        self.assertFalse(trace['training_ready'])
        self.assertFalse(trace['binary_dynamic_mask_is_training_gt'])
        self.assertEqual(trace['actors'][81]['candidate_pixels'], int(truck.sum()))
        self.assertEqual(trace['actors'][81]['candidate_point_support']['support_status'], 'unresolved')
        self.assertEqual(trace['actors'][81]['chosen_kind'], 'unknown')
        self.assertIn('candidate_vehicle_semantics_unresolved', trace['training_blockers'])
        self.assertEqual(trace['background_semantics'], 'unresolved_regions_are_not_known_background')

    def test_unreviewed_sam_sign_is_not_automatically_a_vehicle(self):
        sign = self.mask((1, 1), (2, 1))
        labels, trace = self.prepare({81: FrameMaskEvidence(candidate=sign, candidate_provenance='raw SAM box candidate')})
        self.assertFalse(labels.foreground.any()); self.assertFalse(labels.unknown_identity.any())
        self.assertFalse(trace['training_ready'])
        np.testing.assert_array_equal(trace['semantic_unresolved_regions'], sign)
        resolved, ready = self.prepare({81: FrameMaskEvidence(candidate=sign)},
            certified_nonvehicle_occluder=AcceptedObservation(sign, 'independently certified actual sign surface'))
        self.assertFalse(resolved.foreground.any()); self.assertTrue(ready['training_ready'])
        self.assertFalse(ready['semantic_unresolved_regions'].any())

    def test_independent_vehicle_class_support_keeps_unknown_owner_as_binary_one(self):
        candidate = self.mask((1, 1), (1, 2), (2, 2)); confirmed = self.mask((1, 1), (1, 2))
        item = FrameMaskEvidence(candidate=candidate, candidate_vehicle_support=confirmed)
        partial, blocked = self.prepare({81: item})
        np.testing.assert_array_equal(partial.foreground, confirmed)
        self.assertTrue((partial.owner[confirmed] == -1).all()); self.assertFalse(blocked['training_ready'])
        self.assertTrue(blocked['semantic_unresolved_regions'][2, 2])
        complete, ready = self.prepare({81: item}, vehicle_foreground=self.mask((2, 2)))
        np.testing.assert_array_equal(complete.foreground, candidate)
        self.assertTrue((complete.owner[candidate] == -1).all()); self.assertTrue(ready['training_ready'])
        self.assertAlmostEqual(np.mean(complete.binary_dynamic_mask.astype(float) ** 2), 3 / 20)
        self.assertFalse(ready['loss_weights_modified'])

    def test_no_positive_is_not_a_veto_for_independently_accepted_observation(self):
        mask = self.mask((2, 1), (2, 2))
        labels, trace = self.prepare({81: self.accepted(mask)})
        self.assertTrue((labels.owner[mask] == 81).all())
        self.assertEqual(trace['actors'][81]['candidate_point_support']['positive_count'], 0)
        self.assertTrue(trace['training_ready'])

    def test_certified_sign_changes_only_visible_mask_not_hidden_3d_state(self):
        truck = self.mask((0, 0), (0, 1), (1, 1)); sign = self.mask((0, 1))
        persistent = {81: np.array([[1., 2., 3.], [4., 5., 6.]])}; before = persistent[81].copy()
        source_before = truck.copy()
        labels, trace = self.prepare({81: self.accepted(truck)},
            certified_nonvehicle_occluder=AcceptedObservation(sign, 'independent sign mask and depth audit'))
        self.assertFalse(labels.foreground[0, 1]); self.assertEqual(labels.owner[0, 1], 0)
        self.assertEqual(labels.owner[0, 0], 81)
        np.testing.assert_array_equal(persistent[81], before)
        np.testing.assert_array_equal(truck, source_before)
        self.assertFalse(trace['persistent_3d_state_modified']); self.assertIn(81, trace['persistent_actor_ids'])

    def test_darkcar_weak_candidate_falls_back_to_accepted_same_frame(self):
        old = self.mask((1, 1), (1, 2), (2, 1), (2, 2)); weak = self.mask((1, 1))
        item = FrameMaskEvidence(candidate=weak, weak_candidate=True,
            accepted_same_frame=AcceptedObservation(old, 'reviewed dark glass and tail light'), accepted_timestamp_s=self.t)
        labels, trace = self.prepare({81: item})
        np.testing.assert_array_equal(labels.foreground, old)
        self.assertTrue((labels.owner[old] == 81).all())
        self.assertEqual(trace['actors'][81]['chosen_kind'], 'accepted_same_frame_fallback')
        self.assertTrue(trace['training_ready'])
        with self.assertRaises(ValueError):
            self.prepare({81: FrameMaskEvidence(accepted_same_frame=item.accepted_same_frame,
                                               accepted_timestamp_s=self.t - .1)})

    def test_rejected_candidate_extension_stays_semantically_pending_outside_trusted_old_car(self):
        car = self.mask((1, 1)); possible_sign = self.mask((1, 2)); candidate = car | possible_sign
        item = FrameMaskEvidence(candidate=candidate, weak_candidate=True,
            accepted_same_frame=AcceptedObservation(car, 'same-frame accepted vehicle'), accepted_timestamp_s=self.t)
        labels, trace = self.prepare({81: item})
        np.testing.assert_array_equal(labels.foreground, car)
        self.assertFalse(trace['training_ready']); self.assertTrue(trace['semantic_unresolved_regions'][1, 2])
        _, resolved = self.prepare({81: item},
            certified_nonvehicle_occluder=AcceptedObservation(possible_sign, 'independent nonvehicle image surface'))
        self.assertTrue(resolved['training_ready'])

    def test_supported_old_body_preserved_without_overwriting_other_accepted_owner(self):
        old = self.mask((1, 1), (1, 2), (1, 3)); new = self.mask((1, 1)); other = self.mask((1, 3))
        first = self.accepted(new, accepted_same_frame=AcceptedObservation(old, 'accepted original pixels'),
                              accepted_timestamp_s=self.t, retention_support=old)
        labels, trace = self.prepare({81: first, 7: self.accepted(other)})
        self.assertEqual(labels.owner[1, 2], 81); self.assertEqual(labels.owner[1, 3], 7)
        self.assertEqual(trace['actors'][81]['supported_old_pixels_restored'], 1)

    def test_two_supported_old_silhouettes_do_not_use_iteration_order_as_identity(self):
        old_a = self.mask((1, 1), (1, 2)); old_b = self.mask((1, 2), (1, 3))
        a = self.accepted(self.mask((1, 1)), accepted_same_frame=AcceptedObservation(old_a, 'same-frame QA A'),
                          accepted_timestamp_s=self.t, retention_support=old_a)
        b = self.accepted(self.mask((1, 3)), accepted_same_frame=AcceptedObservation(old_b, 'same-frame QA B'),
                          accepted_timestamp_s=self.t, retention_support=old_b)
        first, _ = self.prepare({81: a, 7: b}); second, _ = self.prepare({7: b, 81: a})
        self.assertEqual(first.owner[1, 2], -1)
        self.assertTrue(first.foreground[1, 2])
        np.testing.assert_array_equal(first.owner, second.owner)

    def test_weak_unknown_candidate_cannot_replace_an_accepted_foreground_owner(self):
        front = self.mask((1, 1), (1, 2)); doubtful = self.mask((1, 2), (1, 3))
        labels, _ = self.prepare({81: self.accepted(front),
                                  7: FrameMaskEvidence(candidate=doubtful, weak_candidate=True,
                                                       candidate_vehicle_support=doubtful)})
        self.assertEqual(labels.owner[1, 2], 81); self.assertEqual(labels.owner[1, 3], -1)

    def test_white_wheel_conflict_without_actual_depth_is_unknown(self):
        white = self.mask((1, 1), (1, 2)); other = self.mask((1, 2), (1, 3))
        labels, trace = self.prepare({81: self.accepted(white), 7: self.accepted(other)})
        self.assertEqual(labels.owner[1, 1], 81); self.assertEqual(labels.owner[1, 3], 7)
        self.assertEqual(labels.owner[1, 2], -1); self.assertTrue(labels.foreground[1, 2])
        self.assertEqual(trace['overlap']['unresolved_overlap_pixels'], 1)

    def test_depth_only_competes_actual_overlap_and_ties_are_unknown(self):
        a = self.mask((0, 0), (1, 1)); b = self.mask((1, 1), (2, 2))
        labels, _ = self.prepare({81: self.accepted(a, **self.depth(10)), 7: self.accepted(b, **self.depth(5))})
        self.assertEqual(labels.owner[1, 1], 7); self.assertEqual(labels.owner[0, 0], 81)
        equal, _ = self.prepare({81: self.accepted(a, **self.depth(5)), 7: self.accepted(b, **self.depth(5))})
        self.assertEqual(equal.owner[1, 1], -1)
        empty_near_box = np.zeros(self.shape, bool)
        solo, _ = self.prepare({81: self.accepted(a, **self.depth(10)),
                                7: self.accepted(empty_near_box, **self.depth(1))})
        self.assertTrue((solo.owner[a] == 81).all())

    def test_box_distance_scalar_is_not_accepted_as_measured_depth(self):
        with self.assertRaises(ValueError):
            self.prepare({81: self.accepted(self.mask((1, 1)), measured_depth=1.,
                measured_depth_valid=np.ones(self.shape, bool), depth_provenance='an OBB distance')})
        with self.assertRaises(ValueError):
            self.prepare({81: self.accepted(self.mask((1, 1)), measured_depth=np.ones(self.shape))})

    def test_vehicle_unknown_support_uses_complete_binary_loss_target(self):
        known = self.mask((0, 0)); vehicle = self.mask((0, 0), (2, 2), (3, 4))
        labels, trace = self.prepare({81: self.accepted(known)}, vehicle_foreground=vehicle)
        self.assertEqual(labels.owner[0, 0], 81)
        self.assertTrue(labels.unknown_identity[2, 2]); self.assertTrue(labels.unknown_identity[3, 4])
        target = labels.binary_dynamic_mask.astype(float)
        self.assertAlmostEqual(np.mean((np.zeros(self.shape) - target) ** 2), 3 / 20)
        self.assertEqual(trace['loss_domain'], 'full_image_raw_alpha_mse'); self.assertFalse(trace['loss_weights_modified'])

    def test_measured_foreground_veto_keeps_unclassified_foreground_unknown(self):
        mask = self.mask((1, 1), (1, 2)); state = np.zeros(self.shape, np.uint8)
        state[1, 1] = Evidence.MEASURED_FOREGROUND; state[1, 2] = Evidence.MIXED
        labels, _ = self.prepare({81: self.accepted(mask, visibility_state=state)})
        self.assertEqual(labels.owner[1, 1], -1); self.assertTrue(labels.foreground[1, 1])
        self.assertEqual(labels.owner[1, 2], 81)

    def test_actual_visibility_module_output_is_accepted(self):
        cfg_shape = self.shape
        vis = visibility_evidence([[1.1, 1.1]], [5.], np.array([True]), cfg_shape, expected_depth=5.)
        labels, _ = self.prepare({81: self.accepted(self.mask((1, 1)), visibility_state=vis['state'])})
        self.assertEqual(labels.owner[1, 1], 81)

    def test_temporal_weak_replay_uses_same_timestamp_accepted_fallback(self):
        old = self.mask((1, 1), (1, 2)); weak = self.mask((1, 1))
        decisions = select_track(81, {self.t: TemporalCandidate(weak, .9, .9, 'propagated', weak_evidence=True)},
                                 {}, previous={self.t: AcceptedObservation(old, 'same-frame reviewed observation')})
        labels, trace = self.prepare({81: FrameMaskEvidence(candidate=weak, temporal_decision=decisions[self.t])})
        self.assertTrue((labels.owner[old] == 81).all())
        self.assertEqual(trace['actors'][81]['chosen_kind'], 'temporal_previous')

    def test_arbitrary_renamed_ids_do_not_change_mask_geometry(self):
        a = self.mask((1, 1), (1, 2)); b = self.mask((1, 2), (2, 2))
        first, _ = self.prepare({81: self.accepted(a), 7: self.accepted(b)})
        renamed = IdentityRegistry({'renamed': 1003, 'other-renamed': 29})
        second, _ = prepare_frame_masks({29: self.accepted(b), 1003: self.accepted(a)}, renamed,
                                        timestamp_s=self.t, shape=self.shape)
        np.testing.assert_array_equal(first.foreground, second.foreground)
        np.testing.assert_array_equal(first.unknown_identity, second.unknown_identity)
        np.testing.assert_array_equal(first.owner == 81, second.owner == 1003)

    def test_missing_source_or_wrong_grid_rejected_instead_of_silent_background(self):
        m = self.mask((1, 1))
        cases = [{81: FrameMaskEvidence(candidate=m, candidate_accepted=True)},
                 {81: FrameMaskEvidence(retention_support=m)},
                 {81: FrameMaskEvidence(candidate=m, candidate_vehicle_support=np.ones((2, 2), bool))},
                 {81: self.accepted(m, visibility_state=np.zeros((2, 2), np.uint8))},
                 {999: self.accepted(m)}]
        for case in cases:
            with self.subTest(case=list(case)), self.assertRaises(ValueError): self.prepare(case)
        with self.assertRaises(ValueError):
            self.prepare({81: self.accepted(m)}, certified_nonvehicle_occluder=AcceptedObservation(m, None))


if __name__ == '__main__':
    unittest.main()
