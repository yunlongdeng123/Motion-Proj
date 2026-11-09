"""纯CPU合同：任意ID、断链、unknown、显式seed与实际轮廓竞争。"""
import unittest
import numpy as np
from v44_temporal import (AcceptedObservation, TemporalCandidate, TemporalConfig,
                          select_track, resolve_actual_overlap)


def pixels(*positions, shape=(3, 5)):
    result = np.zeros(shape, bool)
    for y, x in positions:
        result[y, x] = True
    return result


def candidate(mask, **kwargs):
    values = dict(mask=mask, visibility=.99, iou=.9, provenance="actual_video_candidate")
    values.update(kwargs)
    return TemporalCandidate(**values)


class TemporalContracts(unittest.TestCase):
    def setUp(self):
        self.body = pixels((1, 1), (1, 2))
        self.seed = AcceptedObservation(self.body, {"source": "reviewed_actual_seed"})

    def test_empty_138_breaks_chain_before_high_score_130(self):
        # 数字只作为测试时间戳；核心没有这些帧号。
        candidates = {130: candidate(self.body), 190: candidate(self.body, forced_confidence=True),
                      180: candidate(self.body), 138: candidate(pixels())}
        result = select_track("arbitrary_track", candidates, {190: self.seed})
        self.assertEqual(result[180].status, "accepted")
        self.assertEqual(result[138].status, "rejected")
        self.assertEqual(result[130].status, "rejected")
        self.assertIn("disconnected_after_actual_absence", result[130].reasons)
        self.assertIn((190, 138), result[130].blocked_by)
        self.assertTrue(result[130].identity_unknown)
        self.assertIsNone(result[130].mask)   # unknown不是一张全零背景mask

    def test_absent_score_or_explicit_absence_breaks_forward_chain(self):
        for absent_candidate in (candidate(self.body, visibility=.5), candidate(self.body, absent=True)):
            with self.subTest(absence=absent_candidate):
                result = select_track(801, {200: absent_candidate, 210: candidate(self.body)}, {190: self.seed})
                self.assertEqual(result[210].status, "rejected")
                self.assertIn((190, 200), result[210].blocked_by)

    def test_id_renaming_and_input_order_do_not_change_temporal_decision(self):
        rows = {10: candidate(self.body), 20: candidate(self.body, weak_evidence=True)}
        left = select_track("car_name_is_not_evidence", rows, {30: self.seed})
        right = select_track(("opaque", 937), dict(reversed(list(rows.items()))), {30: self.seed})
        for timestamp in left:
            self.assertEqual(left[timestamp].status, right[timestamp].status)
            self.assertEqual(left[timestamp].reasons, right[timestamp].reasons)
            self.assertEqual(left[timestamp].chosen_provenance, right[timestamp].chosen_provenance)
            np.testing.assert_equal(left[timestamp].mask, right[timestamp].mask)

    def test_forced_seed_scores_require_separate_actual_seed_support(self):
        forced = candidate(self.body, visibility=1., iou=1., forced_confidence=True)
        unsupported = select_track("seed", {10: forced}, {})[10]
        self.assertEqual(unsupported.status, "rejected")
        self.assertIn("forced_confidence_is_not_independent_evidence", unsupported.reasons)
        # 即使模型强制分数低，明确实拍seed依据仍独立成立；不统计为模型置信。
        supported = select_track("seed", {10: candidate(self.body, visibility=0., iou=0., forced_confidence=True)}, {10: self.seed})[10]
        self.assertEqual(supported.status, "accepted")
        self.assertEqual(supported.chosen_kind, "seed_support")
        self.assertEqual(supported.chosen_provenance, self.seed.provenance)
        self.assertFalse(supported.independent_confidence)
        later = select_track("seed", {9: forced}, {10: self.seed})[9]
        self.assertEqual(later.status, "rejected")

    def test_weak_candidate_falls_back_only_to_same_timestamp_observation(self):
        old = AcceptedObservation(pixels((0, 0)), "previously_accepted_source_at_20")
        neighbour = AcceptedObservation(pixels((2, 4)), "other_frame_19")
        weak = candidate(self.body, weak_evidence=True)
        result = select_track("track", {20: weak}, {30: self.seed}, previous={20: old, 19: neighbour})
        self.assertEqual(result[20].status, "rejected")
        self.assertEqual(result[20].chosen_kind, "previous")
        self.assertEqual(result[20].chosen_provenance, old.provenance)
        self.assertFalse(result[20].identity_unknown)
        np.testing.assert_array_equal(result[20].mask, old.mask)
        no_same_frame = select_track("track", {20: weak}, {30: self.seed}, previous={19: neighbour})[20]
        self.assertTrue(no_same_frame.identity_unknown)
        self.assertIsNone(no_same_frame.mask)
        np.testing.assert_array_equal(old.mask, pixels((0, 0)))

    def test_unobserved_mask_and_low_confidence_do_not_become_background(self):
        missing = candidate(None, visibility=None, iou=None)
        rows = {20: missing, 21: candidate(self.body, iou=.1)}
        result = select_track("track", rows, {30: self.seed})
        self.assertEqual(result[20].status, "unknown")
        self.assertEqual(result[21].status, "rejected")
        for timestamp in rows:
            self.assertTrue(result[timestamp].identity_unknown)
            self.assertIsNone(result[timestamp].mask)

    def test_geometry_can_veto_but_never_establish_identity(self):
        compatible = candidate(self.body, geometry_support=np.ones(self.body.shape, bool))
        self.assertEqual(select_track("id", {1: compatible}, {})[1].status, "rejected")
        incompatible = candidate(self.body, geometry_support=pixels())
        old = AcceptedObservation(pixels((0, 0)), "prior")
        result = select_track("id", {1: incompatible}, {2: self.seed}, previous={1: old})[1]
        self.assertIn("geometry_incompatible", result.reasons)
        self.assertEqual(result.chosen_kind, "previous")

    def test_only_new_explicit_seed_can_rebuild_identity_after_gap(self):
        rows = {130: candidate(self.body), 138: candidate(pixels()), 120: candidate(self.body)}
        result = select_track("id", rows, {190: self.seed, 130: AcceptedObservation(self.body, "new_independent_review")})
        self.assertEqual(result[130].chosen_kind, "seed_support")
        self.assertEqual(result[120].status, "accepted")
        self.assertEqual(result[120].connected_seeds, (130,))

    def test_grid_mismatch_and_missing_seed_provenance_are_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            select_track("id", {1: candidate(np.ones((2, 2), bool))}, {2: self.seed})
        with self.assertRaises(ValueError):
            select_track("id", {}, {2: AcceptedObservation(self.body, None)})


class ActualOverlapContracts(unittest.TestCase):
    def test_near_obb_without_actual_mask_does_not_occlude_car(self):
        car = pixels((1, 1), (1, 2))
        for absent_mask in (None, pixels()):
            result = resolve_actual_overlap({"car": car, "near_box": absent_mask}, {"car": 10., "near_box": .1})
            np.testing.assert_array_equal(result.owned_masks["car"], car)
            np.testing.assert_array_equal(result.foreground_mask, car)
            self.assertFalse(result.unknown_mask.any())

    def test_only_actual_overlap_competes_and_renaming_is_invariant(self):
        left, right = pixels((1, 1), (1, 2)), pixels((1, 2), (1, 3))
        result = resolve_actual_overlap({"left": left, "right": right}, {"left": 10., "right": 2.})
        renamed = resolve_actual_overlap({87: left, ("x", 3): right}, {87: 10., ("x", 3): 2.})
        np.testing.assert_array_equal(result.owned_masks["left"], pixels((1, 1)))
        np.testing.assert_array_equal(result.owned_masks["right"], right)
        np.testing.assert_array_equal(result.owned_masks["left"], renamed.owned_masks[87])
        np.testing.assert_array_equal(result.foreground_mask, left | right)

    def test_missing_depth_or_tie_keeps_overlap_unknown(self):
        left, right = pixels((1, 1), (1, 2)), pixels((1, 2), (1, 3))
        for depths in ({"a": 1.}, {"a": 1., "b": 1.}):
            result = resolve_actual_overlap({"a": left, "b": right}, depths)
            self.assertTrue(result.unknown_mask[1, 2])
            self.assertFalse(result.owned_masks["a"][1, 2])
            self.assertFalse(result.owned_masks["b"][1, 2])
            self.assertTrue(result.foreground_mask[1, 2])

    def test_adjacent_component_never_acquires_neighbour_actor_identity(self):
        body, unassigned_piece = pixels((1, 1)), pixels((1, 2))
        result = resolve_actual_overlap({"truck": body}, {"truck": 5.}, unknown_mask=unassigned_piece)
        np.testing.assert_array_equal(result.owned_masks["truck"], body)
        np.testing.assert_array_equal(result.unknown_mask, unassigned_piece)
        # 即使调用方明确给了另一个部件ID，距离近也不把两个ID合并。
        separate = resolve_actual_overlap({"truck": body, "explicit_other_owner": unassigned_piece}, {})
        np.testing.assert_array_equal(separate.owned_masks["truck"], body)
        np.testing.assert_array_equal(separate.owned_masks["explicit_other_owner"], unassigned_piece)


if __name__ == "__main__":
    unittest.main()
