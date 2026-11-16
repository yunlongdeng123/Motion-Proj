"""CPU regressions for measured occluder proposals and independent image QA."""
from copy import deepcopy
import unittest
import numpy as np
from v44_occluders import OccluderConfig, propose_occluders, certify_occluder


class OccluderTests(unittest.TestCase):
    def setUp(self):
        self.shape = (10, 10)
        self.points = np.array([[1.1, 1.1], [2.1, 1.1], [1.1, 2.1], [2.1, 2.1]])
        self.depth = np.array([3., 3.05, 3., 3.1])
        self.own = np.array([[7.1, 7.1], [8.1, 8.1]])
        self.cfg = OccluderConfig(max_pixel_gap=1.5, relative_depth_gap=0., negative_exclusion_pixels=.3)
        self.mask = np.zeros(self.shape, bool); self.mask[1:3, 1:3] = True
        self.sem = self.mask.copy()

    def proposal(self, own=None, **kwargs):
        return propose_occluders(self.points, self.depth, self.own if own is None else own,
                                 self.shape, config=self.cfg, **kwargs)["proposals"][0]

    def certify(self, request=None, mask=None, **kwargs):
        return certify_occluder(self.proposal() if request is None else request,
                                self.mask if mask is None else mask, kwargs.pop("score", .95),
                                semantic_nonvehicle=kwargs.pop("semantic_nonvehicle", self.sem),
                                config=self.cfg, **kwargs)

    def test_foreground_clusters_do_not_cross_depth_jump(self):
        points = np.vstack([self.points, self.points])
        depth = np.r_[self.depth, self.depth + 8.]
        result = propose_occluders(points, depth, [], self.shape, config=self.cfg)
        self.assertEqual(len(result["proposals"]), 2)
        for request in result["proposals"]:
            self.assertLess(np.ptp(depth[request["source_indices"]]), .2)

    def test_spatially_disconnected_surfaces_stay_separate(self):
        result = propose_occluders(np.vstack([self.points, self.points + 5.]),
                                   np.tile(self.depth, 2), [], self.shape, config=self.cfg)
        self.assertEqual(len(result["proposals"]), 2)

    def test_smooth_local_depth_variation_can_form_one_surface(self):
        result = propose_occluders([[1, 1], [2, 1], [3, 1]], [3., 3.4, 3.8], [],
                                   self.shape, config=self.cfg)
        self.assertEqual(len(result["proposals"]), 1)

    def test_empty_measurements_cannot_make_proposal_or_mask(self):
        result = propose_occluders([], [], self.own, self.shape, config=self.cfg)
        self.assertFalse(result["proposals"])
        certified = self.certify(request={"status": "unknown"}, mask=np.ones(self.shape, bool))
        self.assertEqual(certified["status"], "unknown")
        self.assertIsNone(certified["accepted_occluder"])

    def test_sparse_or_repeated_measurements_stay_unknown(self):
        for points, depth in [([[1.1, 1.1]], [3.]), ([[1.1, 1.1]] * 20, [3.] * 20)]:
            result = propose_occluders(points, depth, [], self.shape, config=self.cfg)
            self.assertFalse(result["proposals"])
            self.assertEqual(result["unknown"][0]["reason"], "sparse_measurements")

    def test_proposals_keep_exact_source_indices_after_invalid_inputs(self):
        points = np.vstack([[np.nan, 1], self.points, [-1, 0]])
        result = propose_occluders(points, np.r_[3., self.depth, 3.], [[-1, 0], [8.1, 8.1]],
                                   self.shape, config=self.cfg)
        request = result["proposals"][0]
        np.testing.assert_array_equal(request["positive_uv"], points[request["positive_indices"]])
        self.assertEqual(request["negative_indices"], [1])
        self.assertEqual(request["preserved_own_indices"], [0, 1])
        self.assertEqual(result["invalid_foreground_indices"], [0, 5])

    def test_selected_positive_points_are_real_and_spatially_dispersed(self):
        points = np.array([[x + .1, y + .1] for x in range(1, 5) for y in range(1, 3)])
        cfg = OccluderConfig(max_pixel_gap=1.5, max_positives=3)
        request = propose_occluders(points, np.full(len(points), 3.), [], self.shape, config=cfg)["proposals"][0]
        selected = np.asarray(request["positive_uv"])
        np.testing.assert_array_equal(selected, points[request["positive_indices"]])
        self.assertEqual(len(selected), 3)
        self.assertGreaterEqual(np.ptp(selected[:, 0]), 2.)

    def test_negatives_come_only_from_actual_own_samples(self):
        request = self.proposal()
        np.testing.assert_array_equal(request["negative_uv"], self.own[request["negative_indices"]])
        self.assertEqual(request["preserved_own_indices"], [0, 1])

    def test_duplicate_own_pixels_do_not_repeat_negative_prompts(self):
        own = np.array([[7.1, 7.1], [7.2, 7.2], [8.1, 8.1]])
        request = self.proposal(own=own)
        self.assertEqual(len(request["negative_indices"]), 2)
        self.assertEqual(request["preserved_own_indices"], [0, 1, 2])

    def test_valid_external_image_occluder_is_accepted(self):
        result = self.certify()
        self.assertEqual(result["status"], "accepted_occluder")
        np.testing.assert_array_equal(result["accepted_occluder"], self.mask)

    def test_missing_positive_keeps_unknown(self):
        mask = self.mask.copy(); mask[1, 1] = False
        result = self.certify(mask=mask)
        self.assertEqual(result["reason"], "measured_positive_not_covered")
        self.assertIsNone(result["accepted_occluder"])

    def test_covering_own_negative_keeps_unknown(self):
        mask = self.mask.copy(); mask[7, 7] = True
        result = self.certify(mask=mask, semantic_nonvehicle=np.ones(self.shape, bool))
        self.assertEqual(result["reason"], "own_negative_conflict")
        self.assertIsNone(result["accepted_occluder"])

    def test_low_or_missing_semantic_support_keeps_unknown(self):
        missing = self.certify(semantic_nonvehicle=None)
        weak = self.certify(semantic_nonvehicle=np.zeros(self.shape, bool))
        self.assertEqual(missing["status"], "unknown")
        self.assertEqual(weak["reason"], "insufficient_nonvehicle_support")

    def test_nonvehicle_fraction_is_checked_inside_external_mask(self):
        mask = np.zeros(self.shape, bool); mask[:4, :4] = True
        result = self.certify(mask=mask)
        self.assertEqual(result["reason"], "insufficient_nonvehicle_support")

    def test_own_samples_are_not_rejected_by_their_semantic_class(self):
        support = self.sem.copy(); support[7, 7] = True; support[8, 8] = True
        first, second = self.certify(), self.certify(semantic_nonvehicle=support)
        self.assertEqual(first["status"], "accepted_occluder")
        self.assertEqual(second["status"], first["status"])
        self.assertEqual(second["preserved_own_indices"], [0, 1])

    def test_depth_overlap_alone_makes_no_accepted_mask(self):
        own = np.vstack([[1.2, 1.2], self.own])
        request = self.proposal(own=own)
        self.assertEqual(request["overlapping_own_indices"], [0])
        self.assertNotIn("accepted_occluder", request)
        self.assertNotIn(0, request["negative_indices"])

    def test_independent_image_evidence_can_hide_observation_but_preserve_source(self):
        request = self.proposal(own=np.vstack([[1.2, 1.2], self.own]))
        result = self.certify(request=request)
        self.assertEqual(result["status"], "accepted_occluder")
        self.assertEqual(result["hidden_own_indices"], [0])
        self.assertEqual(result["preserved_own_indices"], [0, 1, 2])

    def test_actor_id_renaming_has_no_effect(self):
        first, second = deepcopy(self.proposal()), deepcopy(self.proposal())
        first["actor_id"], second["actor_id"] = "arbitrary-one", "renamed-982"
        a, b = self.certify(request=first), self.certify(request=second)
        for key in ("status", "reason", "metrics", "hidden_own_indices", "preserved_own_indices"):
            self.assertEqual(a[key], b[key])
        np.testing.assert_array_equal(a["accepted_occluder"], b["accepted_occluder"])

    def test_predictor_score_must_be_finite_and_sufficient(self):
        for score in [.5, np.nan, np.inf, 1.1, True, "high", None, [.95]]:
            self.assertEqual(self.certify(score=score)["status"], "unknown")

    def test_mask_resolution_and_binary_values_are_required(self):
        for mask in [np.ones((5, 5), bool), np.full(self.shape, .2)]:
            self.assertEqual(self.certify(mask=mask)["reason"], "invalid_source_mask")

    def test_raw_semantic_class_ids_are_not_silently_cast_to_support(self):
        result = self.certify(semantic_nonvehicle=np.full(self.shape, 2, np.uint8))
        self.assertEqual(result["reason"], "invalid_semantic_support")

    def test_proposal_budget_records_unknown_instead_of_fabrication(self):
        cfg = OccluderConfig(max_pixel_gap=1.5, max_proposals=1)
        result = propose_occluders(np.vstack([self.points, self.points + 5.]),
                                   np.tile(self.depth, 2), [], self.shape, config=cfg)
        self.assertEqual(len(result["proposals"]), 1)
        self.assertEqual(result["unknown"][0]["reason"], "proposal_limit")

    def test_no_input_or_returned_mask_alias_mutation(self):
        points, own, mask, support = (x.copy() for x in (self.points, self.own, self.mask, self.sem))
        request = self.proposal(semantic_nonvehicle=support)
        result = self.certify(request=request, mask=mask, semantic_nonvehicle=support)
        result["accepted_occluder"][:] = False
        for a, b in [(points, self.points), (own, self.own), (mask, self.mask), (support, self.sem)]:
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
