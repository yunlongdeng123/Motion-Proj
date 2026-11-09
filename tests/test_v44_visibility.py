"""CPU regressions for measured, multilayer visibility evidence."""
import unittest
import numpy as np
from v44_visibility import Evidence, VisibilityConfig, visibility_evidence


class VisibilityTests(unittest.TestCase):
    def evidence(self, uv, depth, own, **kwargs):
        return visibility_evidence(np.asarray(uv, float).reshape(-1, 2),
                                   np.asarray(depth, float), np.asarray(own, bool),
                                   kwargs.pop("image_shape", (4, 4)),
                                   output_shape=kwargs.pop("output_shape", (1, 1)),
                                   own_near=kwargs.pop("own_near", 10.),
                                   own_far=kwargs.pop("own_far", 12.), **kwargs)

    def test_empty_projected_volume_cannot_occlude(self):
        result = self.evidence([], [], [])
        self.assertEqual(result["state"][0, 0], Evidence.UNKNOWN)
        self.assertFalse(result["conflict"].any())
        self.assertEqual(result["counts"]["valid_returns"], 0)

    def test_same_coarse_pixel_with_two_surfaces_is_mixed(self):
        result = self.evidence([[.2, .2], [1.2, .2], [2.2, .2]],
                               [10.5, 3., 3.1], [True, False, False])
        self.assertEqual(result["state"][0, 0], Evidence.MIXED)
        self.assertEqual(result["own_pixels"][0, 0], 1)
        self.assertEqual(result["foreground_pixels"][0, 0], 2)
        self.assertFalse(result["conflict"].any())

    def test_same_native_pixel_with_two_returns_is_also_mixed(self):
        result = self.evidence([[.2, .2], [.2, .2]], [10.5, 3.], [True, False],
                               output_shape=(4, 4))
        self.assertEqual(result["state"][0, 0], Evidence.MIXED)
        self.assertFalse(result["conflict"].any())

    def test_sparse_foreground_measurement_stays_unknown(self):
        result = self.evidence([[.2, .2]], [3.], [False])
        self.assertEqual(result["state"][0, 0], Evidence.UNKNOWN)
        self.assertEqual(result["foreground_pixels"][0, 0], 1)

    def test_duplicate_returns_do_not_fake_dense_surface_support(self):
        result = self.evidence([[.2, .2]] * 10, [3.] * 10, [False] * 10)
        self.assertEqual(result["foreground_returns"][0, 0], 10)
        self.assertEqual(result["foreground_pixels"][0, 0], 1)
        self.assertEqual(result["state"][0, 0], Evidence.UNKNOWN)

    def test_measured_foreground_sign_can_produce_conflict(self):
        result = self.evidence([[.2, .2], [1.2, .2]], [3., 3.1], [False, False])
        self.assertEqual(result["state"][0, 0], Evidence.MEASURED_FOREGROUND)
        self.assertTrue(result["conflict"][0, 0])

    def test_far_measurements_do_not_prove_background(self):
        result = self.evidence([[.2, .2], [1.2, .2]], [30., 31.], [False, False])
        self.assertEqual(result["state"][0, 0], Evidence.UNKNOWN)
        self.assertEqual(result["unassigned_pixels"][0, 0], 2)

    def test_nearby_foreground_does_not_veto_own_or_empty_neighbors(self):
        result = self.evidence([[.2, .2], [1.2, .2], [2.2, .2]],
                               [3., 3., 10.5], [False, False, True],
                               image_shape=(4, 8), output_shape=(2, 4))
        self.assertEqual(result["state"][0, 0], Evidence.MEASURED_FOREGROUND)
        self.assertEqual(result["state"][0, 1], Evidence.SUPPORTED)
        self.assertEqual(result["state"][1, 0], Evidence.UNKNOWN)

    def test_semantic_nonvehicle_hint_does_not_veto_own_body(self):
        result = self.evidence([[.2, .2]], [10.5], [True], semantic_hint=np.array([[2]]))
        self.assertEqual(result["state"][0, 0], Evidence.SUPPORTED)
        self.assertFalse(result["conflict"].any())

    def test_exact_surface_reference_can_find_foreground_inside_volume(self):
        coarse = self.evidence([[.2, .2], [1.2, .2]], [10.8, 10.9], [False, False])
        surface = self.evidence([[.2, .2], [1.2, .2]], [10.8, 10.9], [False, False],
                                expected_depth=11.8)
        self.assertEqual(coarse["state"][0, 0], Evidence.UNKNOWN)
        self.assertEqual(surface["state"][0, 0], Evidence.MEASURED_FOREGROUND)

    def test_known_own_source_cannot_self_veto_when_geometry_is_approximate(self):
        result = self.evidence([[.2, .2]], [9.], [True], expected_depth=11.)
        self.assertEqual(result["state"][0, 0], Evidence.SUPPORTED)
        self.assertEqual(result["foreground_returns"][0, 0], 0)

    def test_renaming_actor_ids_does_not_change_results(self):
        args = ([[.2, .2], [1.2, .2], [2.2, .2]], [10.5, 3., 3.1], [False] * 3)
        first = self.evidence(*args, surface_owner=["object-x", "object-y", None],
                              query_owner="object-x")
        second = self.evidence(*args, surface_owner=["renamed-17", "renamed-88", None],
                               query_owner="renamed-17")
        for name in ("state", "own_pixels", "foreground_pixels", "conflict"):
            np.testing.assert_array_equal(first[name], second[name])
        self.assertEqual(first["counts"], second["counts"])

    def test_unknown_surface_owner_keeps_measured_own_membership(self):
        result = self.evidence([[.2, .2]], [10.5], [True],
                               surface_owner=[None], query_owner="opaque-name")
        self.assertEqual(result["state"][0, 0], Evidence.SUPPORTED)

    def test_explicit_image_scaling_changes_aggregation_not_measurements(self):
        uv = np.array([[.2, .2], [1.2, .2], [2.2, .2]])
        a = self.evidence(uv, [10.5, 3., 3.1], [True, False, False])
        b = self.evidence(uv * 2, [10.5, 3., 3.1], [True, False, False], image_shape=(8, 8))
        np.testing.assert_array_equal(a["state"], b["state"])
        np.testing.assert_array_equal(a["measured_pixels"], b["measured_pixels"])

    def test_invalid_depth_and_uv_are_not_measurements(self):
        result = self.evidence([[.2, .2], [np.nan, 0], [-1, 0], [4, 0], [0, 0]],
                               [np.nan, 2, 2, 2, -1], [False] * 5)
        self.assertEqual(result["counts"]["valid_returns"], 0)
        self.assertEqual(result["counts"]["rejected_returns"], 5)
        self.assertFalse(result["conflict"].any())

    def test_invalid_reference_cannot_claim_foreground(self):
        result = self.evidence([[.2, .2], [1.2, .2]], [1., 1.], [False, False],
                               own_near=np.nan, own_far=np.nan)
        self.assertEqual(result["state"][0, 0], Evidence.UNKNOWN)

    def test_no_input_mutation(self):
        uv = np.array([[.2, .2], [1.2, .2]])
        depth, own = np.array([10.5, 3.]), np.array([True, False])
        copies = [x.copy() for x in (uv, depth, own)]
        visibility_evidence(uv, depth, own, (4, 4), output_shape=(1, 1),
                            own_near=10., own_far=12.)
        for a, b in zip((uv, depth, own), copies):
            np.testing.assert_array_equal(a, b)

    def test_parameter_and_shape_validation(self):
        with self.assertRaises(ValueError):
            VisibilityConfig(min_foreground_pixels=0)
        with self.assertRaises(ValueError):
            VisibilityConfig(foreground_margin=np.nan)
        with self.assertRaises(ValueError):
            self.evidence([], [], [], semantic_hint=np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            visibility_evidence([[0, 0]], [1], [1], (1, 1), expected_depth=2)


if __name__ == "__main__":
    unittest.main()
