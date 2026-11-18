"""最小激活补丁的 CPU 合同；不导入 Torch/native。"""
import unittest
import numpy as np
from road_activation_builder import prepare_arrays


def example():
    n = 6
    road = dict(active=np.array([True, False, False, False, False, False]),
        model_xyz=np.arange(n*3, dtype=np.float64).reshape(n, 3),
        normal_world=np.tile([0., 0., 1.], (n, 1)),
        gaussian_sigma_m=np.tile([.013, .013, .003], (n, 1)),
        road_opacity=np.full(n, .99, np.float32),
        reference_srgb=np.full((n, 3), .25, np.float32),
        minimum_model_depth=np.zeros(n, np.float32),
        model_from_world=np.eye(4), meter_to_model=np.asarray(.01))
    prior = dict(surface_supported=np.ones(n, bool), legal_write=np.ones(n, bool),
        unknown=np.zeros(n, bool), known=np.ones(n, bool), source=np.ones(n, np.uint8),
        guarded_world_rows=np.zeros(n, bool), requested=np.zeros(n, bool),
        observed_reference_srgb=np.full((n, 3), .42, np.float32))
    return road, prior


class ActivationTests(unittest.TestCase):
    def test_activation_only_preserves_every_other_field(self):
        road, prior = example()
        out, p = prepare_arrays(road, prior, np.array([1, 2]))
        np.testing.assert_array_equal(out['active'], [1, 1, 1, 0, 0, 0])
        for key in road:
            if key != 'active':
                np.testing.assert_array_equal(out[key], road[key])
        self.assertFalse(p['activation_material_init'].any())

    def test_new_observed_initialization_only(self):
        road, prior = example()
        out, p = prepare_arrays(road, prior, np.array([1]), initialize_observed=True)
        np.testing.assert_array_equal(out['reference_srgb'][1], prior['observed_reference_srgb'][1])
        np.testing.assert_array_equal(out['reference_srgb'][[0, 2, 3, 4, 5]], road['reference_srgb'][[0, 2, 3, 4, 5]])
        np.testing.assert_array_equal(p['observed_reference_srgb'], prior['observed_reference_srgb'])
        np.testing.assert_array_equal(road['reference_srgb'], np.full((6, 3), .25, np.float32))

    def test_K_and_near_disable_new_rows_and_color_initialization(self):
        road, prior = example()
        K = np.array([1, 1, 0, 0, 0, 0], bool)
        near = np.array([0, 0, 1, 0, 0, 0], bool)
        out, p = prepare_arrays(road, prior, np.array([1, 2, 3]), initialize_observed=True, K_guard=K, near_guard=near)
        np.testing.assert_array_equal(out['active'], [1, 0, 0, 1, 0, 0])
        np.testing.assert_array_equal(out['reference_srgb'][:3], road['reference_srgb'][:3])
        self.assertEqual(int(p['activation_material_init'].sum()), 1)

    def test_old_requested_is_not_an_existence_gate(self):
        road, prior = example()
        out, _ = prepare_arrays(road, prior, np.array([1]))
        self.assertFalse(prior['requested'][1])
        self.assertTrue(out['active'][1])

    def test_original_active_cannot_enter_proposal(self):
        road, prior = example()
        with self.assertRaises(ValueError):
            prepare_arrays(road, prior, np.array([0]))

    def test_inherited_effective_coverage_request_is_preserved(self):
        road, prior = example()
        prior['effective_coverage_requested'] = np.array([0, 0, 0, 0, 1, 0], bool)
        _, p = prepare_arrays(road, prior, np.array([1]))
        np.testing.assert_array_equal(p['effective_coverage_requested'], [0, 1, 0, 0, 1, 0])

    def test_unknown_or_unqualified_material_rejected(self):
        road, prior = example()
        prior['unknown'][1] = True
        with self.assertRaises(ValueError):
            prepare_arrays(road, prior, np.array([1]))
        prior['unknown'][1] = False
        prior['source'][1] = 3
        with self.assertRaises(ValueError):
            prepare_arrays(road, prior, np.array([1]))


if __name__ == '__main__':
    unittest.main()
