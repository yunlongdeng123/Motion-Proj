"""几何、纹理身份与离散ownership测试；不依赖场景或GPU。"""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

import gaussian_patch_transport as g


class TransportGeometryTests(unittest.TestCase):
    def setUp(self):
        self.centers = np.array([[0., 0., .07], [.3, 0., .07], [0., .4, .07], [.3, .4, .07]])
        self.sigma = np.tile([.04, .013, .001], (4, 1))
        self.quaternion = np.tile([1., 0., 0., 0.], (4, 1))
        self.colors = np.array([[.1, .2, .3], [.2, .3, .4], [.3, .4, .5], [.4, .5, .6]], dtype=np.float32)
        self.opacity = np.array([.25, .4, .6, .8], dtype=np.float32)
        self.ids = np.array([91, 9, 53, 82], dtype=np.int64)

    @staticmethod
    def plane(uv):
        frame = Rotation.from_euler('xy', [.2, -.3]).as_matrix()
        return dict(position=np.array([2., -3., .4]) + uv @ frame[:, :2].T,
                    jacobian=np.broadcast_to(frame[:, :2], (len(uv), 3, 2)).copy(),
                    normal=np.broadcast_to(frame[:, 2], (len(uv), 3)).copy(),
                    normal_jacobian=np.zeros((len(uv), 3, 2)))

    @staticmethod
    def curved(uv):
        u, v = uv.T
        height = .17 * u * u + .06 * u * v + .12 * v * v
        du, dv = .34 * u + .06 * v, .06 * u + .24 * v
        derivative = np.zeros((len(uv), 3, 2))
        derivative[:, 0, 0] = 1
        derivative[:, 1, 1] = 1
        derivative[:, 2, 0], derivative[:, 2, 1] = du, dv
        raw_normal = np.stack((-du, -dv, np.ones_like(u)), axis=1)
        length = np.linalg.norm(raw_normal, axis=1)
        normal = raw_normal / length[:, None]
        raw_derivative = np.broadcast_to(np.array([[-.34, -.06], [-.06, -.24], [0., 0.]]), (len(uv), 3, 2))
        normal_derivative = (np.eye(3)[None] - normal[:, :, None] * normal[:, None, :]) @ raw_derivative / length[:, None, None]
        return dict(position=np.stack((u, v, height), axis=1), jacobian=derivative,
                    normal=normal, normal_jacobian=normal_derivative)

    def transport(self, **kwargs):
        defaults = dict(source_origin=np.zeros(3), source_frame=np.eye(3), sample_target=self.plane,
                        patch_id=17, source_asset_id='verified-source-asset')
        defaults.update(kwargs)
        return g.transport_patch(self.centers, self.sigma, self.quaternion, self.colors, self.opacity, self.ids, **defaults)

    def assert_covariance_round_trip(self, result):
        reconstructed = g.covariance_from_sigma_quaternion(result['gaussian_sigma_m'], result['gaussian_rotation_world_wxyz'])
        np.testing.assert_allclose(reconstructed, result['covariance_world'], atol=2e-16, rtol=2e-12)
        actual_normal = g.quaternion_wxyz_to_matrix(result['gaussian_rotation_world_wxyz'])[:, :, 2]
        np.testing.assert_allclose(actual_normal, result['normal_world'], atol=2e-14)

    def test_metric_rigid_anisotropic_covariance_and_texture_identity(self):
        result = self.transport(rotation_rad=.61, translation_uv=(.8, -.5))
        matrix = result['trace']['mapping_2d']
        target_uv = self.centers[:, :2] @ matrix.T + [.8, -.5]
        np.testing.assert_allclose(result['world_xyz'], self.plane(target_uv)['position'], atol=1e-14)
        source_covariance = g.covariance_from_sigma_quaternion(self.sigma, self.quaternion)
        expected = result['jacobian_world'] @ source_covariance @ result['jacobian_world'].transpose(0, 2, 1)
        np.testing.assert_allclose(result['covariance_world'], expected, atol=1e-16)
        self.assert_covariance_round_trip(result)
        source_distance = np.linalg.norm(self.centers[:, None] - self.centers[None], axis=2)
        target_distance = np.linalg.norm(result['world_xyz'][:, None] - result['world_xyz'][None], axis=2)
        np.testing.assert_allclose(source_distance, target_distance, atol=2e-14)
        self.assertEqual(result['reference_srgb'].tobytes(), self.colors.tobytes())
        self.assertEqual(result['road_opacity'].tobytes(), self.opacity.tobytes())
        np.testing.assert_array_equal(result['source_row_id'], self.ids)
        self.assertTrue(result['gaussian_rotation_world_override'].all())
        self.assertEqual(result['trace']['source_asset_id'], 'verified-source-asset')
        np.testing.assert_allclose(result['gaussian_sigma_m'][:, 2], self.sigma[:, 2], atol=1e-15)

    def test_scale_changes_tangent_frequency_and_not_normal_thickness_or_opacity(self):
        result = self.transport(scale=2.)
        np.testing.assert_allclose(result['gaussian_sigma_m'][:, :2], self.sigma[:, :2] * 2, atol=1e-14)
        np.testing.assert_allclose(result['gaussian_sigma_m'][:, 2], self.sigma[:, 2], atol=1e-15)
        np.testing.assert_array_equal(result['road_opacity'], self.opacity)

    def test_curvature_jacobian_matches_independent_finite_difference(self):
        result = self.transport(sample_target=self.curved, rotation_rad=.3, translation_uv=(.1, .2))
        matrix, shift = result['trace']['mapping_2d'], result['trace']['translation_uv']
        for i, center in enumerate(self.centers):
            fixed_normal = result['normal_world'][i]
            def local_extension(x):
                target = self.curved((matrix @ x[:2] + shift)[None])['position'][0]
                return target + fixed_normal * (x[2] - center[2])
            columns = []
            for axis in range(3):
                delta = np.eye(3)[axis] * 1e-6
                columns.append((local_extension(center + delta) - local_extension(center - delta)) / 2e-6)
            np.testing.assert_allclose(result['jacobian_world'][i], np.stack(columns, axis=1), atol=3e-10, rtol=1e-9)
        self.assert_covariance_round_trip(result)
        np.testing.assert_allclose(result['world_xyz'], self.curved(result['target_uv'])['position'], atol=1e-14)

    def test_explicit_normal_residual_includes_curvature_derivative(self):
        result = self.transport(sample_target=self.curved, normal_residual=True, rotation_rad=.21)
        matrix = result['trace']['mapping_2d']
        for i, center in enumerate(self.centers):
            def offset_surface(x):
                sample = self.curved((matrix @ x[:2])[None])
                return sample['position'][0] + x[2] * sample['normal'][0]
            np.testing.assert_allclose(result['world_xyz'][i], offset_surface(center), atol=1e-14)
            columns = []
            for axis in range(3):
                delta = np.eye(3)[axis] * 1e-6
                columns.append((offset_surface(center + delta) - offset_surface(center - delta)) / 2e-6)
            np.testing.assert_allclose(result['jacobian_world'][i], np.stack(columns, axis=1), atol=3e-10, rtol=1e-9)
        self.assert_covariance_round_trip(result)

    def test_nonrepresentable_normal_coupling_is_rejected(self):
        covariance = np.array([[[.04, 0., .01], [0., .03, 0.], [.01, 0., .02]]])
        with self.assertRaisesRegex(ValueError, '耦合'):
            g.normal_preserving_decomposition(covariance, [[0., 0., 1.]])

    def test_invalid_target_drop_retains_input_identity(self):
        def surface(uv):
            sample = self.plane(uv)
            sample['position'][2] = np.nan
            sample['valid'] = np.array([False, True, True, True])
            return sample
        with self.assertRaisesRegex(ValueError, '无效行数'):
            self.transport(sample_target=surface)
        result = self.transport(sample_target=surface, invalid='drop')
        np.testing.assert_array_equal(result['source_row_id'], self.ids[[1, 3]])
        np.testing.assert_array_equal(result['trace']['dropped_input_indices'], [0, 2])
        np.testing.assert_array_equal(result['reference_srgb'], self.colors[[1, 3]])

    def test_missing_normal_derivative_cannot_silently_carry_height(self):
        def surface(uv):
            sample = self.plane(uv); del sample['normal_jacobian']; return sample
        with self.assertRaisesRegex(ValueError, 'normal_jacobian'):
            self.transport(sample_target=surface, normal_residual=True)

    def test_overlap_ownership_keeps_whole_patch_rows_without_color_average(self):
        first = self.transport(patch_id=7)
        self.colors = (self.colors * .5).astype(np.float32)
        second = self.transport(patch_id=9)
        merged = g.merge_transported_patches([first, second], cell_size=.2)
        np.testing.assert_array_equal(merged['source_row_id'], first['source_row_id'])
        np.testing.assert_array_equal(merged['reference_srgb'], first['reference_srgb'])
        np.testing.assert_array_equal(merged['covariance_world'], first['covariance_world'])
        self.assertTrue((merged['patch_id'] == 7).all())
        self.assertEqual(len(merged['trace']['discarded_candidate_indices']), 4)
        with self.assertRaisesRegex(ValueError, 'patch_id'):
            g.merge_transported_patches([first, first], cell_size=.2)

    def test_potts_ownership_is_coherent_and_energy_nonincreasing(self):
        points = np.array([[.25, .25], [1.25, .25], [2.25, .25]] * 2)
        labels = np.array([10, 10, 10, 20, 20, 20])
        result = g.world_grid_ownership(points, labels, cell_size=1., unary_cost=[0., 1.6, 0., 1., 0., 1.], smoothness=2.)
        self.assertTrue(np.all(np.diff(result['energy_history']) <= 1e-12))
        self.assertEqual(len(set(result['cell_patch_id'])), 1)
        for cell in np.unique(result['candidate_cell_index']):
            selected = result['keep_mask'] & (result['candidate_cell_index'] == cell)
            self.assertEqual(len(np.unique(labels[selected])), 1)
        self.assertFalse(result['global_graphcut_optimum'])

    def test_replacement_respects_full_support_and_protected_rows(self):
        centers = np.array([[0., 0., 0.], [.6, 0., 0.], [.9, 0., 0.], [1.2, 0., 0.]])
        covariance = np.tile(np.diag([.1, .05, .01]) ** 2, (4, 1, 1))
        result = g.target_replacement_mask(centers, covariance, lambda x: 1 - np.abs(x[:, 0]),
                                           protected_mask=[True, False, False, False])
        np.testing.assert_array_equal(result['replace_mask'], [False, True, False, False])
        np.testing.assert_allclose(result['support_radius_m'], .3)

    def test_deduplication_preserves_nontransitive_texture_points_and_fixed_rows(self):
        centers = np.array([[0., 0., 0.], [.75, 0., 0.], [1.5, 0., 0.], [0., 0., 0.]])
        result = g.deduplicate_centers(centers, tolerance=1.)
        np.testing.assert_array_equal(result['kept_indices'], [0, 2])
        np.testing.assert_array_equal(result['representative'], [0, 0, 2, 0])
        ranked = g.deduplicate_centers(centers, tolerance=0., priority=[5., 1., 2., 0.])
        np.testing.assert_array_equal(ranked['kept_indices'], [1, 2, 3])
        self.assertEqual(ranked['representative'][0], 3)
        fixed = g.deduplicate_centers(centers, tolerance=.1, fixed_centers=[[1.5, 0., 0.]])
        np.testing.assert_array_equal(fixed['kept_indices'], [0, 1])
        self.assertEqual(fixed['representative'][2], -1)
        self.assertEqual(fixed['fixed_row'][2], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
