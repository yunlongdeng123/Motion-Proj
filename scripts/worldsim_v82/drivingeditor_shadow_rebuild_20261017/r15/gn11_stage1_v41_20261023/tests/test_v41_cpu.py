"""机制/隔离/恢复/证据边界；CPU 解析光栅替身不冒充 CUDA 验收。"""

import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('V4_PACKAGE_PATH', ROOT.parent/'gn11_stage1_v4_20261022'))
ENGINE = ROOT/'engine' if (ROOT/'engine').exists() else ROOT/'engine_overlay'
sys.path[:0] = [str(ENGINE), str(BASE/'engine_overlay'), str(ROOT/'tools'), str(BASE/'tools'), str(BASE/'tests')]
from utils.v41_objectives import dense_objective, valid_windows, ssim_map
from utils.v41_render import native_snapshot, render_snapshot
from utils.v41_road import DenseRoadTraining
from utils.v4_road import RoadTraining, RoadLayer
from v41_dense_geometry import FixedSurface, finalize_masks, project, ray_box_blocked, measured_patch_evidence
from v41_train_hooks import install_hooks, strip_hooks
from utils.v41_vehicle import vehicle_domains
from v41_acceptance import road_query_metrics, regression_summary
from compare_v1_states import differences
from test_v4_cpu import fixture, prepare, Config


def analytic_raster(camera, pipe, degree, rows, background):
    """精确 front-to-back 三维中心排序，测试接口的梯度连接，不近似声明 CUDA 结果。"""
    order = torch.argsort(rows['means3D'][:, 2])
    colors = rows['colors_precomp']
    if colors is None:
        colors = (rows['shs'][:, 0]*.28209479177387814+.5).clamp_min(0)
    transmittance, color = colors.new_tensor(1.), colors.new_zeros(3)
    features = rows['features'].new_zeros(rows['features'].shape[1])
    for index in order:
        alpha = rows['opacities'][index, 0]*rows['mask'][index]
        weight = transmittance*alpha
        color = color+weight*colors[index]
        features = features+weight*rows['features'][index]
        transmittance = transmittance*(1-alpha)
    color = color+transmittance*background
    return dict(rgb=color[:, None, None], alpha=features[1:2, None, None],
        depth_sum=features[:1, None, None], extra=features[2:, None, None], radii=colors.new_ones(len(colors)))


def inputs(back_opacity=1.):
    rgb = torch.nn.Parameter(torch.tensor([[0., 0., 0.], [.7, .7, .7]]))
    alpha = torch.nn.Parameter(torch.tensor([[.5], [back_opacity]]))
    xyz = torch.nn.Parameter(torch.tensor([[0., 0., 1.], [0., 0., 3.]]))
    rows = dict(means3D=xyz, means2D=torch.zeros_like(xyz), shs=None, colors_precomp=rgb,
        features=torch.tensor([[1., 1.], [3., 1.]]), opacities=alpha,
        scales=torch.ones(2, 3), rotations=torch.tensor([[1., 0., 0., 0.]]*2),
        cov3D_precomp=None, mask=torch.ones(2, dtype=torch.bool))
    camera = SimpleNamespace(world_view_transform=torch.eye(4), get_world_directions=lambda _:torch.ones(1, 1, 3))
    static = SimpleNamespace(active_sh_degree=0)
    road = RoadLayer(fixture(), .1, 'R')
    with torch.no_grad():
        road.xyz[:, 2] = 2
        road.observed[:] = False; road.observed[0] = True
        road.dc.fill_((.35-.5)/.28209479177387814)
        road.opacity.fill_(float(np.log(.95/.05)))
    pipe = SimpleNamespace(compute_cov3D_python=False)
    return rgb, alpha, xyz, rows, camera, static, road, pipe


class ObjectiveTests(unittest.TestCase):
    def test_dense_error_between_centers_is_detected(self):
        target = torch.full((3, 24, 24), .4)
        joint = target.clone(); joint[:, 7:17, 7:17] = .1; joint.requires_grad_()
        visible = torch.ones(24, 24, dtype=torch.bool)
        total, terms = dense_objective(joint, target, dict(rgb=target, alpha=torch.ones(1, 24, 24)),
            target, visible, ~visible)
        self.assertGreater(float(terms['joint_rgb']), .01)
        total.backward()
        self.assertGreater(float(joint.grad[:, 9:15, 9:15].abs().sum()), 0)

    def test_ssim_never_reads_hidden_pixels(self):
        mask = torch.zeros(32, 32, dtype=torch.bool); mask[4:28, 4:28] = True
        windows = valid_windows(mask)
        a = torch.rand(3, 32, 32); b = a.clone(); b[:, ~mask] = 100
        torch.testing.assert_close(ssim_map(a, a, mask)[windows], ssim_map(b, a, mask)[windows])
        self.assertGreater(int(windows.sum()), 14*14)

    def test_unknown_has_no_rgb_gradient_and_protection_uses_native(self):
        target = torch.zeros(3, 24, 24)
        joint = torch.full_like(target, .3, requires_grad=True)
        native = torch.full_like(target, .3)
        visible = torch.zeros(24, 24, dtype=torch.bool)
        protect = visible.clone(); protect[:5] = True
        road = dict(rgb=torch.ones_like(target)*.9, alpha=torch.ones(1, 24, 24))
        total, terms = dense_objective(joint, native, road, target, visible, protect)
        self.assertEqual(float(total), 0.)
        total.backward(); self.assertEqual(float(joint.grad.abs().sum()), 0.)
        self.assertEqual(int(terms['ssim_windows']), 0)
        changed = torch.full_like(target, .1, requires_grad=True)
        loss, _ = dense_objective(changed, native, road, target, visible, ~visible)
        loss.backward()
        self.assertLess(float(changed.grad.mean()), 0)  # 指向 V1 的 .3，而非未知路面的 GT 0。

    def test_overlapping_domains_rejected(self):
        rgb = torch.zeros(3, 12, 12); m = torch.ones(12, 12, dtype=torch.bool)
        with self.assertRaises(ValueError):
            dense_objective(rgb, rgb, dict(rgb=rgb, alpha=rgb[:1]), rgb, m, m)

    def test_out_of_range_rgb_keeps_gradient(self):
        target = torch.ones(3, 12, 12); pred = torch.full_like(target, 1.2, requires_grad=True)
        mask = torch.ones(12, 12, dtype=torch.bool)
        loss, _ = dense_objective(pred, target, dict(rgb=target, alpha=target[:1]), target, mask, ~mask)
        loss.backward(); self.assertGreater(float(pred.grad.mean()), 0)

    def test_small_vehicle_not_diluted_by_large_background_or_actor(self):
        gt = torch.zeros(3, 32, 32)
        joint = gt.clone(); joint[:, 4:6, 4:6] = .3; joint.requires_grad_()
        body = torch.zeros(2, 32, 32, dtype=torch.bool)
        body[0, 4:6, 4:6] = True; body[1, 10:30, 10:30] = True
        empty = torch.zeros(32, 32, dtype=torch.bool)
        loss, terms = dense_objective(joint, gt, dict(rgb=gt, alpha=gt[:1]), gt, empty, ~empty,
            bodies=body, edges=torch.zeros_like(body))
        self.assertAlmostEqual(float(terms['vehicle_body_preservation']), .15, places=6)
        loss.backward(); self.assertGreater(float(joint.grad[:, 4:6, 4:6].sum()), 0)

    def test_vehicle_edge_has_separate_gradient_and_no_road_target(self):
        labels = np.zeros((24, 24), np.uint8); labels[8:16, 8:16] = 1
        body, edge = map(torch.from_numpy, vehicle_domains(labels, labels > 0))
        vehicle = body.any(0) | edge.any(0)
        pred = torch.zeros(3, 24, 24); pred[:, edge[0]] = .2; pred.requires_grad_()
        gt = torch.zeros_like(pred)
        loss, terms = dense_objective(pred, gt, dict(rgb=gt, alpha=gt[:1]), gt, ~vehicle, vehicle,
            bodies=body, edges=edge)
        self.assertAlmostEqual(float(terms['vehicle_edge_preservation']), .2, places=6)
        loss.backward(); self.assertGreater(float(pred.grad[:, edge[0]].sum()), 0)
        with self.assertRaises(ValueError):
            dense_objective(pred, gt, dict(rgb=gt, alpha=gt[:1]), gt, ~vehicle, body.any(0), bodies=body, edges=edge)


class RenderGradientTests(unittest.TestCase):
    def test_interleaving_darkening_gradient_only_road(self):
        rgb, alpha, xyz, rows, camera, static, road, pipe = inputs()
        fake = SimpleNamespace(prepare_rasterization=lambda *a, **k:rows)
        with patch.dict(sys.modules, gaussian_renderer=fake), patch('utils.v41_render.raster', analytic_raster):
            snapshot = native_snapshot(camera, static, None, pipe, torch.zeros(3), None)
            native = render_snapshot(camera, pipe, snapshot)['rgb']
            joint = render_snapshot(camera, pipe, snapshot, road, True)['rgb']
            self.assertAlmostEqual(float(native.mean()), .35, places=6)
            self.assertAlmostEqual(float(joint.mean()), .18375, places=6)
            (joint-native).square().mean().backward()
            self.assertGreater(float(road.opacity.grad[0]), 0.)
            self.assertGreater(float(road.dc.grad[0].abs().sum()), 0.)
            self.assertTrue(all(p.grad is None for p in (rgb, alpha, xyz)))
            before = float((joint-native).square().mean())
            with torch.no_grad():
                road.opacity -= .1*road.opacity.grad
            after = render_snapshot(camera, pipe, snapshot, road, True)['rgb']
            self.assertLess(float((after-native).square().mean()), before)

    def test_environment_transmittance_stays_differentiable(self):
        _, _, _, rows, camera, static, road, pipe = inputs(back_opacity=.2)
        env = torch.nn.Parameter(torch.full((1, 1, 3), .8))
        fake = SimpleNamespace(prepare_rasterization=lambda *a, **k:rows)
        with patch.dict(sys.modules, gaussian_renderer=fake), patch('utils.v41_render.raster', analytic_raster):
            snapshot = native_snapshot(camera, static, None, pipe, torch.zeros(3), lambda _:env)
            value = render_snapshot(camera, pipe, snapshot, road, True)['rgb'].sum()
            derivative = torch.autograd.grad(value, road.opacity)[0][0].item()
            original = road.opacity[0].item(); results = []
            for delta in (-.002, .002):
                with torch.no_grad(): road.opacity[0] = original+delta
                results.append(float(render_snapshot(camera, pipe, snapshot, road, True)['rgb'].sum()))
            self.assertAlmostEqual(derivative, (results[1]-results[0])/.004, places=4)
            self.assertIsNone(env.grad)


def tiny_training(path):
    args, scene = prepare(path, 'R')
    training = DenseRoadTraining.__new__(DenseRoadTraining)
    RoadTraining.__init__(training, args, scene, 0)
    training.args = args; training.dense_path = '/fixed/synthetic/manifest.json'; training.mask_policy = {'test': True}
    for camera in training.cameras:
        camera.original_image = torch.full((3, 8, 8), .4)
    training.dense = [dict(visible=torch.ones(8, 8, dtype=torch.bool), protect=torch.zeros(8, 8, dtype=torch.bool),
        ssim_valid=torch.zeros(8, 8, dtype=torch.bool), observed_rows=torch.tensor([0, 5]),
        vehicle_body=torch.zeros(6, 8, 8, dtype=torch.bool), vehicle_edge=torch.zeros(6, 8, 8, dtype=torch.bool)) for _ in training.cameras]
    return training, scene.gaussians.get_xyz


def tiny_road(camera, road, pipe, **kwargs):
    random.random(); np.random.rand(); torch.rand(1)
    alpha = 1-(1-road.opacity.sigmoid()[:, 0]*road.active).prod()
    color = road.colors()[road.active].mean(0)
    return dict(rgb=(color*alpha)[:, None, None].expand(3, 8, 8),
        alpha=alpha.expand(1, 8, 8), depth_sum=alpha.expand(1, 8, 8))


def tiny_pair(camera, native, dynamic, road, pipe, background, env):
    detached = native.detach().expand(3, 8, 8)/5
    rendered = tiny_road(camera, road, pipe)
    return detached, dict(rgb=rendered['rgb']+(1-rendered['alpha'])*detached, alpha=rendered['alpha'])


class IsolationRecoveryTests(unittest.TestCase):
    def test_native_optimizer_rng_trajectory_unchanged(self):
        result = []
        with tempfile.TemporaryDirectory() as temp:
            for enabled in (False, True):
                random.seed(2); np.random.seed(2); torch.manual_seed(2)
                road, native = tiny_training(Path(temp)/str(enabled))
                optimizer = torch.optim.Adam([native], lr=.01)
                samples = []
                for step in (1, 2, 3):
                    value = random.random()+np.random.rand()+torch.rand(1)
                    samples.append(value.clone())
                    (native-value).square().mean().backward()
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    if enabled:
                        with torch.no_grad():
                            road.step(step, native, None, None, None, tiny_road, tiny_pair)
                result.append((native.detach().clone(), optimizer.state_dict(), samples,
                    torch.get_rng_state(), np.random.get_state(), random.getstate()))
            self.assertEqual(differences(*result), [])

    def test_checkpoint_restores_exact_next_update_and_no_geometry_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            road, native = tiny_training(temp)
            initial = {k:getattr(road.layer, k).clone() for k in ('xyz', 'scaling', 'rotation')}
            road.step(1, native, None, None, None, tiny_road, tiny_pair)
            path = Path(temp)/'road_chkpnt1.pth'; torch.save(road.state(1), path)
            road.step(2, native, None, None, None, tiny_road, tiny_pair)
            restored, reference = tiny_training(temp)
            restored.restore(path, 1)
            restored.step(2, reference, None, None, None, tiny_road, tiny_pair)
            for key in ('layer', 'optimizer', 'sampler', 'stack'):
                self.assertEqual(differences(road.state(2)[key], restored.state(2)[key]), [])
            for key, value in initial.items():
                self.assertTrue(torch.equal(value, getattr(road.layer, key)))

    def test_old_V4_state_rejected_and_manifest_requires_native_four(self):
        with tempfile.TemporaryDirectory() as temp:
            road, native = tiny_training(temp)
            road.step(1, native, None, None, None, tiny_road, tiny_pair)
            with self.assertRaises(ValueError): road.save(1)
            path = Path(temp)/'old.pth'; state = road.state(1); state['version'] = 'V4'
            torch.save(state, path)
            with self.assertRaises(ValueError): road.restore(path, 1)


class AcceptanceTests(unittest.TestCase):
    def test_independent_alpha_cannot_pass_deleted_effective_coverage(self):
        arrays = dict(query_frame=np.array([0, 0]), query_actor=np.array([37, 37]),
            query_pixel=np.array([[3, 3], [4, 3]]), query_unit_id=np.array([0, 1]),
            surface_supported=np.array([True, False]), provenance=np.ones(2), base_weight=np.ones((49, 2)))
        zero = np.zeros((320, 480)); one = np.ones_like(zero)
        result = road_query_metrics(arrays, 0, 37, zero, zero, one, np.zeros(2), np.ones(2))
        self.assertEqual(result['unsupported_pixels'], 1)
        scores = result['thresholds']['0.05']
        self.assertEqual(scores['independent']['coverage_fraction'], 1)
        self.assertEqual(scores['delete_with_verified_source_color']['coverage_fraction'], 0)

    def test_one_actor_regression_cannot_be_hidden_by_full_image_gain(self):
        before = dict(pixels=12, mse=.02, mae=.1, psnr=17.)
        record = dict(frame='example', regions=dict(v1={'full':before, 'a6_edge':before},
            composite={'full':dict(before, mse=.01, psnr=20), 'a6_edge':dict(before, mse=.04, psnr=14)}))
        result = regression_summary([record])
        self.assertEqual(result['full']['worsened_frames'], 0)
        self.assertEqual(result['a6_edge']['worsened_frames'], 1)


class GeometryTests(unittest.TestCase):
    def test_second_view_and_occluder_required(self):
        current = np.ones((32, 32), bool); count = np.full((32, 32), 2, np.uint8)
        protect = np.zeros((32, 32), bool); protect[:, 15:18] = True; count[:6] = 1
        visible, windows = finalize_masks(current, count, protect)
        self.assertFalse(visible[:6].any() or visible[:, 15:18].any())
        self.assertFalse(windows[:, 15:18].any())

    def test_ray_boxes_block_background_road(self):
        box = dict(center=np.array([0., 0., 2.]), half=np.ones(3)*.5, axes=np.eye(3))
        np.testing.assert_array_equal(ray_box_blocked(np.array([[0., 0., 4.], [3., 0., 4.]]),
            np.zeros(3), [box]), [True, False])

    def test_native_half_pixel_convention(self):
        uv, depth = project(np.array([[0., 0., 2.]]), np.array([[20., 0, 10.5], [0, 20, 8.5], [0, 0, 1]]), np.eye(4))
        np.testing.assert_array_equal(uv, [[10., 8.]])

    def test_fixed_surface_missing_cell_does_not_fill(self):
        keys = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        units = dict(keys=keys, world=np.c_[(keys+.5)*.2, np.zeros(4)],
            surface_supported=np.array([True, True, True, False]), provenance=np.ones(4),
            base_weight=np.ones((49, 4)), normal_world=np.tile([0., 0., 1.], (4, 1)))
        surface = FixedSurface(dict(coordinate_anchor=np.eye(4).tolist()), units)
        self.assertFalse(surface.height(np.array([[.2, .2]]))[1][0])

    def test_measured_patch_no_extrapolation_and_near_occluder_rejected(self):
        K = np.array([[20., 0, 20.5], [0, 20, 20.5], [0, 0, 1.]])
        cloud = np.array([[0., 0., 2.], [.6, 0., 2.], [0., .6, 2.], [.15, .15, 1.]])
        query = np.array([[.3, .3, 2.], [1., 1., 2.]])
        uv, _ = project(query, K, np.eye(4))
        result, _, _, conflict = measured_patch_evidence(cloud, np.array([1, 1, 1, 0], bool), K,
            np.eye(4), query, uv, np.tile([0., 0., 1.], (2, 1)))
        self.assertFalse(result.any()); self.assertTrue(conflict[0])

    def test_hooks_are_reversible(self):
        source = ('def training(args):\n'
            '    progress_bar = tqdm(range(first_iter + 1, args.iterations + 1), desc="Training progress")\n'
            '    for i in range(10):\n        with torch.no_grad():\n            pass\n'
            '# def mkdir_p(folder_path):\n')
        modified = install_hooks(source)
        self.assertEqual(strip_hooks(modified), source)
        self.assertIn('gaussians, dynamic_gaussians, background, env_map', modified)


if __name__ == '__main__':
    unittest.main(verbosity=2)
