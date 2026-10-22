"""只验证隔离合同和 CPU 数值，不运行真实场景实验或 CUDA。"""

from pathlib import Path
from types import SimpleNamespace
import json
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'engine' if (ROOT/'engine').exists() else ROOT/'engine_overlay'
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ROOT/'tools'))
from utils.v4_road import (RoadLayer, RoadTraining, classify_units, load_bundle,
                           create_road_training, POLICY, INPUT_VERSION, FRAMES)
from utils.v4_road_render import road_kwargs, merge_rows, render_composite
from compare_v1_states import differences


class Config(dict):
    __getattr__ = dict.__getitem__


def fixture():
    n = 7
    arrays = dict(model_xyz=np.zeros((n, 3)), normal_model=np.tile([0., 0., 1.], (n, 1)),
        surface_supported=np.array([True]*6+[False]), provenance=np.array([1, 2, 2, 0, 2, 1, 2]),
        edges=np.array([[0, 1], [1, 2], [2, 3], [3, 4], [5, 6]], np.int64),
        source_rgb=np.full((49, n, 3), .65, np.float32), source_depth=np.ones((49, n), np.float32),
        base_weight=np.zeros((49, n), np.float32))
    arrays['base_weight'][:2, [0, 5]] = 1.
    arrays['base_weight'][0, 3] = 1.  # 单来源 UNKNOWN 不能变成传播桥。
    for index in range(9):
        arrays[f'gate_test{index}'] = arrays['base_weight'] > 0
    return arrays


def prepare(directory, arm):
    arrays = fixture()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory/'units.npz', **arrays)
    intrinsic = [[1., 0, 4.], [0, 1., 4.], [0, 0, 1.]]
    meta = dict(status='CPU_MULTIVIEW_ROAD_INPUT_VERIFIED', version=INPUT_VERSION,
        units='units.npz', frame_order=FRAMES, meter_to_model=.1,
        world_to_model=np.diag([.1, .1, .1, 1.]).tolist(),
        frames={key: dict(intrinsic=intrinsic, world_to_camera=np.eye(4).tolist()) for key in FRAMES})
    (directory/'bundle.json').write_text(json.dumps(meta))
    args = Config(v4_arm=arm, v4_bundle=str(directory/'bundle.json'),
        seed=0, model_path=str(directory), checkpoint_iterations=[], save_iterations=[])
    cameras = [SimpleNamespace(image_path='/input/image_0/'+key+'.png', image_name=key,
        fx=1., fy=1., cx=4., cy=4., full_proj_transform=torch.eye(4),
        world_view_transform=torch.eye(4),
        image_width=8, image_height=8) for key in FRAMES]
    baseline = torch.nn.Parameter(torch.tensor([2.]))
    scene = SimpleNamespace(gaussians=SimpleNamespace(get_xyz=baseline), getTrainCameras=lambda: cameras)
    return args, scene


def fake_renderer(camera, layer, args, for_training=False):
    # 故意消耗全局 RNG，验证道路调用即使以后新增随机操作也不扰动 V1。
    random.random(); np.random.random(); torch.rand(2)
    opacity = layer.opacity.sigmoid()[:, 0]*layer.active
    color = (layer.colors()*opacity[:, None]).sum(0)/opacity.sum().clamp_min(1e-5)
    alpha = 1-(1-opacity).prod()
    return dict(rgb=(color*alpha)[:, None, None].expand(3, 8, 8),
        alpha=alpha.reshape(1, 1, 1).expand(1, 8, 8),
        depth_sum=alpha.reshape(1, 1, 1).expand(1, 8, 8))


class IsolationTests(unittest.TestCase):
    def test_graph_rejects_unknown_and_unsupported_bridges(self):
        observed, prior = classify_units(fixture())
        np.testing.assert_array_equal(np.flatnonzero(observed), [0, 5])
        np.testing.assert_array_equal(np.flatnonzero(prior), [1, 2])

    def test_only_two_parameter_groups_and_fixed_geometry(self):
        layer = RoadLayer(fixture(), .1, 'RP')
        self.assertEqual(set(dict(layer.named_parameters())), {'dc', 'opacity'})
        self.assertFalse(layer.xyz.requires_grad or layer.scaling.requires_grad or layer.rotation.requires_grad)
        self.assertEqual(len(layer.xyz), 6)
        torch.testing.assert_close(layer.scaling[0], torch.tensor([.013, .013, .0015]))

    def test_unknown_and_unvisited_gray_not_output(self):
        layer = RoadLayer(fixture(), .1, 'RP')
        self.assertFalse(layer.output_mask().any())
        layer.source_seen.fill_(True)
        layer.appearance_updated.fill_(True)
        self.assertFalse(layer.output_mask().any())
        layer.steps.fill_(1)
        self.assertEqual(layer.output_mask().tolist(), [True, True, True, False, False, True])
        real = RoadLayer(fixture(), .1, 'R')
        real.source_seen.fill_(True); real.appearance_updated.fill_(True)
        real.steps.fill_(1)
        self.assertEqual(real.output_mask().tolist(), [True, False, False, False, False, True])

    def test_hard_gates_cannot_be_overridden(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = prepare(temp, 'R')
            arrays = fixture()
            arrays['gate_test0'][0, 0] = False
            np.savez(Path(temp)/'units.npz', **arrays)
            with self.assertRaises(ValueError):
                load_bundle(args.v4_bundle)

    def test_199_frame_bundle_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = prepare(temp, 'R')
            path = Path(args.v4_bundle)
            meta = json.loads(path.read_text()); meta['frame_order'].append('0008198')
            path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                load_bundle(path)

    def test_B_does_not_load_or_allocate_road(self):
        self.assertIsNone(create_road_training(Config(v4_arm='B'), None))

    def test_training_inside_no_grad_and_no_V1_gradient(self):
        with tempfile.TemporaryDirectory() as temp:
            args, scene = prepare(temp, 'RP')
            road = create_road_training(args, scene)
            baseline = scene.gaussians.get_xyz
            baseline.grad = torch.full_like(baseline, 17.)
            before = {name: value.clone() for name, value in road.layer.named_buffers()
                      if name in ('xyz', 'rotation', 'scaling', 'unit_ids')}
            with torch.no_grad():
                road.step(1, fake_renderer)
            self.assertEqual(float(baseline), 2.)
            self.assertEqual(float(baseline.grad), 17.)
            for name, value in before.items():
                self.assertTrue(torch.equal(value, getattr(road.layer, name)))
            self.assertEqual(int(road.layer.steps), 1)
            self.assertEqual(road.layer.output_mask()[3:5].tolist(), [False, False])

    def test_B_R_RP_preserve_toy_V1_trajectory_and_rng(self):
        results = []
        with tempfile.TemporaryDirectory() as temp:
            for arm in ('B', 'R', 'RP'):
                random.seed(12); np.random.seed(12); torch.manual_seed(12)
                args, scene = prepare(Path(temp)/arm, arm)
                baseline = scene.gaussians.get_xyz
                optimizer = torch.optim.Adam([baseline], lr=.01)
                road = create_road_training(args, scene)
                samples = []
                for step in range(1, 4):
                    target = torch.rand(1)+random.random()+np.random.random()
                    samples.append(float(target))
                    (baseline-target).square().sum().backward()
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    if road:
                        road.step(step, fake_renderer)
                results.append((baseline.detach().clone(), samples, torch.rand(3), random.random(), np.random.random()))
            self.assertEqual(differences(results[0], results[1]), [])
            self.assertEqual(differences(results[0], results[2]), [])

    def test_sidecar_restores_sampler_optimizer_and_rejects_wrong_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            args, scene = prepare(temp, 'RP')
            road = create_road_training(args, scene)
            road.step(1, fake_renderer); road.save(1)
            expected = road.sample()
            restored = create_road_training(args, scene, 1)
            self.assertEqual(restored.sample(), expected)
            self.assertEqual(differences(road.optimizer.state_dict(), restored.optimizer.state_dict()), [])
            args['v4_arm'] = 'R'
            with self.assertRaises(ValueError):
                create_road_training(args, scene, 1)

    def test_double_or_skipped_step_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args, scene = prepare(temp, 'R')
            road = create_road_training(args, scene)
            with self.assertRaises(ValueError):
                road.step(2, fake_renderer)
            road.step(1, fake_renderer)
            with self.assertRaises(ValueError):
                road.step(1, fake_renderer)

    def test_restore_rejects_geometry_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            args, scene = prepare(temp, 'R')
            road = create_road_training(args, scene)
            road.step(1, fake_renderer); road.save(1)
            path = Path(temp)/'road_chkpnt1.pth'
            state = torch.load(path)
            state['layer']['xyz'][0, 2] += .1
            torch.save(state, path)
            with self.assertRaises(ValueError):
                create_road_training(args, scene, 1)

    def test_R_no_prior_objective(self):
        with tempfile.TemporaryDirectory() as temp:
            args, scene = prepare(temp, 'R')
            road = create_road_training(args, scene)
            _, terms = road.objective([0, 1], fake_renderer)
            self.assertEqual(terms['prior_continuity'], 0.)
            self.assertEqual(len(road.edges), 0)

    def test_merge_does_not_silently_drop_inconsistent_fields(self):
        with self.assertRaises(ValueError):
            merge_rows(dict(shs=None), dict(shs=torch.zeros(1, 1, 3)))

    def test_road_SH_padding_and_true_depth(self):
        layer = RoadLayer(fixture(), .1, 'RP')
        layer.xyz[:, 2] = 3.
        layer.source_seen.fill_(True)
        camera = SimpleNamespace(world_view_transform=torch.eye(4))
        kwargs = road_kwargs(camera, layer, SimpleNamespace(compute_cov3D_python=False), sh_channels=16)
        self.assertEqual(tuple(kwargs['shs'].shape), (6, 16, 3))
        self.assertEqual(int(torch.count_nonzero(kwargs['shs'][:, 1:])), 0)
        self.assertTrue((kwargs['features'][:, 0] == 3).all())
        self.assertTrue(torch.equal(kwargs['means3D'], layer.xyz))

    def test_checkpoint_comparison_detects_shape_and_value_drift(self):
        self.assertTrue(differences(torch.tensor([1.]), torch.tensor([2.])))
        self.assertTrue(differences(torch.zeros(1), torch.zeros(2)))
        self.assertFalse(differences({'x': torch.ones(2)}, {'x': torch.ones(2)}))

    def test_composite_preserves_depth_order_in_CPU_analytic_raster(self):
        layer = RoadLayer(fixture(), .1, 'R')
        layer.xyz[:, 2] = 2.
        layer.source_seen.fill_(True)
        layer.steps.fill_(1)
        camera = SimpleNamespace(world_view_transform=torch.eye(4))
        pipe = SimpleNamespace(compute_cov3D_python=False)
        static = SimpleNamespace(active_sh_degree=0)
        static.values = dict(means3D=torch.tensor([[0., 0., 1.]]), shs=None,
            colors_precomp=torch.zeros(1, 3), features=torch.tensor([[1., 1.]]),
            opacities=torch.ones(1, 1), scales=torch.ones(1, 3),
            rotations=torch.tensor([[1., 0., 0., 0.]]), cov3D_precomp=None,
            mask=torch.ones(1, dtype=torch.bool))

        def analytic(camera, pipe, degree, kwargs, background):
            color = torch.zeros(3); feature = torch.zeros(kwargs['features'].shape[1]); trans = 1.
            for row in kwargs['means3D'][:, 2].argsort().tolist():
                if not kwargs['mask'][row]:
                    continue
                alpha = float(kwargs['opacities'][row, 0]); weight = trans*alpha
                color += weight*kwargs['colors_precomp'][row]
                feature += weight*kwargs['features'][row]
                trans *= 1-alpha
            return dict(rgb=color[:, None, None], depth_sum=feature[:1, None, None],
                        alpha=feature[1:2, None, None], extra=feature[2:, None, None])

        fake_native = SimpleNamespace(prepare_rasterization=lambda camera, model, pipe: dict(model.values))
        with patch.dict(sys.modules, {'gaussian_renderer': fake_native}), patch('utils.v4_road_render.raster', analytic):
            hidden = render_composite(camera, static, None, layer, pipe, torch.zeros(3))
            self.assertEqual(float(hidden['rgb'].sum()), 0.)
            self.assertEqual(float(hidden['road_visible_alpha']), 0.)
            static.values['means3D'][:, 2] = 3.
            visible = render_composite(camera, static, None, layer, pipe, torch.zeros(3))
            self.assertGreater(float(visible['rgb'].sum()), 0.)
            self.assertGreater(float(visible['road_visible_alpha']), 0.)
            self.assertTrue((layer.xyz[:, 2] == 2.).all())


if __name__ == '__main__':
    unittest.main()
