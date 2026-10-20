"""道路世界身份、可微多来源目标及来源失效边界；不导入 CUDA 光栅器。"""

import ast
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'engine_overlay' if (ROOT/'engine_overlay').is_dir() else ROOT/'engine'
sys.path.insert(0, str(ENGINE))
from test_v3_temporal_cpu import model_on_cpu
from utils.v3_multiview_road import (MultiViewRoad, install_road_units, geometry_objective,
                                    sample_map, robust_rgb, updated_reliability)


def toy():
    model, args = model_on_cpu()
    args.densify_until_num_points = 100
    args.v3_road_meter_to_model = .1
    arrays = dict(surface_supported=np.ones(2, bool), model_xyz=np.array([[0., 0, 0], [.02, 0, 0]]),
                  normal_model=np.array([[0., 0, 1.], [0, 0, 1.]]))
    install_road_units(model, arrays, args)
    road = object.__new__(MultiViewRoad)
    road.data = {key: torch.as_tensor(value, dtype=torch.float32 if key != 'surface_supported' else torch.bool)
                 for key, value in arrays.items()}
    road.data.update(base_weight=torch.tensor([[1., 1.], [3., 3.]]),
                     source_rgb=torch.tensor([[[.8]*3]*2, [[.05]*3]*2]),
                     source_depth=torch.full((2, 2), .1), source_uv=torch.zeros(2, 2, 2),
                     provenance=torch.ones(2, dtype=torch.long), is_query=torch.ones(2, dtype=torch.bool),
                     edges=torch.tensor([[0, 1]]))
    road.reliability = torch.ones(2, 2)
    road.cameras, road.scale = [0, 1], .1
    road.visits, road.gradient_samples = np.zeros((2, 2), np.int32), np.zeros((2, 2), np.int32)
    road.gradient_l2_cumulative = np.zeros((2, 2))
    road.coverage_uv = torch.zeros(2, 2, 9, 2)
    road.coverage_valid = torch.zeros(2, 2, 9, dtype=torch.bool)
    road.coverage_denominator = road.coverage_valid.sum()
    return model, args, road


def toy_render(camera, model, args):
    rows = torch.nonzero(model.road_unit_mask).flatten()
    rgb = (model._features_dc[rows, 0]*.28209479177387814+.5).mean(0)[:, None, None].expand(3, 2, 2)
    proxy = torch.zeros(2, 3, requires_grad=True)
    depth = (.1+model._xyz[rows, 2].mean()).expand(1, 2, 2)
    return dict(rgb=rgb, alpha=torch.ones(1, 2, 2), depth_sum=depth, proxy=proxy)


WEIGHTS = dict(observed_rgb=1., observed_depth=0., prior_rgb=0., prior_depth=0.)


class MultiviewRoadTests(unittest.TestCase):
    def test_complete_checkpoint_carries_live_source_reliability_and_audit_state(self):
        from utils import paper_checkpoint, v3_runtime_context
        model, args, road = toy()
        road.path, road.exclusions = Path('/test-world-units.json'), []
        road.objective_with_renderer(model, args, 1, WEIGHTS, toy_render)
        reliability, visits = road.reliability.clone(), road.visits.copy()
        scene = SimpleNamespace(train_cameras={1.: []}, scale_index=0, resolution_scales=[1.],
                                 getTrainCameras=lambda: [])
        with patch.object(v3_runtime_context, 'ACTIVE_CONTEXT', SimpleNamespace(road=road)):
            state = paper_checkpoint.capture_training_state(1, scene, [], {}, {'v3_bundle': str(road.path)})
            self.assertIn('v3_world_road_runtime', state)
            road.reliability.fill_(.1)
            road.visits.fill(0)
            with tempfile.TemporaryDirectory() as temp, patch.object(paper_checkpoint, '_validate_continuation', lambda *a: None):
                # 这里只测新增 runtime hook；原有完整场景/续训配置校验不属于此桩。
                destination = Path(temp)/'training_state.pth'
                torch.save(state, destination)
                paper_checkpoint.restore_training_state(destination, 1, scene, {}, {'v3_bundle': str(road.path)})
            torch.testing.assert_close(road.reliability, reliability)
            np.testing.assert_array_equal(road.visits, visits)

    def test_road_only_preparation_uses_exact_same_parameters_as_normal_forward(self):
        tree = ast.parse((ENGINE/'gaussian_renderer/__init__.py').read_text(encoding='utf-8'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {'prepare_rasterization', 'render_road_only'}]
        captured = {}
        class RasterStub:
            def __init__(self, **kwargs):
                pass
            def __call__(self, **kwargs):
                captured.update(kwargs)
                return None, torch.zeros(3, 2, 2), torch.zeros(2, 2, 2), torch.ones(len(kwargs['means3D']))
        from utils.v3_render_audit import raster_call
        namespace = dict(torch=torch, math=math, GaussianModel=object, raster_call=raster_call,
                         GaussianRasterizationSettings=lambda **kw: kw, GaussianRasterizer=RasterStub)
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'actual_renderer_cpu_preparation', 'exec'), namespace)
        model, _, _ = toy()
        camera = SimpleNamespace(timestamp=.1, FoVx=1., FoVy=1., image_width=2, image_height=2,
                                 world_view_transform=torch.eye(4), full_proj_transform=torch.eye(4), camera_center=torch.zeros(3))
        pipe = SimpleNamespace(neg_fov=False, debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        normal = namespace['prepare_rasterization'](camera, model, pipe)
        result = namespace['render_road_only'](camera, model, pipe)
        for key in normal:
            if normal[key] is None:
                self.assertIsNone(captured[key])
            else:
                torch.testing.assert_close(captured[key], normal[key][result['rows']], rtol=0, atol=0)
        with torch.no_grad():
            evaluated = namespace['render_road_only'](camera, model, pipe)
        self.assertFalse(evaluated['proxy'].requires_grad)

    def test_world_identity_is_time_shared_but_ordinary_background_can_move(self):
        model, _, _ = toy()
        first, second = model.get_xyz_SHM(.01), model.get_xyz_SHM(.07)
        self.assertFalse(torch.equal(first[:2], second[:2]))
        self.assertTrue(torch.equal(first[2:], second[2:]))
        self.assertTrue(torch.equal(model.get_marginal_t(.48)[2:], torch.ones(2, 1)))
        self.assertTrue(torch.equal(model.get_features[2:, 1:], torch.zeros_like(model.get_features[2:, 1:])))

    def test_append_prune_clone_and_checkpoint_keep_world_ids_and_budget(self):
        model, args, _ = toy()
        self.assertEqual(len(model._xyz), 4)
        model._opacity.sum().backward()
        model.optimizer.step()
        opacity_before = model._opacity[model.road_unit_mask].detach().clone()
        model.reset_opacity()
        torch.testing.assert_close(model._opacity[model.road_unit_mask], opacity_before)
        self.assertLessEqual(float(model.get_opacity[~model.road_unit_mask].max()), .010001)
        model.densify_and_clone(torch.ones(4, 1), .1, 1., torch.ones(4, 1), .1)
        self.assertEqual(int(model.road_unit_mask.sum()), 2)
        model.prune_points(torch.ones(len(model._xyz), dtype=torch.bool))
        self.assertTrue(torch.equal(model._road_ids(), torch.tensor([0, 1])))
        restored, _ = model_on_cpu()
        restored.restore(model.capture(), args)
        self.assertTrue(torch.equal(restored._road_ids(), torch.tensor([0, 1])))
        self.assertTrue(torch.equal(restored.get_xyz_SHM(.01), restored.get_xyz_SHM(.1)))

    def test_matched_multi_source_loss_keeps_raw_colors_and_global_weight_denominator(self):
        model, args, road = toy()
        original = road.data['source_rgb'].clone()
        with patch('utils.v3_multiview_road.updated_reliability', lambda a, b: torch.ones_like(a)):
            result, terms = road.objective_with_renderer(model, args, 1, WEIGHTS, toy_render)
        predicted = toy_render(0, model, args)['rgb'][:, 0, 0]
        expected = .25*robust_rgb(predicted, original[0, 0])+.75*robust_rgb(predicted, original[1, 0])
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(road.data['source_rgb'], original)
        result.backward()
        self.assertGreater(float(model._features_dc.grad[2:].abs().sum()), 0)
        self.assertTrue((road.gradient_l2_cumulative > 0).all())
        self.assertTrue(np.array_equal(road.visits, np.ones((2, 2))))

    def test_hard_rejected_source_cannot_get_weight_from_residual_reliability(self):
        model, args, road = toy()
        road.data['base_weight'][0] = 0
        first, _ = road.objective_with_renderer(model, args, 1, WEIGHTS, toy_render)
        road.data['source_rgb'][0] = 1e6
        second, _ = road.objective_with_renderer(model, args, 1, WEIGHTS, toy_render)
        torch.testing.assert_close(first, second)
        self.assertTrue(np.array_equal(road.visits[0], [0, 0]))
        self.assertEqual(float(road.gradient_l2_cumulative[0].sum()), 0.)

    def test_reliability_is_bounded_detached_and_does_not_create_exposure(self):
        color = torch.tensor([0., 100.], requires_grad=True)
        depth = torch.tensor([0., 100.], requires_grad=True)
        weight = updated_reliability(color, depth)
        self.assertFalse(weight.requires_grad)
        torch.testing.assert_close(weight, torch.tensor([1., .1]))

    def test_surface_normal_thin_layer_terms_have_parameter_gradients(self):
        model, _, road = toy()
        with torch.no_grad():
            model._xyz[2:, 2] += .01
            model._scaling[2:, 2] = np.log(.1*.1)
            model._rotation[2:, 1] = .3
        loss, terms = geometry_objective(model, road.data, .1)
        loss.backward()
        for field in ['_xyz', '_rotation', '_scaling']:
            self.assertGreater(float(getattr(model, field).grad[2:].abs().sum()), 0)
        self.assertGreater(float(terms['surface']), 0)

    def test_unexposed_prior_uses_continuity_not_fake_ground_truth(self):
        model, args, road = toy()
        road.data['provenance'] = torch.tensor([1, 2])
        road.data['base_weight'][:, 1] = 0
        with torch.no_grad():
            model._features_dc[2, 0] += 1
        result, terms = road.objective_with_renderer(model, args, 1,
                                                     dict(observed_rgb=0., observed_depth=0., prior_rgb=.1, prior_depth=.01), toy_render)
        result.backward()
        self.assertGreater(terms['prior_continuity'], 0)
        self.assertGreater(float(model._features_dc.grad[3].abs().sum()), 0)
        self.assertTrue(np.array_equal(road.visits[:, 1], [0, 0]))

    def test_pixel_center_bilinear_sampling_is_explicit_and_differentiable(self):
        image = torch.tensor([[[1., 3.], [5., 7.]]], requires_grad=True)
        value = sample_map(image, torch.tensor([[.5, .5], [1., 0.]]))
        torch.testing.assert_close(value[:, 0], torch.tensor([4., 3.]))
        value.sum().backward()
        self.assertTrue(torch.isfinite(image.grad).all())


if __name__ == '__main__':
    unittest.main()
