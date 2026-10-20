import importlib.util
import ast
from collections import namedtuple
import sys
import tempfile
import json
from unittest.mock import patch
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'engine_overlay' if (ROOT/'engine_overlay').is_dir() else ROOT/'engine'
sys.path.insert(0, str(ENGINE))
from utils.v3_objectives import body_joint_rgb_loss
from utils.v3_runtime_context import V3Context, _autograd
from utils.v3_supervision_diagnostics import center_in_boxes, sampled_front_contributions, supervision_metrics


def model(parameters, road=None):
    result = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[
        dict(name=name, params=[value]) for name, value in parameters.items()]))
    if road is not None:
        result.road_unit_mask = road
    return result


class ResponsibilityTests(unittest.TestCase):
    def test_l1_main_exactly_matches_masked_v1_scale(self):
        pred = torch.ones(3, 4, 4, requires_grad=True)
        target = torch.zeros_like(pred)
        mask = torch.zeros(1, 4, 4)
        mask[:, 1:3, 1:3] = 1
        similarity = lambda a, b: 1-(a-b).square().mean()
        main = body_joint_rgb_loss(pred, target, mask, similarity, dssim=0)
        area = body_joint_rgb_loss(pred, target, mask, similarity, dssim=0, l1_normalization='body_area')
        self.assertEqual(float(main), .25)
        self.assertEqual(float(area), 1.)
        self.assertTrue(torch.equal(torch.autograd.grad(main, pred, retain_graph=True)[0], mask.expand_as(pred)/48))
        # 仅 L1 缩放，掩码结构项不变。
        self.assertEqual(float(body_joint_rgb_loss(pred, target, mask, similarity, 1)),
                         float(body_joint_rgb_loss(pred, target, mask, similarity, 1, 'body_area')))
        self.assertEqual(float(body_joint_rgb_loss(pred, target, mask*0, similarity)), 0.)

    def test_nonleaf_retain_grad_reproduces_old_double_count(self):
        proxy = torch.zeros(1, 3, requires_grad=True)+0
        proxy.retain_grad()
        gradient = torch.autograd.grad(7*proxy.sum(), [proxy], retain_graph=True)[0]
        self.assertTrue(torch.equal(proxy.grad, torch.full_like(proxy, 7)))
        proxy.grad += gradient
        self.assertTrue(torch.equal(proxy.grad, torch.full_like(proxy, 14)))

    def test_autograd_preserves_all_retained_buffers_and_existing_gradients(self):
        first = torch.ones(1, 3, requires_grad=True)+0
        second = torch.ones(1, 3, requires_grad=True)+0
        first.retain_grad(); second.retain_grad()
        first.grad = torch.full_like(first, 4)
        gradient = _autograd(2*first.sum()+3*second.sum(), [first], [first, second])[0]
        self.assertTrue(torch.equal(gradient, torch.full_like(first, 2)))
        self.assertTrue(torch.equal(first.grad, torch.full_like(first, 4)))
        self.assertIsNone(second.grad)

    def test_nonleaf_runtime_body_road_base_once_and_road_rows_protected(self):
        names = ['xyz', 'opacity', 'scaling', 'rotation', 't', 'scaling_t', 'velocity', 'f_dc', 'f_rest']
        parameters = {key: torch.ones(2, 1, requires_grad=True) for key in names}
        dynamic = torch.ones(1, requires_grad=True)
        environment = torch.ones(1, requires_grad=True)
        proxies = [torch.zeros(3 if i == 0 else (2 if i == 1 else 1), 3, requires_grad=True)+0 for i in range(3)]
        for proxy in proxies:
            proxy.retain_grad()
            proxy.grad = torch.full_like(proxy, .5)
        body = sum(value.sum()*10 for value in parameters.values())+2*dynamic.sum()+100*environment.sum()
        body += 3*proxies[0].sum()
        road = sum(value.sum()*4 for value in parameters.values())+100*dynamic.sum()+100*environment.sum()+5*proxies[1].sum()
        base = sum(value.sum()*7 for value in parameters.values())+11*environment.sum()
        base += sum((index+1)*proxy.sum() for index, proxy in enumerate(proxies))
        context = object.__new__(V3Context)
        context.gate_only = False
        context.backward(base, body, road, model(parameters, torch.tensor([False, True])),
                         model({'all_dynamic': dynamic}), dict(zip(
                             ['viewspace_points', 'static_viewspace_points', 'dynamic_viewspace_points'], proxies)), 1)
        for key in names:
            expected = torch.tensor([[21.], [11.]]) if key in ['xyz', 'opacity', 'scaling', 'rotation'] else torch.full((2, 1), 11.)
            self.assertTrue(torch.equal(parameters[key].grad, expected), key)
        self.assertEqual(float(dynamic.grad), 2.)
        self.assertEqual(float(environment.grad), 11.)
        expected_joint = torch.tensor([[4.5]*3, [1.5]*3, [4.5]*3])
        self.assertTrue(torch.equal(proxies[0].grad, expected_joint))
        self.assertTrue(torch.equal(proxies[1].grad, torch.full_like(proxies[1], 7.5)))
        self.assertTrue(torch.equal(proxies[2].grad, torch.full_like(proxies[2], 3.5)))


class DiagnosticTests(unittest.TestCase):
    def package(self):
        alpha = torch.full((1, 1, 1), .9)
        # 5m与20m静态贡献各.45，均值12.5m，实际前方alpha=.45。
        return dict(alpha=alpha, static_alpha=alpha, dynamic_alpha=torch.ones_like(alpha),
                    depth=torch.full_like(alpha, 9.), static_depth=torch.full_like(alpha, 11.25),
                    dynamic_depth=torch.full_like(alpha, 10.), render=torch.ones(3, 1, 1),
                    dynamic_premultiplied_rgb=torch.ones(3, 1, 1),
                    static_raster_kwargs={'features': torch.tensor([[5., 1.], [20., 1.]])})

    def test_native_feature_contract_detects_mean_depth_cancellation_and_outside_centers(self):
        package = self.package()
        def raster(flags, joint):
            weights = torch.tensor([.45, .45])
            values = (flags*weights[:, None]).sum(0)
            return torch.cat((values, torch.tensor([11.25, .9])))[:, None, None]
        arrays, metrics = sampled_front_contributions(package, torch.ones(1, 1, 1),
                torch.tensor([[10.]]), torch.tensor([False, True]), torch.tensor([False, True]), raster)
        self.assertFalse(bool(arrays['sample_mean_front'][0]))
        self.assertAlmostEqual(float(arrays['static_front_alpha'][0]), .45)
        self.assertEqual(float(arrays['static_front_center_inside_alpha'][0]), 0)
        self.assertEqual(float(arrays['static_front_road_alpha'][0]), 0)
        self.assertAlmostEqual(metrics['static_front_center_outside_alpha_mean'], .45)
        self.assertEqual(metrics['mean_depth_missed_pixels_alpha_gt_1over255'], 1)

    def test_feature_pass_fails_if_transmittance_changed(self):
        with self.assertRaisesRegex(RuntimeError, 'transmittance'):
            sampled_front_contributions(self.package(), torch.ones(1, 1, 1), torch.tensor([[10.]]),
                torch.tensor([False, True]), torch.tensor([False, True]),
                lambda flags, joint: torch.zeros(flags.shape[1]+2, 1, 1))

    def test_no_lidar_is_missing_not_a_clean_front_claim(self):
        arrays, metrics = sampled_front_contributions(self.package(), torch.ones(1, 1, 1), torch.zeros(1, 1),
                torch.tensor([False, True]), torch.tensor([False, True]), lambda *_: self.fail())
        self.assertEqual(metrics['status'], 'NO_VALID_BODY_LIDAR')
        self.assertEqual(metrics['extra_raster_calls'], 0)
        self.assertEqual(len(arrays['sample_pixels_yx']), 0)

    def test_depth_rgb_alpha_and_boundary_diagnostics_do_not_build_gradients(self):
        package = self.package()
        metrics, maps = supervision_metrics(package, torch.zeros(3, 1, 1), torch.ones(1, 1, 1),
                                            torch.tensor([[10.]]), .1)
        self.assertEqual(metrics['joint_body_depth_mae_m'], 0.)
        self.assertEqual(metrics['dynamic_body_depth_mae_m'], 0.)
        self.assertEqual(metrics['static_body_depth_mae_m'], 25.)
        self.assertEqual(metrics['body_dynamic_raw_rgb_l1'], 1.)
        self.assertTrue(all(not value.requires_grad for value in maps.values()))
        package['dynamic_alpha'].zero_()
        metrics, _ = supervision_metrics(package, torch.zeros(3, 1, 1), torch.ones(1, 1, 1),
                                         torch.zeros(1, 1), .1)
        self.assertIsNone(metrics['joint_body_depth_mae_m'])

    def test_center_box_test_has_no_2d_deletion_semantics(self):
        actual = center_in_boxes(torch.tensor([[0., 0., 0.], [2., 0., 0.]]), torch.zeros(1, 3),
                                 torch.eye(3)[None], torch.ones(1, 3))
        self.assertTrue(torch.equal(actual, torch.tensor([True, False])))

    def test_real_feature_renderer_keeps_geometry_and_does_not_mutate_main_package(self):
        tree = ast.parse((ENGINE/'gaussian_renderer/__init__.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'render_contribution_features')
        calls = []
        class Raster:
            def __init__(self, raster_settings):
                self.settings = raster_settings
            def __call__(self, **kwargs):
                calls.append(kwargs)
                self.assert_background = self.settings.bg
                return None, None, kwargs['features'].sum(0)[:, None, None], None
        namespace = dict(torch=torch, GaussianRasterizer=Raster)
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'actual_feature_render', 'exec'), namespace)
        def kwargs(count):
            return dict(means3D=torch.ones(count, 3), means2D=torch.zeros(count, 3),
                        opacities=torch.full((count, 1), .2), features=torch.ones(count, 6),
                        mask=torch.ones(count, dtype=torch.bool), shs=torch.ones(count, 1, 3),
                        scales=torch.ones(count, 3), rotations=torch.ones(count, 4), cov3D_precomp=None)
        static, joint = kwargs(2), kwargs(3)
        package = dict(static_raster_kwargs=static, joint_raster_kwargs=joint,
                       background_means3D=static['means3D'], raster_settings=namedtuple('Settings', ['bg'])(torch.ones(3)))
        flags = torch.tensor([[1., 0.], [0., 1.]])
        namespace['render_contribution_features'](package, flags, joint=True)
        actual = calls[0]
        for key in ['means3D', 'opacities', 'mask', 'scales', 'rotations']:
            self.assertEqual(actual[key].data_ptr(), joint[key].data_ptr())
        self.assertTrue(torch.equal(actual['features'][:, :2], torch.cat((flags, torch.zeros(1, 2)))))
        self.assertTrue(torch.equal(joint['features'], torch.ones(3, 6)))
        self.assertIsNotNone(joint['shs'])
        self.assertFalse(actual['means2D'].requires_grad)
        with self.assertRaisesRegex(ValueError, 'channel budget'):
            namespace['render_contribution_features'](package, torch.ones(2, 23))

    def test_offline_tool_never_calls_unexecuted_data_zero_success(self):
        path = ROOT/'tools/report_supervision_audit.py'
        if not path.exists():
            path = ROOT/'report_supervision_audit.py'
        spec = importlib.util.spec_from_file_location('supervision_report_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            result = module.export(Path(temporary)/'not_run', Path(temporary)/'output')
            self.assertEqual(result['status'], 'NOT_EXECUTED_NO_DIAGNOSTIC_FILES')
            self.assertEqual(result['records'], [])
            with self.assertRaises(FileExistsError):
                module.export(Path(temporary)/'not_run', Path(temporary)/'output')

    def test_actual_objective_adapter_exports_preupdate_diagnostics_on_cpu_stub(self):
        package = self.package()
        package.update(background_means3D=torch.tensor([[0., 0., 5.], [0., 0., 20.]]),
                       background_opacities=torch.full((2, 1), .5))
        def render(_, flags, joint):
            return torch.cat(((flags*.45).sum(0), torch.tensor([11.25, .9])))[:, None, None]
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {
                'gaussian_renderer': SimpleNamespace(render_contribution_features=render)}):
            context = object.__new__(V3Context)
            context.gate_only = True
            context.gate_details = {}
            from utils.v3_runtime_context import V3_POLICY
            context.policy = dict(V3_POLICY)
            context.arm = 'full'
            context.config = dict(training=dict(road_clearance_m=.1, volume_edge_m=.1, volume_exclusion_weight=.1), road=dict(objective_weights={}))
            context.road = SimpleNamespace(objective=lambda *args: (torch.tensor(0.), {}))
            context.output = Path(temporary)
            context.log = Path(temporary)/'objectives.jsonl'
            context.meta = dict(ground_plane_model=[0, 0, 1, 0], meter_to_model=.1)
            context.frame = lambda _: ('0008089', {'selected_boxes_model': [
                dict(center=[0, 0, 10], axes=torch.eye(3).tolist(), half=[1, 1, 1])]})
            result = context.objectives(SimpleNamespace(pts_depth=torch.tensor([[10.]])), package,
                SimpleNamespace(_xyz=torch.ones(2, 3), road_unit_mask=torch.tensor([False, True])),
                torch.ones(1, 1, 1), torch.zeros(3, 1, 1), SimpleNamespace(lambda_dssim=.2),
                lambda a, b: 1-(a-b).square().mean(), iteration=1)
            self.assertEqual(set(result), {'base_extra', 'body_joint', 'road'})
            record = json.loads(next(context.output.glob('supervision_audit/*/metadata.json')).read_text())
            self.assertEqual(record['completed_optimizer_updates'], 0)
            self.assertEqual(record['sampled_front']['extra_raster_calls'], 2)
            self.assertEqual(record['static_centers_inside_selected_boxes'], 0)
            self.assertEqual(context.gate_details['sampled_native_front_diagnostic']['sampled_pixels'], 1)


if __name__ == '__main__':
    unittest.main()
