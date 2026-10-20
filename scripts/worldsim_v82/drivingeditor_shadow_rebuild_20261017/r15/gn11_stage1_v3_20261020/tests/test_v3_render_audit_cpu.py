"""本文件只验证审计和数值比较器，不把CPU桩冒充CUDA实测。"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'engine_overlay' if (ROOT/'engine_overlay').is_dir() else ROOT/'engine'
sys.path.insert(0, str(ENGINE))
from utils.v3_render_audit import (RenderAudit, audit_span, raster_call, capacity_snapshot,
    expected_screen_buffers, check_screen_buffers, check_densification_numeric)
from utils.v3_densification import accumulate_max_screen_gradient
from utils.v3_runtime_context import _autograd
from utils.v3_road_diagnostics import road_visibility
from test_v3_cpu import _ModelStub
from test_v3_multiview_road_cpu import toy, toy_render, WEIGHTS


class RenderAuditTests(unittest.TestCase):
    def test_cpu_profile_never_touches_cuda_and_counts_inclusive_spans(self):
        with patch.object(torch.cuda, 'Event', side_effect=AssertionError('CPU must not create events')):
            audit = RenderAudit('cpu')
            audit.begin(1)
            with audit_span('outer'):
                result = raster_call('road_only', lambda **kw: kw['value'], value=3)
            record = audit.finish()
            self.assertEqual(result, 3)
            self.assertEqual(record['raster_calls'], {'road_only': 1})
            self.assertEqual(set(record['spans']), {'outer', 'raster/road_only'})
            self.assertIsNone(record['spans']['outer']['cuda_stream_elapsed_ms'])
            self.assertIsNone(record['memory_peak_allocated_bytes'])
            self.assertEqual(record['backend'], 'cpu_only_no_cuda_measurement')
            audit.begin(2)
            raster_call('joint', lambda: None)
            second = audit.finish()
            self.assertFalse(second['profiled'])
            self.assertEqual(second['cumulative_raster_calls_all_training_iterations'], {'road_only': 1, 'joint': 1})
            self.assertEqual(second['spans'], {})
            # SGCM原生offset50与整100步错开，不能漏计真实细化事件。
            audit.begin(50, topology_due=True)
            with audit_span('topology/SGCM'):
                pass
            sgcm = audit.finish()
            self.assertTrue(sgcm['profiled'] and sgcm['SGCM_due'])
            self.assertIn('topology/SGCM', sgcm['spans'])

    def test_numeric_reference_preserves_nonleaf_buffers_detects_double_add(self):
        proxies = [torch.zeros(n, 3, requires_grad=True)+0 for n in [3, 2, 1]]
        for p in proxies:
            p.retain_grad()
            p.grad = torch.full_like(p, .25)
        base = sum((i+1)*p.sum() for i, p in enumerate(proxies))
        body, road = 7*proxies[0].sum(), 11*proxies[1].sum()
        expected = expected_screen_buffers((base, body, road), proxies, torch.tensor([False, True]), _autograd)
        self.assertTrue(all(torch.equal(p.grad, torch.full_like(p, .25)) for p in proxies))
        body_grad = _autograd(body, [proxies[0]], proxies)[0].clone()
        road_grad = _autograd(road, [proxies[1]], proxies)[0]
        body_grad[1] = 0
        base.backward()
        proxies[0].grad += body_grad
        proxies[1].grad += road_grad
        checked = check_screen_buffers(proxies, expected)
        self.assertEqual(checked['joint']['max_absolute_error'], 0)
        proxies[0].grad += body_grad
        with self.assertRaisesRegex(RuntimeError, 'numeric mismatch'):
            check_screen_buffers(proxies, expected)

    def test_max_fusion_scratch_checks_real_function_and_does_not_change_model(self):
        model = _ModelStub(4)
        model.road_unit_mask = torch.tensor([False, False, False, True])
        b, j = [torch.zeros(4, 3) for _ in range(2)]
        b.grad = torch.tensor([[3., 4, 0], [1, 0, 0], [9, 0, 0], [99, 0, 0]])
        j.grad = torch.tensor([[0., 2, 0], [0, 6, 0], [0, 0, 0], [99, 0, 0]])
        visible_b, visible_j = torch.tensor([True, True, False, True]), torch.ones(4, dtype=torch.bool)
        result = check_densification_numeric(model, b, visible_b, torch.ones(4), j, visible_j, torch.ones(4), accumulate_max_screen_gradient)
        self.assertTrue(result['denominator_exact'])
        self.assertEqual(result['fixed_road_excluded'], 1)
        self.assertEqual(float(model.denom.sum()), 0)
        accumulate_max_screen_gradient(model, b, visible_b, torch.ones(4), j, visible_j, torch.ones(4))
        torch.testing.assert_close(model.xyz_gradient_accum[:, 0], torch.tensor([5., 6., 0., 0.]))
        torch.testing.assert_close(model.denom[:, 0], torch.tensor([1., 1., 1., 0.]))
        def double_count(*args):
            accumulate_max_screen_gradient(*args)
            args[0].denom += 1
        with self.assertRaisesRegex(RuntimeError, 'denominator'):
            check_densification_numeric(model, b, visible_b, torch.ones(4), j, visible_j, torch.ones(4), double_count)

    def test_capacity_id_guard_and_native_trigger_not_hidden_hard_cap(self):
        static = SimpleNamespace(get_xyz=torch.zeros(5, 3), _road_ids=lambda: torch.tensor([-1, 0, -1, 1, -1]))
        dynamic = SimpleNamespace(get_xyz=torch.zeros(3, 3))
        args = SimpleNamespace(densify_until_num_points=4, dynamic_densify_until_num_points=2)
        before = static.get_xyz.clone()
        record = capacity_snapshot(static, dynamic, args, torch.tensor([0, 1]))
        self.assertEqual(record['ordinary_background'], 3)
        self.assertEqual(record['static_trigger_overshoot'], 1)
        self.assertEqual(record['dynamic_trigger_overshoot'], 1)
        torch.testing.assert_close(before, static.get_xyz)
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            capacity_snapshot(static, dynamic, args, torch.tensor([0, 2]))
        static._road_ids = lambda: torch.tensor([-1, 0, -1, 0, -1])
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            capacity_snapshot(static, dynamic, args, torch.tensor([0, 1]))

    def test_actual_zero_update_backward_runs_numeric_checks_before_stop_CPU_stub(self):
        from utils.v3_runtime_context import V3Context, ZeroUpdateGateComplete
        from test_v3_responsibility_cpu import model
        parameters = {key: torch.ones(2, 1, requires_grad=True) for key in [
            'xyz', 'opacity', 'scaling', 'rotation', 't', 'scaling_t', 'velocity', 'f_dc', 'f_rest']}
        dp = torch.ones(1, 1, requires_grad=True)
        static = model(parameters, torch.tensor([False, True]))
        static.get_xyz, static._t = torch.zeros(2, 3), parameters['t']
        static._scaling_t, static._velocity = parameters['scaling_t'], parameters['velocity']
        dynamic = model({'all': dp}); dynamic.get_xyz, dynamic._t = torch.zeros(1, 3), dp
        proxies = [torch.zeros(n, 3, requires_grad=True)+0 for n in [3, 2, 1]]
        for p in proxies:
            p.retain_grad()
        package = dict(zip(['viewspace_points', 'static_viewspace_points', 'dynamic_viewspace_points'], proxies))
        for prefix, count in [('', 3), ('static_', 2), ('dynamic_', 1)]:
            package[prefix+'visibility_filter'] = torch.ones(count, dtype=torch.bool)
            package[prefix+'radii'] = torch.ones(count)
        base = sum(p.sum() for p in parameters.values())+dp.sum()+sum(p.sum() for p in proxies)
        body = sum(p.sum()*7 for p in parameters.values())+dp.sum()+proxies[0].sum()*7
        road = parameters['xyz'].sum()*11+proxies[1].sum()*11
        context = object.__new__(V3Context)
        context.gate_only, context.gate_details, context.models = True, {}, [static, dynamic]
        context.optimizer_steps = dict(static=0, dynamic=0)
        context.finish_iteration = lambda _: {'CPU_stub_only': True}
        with tempfile.TemporaryDirectory() as temporary:
            context.output = Path(temporary)
            with self.assertRaises(ZeroUpdateGateComplete):
                context.backward(base, body, road, static, dynamic, package, 1)
            result = json.loads((context.output/'gpu_gate_details.json').read_text())
            self.assertEqual(set(result['screen_buffers_numeric']), {'joint', 'static', 'dynamic'})
            self.assertEqual(result['densification_numeric']['static']['fixed_road_excluded'], 1)
            self.assertTrue(result['densification_numeric']['dynamic']['denominator_exact'])
            self.assertEqual(result['optimizer_updates'], 0)

    def test_road_visibility_keeps_joint_and_background_T_and_reports_no_GT(self):
        package = dict(background_means3D=torch.zeros(2, 3), alpha=torch.ones(1, 2, 2), static_alpha=torch.full((1, 2, 2), .9))
        road_package = dict(alpha=torch.full((1, 2, 2), .8))
        called = []
        def render(flags, joint):
            called.append((flags.clone(), joint))
            return torch.cat((torch.full((1, 2, 2), .1 if joint else .2), torch.zeros(1, 2, 2), package['alpha' if joint else 'static_alpha']))
        arrays, metrics = road_visibility(package, road_package, torch.tensor([False, True]), torch.arange(3),
            torch.tensor([[0., 0], [1, 1], [0, 0]]), torch.ones(3, dtype=torch.bool), torch.tensor([1, 2, 0]),
            torch.ones(3, dtype=torch.bool), torch.tensor([True, True, False]), render)
        self.assertEqual(len(called), 2)
        torch.testing.assert_close(called[0][0], torch.tensor([[0.], [1.]]))
        self.assertAlmostEqual(metrics['cohorts']['observed']['background_attenuation'], .6)
        self.assertAlmostEqual(metrics['cohorts']['prior']['joint_attenuation'], .7)
        self.assertIsNone(metrics['cohorts']['unknown']['road_only_alpha'])
        self.assertTrue(metrics['attenuation_is_not_illegal_occlusion_ground_truth'])
        self.assertFalse(arrays['road_alpha_in_joint'].requires_grad)
        def wrong(flags, joint):
            result = render(flags, joint)
            result[-1] *= .5
            return result
        with self.assertRaisesRegex(RuntimeError, 'transmittance'):
            road_visibility(package, road_package, torch.tensor([False, True]), torch.arange(3),
                torch.zeros(3, 2), torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.long),
                torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool), wrong)

    def test_source_selection_and_gradient_reach_are_separate(self):
        road_model, args, road = toy()
        road.objective_with_renderer(road_model, args, 1, WEIGHTS, toy_render)
        self.assertEqual(len(road.last_source_audit), 2)
        for row in road.last_source_audit:
            self.assertEqual(row['active_RGB_observations'], 2)
            self.assertGreater(row['sampled_DC_gradient_L2_sum'], 0)
        zero = dict(observed_rgb=0., observed_depth=0., prior_rgb=0., prior_depth=0.)
        road.objective_with_renderer(road_model, args, 1, zero, toy_render)
        for row in road.last_source_audit:
            self.assertEqual(row['active_RGB_observations'], 0)
            self.assertEqual(row['weighted_RGB_objective'], 0.)
            self.assertIsNone(row['robust_RGB_error_mean'])
            self.assertEqual(row.get('sampled_DC_gradient_L2_sum', 0), 0)

    def test_offline_cost_report_keeps_gate_separate_and_missing_data_unknown(self):
        path = ROOT/'tools/report_render_audit.py'
        if not path.exists():
            path = ROOT/'report_render_audit.py'
        spec = importlib.util.spec_from_file_location('render_report_test', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = module.export(root/'not_run', root/'empty')
            self.assertEqual(result['status'], 'NOT_EXECUTED_NO_RENDER_AUDIT')
            self.assertEqual(result['CUDA_phase_sample_ms'], {})
            with self.assertRaises(FileExistsError):
                module.export(root/'not_run', root/'empty')
            audit = RenderAudit('cpu'); audit.begin(1, gate=True)
            record = audit.finish()
            (root/'v3_render_cost.jsonl').write_text(json.dumps(record)+'\n')
            result = module.export(root, root/'gate')
            self.assertEqual(result['training_profiled_iterations'], 0)
            self.assertEqual(result['gate_profiled_iterations'], 1)


if __name__ == '__main__':
    unittest.main()
