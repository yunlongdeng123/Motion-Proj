"""本轮五项代码审查反例；CPU解析alpha不是CUDA渲染通过声明。"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'engine_overlay' if (ROOT/'engine_overlay').exists() else ROOT/'engine'
TOOLS = ROOT/'tools' if (ROOT/'tools').exists() else ROOT
sys.path[:0] = [str(ENGINE), str(TOOLS)]
from utils.v3_road_prior import prior_graph, rendered_coverage, fixed_subcells
from utils.v3_road_geometry import required_occluder_masks
from v3_gpu_authorization import GPU_ENTRY_VERSION, require_gpu_authorization


class RoadReviewTests(unittest.TestCase):
    def test_high_point_opacity_same_color_surface_can_still_have_subcell_holes(self):
        uv = torch.tensor([[[x, y] for x in [8., 16., 24.] for y in [8., 16., 24.]]])
        valid = torch.ones(1, 9, dtype=torch.bool)
        yy, xx = torch.meshgrid(torch.arange(33.), torch.arange(33.), indexing='ij')
        scale = torch.tensor(2., requires_grad=True)
        opacity = torch.tensor(.999)
        alpha = (opacity*torch.exp(-((xx-16)**2+(yy-16)**2)/(2*scale**2)))[None]
        old = torch.relu(.95-opacity)**2
        self.assertEqual(float(old), 0.)
        loss, samples = rendered_coverage(alpha, uv, valid, valid.sum(), 1.)
        self.assertGreater(float(samples.max()), .95)
        self.assertLess(float(samples.min()), .01)
        self.assertGreater(float(loss), .5)
        loss.backward(retain_graph=True)
        self.assertLess(float(scale.grad), 0.)
        wide = (opacity*torch.exp(-((xx-16)**2+(yy-16)**2)/(2*16.**2)))[None]
        improved, _ = rendered_coverage(wide, uv, valid, valid.sum(), 1.)
        self.assertLess(float(improved), float(loss))
        # 先验关时该新增目标梯度为0；没有第三个实验因素。
        scale.grad = None
        off, _ = rendered_coverage(alpha, uv, valid, valid.sum(), 1.)
        (off*0).backward()
        self.assertEqual(float(scale.grad), 0.)

    def test_context_prior_can_connect_query_to_measured_context_but_unknown_cannot_bridge(self):
        p = np.array([2, 2, 1, 2, 0, 1])
        query = np.array([True, False, False, True, False, False])
        graph, components = prior_graph(p, np.ones(6, bool), query, np.array([[0, 1], [1, 2], [3, 4], [4, 5]]))
        self.assertEqual(graph['context_hops'][0], 2)
        self.assertEqual(graph['context_hops'][3], -1)
        self.assertTrue(components[0]['anchored_to_context'])
        self.assertFalse(components[1]['anchored_to_context'])
        np.testing.assert_array_equal(graph['edge_valid'], [True, True, False, False])

    def test_world_subcells_fixed_on_local_tangent_not_trainable_centers(self):
        arrays = dict(normal_world=np.array([[0., 0, 1.]]), world=np.array([[1., 2, 3.]]))
        metadata = dict(coordinate_anchor=np.eye(4), world_to_model=np.eye(4))
        points = fixed_subcells(arrays, metadata, dict(coverage_offsets=[-.4, 0., .4], cell_m=.2))
        self.assertEqual(points.shape, (1, 9, 3))
        np.testing.assert_allclose(points[0, 4], [1, 2, 3])
        np.testing.assert_allclose(points[0, 0], [.92, 1.92, 3])

    def test_all_three_original_masks_required_and_native_only_occluder_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fid = '0008089'
            paths = [root/'full5_pipeline/inputs/scene0008_089_137/matched_training_labels',
                root/'all_vehicle_pipeline/inputs/scene0008_089_137/dynamic_masks/dynamic_mask_0',
                root/'all_vehicle_native49/native_predictions']
            for path in paths:
                path.mkdir(parents=True)
            blank = np.zeros((320, 480), np.uint8)
            for path in paths[:2]:
                Image.fromarray(blank).save(path/(fid+'.png'))
            with self.assertRaises(FileNotFoundError):
                required_occluder_masks(root, fid)
            proposal = blank.astype(bool)[None]; proposal[0, 23, 17] = True
            np.savez(paths[2]/(fid+'.npz'), masks=proposal)
            union, audit = required_occluder_masks(root, fid)
            self.assertTrue(union[23, 17]); self.assertEqual(audit['native_extra_pixels'], 1)

    def test_single_entry_gate_and_train_refuse_missing_or_stale_approval_before_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp)
            plan = dict(campaign=str(campaign), method_version='m', road_version='r',
                common_v3_policy={'supervision_version': 's'}, render_audit_version='a',
                input_bundle='/fixed49/bundle.json', arm_order=['full'], road_source_exclusions=[])
            plan_path = campaign/'plan.json'; plan_path.write_text(json.dumps(plan))
            for mode in ['gate', 'train']:
                for acknowledged in [False, True]:
                    with self.assertRaises(RuntimeError):
                        require_gpu_authorization(plan_path, plan, acknowledged, 'full', mode)
                command = [sys.executable, str(TOOLS/'run_v3_arm.py')]
                for key in ['engine', 'output', 'bundle', 'baseline-config', 'readiness', 'binding-tools', 'gpu-lock']:
                    command += ['--'+key, str(campaign/key)]
                command += ['--plan', str(plan_path), '--arm', 'full', '--mode', mode]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('explicit post-review GPU execution flag', result.stderr)
                self.assertFalse((campaign/'output').exists())
            approval = dict(status='USER_AUTHORIZED_GPU_V3', scope='zero_update_gates_and_four_fresh_30k',
                method_version='m', road_version='r', supervision_version='s', render_audit_version='a',
                gpu_entry_version=GPU_ENTRY_VERSION, input_bundle=plan['input_bundle'], arm_order=['full'])
            path = campaign/'GPU_APPROVAL.json'; path.write_text(json.dumps(approval))
            require_gpu_authorization(plan_path, plan, True, 'full', 'gate')
            approval['input_bundle'] = '/old/bundle.json'; path.write_text(json.dumps(approval))
            with self.assertRaises(RuntimeError):
                require_gpu_authorization(plan_path, plan, True, 'full', 'gate')


if __name__ == '__main__':
    unittest.main()
