"""原生时间生命周期测试；选择性证据相关四项仅保留为离线历史测试。"""

import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine_overlay" if (ROOT / "engine_overlay").is_dir() else ROOT / "engine"
sys.path.insert(0, str(ENGINE))
from utils.v3_temporal import (
    TEMPORAL_POLICY, TEMPORAL_VERSION, TemporalEvidence, repeated_surface_cells,
    selective_temporal_loss, voxel_keys_numpy,
)
from utils.v3_densification import accumulate_max_screen_gradient


def native_class_on_cpu():
    """读取实际类，只把设备常量 cuda 换为 cpu；公式/优化器/拓扑不改。"""
    tree = ast.parse((ENGINE / "scene" / "gaussian_model.py").read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"NativeGaussianModel", "GaussianModel"}]
    class CPUDevice(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant("cpu"), node) if node.value == "cuda" else node
    module = CPUDevice().visit(ast.Module(body=classes, type_ignores=[]))
    namespace = dict(
        torch=torch, nn=torch.nn, np=np, math=math, BasicPointCloud=object,
        inverse_sigmoid=torch.logit, build_rotation=lambda q: None,
        build_scaling_rotation=lambda s, r: None,
        get_expon_lr_func=lambda **kw: lambda i: kw["lr_init"],
        capture_sgcm_optimizer=lambda model: model.optimizer.state_dict(),
        restore_sgcm_optimizer=lambda model, state: state,
        lineage_after_append=lambda *args: None, lineage_after_prune=lambda *args: None,
    )
    exec(compile(ast.fix_missing_locations(module), "native_gaussian_cpu_test", "exec"), namespace)
    return namespace["GaussianModel"]


def model_on_cpu():
    cls = native_class_on_cpu()
    args = SimpleNamespace(
        sh_degree=0, time_duration=[-0.48, 0.48], no_time_split=True, t_grad=True,
        contract=False, t_init=0.1, big_point_threshold=0.1, cycle=0.2,
        velocity_decay=1.0, random_init_point=0, dynamic_random_init_point=0,
        percent_dense=0.01, position_lr_init=0.00016, position_lr_final=1.6e-6,
        position_lr_delay_mult=0.01, iterations=30000, feature_lr=0.0025,
        opacity_lr=0.005, scaling_lr=0.005, rotation_lr=0.001,
        t_lr_init=0.0008, scaling_t_lr=0.002, velocity_lr=0.001,
    )
    model = cls(args)
    for name, value in {
        "_xyz": torch.zeros(2, 3), "_features_dc": torch.zeros(2, 1, 3),
        "_features_rest": torch.zeros(2, 0, 3), "_scaling": torch.full((2, 3), -5.0),
        "_rotation": torch.tensor([[1., 0, 0, 0]]).repeat(2, 1),
        "_opacity": torch.zeros(2, 1), "_t": torch.zeros(2, 1),
        "_scaling_t": torch.zeros(2, 1), "_velocity": torch.ones(2, 3),
    }.items():
        setattr(model, name, torch.nn.Parameter(value))
    model.max_radii2D = torch.zeros(2)
    model.spatial_lr_scale = 1.0
    model.training_setup(args)
    return model, args


class TemporalTests(unittest.TestCase):
    def test_repeated_returns_need_distinct_frames_and_time_span(self):
        stable, _, _ = repeated_surface_cells(
            [np.array([1, 1, 2]), np.array([1, 2]), np.array([1, 3])],
            [0.0, 0.1, 0.3], 3, 0.2,
        )
        np.testing.assert_array_equal(stable, [1])
        short, _, _ = repeated_surface_cells(
            [np.array([1]), np.array([1]), np.array([1])], [0, 0.01, 0.02], 3, 0.2,
        )
        self.assertEqual(len(short), 0)

    def test_field_requeries_current_geometry_and_topology(self):
        xyz = np.array([[0.005, 0.005, 0.005], [0.025, 0.005, 0.005]])
        keys = voxel_keys_numpy(xyz, 0.02)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez(root / "field.npz", keys=keys, confidence=np.array([1.0, 0.5]))
            (root / "metadata.json").write_text(json.dumps({
                "version": TEMPORAL_VERSION, "policy": TEMPORAL_POLICY,
                "field": "field.npz", "meter_to_model": 0.1,
            }))
            field = TemporalEvidence(root / "metadata.json", "cpu")
            points = torch.tensor(xyz, dtype=torch.float32)
            scales = torch.full((2, 3), 0.005)
            self.assertTrue(torch.equal(field.weights(points, scales), torch.tensor([1., .5])))
            # clone在原位继承空间证据；split移离表面/大支撑只能成为未知。
            children = torch.cat([points, points[:1], torch.full((1, 3), 10.)])
            child_scales = torch.full((4, 3), 0.005)
            child_scales[1] = 1.0
            self.assertTrue(torch.equal(
                field.weights(children, child_scales), torch.tensor([1., 0., 1., 0.])
            ))

    def test_native_background_moves_and_has_temporal_gradients(self):
        model, _ = model_on_cpu()
        self.assertFalse(torch.equal(model.get_xyz_SHM(0.01), model.get_xyz_SHM(0.05)))
        self.assertFalse(torch.equal(model.get_marginal_t(-0.4), model.get_marginal_t(0.1)))
        loss = model.get_xyz_SHM(0.03).sum() + model.get_marginal_t(0.1).sum()
        loss.backward()
        for field in ("_t", "_scaling_t", "_velocity"):
            parameter = getattr(model, field)
            self.assertTrue(parameter.requires_grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(all(group["lr"] > 0 for group in model.optimizer.param_groups))

    def test_stable_loss_is_stronger_and_raw_motion_cannot_hide_in_lifetime(self):
        model, _ = model_on_cpu()
        with torch.no_grad():
            model._scaling_t.fill_(8.0)
        self.assertEqual(float(model.get_inst_velocity.abs().max()), 0.0)
        loss, metrics = selective_temporal_loss(model, torch.tensor([1., 0.]), [-.48, .48], .1)
        loss.backward()
        self.assertGreater(float(metrics["stable_motion"]), 0)
        self.assertGreater(
            float(model._velocity.grad[0].abs().sum()),
            50 * float(model._velocity.grad[1].abs().sum()),
        )
        self.assertGreater(float(model._velocity.grad[1].abs().sum()), 0)

    def test_native_append_prune_and_checkpoint_preserve_temporal_parameters(self):
        model, args = model_on_cpu()
        (sum(parameter.square().sum() for group in model.optimizer.param_groups
             for parameter in group["params"])).backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        fields = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
                  "_rotation", "_t", "_scaling_t", "_velocity")
        model.densification_postfix(*[getattr(model, field)[:1].detach().clone() for field in fields])
        model.prune_points(torch.tensor([False, True, False]))
        captured = model.capture()
        restored, _ = model_on_cpu()
        restored.restore(captured, args)
        for field in ("_t", "_scaling_t", "_velocity"):
            self.assertTrue(torch.equal(getattr(model, field), getattr(restored, field)))
            self.assertTrue(getattr(restored, field).requires_grad)

    def test_background_temporal_densification_statistics_are_counted_once(self):
        model, _ = model_on_cpu()
        model._t.grad = torch.tensor([[2.], [3.]])
        branch = torch.zeros(2, 3, requires_grad=True)
        joint = torch.zeros(2, 3, requires_grad=True)
        branch.grad = torch.ones(2, 3)
        joint.grad = torch.full((2, 3), 2.)
        visible = torch.ones(2, dtype=torch.bool)
        radii = torch.ones(2)
        accumulate_max_screen_gradient(model, branch, visible, radii, joint, visible, radii)
        self.assertTrue(torch.equal(model.t_gradient_accum, torch.tensor([[2.], [3.]])))
        self.assertTrue(torch.equal(model.denom, torch.ones(2, 1)))

    def test_object_veto_is_three_dimensional_and_does_not_clear_road(self):
        builder = ROOT / "tools" / "prepare_temporal_evidence.py"
        if not builder.is_file():
            builder = ROOT / "prepare_temporal_evidence.py"
        if not builder.is_file():
            self.skipTest("historical offline builder is not a current campaign dependency")
        spec = importlib.util.spec_from_file_location("temporal_builder_test", builder)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        box = {"center": np.array([0., 0., 1.]), "axes": np.eye(3), "half": np.array([2., 1., .8])}
        points = np.array([[0, 0, 1.], [0, 0, 0.], [0, 2, 1.]])
        np.testing.assert_array_equal(module.inside_boxes(points, [box]), [True, False, False])
        centers = module.box_voxel_centers(box, .2)
        self.assertTrue((centers[:, 2] > 0).all())


if __name__ == "__main__":
    unittest.main()
