import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine_overlay" if (ROOT / "engine_overlay").is_dir() else ROOT / "engine"
sys.path.insert(0, str(ENGINE))

from utils.v3_densification import accumulate_max_screen_gradient
from utils.v3_objectives import (
    STATIC_BODY_RGB_GROUPS,
    body_joint_rgb_loss,
    dynamic_color_leak_losses,
    observed_rgb_loss,
    static_front_occlusion_loss,
)

if (ENGINE / "utils" / "v2_road_objective.py").is_file():
    from utils.v3_runtime_context import V3Context
else:
    V3Context = None


def differentiable_similarity(first, second):
    return 1.0 - ((first - second) ** 2).mean()


class ObjectiveTests(unittest.TestCase):
    def test_observed_and_body_rgb_have_disjoint_direct_pixel_domains(self):
        mask = torch.zeros(1, 4, 4)
        mask[:, 1:3, 1:3] = 1
        target = torch.zeros(3, 4, 4)

        outside_prediction = torch.ones(3, 4, 4, requires_grad=True)
        outside_loss = observed_rgb_loss(
            outside_prediction, target, mask, differentiable_similarity, dssim=0.2
        )
        outside_loss.backward()
        self.assertEqual(float(outside_prediction.grad[:, 1:3, 1:3].abs().max()), 0.0)
        self.assertGreater(float(outside_prediction.grad[:, 0, 0].abs().max()), 0.0)

        body_prediction = torch.ones(3, 4, 4, requires_grad=True)
        body_loss = body_joint_rgb_loss(
            body_prediction, target, mask, differentiable_similarity, dssim=0.2
        )
        body_loss.backward()
        self.assertGreater(float(body_prediction.grad[:, 1:3, 1:3].abs().max()), 0.0)
        self.assertEqual(float(body_prediction.grad[:, 0, 0].abs().max()), 0.0)

    def test_front_occlusion_uses_depth_order_and_is_differentiable(self):
        mask = torch.ones(1, 2, 2)
        measured = torch.full((1, 2, 2), 10.0)
        alpha = torch.full((1, 2, 2), 0.5, requires_grad=True)
        front_sum = torch.full((1, 2, 2), 2.5, requires_grad=True)
        front, metrics = static_front_occlusion_loss(front_sum, alpha, measured, mask)

        behind_alpha = torch.full((1, 2, 2), 0.5)
        behind_sum = torch.full((1, 2, 2), 12.0) * behind_alpha
        behind, _ = static_front_occlusion_loss(behind_sum, behind_alpha, measured, mask)
        self.assertGreater(float(front), float(behind) + 0.1)
        self.assertEqual(float(metrics["front_valid_pixels"]), 4.0)
        front.backward()
        self.assertTrue(torch.isfinite(alpha.grad).all())
        self.assertTrue(torch.isfinite(front_sum.grad).all())

    def test_dynamic_leak_penalizes_low_alpha_high_color_outside_mask(self):
        mask = torch.zeros(1, 10, 10)
        mask[:, 4:6, 4:6] = 1
        alpha = torch.zeros(1, 10, 10)
        alpha[:, 0, 0] = 0.0428
        raw = torch.zeros(3, 10, 10)
        raw[0, 0, 0] = 0.5495
        color_loss, ratio_loss, metrics = dynamic_color_leak_losses(raw, alpha, mask)
        self.assertGreater(float(color_loss), 0.0)
        self.assertGreater(float(ratio_loss), 0.0)
        self.assertGreater(float(metrics["dynamic_ratio_excess_top"]), 0.0)

    def test_dynamic_leak_ignores_two_pixel_boundary_band(self):
        mask = torch.zeros(1, 10, 10)
        mask[:, 4:6, 4:6] = 1
        alpha = torch.zeros(1, 10, 10)
        raw = torch.zeros(3, 10, 10)
        raw[:, 3, 4] = 1.0
        color_loss, ratio_loss, _ = dynamic_color_leak_losses(raw, alpha, mask)
        self.assertEqual(float(color_loss), 0.0)
        self.assertEqual(float(ratio_loss), 0.0)


class _ModelStub:
    def __init__(self, count):
        self.get_xyz = torch.zeros(count, 3)
        self.max_radii2D = torch.zeros(count)
        self.xyz_gradient_accum = torch.zeros(count, 1)
        self.denom = torch.zeros(count, 1)
        self.t_gradient_accum = torch.ones(count, 1)
        self._t = torch.zeros(count, 1, requires_grad=False)


class DensificationTests(unittest.TestCase):
    def test_branch_and_joint_use_max_with_single_denominator_increment(self):
        model = _ModelStub(3)
        branch = torch.zeros(3, 3, requires_grad=True)
        joint = torch.zeros(3, 3, requires_grad=True)
        branch.grad = torch.tensor([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        joint.grad = torch.tensor([[0.0, 2.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 0.0]])
        statistics = accumulate_max_screen_gradient(
            model,
            branch,
            torch.tensor([True, True, False]),
            torch.tensor([2.0, 3.0, 1.0]),
            joint,
            torch.tensor([True, True, True]),
            torch.tensor([4.0, 1.0, 5.0]),
        )
        self.assertTrue(torch.equal(model.xyz_gradient_accum[:, 0], torch.tensor([5.0, 6.0, 0.0])))
        self.assertTrue(torch.equal(model.denom[:, 0], torch.ones(3)))
        self.assertTrue(torch.equal(model.max_radii2D, torch.tensor([4.0, 3.0, 5.0])))
        self.assertEqual(float(statistics["joint_selected_points"]), 1.0)
        self.assertTrue(torch.equal(model.t_gradient_accum, torch.zeros(3, 1)))


class _OptimizerStub:
    def __init__(self, named_parameters):
        self.param_groups = [
            {"name": name, "params": [parameter]}
            for name, parameter in named_parameters.items()
        ]


class _RouteModelStub:
    def __init__(self, named_parameters):
        self.optimizer = _OptimizerStub(named_parameters)


class GradientRoutingTests(unittest.TestCase):
    @unittest.skipIf(V3Context is None, "full frozen v2 engine is required")
    def test_runtime_accepts_omegaconf_temporal_policy(self):
        from omegaconf import OmegaConf
        from utils.v3_runtime_context import _json_clone
        from utils.v3_regularizer_contract import TEMPORAL_POLICY
        self.assertEqual(_json_clone(OmegaConf.create(TEMPORAL_POLICY)), TEMPORAL_POLICY)

    @unittest.skipIf(V3Context is None, "full frozen v2 engine is required")
    def test_body_and_road_parameter_responsibility_is_explicit(self):
        self.assertEqual(
            STATIC_BODY_RGB_GROUPS,
            frozenset({"xyz", "opacity", "scaling", "rotation", "t", "scaling_t", "velocity"}),
        )
        static_parameters = {
            name: torch.tensor(1.0, requires_grad=True)
            for name in ["xyz", "opacity", "scaling", "rotation", "f_dc", "f_rest", "t", "scaling_t", "velocity"]
        }
        dynamic_parameter = torch.tensor(1.0, requires_grad=True)
        environment_parameter = torch.tensor(1.0, requires_grad=True)
        static = _RouteModelStub(static_parameters)
        dynamic = _RouteModelStub({"dynamic": dynamic_parameter})
        joint_screen = torch.ones(1, 3, requires_grad=True)
        static_screen = torch.ones(1, 3, requires_grad=True)
        dynamic_screen = torch.ones(1, 3, requires_grad=True)

        body = (
            2 * static_parameters["xyz"]
            + 3 * static_parameters["opacity"]
            + 4 * static_parameters["scaling"]
            + 5 * static_parameters["rotation"]
            + 15 * static_parameters["t"]
            + 16 * static_parameters["scaling_t"]
            + 17 * static_parameters["velocity"]
            + 100 * static_parameters["f_dc"]
            + 100 * static_parameters["f_rest"]
            + 6 * dynamic_parameter
            + 100 * environment_parameter
            + 7 * joint_screen.sum()
        )
        road = (
            8 * static_parameters["xyz"]
            + 9 * static_parameters["f_dc"]
            + 10 * static_parameters["f_rest"]
            + 100 * dynamic_parameter
            + 100 * environment_parameter
            + 11 * static_screen.sum()
        )
        base = (
            12 * static_parameters["f_dc"]
            + 13 * environment_parameter
            + 14 * dynamic_screen.sum()
        )
        context = object.__new__(V3Context)
        context.gate_only = False
        context.backward(
            base,
            body,
            road,
            static,
            dynamic,
            {
                "viewspace_points": joint_screen,
                "static_viewspace_points": static_screen,
                "dynamic_viewspace_points": dynamic_screen,
            },
            iteration=1,
        )

        self.assertEqual(float(static_parameters["xyz"].grad), 10.0)
        self.assertEqual(float(static_parameters["opacity"].grad), 3.0)
        self.assertEqual(float(static_parameters["scaling"].grad), 4.0)
        self.assertEqual(float(static_parameters["rotation"].grad), 5.0)
        self.assertEqual(float(static_parameters["t"].grad), 15.0)
        self.assertEqual(float(static_parameters["scaling_t"].grad), 16.0)
        self.assertEqual(float(static_parameters["velocity"].grad), 17.0)
        self.assertEqual(float(static_parameters["f_dc"].grad), 21.0)
        self.assertEqual(float(static_parameters["f_rest"].grad), 10.0)
        self.assertEqual(float(dynamic_parameter.grad), 6.0)
        self.assertEqual(float(environment_parameter.grad), 13.0)
        self.assertTrue(torch.equal(joint_screen.grad, torch.full((1, 3), 7.0)))
        self.assertTrue(torch.equal(static_screen.grad, torch.full((1, 3), 11.0)))
        self.assertTrue(torch.equal(dynamic_screen.grad, torch.full((1, 3), 14.0)))


if __name__ == "__main__":
    unittest.main()
