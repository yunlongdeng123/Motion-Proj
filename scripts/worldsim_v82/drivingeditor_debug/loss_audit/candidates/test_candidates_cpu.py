"""Verify candidate invariants and camera integration on CPU; no formal mutations."""
import argparse
import ast
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, source = Path(args.root), Path(args.source_root)
    helper = load(root / "nearest_depth/utils/sparse_depth.py", "candidate_sparse_depth")
    candidate_loss = load(root / "opacity_outside/utils/paper_losses.py", "candidate_opacity")
    original_loss = load(source / "utils/paper_losses.py", "original_losses")
    image_loss = load(source / "utils/loss_utils.py", "frozen_image_losses")
    tests = []

    def check(name, passed, details=None):
        tests.append({"test": name, "passed": bool(passed), "details": details})
        assert passed, name

    uv = np.array([[1, 1], [1, 1], [1, 1], [2, 2]])
    z = np.array([12., 3., 7., 8.])
    expected = helper.nearest_positive_depth(uv, z, 4, 4)
    check("collision_nearest", expected[0, 1, 1] == 3.)
    values = [helper.nearest_positive_depth(uv[list(p)], z[list(p)], 4, 4)
              for p in itertools.permutations(range(4))]
    check("all24_input_permutations_exact_equal", all(np.array_equal(v, expected) for v in values))
    uv_unique = np.array([[1, 1], [2, 2], [0, 3]])
    z_unique = np.array([3., 8., 11.])
    old = np.zeros((1, 4, 4))
    old[0, uv_unique[:, 1], uv_unique[:, 0]] = z_unique
    check("noncollision_identical_to_original", np.array_equal(helper.nearest_positive_depth(uv_unique, z_unique, 4, 4), old))
    invalid_uv = np.array([[1., 1.], [1., 1.], [1., 1.], [1., 1.],
                           [1., 1.], [np.nan, 1.], [-1., 1.], [4., 1.]])
    invalid_z = np.array([5., 0., np.nan, np.inf, -3., 1., 1., 1.])
    invalid_result = helper.nearest_positive_depth(invalid_uv, invalid_z, 4, 4)
    check("zero_nan_inf_behind_camera_and_invalid_uv_ignored",
          np.isfinite(invalid_result).all() and np.count_nonzero(invalid_result) == 1
          and invalid_result[0, 1, 1] == 5.)
    check("empty_returns_allzero", np.array_equal(helper.nearest_positive_depth(np.empty((0, 2)), np.empty(0), 4, 4), np.zeros((1, 4, 4))))

    # Execute the exact patched loadCam function with a CPU-only Camera stub.
    camera_source = (root / "nearest_depth/utils/camera_utils.py").read_text(encoding="utf-8")
    camera_ast = ast.parse(camera_source)
    function = next(n for n in camera_ast.body if isinstance(n, ast.FunctionDef) and n.name == "loadCam")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"np": np, "torch": torch, "CameraInfo": object,
                 "nearest_positive_depth": helper.nearest_positive_depth,
                 "Camera": lambda **kwargs: kwargs}
    exec(compile(module, "candidate_camera_utils.py", "exec"), namespace)
    camera_info = SimpleNamespace(width=8, height=8, cx=4., cy=4., fx=4., fy=4.,
                                  image=np.zeros((8, 8, 3)), dynamic_image=None, sky_mask=None,
                                  pointcloud_camera=np.array([[0., 0., 5.], [0., 0., 10.]]),
                                  image_path="image_0/test.png", R=np.eye(3), T=np.zeros(3),
                                  FovX=-1., FovY=-1., uid=0, image_name="test", timestamp=0.,
                                  pointcloud_front=None, pointcloud_left=None, pointcloud_right=None,
                                  pointcloud_back=None, bbx_info_1=None)
    camera_args = SimpleNamespace(resolution=-1, data_device="cpu")
    current = namespace["loadCam"](camera_args, 0, camera_info, 1.)["pts_depth"]
    camera_info.pointcloud_camera = camera_info.pointcloud_camera[::-1].copy()
    reversed_points = namespace["loadCam"](camera_args, 0, camera_info, 1.)["pts_depth"]
    check("exact_candidate_loadCam_collision_integration",
          current[0, 4, 4].item() == 5. and torch.equal(current, reversed_points)
          and current.device.type == "cpu")
    camera_info.pointcloud_camera = np.array([[0., 0., 0.], [0., 0., -5.],
                                             [0., 0., np.nan], [np.nan, 0., 2.], [0., 0., 5.]])
    invalid_current = namespace["loadCam"](camera_args, 0, camera_info, 1.)["pts_depth"]
    check("exact_candidate_loadCam_zero_nan_behind_integration",
          torch.isfinite(invalid_current).all().item() and torch.count_nonzero(invalid_current).item() == 1
          and invalid_current[0, 4, 4].item() == 5.)

    # Existing foreground Eq13 remains numerically and gradient-identical.
    alpha = torch.tensor([[[.8, .8]]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[[1., 0.]]], dtype=torch.float64)
    foreground = candidate_loss.dynamic_opacity_loss(alpha, mask)
    original = original_loss.dynamic_opacity_loss(alpha, mask)
    fg = torch.autograd.grad(foreground, alpha, retain_graph=True)[0]
    og = torch.autograd.grad(original, alpha, retain_graph=True)[0]
    background = candidate_loss.diagnostic_dynamic_background_opacity_loss(alpha, mask)
    bg = torch.autograd.grad(background, alpha, retain_graph=True)[0]
    check("foreground_eq13_unchanged", torch.equal(foreground, original) and torch.equal(fg, og))
    check("candidate_only_supplies_outside_gradient", bg[0, 0, 0].item() == 0. and bg[0, 0, 1].item() > 0.)
    check("complement_background_gradcheck",
          torch.autograd.gradcheck(lambda a: candidate_loss.diagnostic_dynamic_background_opacity_loss(a, mask),
                                   (alpha,), eps=1e-6, atol=1e-5))
    check("default_zero_coefficient_exact_baseline",
          torch.equal(original + 0. * background, original))
    opaque = torch.tensor([[[1.]]], dtype=torch.float64, requires_grad=True)
    opaque_penalty = candidate_loss.diagnostic_dynamic_background_opacity_loss(opaque, torch.zeros_like(opaque))
    opaque_grad = torch.autograd.grad(opaque_penalty, opaque)[0]
    check("background_term_finite_nonzero_gradient_at_full_opacity",
          torch.isfinite(opaque_penalty).item() and torch.isfinite(opaque_grad).all().item()
          and opaque_grad.item() > 0.)

    # Full implemented objective at two conceptual per-pixel layer allocations.
    def objective(all_background_in_dynamic):
        dtype = torch.float64
        mask = torch.zeros((1, 4, 4), dtype=dtype)
        mask[:, 1:3, 1:3] = 1
        gt = torch.full((3, 4, 4), .5, dtype=dtype)
        alpha_full = .99995
        dynamic_alpha = torch.full_like(mask, alpha_full) if all_background_in_dynamic else mask * alpha_full
        static_alpha = torch.zeros_like(mask) if all_background_in_dynamic else (1 - mask) * alpha_full
        composite_alpha = static_alpha + (1 - static_alpha) * dynamic_alpha
        # Both layers and the shared environment match the observed color.
        composite = composite_alpha * gt + (1 - composite_alpha) * gt
        dynamic_rgb = (dynamic_alpha * gt + (1 - dynamic_alpha) * gt) * mask
        masked_gt = gt * mask
        global_rgb = .8 * F.l1_loss(composite, gt) + .2 * (1 - image_loss.ssim(composite, gt))
        dynamic_rgb_loss = .8 * F.l1_loss(dynamic_rgb, masked_gt) + .2 * (1 - image_loss.ssim(dynamic_rgb, masked_gt))
        velocity_features = torch.zeros((3, 4, 4), dtype=dtype)
        static_vmap = velocity_features / static_alpha.clamp_min(1e-5)
        nps = original_loss.nps_loss(static_vmap, mask)
        sky = original_loss.sky_loss(static_alpha, torch.zeros_like(mask))
        opacity = original_loss.dynamic_opacity_loss(dynamic_alpha, mask)
        # Choose conditional Gaussian depth so the unchanged harmonic mixture matches20.
        conditional_depth = composite_alpha / (1 / 20. - (1 - composite_alpha) / 900.)
        harmonic_depth = 1 / (composite_alpha / conditional_depth + (1 - composite_alpha) / 900.)
        lidar = torch.zeros_like(mask)
        lidar[:, 0, 0] = 20.
        lidar[:, 1, 1] = 20.
        depth = original_loss.sparse_inverse_depth_loss(harmonic_depth, lidar)
        total = original_loss.compose_paper_loss(global_rgb, dynamic_rgb_loss, sky, nps, opacity, depth,
                                                lambda_sky=.05, lambda_nps=.01, lambda_opacity=.05,
                                                lambda_depth=.1, lambda_dynamic_rgb=1.)
        complement = candidate_loss.diagnostic_dynamic_background_opacity_loss(dynamic_alpha, mask)
        return {"allocation": "dynamic_explains_whole_background" if all_background_in_dynamic else "proper_split",
                "components_raw": {"global_rgb": global_rgb.item(), "dynamic_rgb": dynamic_rgb_loss.item(),
                                   "sky": sky.item(), "nps": nps.item(), "foreground_opacity": opacity.item(),
                                   "sparse_inverse_depth": depth.item()},
                "original_total": total.item(), "diagnostic_complement_raw": complement.item(),
                "total_plus_lambda_0_05_complement": (total + .05 * complement).item()}
    proper, collapsed = objective(False), objective(True)
    check("actual_full_objective_layer_degeneracy",
          abs(proper["original_total"] - collapsed["original_total"]) < 1e-14
          and collapsed["components_raw"]["sky"] == 0. and collapsed["components_raw"]["nps"] == 0.)
    check("single_complement_eliminates_counterexample_objective_equality",
          collapsed["total_plus_lambda_0_05_complement"] > proper["total_plus_lambda_0_05_complement"])
    for filename in (root / "opacity_outside/train.py", root / "opacity_outside/utils/paper_losses.py",
                     root / "nearest_depth/utils/camera_utils.py", root / "nearest_depth/utils/sparse_depth.py"):
        ast.parse(filename.read_text(encoding="utf-8"))
    check("all_candidate_python_syntax_valid", True)
    candidate_tree = ast.parse((root / "opacity_outside/train.py").read_text(encoding="utf-8"))
    original_tree = ast.parse((source / "train.py").read_text(encoding="utf-8"))
    def compose_node(tree):
        return next(node for node in ast.walk(tree) if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "compose_paper_loss")
    composed = compose_node(candidate_tree)
    captured = next(node for node in ast.walk(candidate_tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and target.slice.value == "loss_original_objective_before_diagnostic" for target in node.targets))
    augmented = next(node for node in ast.walk(candidate_tree) if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add) and any(isinstance(n, ast.Name) and n.id == "diagnostic_lambda_background" for n in ast.walk(node.value)))
    backward = next(node for node in ast.walk(candidate_tree) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "backward")
    check("exact_original_objective_captured_before_addition_and_backward",
          ast.dump(composed, include_attributes=False) == ast.dump(compose_node(original_tree), include_attributes=False)
          and ast.dump(captured.value, include_attributes=False) == ast.dump(ast.parse("loss.item()", mode="eval").body, include_attributes=False)
          and composed.lineno < captured.lineno < augmented.lineno < backward.lineno)
    payload = {
        "scope": "CPU invariance and exact candidate camera-function integration; no GPU/real model/train/checkpoint/source mutation",
        "torch": torch.__version__, "tests": tests, "all_passed": all(x["passed"] for x in tests),
        "opacity_two_pixel_evidence": {"alpha": alpha.detach().flatten().tolist(), "mask": mask.flatten().tolist(),
                                       "foreground_raw_loss": foreground.item(), "foreground_raw_gradient": fg.flatten().tolist(),
                                       "background_raw_loss": background.item(), "background_raw_gradient": bg.flatten().tolist(),
                                       "lambda_0_05_foreground_gradient": (.05 * fg).flatten().tolist(),
                                       "lambda_0_05_background_gradient": (.05 * bg).flatten().tolist()},
        "full_objective_layer_degeneracy_counterexample": {
            "cases": [proper, collapsed],
            "interpretation": "Conceptual per-pixel alpha/color/depth construction with actual frozen loss/SSIM/compose functions. Confirms an implemented-objective identifiability gap; does not prove realizability by a finite shared3D Gaussian set, the actual optimizer path, or author intent.",
        },
        "limitations": "Nearest z per coarse pixel is an explicit geometric policy; experimental complementary opacity objective is not certified author formula. Neither CPU result proves the cause of the real branch contamination.",
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
