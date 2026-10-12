"""Small CPU numerical audit of unchanged frozen losses and source dataflow."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor(values, grad=False):
    return torch.tensor(values, dtype=torch.float64, device="cpu", requires_grad=grad)


def literal_nps(features, mask, distances):
    result = features.sum() * 0.
    _, height, width = features.shape
    for y in range(height):
        for x in range(width):
            if mask[0, y, x] <= .5 or distances[0, y, x] <= 0:
                continue
            for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width:
                    result = result + torch.linalg.vector_norm(
                        features[:, y, x] - features[:, yy, xx]) / distances[0, y, x]
    return result


def extracted_property(source, cls_name, name):
    parsed = ast.parse(Path(source).read_text())
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    isolated = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {"np": np, "torch": torch}
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.source_root)
    losses = load_module("frozen_paper_losses", root / "utils/paper_losses.py")
    image_losses = load_module("frozen_loss_utils", root / "utils/loss_utils.py")
    torch.set_num_threads(1)
    torch.manual_seed(8412)
    payload = {
        "scope": "CPU double-precision toys on unchanged frozen functions; no full models, GPU, training or checkpoint writes",
        "torch": torch.__version__, "source_root": str(root),
        "source_pins": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [root / "utils/paper_losses.py", root / "utils/loss_utils.py",
                                  root / "train.py", root / "gaussian_renderer/__init__.py",
                                  root / "scene/gaussian_model.py", root / "utils/camera_utils.py",
                                  root / "scene/cameras.py", root / "utils/paper_masks.py"]},
        "checks": {}, "demonstrations": {},
    }

    # Scalar formulas and signs against analytic gradients, in both mask domains.
    alpha = tensor([[[.1, .8, .6, .4]]], True)
    mask = tensor([[[1., 1., 0., 0.]]])
    sky = losses.sky_loss(alpha, mask)
    opacity = losses.dynamic_opacity_loss(alpha, mask)
    sky_grad = torch.autograd.grad(sky, alpha, retain_graph=True)[0]
    opacity_grad = torch.autograd.grad(opacity, alpha)[0]
    sky_reference_grad = mask * (-torch.log1p(-alpha) + alpha / (1 - alpha)) / 4
    opacity_reference_grad = mask * (torch.log(alpha) - (1 - alpha) / alpha) / 4
    payload["checks"]["eq7_eq13_analytic_gradients"] = {
        "sky_value": sky.item(), "sky_grad": sky_grad.flatten().tolist(),
        "opacity_value": opacity.item(), "opacity_grad": opacity_grad.flatten().tolist(),
        "sky_gradient_max_error": (sky_grad - sky_reference_grad).abs().max().item(),
        "opacity_gradient_max_error": (opacity_grad - opacity_reference_grad).abs().max().item(),
        "passed": torch.allclose(sky_grad, sky_reference_grad) and torch.allclose(opacity_grad, opacity_reference_grad),
    }

    # Independent nested-loop Eq9 reference; no implementation-mirroring edge sum.
    nps_rows = []
    masks = {
        "center_rect": tensor([[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0],
                                 [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]]),
        "image_corner": tensor([[[1, 1, 0, 0, 0], [1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
        "checkerboard": tensor([[[0, 1, 0, 1, 0], [1, 0, 1, 0, 1],
                                  [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]]),
        "empty": torch.zeros((1, 4, 5), dtype=torch.float64),
    }
    for name, m in masks.items():
        features = torch.randn((3, 4, 5), dtype=torch.float64, requires_grad=True)
        distances = losses.boundary_distance(m)
        actual = losses.nps_loss(features, m, distances)
        expected = literal_nps(features, m, distances)
        actual_grad = torch.autograd.grad(actual, features, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, features)[0]
        gradcheck = torch.autograd.gradcheck(lambda f: losses.nps_loss(f, m, distances),
                                            (features,), eps=1e-6, atol=1e-5, rtol=1e-4)
        nps_rows.append({"name": name, "actual": actual.item(), "literal_reference": expected.item(),
                         "value_error": abs(actual.item() - expected.item()),
                         "gradient_max_error": (actual_grad - expected_grad).abs().max().item(),
                         "gradcheck": bool(gradcheck), "passed": torch.allclose(actual, expected)
                         and torch.allclose(actual_grad, expected_grad) and gradcheck})
    payload["checks"]["eq9_four_axial_neighbors_full_source_sum"] = nps_rows
    full_mask = torch.ones((1, 4, 5), dtype=torch.float64)
    full_feat = torch.randn((3, 4, 5), dtype=torch.float64, requires_grad=True)
    full_value = losses.nps_loss(full_feat, full_mask)
    payload["demonstrations"]["nps_full_image_mask_policy"] = {
        "loss": full_value.item(), "gradient_max": torch.autograd.grad(full_value, full_feat)[0].abs().max().item(),
        "interpretation": "No static boundary exists; code explicitly returns zero. Paper does not define this edge case.",
    }

    # Uniform nonzero velocity is an exact null direction of Eq9.
    uniform = tensor([1.5, -2., .7], True).reshape(3, 1, 1).expand(3, 4, 5)
    uniform_value = losses.nps_loss(uniform, masks["center_rect"])
    uniform_grad = torch.autograd.grad(uniform_value, uniform)[0]
    payload["demonstrations"]["nps_uniform_nonzero_velocity_nullspace"] = {
        "velocity_vector": [1.5, -2., .7], "loss": uniform_value.item(),
        "gradient_max": uniform_grad.abs().max().item(),
        "interpretation": "Eq9 is local consistency, not an absolute zero-velocity or static-RGB anchor. This is a property of the printed loss, not a missing mandatory paper term.",
    }

    # NPS feature opacity normalization cancels scale for a one-Gaussian ray.
    alpha_map = torch.full((1, 4, 5), .2, dtype=torch.float64, requires_grad=True)
    velocity = torch.randn((3, 4, 5), dtype=torch.float64, requires_grad=True)
    normed = alpha_map * velocity / alpha_map.clamp_min(1e-5)
    normed_loss = losses.nps_loss(normed, masks["center_rect"])
    ga, gv = torch.autograd.grad(normed_loss, (alpha_map, velocity))
    payload["demonstrations"]["nps_single_gaussian_ray_alpha_scale_invariance"] = {
        "alpha_gradient_max": ga.abs().max().item(), "velocity_gradient_l1": gv.abs().sum().item(),
        "interpretation": "With alpha above its floor, NPS regularizes expected velocity and provides no independent opacity support constraint in this toy.",
    }

    # The exact frozen damped velocity property permits beta to suppress v_map.
    damped_velocity = extracted_property(root / "scene/gaussian_model.py", "GaussianModel", "get_inst_velocity")
    class Minimal:
        pass
    dummy = Minimal()
    dummy._velocity = tensor([[1., 0., 0.], [0., 0., 0.]], True)
    dummy.get_scaling_t = tensor([[.2], [.2]], True)
    dummy.T, dummy.velocity_decay = .2, 1.
    feature = damped_velocity(dummy).T.reshape(3, 1, 2)
    m = tensor([[[1., 0.]]])
    beta_loss = losses.nps_loss(feature, m)
    beta_grad, velocity_grad = torch.autograd.grad(beta_loss, (dummy.get_scaling_t, dummy._velocity))
    payload["demonstrations"]["nps_temporal_scale_damping_direction"] = {
        "loss": beta_loss.item(), "beta_gradient": beta_grad.flatten().tolist(),
        "raw_velocity_gradient": velocity_grad.tolist(),
        "interpretation": "Negative beta gradient permits increasing temporal scale to reduce the damped NPS feature. Other objectives may counteract this; this test does not establish the real trajectory.",
    }

    # Sum normalization and image resolution scaling with independent tile copies.
    scaling = []
    base_m = torch.zeros((1, 8, 8), dtype=torch.float64)
    base_m[:, 2:6, 2:6] = 1
    base_f = torch.zeros((3, 8, 8), dtype=torch.float64)
    base_f[0, 3:5, 3:5] = 1
    for repeats in (1, 2, 4):
        fm, mm = base_f.repeat(1, repeats, repeats), base_m.repeat(1, repeats, repeats)
        value = losses.nps_loss(fm, mm).item()
        scaling.append({"shape": list(mm.shape), "independent_patch_count": repeats ** 2,
                        "nps_sum": value, "lambda_nps_times_sum": .01 * value,
                        "nps_per_full_image_pixel": value / mm.numel()})
    payload["demonstrations"]["nps_sum_scale"] = {
        "rows": scaling,
        "interpretation": "The printed Eq9 sum is implemented; its coefficient cannot be compared to mean RGB losses without scene/resolution scale. Unpublished lambda is not independently recovered.",
    }

    # Inverse depth: independent valid sparse residual and meaningful gradient.
    rendered = tensor([[[2., 7.], [4., 99.]]], True)
    lidar = tensor([[[4., 0.], [2., float('nan')]]])
    actual_depth = losses.sparse_inverse_depth_loss(rendered, lidar)
    reference_depth = ((1 / (4 + 1e-5) - 1 / (rendered[0, 0, 0] + 1e-5)).abs()
                       + (1 / (2 + 1e-5) - 1 / (rendered[0, 1, 0] + 1e-5)).abs()) / 4
    gd = torch.autograd.grad(actual_depth, rendered)[0]
    payload["checks"]["eq16_sparse_inverse_depth"] = {
        "value": actual_depth.item(), "reference": reference_depth.item(), "gradient": gd.tolist(),
        "passed": torch.allclose(actual_depth, reference_depth) and torch.isfinite(gd).all().item()
        and gd[0, 0, 1].item() == 0 and gd[0, 1, 1].item() == 0,
    }

    # Exact Eq6 dynamic RGB path on a small image, using real frozen SSIM.
    rgb_cases = []
    for env_value, gaussian_value in ((.5, .5), (.5, .75), (.75, .5)):
        a = torch.full((1, 12, 12), .25, dtype=torch.float64, requires_grad=True)
        env = torch.full((3, 12, 12), env_value, dtype=torch.float64, requires_grad=True)
        color = torch.full((3, 12, 12), gaussian_value, dtype=torch.float64, requires_grad=True)
        target = torch.full((3, 12, 12), .5, dtype=torch.float64)
        mask = torch.zeros((1, 12, 12), dtype=torch.float64)
        mask[:, 4:8, 4:8] = 1
        rendered = (a * color + (1 - a) * env) * mask
        target = target * mask
        rgb = .8 * F.l1_loss(rendered, target) + .2 * (1 - image_losses.ssim(rendered, target))
        soft_opacity = .05 * losses.dynamic_opacity_loss(a, mask)
        rgb_grads = torch.autograd.grad(rgb, (a, env, color), retain_graph=True)
        opacity_grad = torch.autograd.grad(soft_opacity, a)[0]
        rgb_cases.append({"env": env_value, "gaussian_rgb": gaussian_value, "target": .5,
                          "alpha": .25, "mask_fraction": mask.mean().item(), "rgb_loss": rgb.item(),
                          "rgb_alpha_gradient_inside_mean": rgb_grads[0][mask.bool()].mean().item(),
                          "rgb_alpha_gradient_outside_max": rgb_grads[0][~mask.bool()].abs().max().item(),
                          "rgb_env_gradient_l1": rgb_grads[1].abs().sum().item(),
                          "rgb_gaussian_gradient_l1": rgb_grads[2].abs().sum().item(),
                          "weighted_opacity_gradient_inside_mean": opacity_grad[mask.bool()].mean().item()})
    payload["demonstrations"]["dynamic_eq6_shared_env_rgb_gradients"] = {
        "rows": rgb_cases,
        "interpretation": "Masked dynamic RGB has exactly zero outside gradients. When env and Gaussian both match target, RGB is independent of alpha; only masked soft opacity encourages coverage. Shared EnvMap is explicit paper behavior.",
    }

    # Shared two-branch foreground/background assignment is underdetermined outside mask.
    layer_rows = []
    for static_alpha, dynamic_alpha in ((.9, .05), (.05, .9)):
        total_alpha = static_alpha + (1 - static_alpha) * dynamic_alpha
        lidar_z = 20.
        # Choose a conditional Gaussian depth that exactly matches the existing harmonic depth.
        gaussian_z = total_alpha / (1 / lidar_z - (1 - total_alpha) / 900.)
        depth_blended = 1 / (total_alpha / gaussian_z + (1 - total_alpha) / 900.)
        alpha_d = tensor([[[dynamic_alpha]]], True)
        outside = tensor([[[0.]]])
        opacity = losses.dynamic_opacity_loss(alpha_d, outside)
        outside_grad = torch.autograd.grad(opacity, alpha_d)[0].item()
        layer_rows.append({"static_alpha": static_alpha, "dynamic_alpha": dynamic_alpha,
                           "composite_alpha": total_alpha, "composite_rgb": .4,
                           "masked_dynamic_rgb_loss": 0., "nps_with_uniform_zero_velocity": 0.,
                           "sky_loss_non_sky_pixel": 0., "dynamic_opacity_loss": opacity.item(),
                           "dynamic_opacity_gradient": outside_grad,
                           "gaussian_conditional_depth": gaussian_z,
                           "harmonic_blended_depth": depth_blended, "lidar_depth": lidar_z})
    payload["demonstrations"]["outside_mask_layer_assignment_nullspace"] = {
        "rows": layer_rows,
        "interpretation": "Two opposite branch allocations can satisfy all currently active scalar losses on this artificial outside-mask pixel. This is a falsifiable identifiability concern, not proof of what the author intended or a confirmed cause in the real run.",
    }

    # Collision target counterexample to camera_utils.py last-write indexing.
    collision_rows = []
    for depths in ((5., 10.), (10., 5.)):
        sparse = np.zeros((1, 1, 1))
        uv = np.array([[0, 0], [0, 0]])
        sparse[0, uv[:, 1], uv[:, 0]] = np.array(depths)
        r = tensor([[[7.]]], True)
        target = tensor(sparse)
        loss = losses.sparse_inverse_depth_loss(r, target)
        gradient = torch.autograd.grad(loss, r)[0].item()
        collision_rows.append({"input_z_order": list(depths), "current_target_z": sparse.item(),
                               "nearest_z_target": min(depths), "rendered_z": 7.,
                               "depth_loss": loss.item(), "rendered_depth_gradient": gradient})
    payload["demonstrations"]["sparse_depth_duplicate_pixel_order_bug"] = {
        "rows": collision_rows,
        "confirmed_implementation_behavior": "Identical projected point set produces different sparse depth and opposite descent direction when point order reverses.",
        "paper_reference": "Eq16, PDF pages7-8 (printed1702-1703), sparse LiDAR depth sampled against rendered visible depth",
        "source_reference": "utils/camera_utils.py117-133; nearest-visibility selection absent at133",
        "causal_attribution": "Not established as cause of branch collapse; assess collision count and z spread separately.",
    }

    # Harmonic 900-depth behavior carried into paper branch, explicitly record rather than misreport.
    harmonic_rows = []
    for alpha_value in (.1, .5, .9, 1.):
        a = tensor(alpha_value, True)
        conditional_z = tensor(20., True)
        blended = 1 / (a / conditional_z.clamp_min(1e-5) + (1 - a) / 900.).clamp_min(1e-5)
        residual = (1 / (blended + 1e-5) - 1 / (tensor(20.) + 1e-5)).abs()
        grads = torch.autograd.grad(residual, (a, conditional_z))
        harmonic_rows.append({"alpha": alpha_value, "normalized_gaussian_depth": 20.,
                              "depth_after_harmonic_sky_blend": blended.item(),
                              "inverse_depth_residual": residual.item(),
                              "alpha_gradient": grads[0].item(), "depth_gradient": grads[1].item()})
    payload["demonstrations"]["harmonic_sky_depth_adds_opacity_gradient"] = {
        "rows": harmonic_rows,
        "source_reference": "train.py360-373 before paper-depth at399; frozen config depth_blend_mode=0",
        "interpretation": "Opacity-normalized Gaussian depth is already correct. The additional900sky harmonic mixture makes LiDAR loss constrain total opacity too; exact author depth convention is undisclosed.",
    }
    xyz_shm = extracted_property(root / "scene/gaussian_model.py", "GaussianModel", "get_xyz_SHM")
    marginal_t = extracted_property(root / "scene/gaussian_model.py", "GaussianModel", "get_marginal_t")
    static = Minimal()
    static._xyz, static._velocity, static._t = tensor([[0., 0., 0.]]), tensor([[10., 0., 0.]]), tensor([[0.]])
    static.get_scaling_t, static.get_t = tensor([[10.]]), static._t
    static.T, static.velocity_decay = .2, 1.
    static_points = []
    for timestamp in (-.48, -.05, 0., .05, .48):
        static_points.append({"timestamp": timestamp, "xyz": xyz_shm(static, timestamp).flatten().tolist(),
                              "marginal_t": marginal_t(static, timestamp).item()})
    payload["demonstrations"]["static_vmap_damping_does_not_imply_time_invariant_geometry"] = {
        "raw_velocity": static._velocity.flatten().tolist(), "beta": 10., "cycle": .2,
        "reported_damped_velocity": damped_velocity(static).flatten().tolist(),
        "maximum_periodic_position_amplitude_model_units": 10. * .2 / (2 * np.pi),
        "timestamp_samples": static_points,
        "source_reference": "static renderer prepare_rasterization else branch259-273; GaussianModel147-178; train.py300,333-335,394",
        "paper_reference": "PDF5/printed1700 Eq4 says time-invariant G_s, while PDF4/printed1699 Eq1-3 gives shared4D base parameterization",
        "interpretation": "A static feature can be nearly zero because beta is large even while the exact SHM positions move substantially. Demonstrates a real semantic discrepancy; actual checkpoint beta/raw-velocity/displacement distribution remains to be measured.",
    }
    all_checks = [payload["checks"]["eq7_eq13_analytic_gradients"]["passed"],
                  payload["checks"]["eq16_sparse_inverse_depth"]["passed"]]
    all_checks.extend(row["passed"] for row in nps_rows)
    payload["all_formula_checks_passed"] = bool(all(all_checks))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({"all_formula_checks_passed": payload["all_formula_checks_passed"],
                      "source_root": str(root), "output": str(out), "torch": torch.__version__}))
    for name, values in payload["demonstrations"].items():
        print(name, json.dumps(values, ensure_ascii=False))
    assert payload["all_formula_checks_passed"]


if __name__ == "__main__":
    main()
