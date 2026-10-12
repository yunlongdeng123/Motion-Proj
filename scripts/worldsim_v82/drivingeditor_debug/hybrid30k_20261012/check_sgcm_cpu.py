"""真实 source methods 的 CPU SGCM 回归；不加载模型 checkpoint 或 CUDA。

python check_sgcm_cpu.py --source <local-small-source-snapshot>
可选 --report <work-json> 保存结果；可选 --region-views <small-json>
测量 Hd 投影与实际 mask 的交集，不把二维外包络当精确三维遮挡区域。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import ast
import copy
import importlib.util
import itertools
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from torch import nn

from sgcm_patch import patch_sgcm_source, patch_gaussian_source
import sgcm_helpers as helper

torch.set_num_threads(1)

FIELDS = ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
          "_rotation", "_t", "_scaling_t", "_velocity"]
GROUPS = ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation", "t", "scaling_t", "velocity"]
STATISTICS = ["max_radii2D", "xyz_gradient_accum", "t_gradient_accum", "denom"]


class TorchCPUAdapter:
    def __getattr__(self, name):
        return getattr(torch, name)

    def zeros(self, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        kwargs.setdefault("dtype", torch.float64)
        return torch.zeros(*args, **kwargs)


def paper_module(source, name):
    module = ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, name, "exec"), module.__dict__)
    return module


def model_harness(source):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GaussianModel")
    method_names = ["capture", "restore", "_prune_optimizer", "prune_points",
                    "cat_tensors_to_optimizer", "densification_postfix", "densify_and_clone", "densify_and_split"]
    namespace = {"torch": TorchCPUAdapter(), "nn": nn, "np": np,
                 "lineage_after_append": helper.lineage_after_append,
                 "lineage_after_prune": helper.lineage_after_prune,
                 "capture_sgcm_optimizer": helper.capture_sgcm_optimizer,
                 "restore_sgcm_optimizer": helper.restore_sgcm_optimizer,
                 "build_rotation": lambda quat: torch.eye(3, dtype=quat.dtype).repeat(len(quat), 1, 1)}
    lines = source.splitlines()
    for name in method_names:
        node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
        body = textwrap.dedent("\n".join(lines[node.lineno-1:node.end_lineno]))
        exec(compile(body, "GaussianModel."+name, "exec"), namespace)

    class Harness:
        get_xyz = property(lambda self: self._xyz)
        get_scaling = property(lambda self: torch.exp(self._scaling))
        get_opacity = property(lambda self: torch.sigmoid(self._opacity))
        get_t = property(lambda self: self._t)
        get_scaling_t = property(lambda self: torch.exp(self._scaling_t))
        get_inst_velocity = property(lambda self: self._velocity * torch.exp(-self.get_scaling_t/self.T/2*self.velocity_decay))
        scaling_inverse_activation = staticmethod(torch.log)
        scaling_t_inverse_activation = staticmethod(torch.log)

        def setup_functions(self):
            pass

        def training_setup(self, unused):
            self.optimizer = torch.optim.Adam([{"params": [getattr(self, field)], "name": name, "lr": .001}
                                               for field, name in zip(FIELDS, GROUPS)])
            for name in STATISTICS:
                if name == "max_radii2D" and hasattr(self, name):
                    continue
                value = torch.zeros(len(self.get_xyz)) if name == "max_radii2D" else torch.zeros(len(self.get_xyz), 1)
                setattr(self, name, value)

    for name in method_names:
        setattr(Harness, name, namespace[name])
    return Harness


def make_model(harness, xyz=None, times=None, sigmas=None, opacity=None, scales=None):
    xyz = np.asarray([[0, 0, .3], [10, 0, .1]] if xyz is None else xyz, dtype=np.float64)
    count = len(xyz)
    times = np.zeros(count) if times is None else np.asarray(times)
    sigmas = np.full(count, .31) if sigmas is None else np.asarray(sigmas)
    opacity = np.full(count, .1) if opacity is None else np.asarray(opacity)
    scales = np.tile([.1, .2, .3], (count, 1)) if scales is None else np.asarray(scales)
    values = [xyz, np.arange(count).reshape(count, 1, 1)+10,
              np.arange(count*6).reshape(count, 2, 3), np.log(opacity/(1-opacity))[:, None],
              np.log(scales), np.tile([1., 0, 0, 0], (count, 1)), times[:, None],
              np.log(sigmas)[:, None], np.zeros((count, 3))]
    model = harness()
    for field, value in zip(FIELDS, values):
        setattr(model, field, nn.Parameter(torch.tensor(value, dtype=torch.float64)))
    model.training_setup(None)
    # 有 Adam moments 才能检查 append/prune 真正保留了 optimizer 行。
    for field in FIELDS:
        getattr(model, field).grad = torch.full_like(getattr(model, field), .125)
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for field, value in zip(FIELDS, values):
            getattr(model, field).copy_(torch.tensor(value, dtype=torch.float64))
    for name in STATISTICS:
        values = torch.arange(count, dtype=torch.float64)+1
        setattr(model, name, values if name == "max_radii2D" else values[:, None])
    model.active_sh_degree, model.spatial_lr_scale = 0, 1.0
    model.T, model.velocity_decay, model.time_duration = 1.0, .5, [0., 1.]
    model.percent_dense, model.contract = 1.0, False
    model.no_time_split = True
    return model


def make_view(timestamp=0., x_shift=0., depth_shift=3., name="fixture"):
    worldview = np.eye(4)
    worldview[3, 0], worldview[3, 2] = x_shift, depth_shift
    # 夹具使用正交投影，以明确控制中心图像边界；真实检查使用 Camera.full_proj。
    return SimpleNamespace(timestamp=timestamp, world_view_transform=torch.tensor(worldview),
                           full_proj_transform=torch.tensor(worldview), image_width=64, image_height=48,
                           image_path=name)


def controller(module, model=None, guard=True, cameras=None, plane=None, bounds=None, scale=1.0, affine=None):
    values = dict(start=0, end=1000, interval=100, offset=0, max_new_points=10000)
    if hasattr(module.SGCMConfig(), "guard_enabled"):
        values["guard_enabled"] = guard
    result = module.SGCMController(module.SGCMConfig(**values), scale,
                                   np.asarray([0., 0, 1, 0] if plane is None else plane),
                                   np.eye(4) if affine is None else affine,
                                   [np.asarray([[-1., -1., -1.], [1., 1., 2.]] if bounds is None else bounds)], {})
    if model is not None and guard:
        helper.bind_sgcm_guard(result, [make_view()] if cameras is None else cameras, model)
    return result


def append_from(model, ids):
    parents = torch.tensor(ids, dtype=torch.int64)
    children = [getattr(model, field)[parents].detach().clone() for field in FIELDS]
    model.densification_postfix(*children, sgcm_parent_indices=parents)


def assert_model_equal(first, second):
    for name in FIELDS+STATISTICS:
        assert torch.equal(getattr(first, name), getattr(second, name)), name
    assert first.optimizer.state_dict()["param_groups"] == second.optimizer.state_dict()["param_groups"]
    for left, right in zip(first.optimizer.param_groups, second.optimizer.param_groups):
        state_left = first.optimizer.state[left["params"][0]]
        state_right = second.optimizer.state[right["params"][0]]
        assert set(state_left) == set(state_right)
        for key in state_left:
            assert torch.equal(state_left[key], state_right[key]), key


def regressions(source):
    sgcm_text = (source/"scene/paper_sgcm.py").read_text(encoding="utf-8")
    gaussian_text = (source/"scene/gaussian_model.py").read_text(encoding="utf-8")
    package = ModuleType("scene")
    package.__path__ = []
    sys.modules.setdefault("scene", package)
    sys.modules["scene.hybrid_sgcm_helpers"] = helper
    original = paper_module(sgcm_text, "sgcm_original_cpu")
    corrected = paper_module(patch_sgcm_source(sgcm_text), "sgcm_corrected_cpu")
    original_harness = model_harness(gaussian_text)
    harness = model_harness(patch_gaussian_source(gaussian_text))
    passed = []

    for raw in [sgcm_text, sgcm_text.replace("\n", "\r\n")]:
        assert patch_sgcm_source(raw, enabled=False) == raw
    for raw in [gaussian_text, gaussian_text.replace("\n", "\r\n")]:
        assert patch_gaussian_source(raw, enabled=False) == raw
    passed.append("source_patch_disabled_exact_text_and_newlines")

    baseline, replay = make_model(original_harness), make_model(harness)
    ca, cb = controller(original, guard=False), controller(corrected, guard=False)
    for iteration in [100, 200, 300]:
        radius = torch.full((len(baseline.get_xyz),), 30., dtype=torch.float64)
        assert ca.refine(baseline, radius, iteration) == cb.refine(replay, radius, iteration)
        assert_model_equal(baseline, replay)
    assert len(baseline.capture()) == len(replay.capture()) == 18
    assert helper.LINEAGE_KEY not in replay.capture()[14]
    passed.append("runtime_guard_disabled_parameters_statistics_adam_and_legacy_capture_match")

    hidden = make_model(harness, times=[10, 0], sigmas=[.01, .31])
    ch = controller(corrected, hidden, cameras=[make_view(0), make_view(1)])
    result = ch.refine(hidden, torch.tensor([50., 0.]), 100)
    assert result["cloned"] == 0 and result["excluded_clone_training_view_support"] == 1
    passed.append("temporal_prefilter_hidden_parent_not_cloned_despite_positive_fixture_radius")

    under_alpha = make_model(harness, opacity=[.001, .1])
    result = controller(corrected, under_alpha).refine(under_alpha, torch.tensor([50., 0.]), 100)
    assert result["cloned"] == 0
    passed.append("effective_opacity_under_kernel_cutoff_not_cloned")

    outside = make_model(harness)
    result = controller(corrected, outside, cameras=[make_view(x_shift=100)]).refine(outside, torch.tensor([0., 0.]), 100)
    assert result["cloned"] == 0 and len(outside.get_xyz) == 2
    passed.append("no_training_view_support_not_cloned")

    elsewhere = make_model(harness, times=[1, 0], sigmas=[.1, .31])
    cc = controller(corrected, elsewhere, cameras=[make_view(0, x_shift=100), make_view(1)])
    result = cc.refine(elsewhere, torch.tensor([0., 0.]), 100)
    assert result["cloned"] == 1 and len(elsewhere.get_xyz) == 3
    passed.append("current_view_hidden_but_other_actual_training_timestamp_supported_clone_allowed")

    live = make_model(harness)
    cl = controller(corrected, live)
    counts = [len(live.get_xyz)]
    for iteration in [100, 200, 300, 400]:
        cl.refine(live, torch.full((len(live.get_xyz),), 30.), iteration)
        counts.append(len(live.get_xyz))
    assert counts == [2, 3, 3, 3, 3]
    with torch.no_grad():
        live._xyz[0, 0] += .1
        live._xyz[-1, 0] -= .1
    assert cl.refine(live, torch.full((len(live.get_xyz),), 30.), 500)["cloned"] == 0
    passed.append("clone_lineage_once_including_moved_clone_descendants")

    siblings = make_model(harness)
    cs = controller(corrected, siblings)
    append_from(siblings, [0])
    result = cs.refine(siblings, torch.tensor([30., 0., 40.]), 100)
    assert result["cloned"] == 1 and result["excluded_duplicate_lineage_in_step"] == 1
    assert torch.equal(siblings._sgcm_clone_blocked, torch.tensor([True, False, True, True]))
    assert cs.refine(siblings, torch.full((4,), 30.), 200)["cloned"] == 0
    passed.append("unspent_generic_siblings_share_one_source_clone_budget")

    # 运行实际 generic clone 方法，不以测试手写复制代替 source 行为。
    generic = make_model(harness)
    cg = controller(corrected, generic)
    cg.refine(generic, torch.tensor([30., 0.]), 100)
    generic.densify_and_clone(torch.ones(3, 1), .5, 1., torch.zeros(3, 1), .5)
    assert len(generic.get_xyz) == 6
    assert torch.equal(generic._sgcm_lineage_ids, torch.tensor([0, 1, 0, 0, 1, 0]))
    assert cg.refine(generic, torch.full((6,), 30.), 200)["cloned"] == 0
    passed.append("real_generic_clone_inherits_spent_lineage")

    generic_split = make_model(harness)
    cgs = controller(corrected, generic_split)
    cgs.refine(generic_split, torch.tensor([30., 0.]), 100)
    generic_split.percent_dense = .05
    generic_split.densify_and_split(torch.ones(3, 1), .5, 1., torch.zeros(3, 1), .5)
    assert len(generic_split.get_xyz) == 6
    assert torch.equal(generic_split._sgcm_lineage_ids, torch.tensor([0, 1, 0, 0, 1, 0]))
    assert torch.equal(generic_split._sgcm_clone_blocked, torch.tensor([True, False, True, True, False, True]))
    passed.append("real_generic_spatial_temporal_split_inherits_spent_lineage")

    edge = make_model(harness, xyz=[[.9, 0, .1], [10, 0, .1]], scales=[[.4, .1, .1], [.1, .2, .3]])
    ce = controller(corrected, edge)
    result = ce.refine(edge, torch.tensor([30., 0.]), 100)
    assert result["split_parents"] == 0 and result["excluded_split_child_training_view_support"] == 1
    assert len(edge.get_xyz) == 2
    passed.append("split_rejected_if_one_proposed_child_has_no_training_view_support")

    split = make_model(harness, xyz=[[0, 0, .1], [10, 0, .1]])
    cs = controller(corrected, split)
    proposed = helper.split_child_geometry(split, torch.tensor([0]))
    old = split.get_xyz.detach().clone()
    result = cs.refine(split, torch.tensor([30., 0.]), 100)
    assert result["split_parents"] == 1 and torch.equal(split.get_xyz[0], old[1])
    assert torch.allclose(split.get_xyz[1:], proposed, rtol=0, atol=1e-15)
    assert torch.equal(split._sgcm_lineage_ids, torch.tensor([1, 0, 0]))
    passed.append("eligible_split_exact_original_child_geometry_and_parent_lineage")

    prune = make_model(harness, xyz=[[0, 0, .8], [0, 0, .3], [0, 0, .1], [10, 0, .1]])
    cp = controller(corrected, prune)
    old_fields = {field: getattr(prune, field).detach().clone() for field in FIELDS}
    old_stats = {field: getattr(prune, field).clone() for field in STATISTICS}
    old_adam = copy.deepcopy(prune.optimizer.state_dict())
    result = cp.refine(prune, torch.tensor([0., 30., 30., 0.]), 100)
    # prune row0, remove split row2, surviving row1/3, clone1, split2/2.
    assert result["pruned"] == 1 and result["cloned"] == 1 and result["split_parents"] == 1
    assert torch.equal(prune._sgcm_lineage_ids, torch.tensor([1, 3, 1, 2, 2]))
    assert torch.equal(prune._sgcm_clone_blocked, torch.tensor([True, False, True, False, False]))
    for field in FIELDS:
        assert torch.equal(getattr(prune, field)[:2], old_fields[field][[1, 3]])
    for field in STATISTICS:
        assert torch.equal(getattr(prune, field)[:2], old_stats[field][[1, 3]])
        assert torch.count_nonzero(getattr(prune, field)[2:]) == 0
    for group_index, group in enumerate(prune.optimizer.param_groups):
        current = prune.optimizer.state[group["params"][0]]
        previous = old_adam["state"][group_index]
        assert current["exp_avg"].shape == group["params"][0].shape
        assert torch.equal(current["exp_avg"][:2], previous["exp_avg"][[1, 3]])
        assert torch.count_nonzero(current["exp_avg"][2:]) == 0
    passed.append("prune_clone_split_fields_statistics_optimizer_and_lineage_rows_aligned")

    payload = live.capture()
    assert len(payload) == 18 and helper.LINEAGE_KEY in payload[14]
    restored = make_model(harness)
    restored.restore(payload, SimpleNamespace())
    assert torch.equal(restored._sgcm_lineage_ids, live._sgcm_lineage_ids)
    assert torch.equal(restored._sgcm_clone_blocked, live._sgcm_clone_blocked)
    cr = controller(corrected)
    helper.bind_sgcm_guard(cr, [make_view()], restored, resuming=True)
    assert cr.refine(restored, torch.full((len(restored.get_xyz),), 30.), 600)["cloned"] == 0
    assert_model_equal(live, restored)
    passed.append("legacy_18_field_checkpoint_restores_lineage_and_prevents_budget_reset")

    legacy = make_model(harness)
    try:
        helper.enable_lineage(legacy, resuming=True)
    except ValueError as error:
        assert "requires saved" in str(error)
    else:
        raise AssertionError("Missing resume lineage must be rejected")
    malformed = dict(payload[14])
    malformed[helper.LINEAGE_KEY] = dict(malformed[helper.LINEAGE_KEY], lineage_ids=torch.tensor([0]))
    try:
        helper.restore_sgcm_optimizer(restored, malformed)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Misaligned checkpoint lineage must be rejected")
    passed.append("resume_missing_or_misaligned_lineage_fails_closed")

    bad_filter = make_model(harness)
    cf = controller(corrected, bad_filter)
    try:
        cf.refine(bad_filter, torch.tensor([30., 0.]), 100, visibility_filter=torch.tensor([False, True]))
    except RuntimeError as error:
        assert "aligned" in str(error)
    else:
        raise AssertionError("Misaligned current renderer filter must be rejected")
    passed.append("current_renderer_filter_must_match_static_radius_rows")

    metric_masks = []
    world = np.array([[0, 0, .249], [0, 0, .251], [0, 0, .499], [0, 0, .501], [2, 0, .501]])
    for scale in [1., .1]:
        transform = np.eye(4)
        transform[:3, :3] *= scale
        transform[:3, 3] = [3, -2, 7]
        plane = corrected.transform_plane([0, 0, 1, 0], transform)
        cm = controller(corrected, guard=False, scale=scale, plane=plane, affine=np.linalg.inv(transform))
        classification = cm.classify_numpy(corrected.transform_points(world, transform), np.array([30., 30., 30., 30., 30.]))
        metric_masks.append({name: classification[name].tolist() for name in ["prune", "clone", "split", "inside"]})
    assert metric_masks[0] == metric_masks[1]
    assert metric_masks[0] == {"prune": [False, False, False, True, False],
                               "clone": [False, True, True, False, False],
                               "split": [True, False, False, False, False],
                               "inside": [True, True, True, True, False]}
    passed.append("meter_model_scale_point1_and_hd_mask_keep_exact_eq8_regions")

    # 实际 source 语义核对，避免误把 per-pixel contrib 当成逐高斯向量。
    kernel = (source/"rasterize_points.cu").read_text(encoding="utf-8")
    forward = (source/"rasterizer_forward.cu").read_text(encoding="utf-8")
    renderer = (source/"renderer.py").read_text(encoding="utf-8")
    assert "out_contrib = torch::full({1, H, W}" in kernel
    assert "n_contrib[pix_id] = last_contributor" in forward
    assert "alpha < 1.0f / 255.0f" in forward
    assert "mask = marginal_t[:, 0] > 0.05" in renderer
    assert '"contrib": contrib' in renderer
    auxiliary = (source/"rasterizer_auxiliary.h").read_text(encoding="utf-8")
    assert "if (p_view.z <= 0.2f)" in auxiliary
    passed.append("actual_renderer_contrib_is_per_pixel_and_temporal_alpha_cutoffs_verified")

    return {"tests_passed": len(passed), "cases": passed, "eligible_clone_point_counts": counts,
            "guard_disabled_matches_upstream": True, "checkpoint_capture_tuple_length": 18,
            "contribution_semantics": "per-pixel last contributor; not used as per-Gaussian eligibility",
            "guard_semantics": "actual training-view center/SHM/temporal-alpha support proxy; occlusion unmeasured",
            "gpu_used": False, "cuda_initialized": torch.cuda.is_initialized()}, corrected


def actual_geometry_measurement(source, module):
    metadata = json.loads((source/"sgcm_runtime_geometry.json").read_text(encoding="utf-8"))
    scale = module.validate_similarity(metadata["world_to_model"])
    assert np.isclose(scale, .1, atol=1e-12)
    assert np.isclose(metadata["dr_model"], .05, atol=1e-12)
    plane = np.asarray(metadata["ground_fit"]["plane_anchor"])
    regions = np.asarray(metadata["regions_anchor"])
    measurements = []
    for bounds in regions:
        corners = np.asarray(list(itertools.product(bounds[:, 0], bounds[:, 1])))
        ground = -(corners@plane[:2]+plane[3])/plane[2]
        assert bounds[0, 2] <= ground.min() and bounds[1, 2] >= ground.max()
        extent = bounds[1]-bounds[0]
        measurements.append({"extent_anchor_m": extent.tolist(), "xy_area_m2": float(extent[0]*extent[1]),
                             "volume_m3": float(np.prod(extent)), "ground_corner_z_m": ground.tolist(),
                             "region_lower_z_m": float(bounds[0, 2]), "region_upper_z_m": float(bounds[1, 2])})
    fit = metadata["ground_fit"]
    assert fit["inlier_cells"] >= metadata["config"]["ground_min_cells"]
    assert fit["rmse_m"] <= metadata["config"]["ground_max_rmse_m"]
    return {"source": str(source/"sgcm_runtime_geometry.json"), "meter_to_model_scale": scale,
            "dr_m": metadata["config"]["dr_m"], "dr_model": metadata["dr_model"],
            "ground_rmse_m": fit["rmse_m"], "ground_inlier_cells": fit["inlier_cells"],
            "regions": measurements, "geometry_scope": "existing validated time-aggregated LiDAR AABB; no checkpoint loaded"}


def region_mask_measurement(view_manifest, metadata):
    """小档提供 model_to_anchor/实际 camera 小矩阵与 mask path；无 GT 3D boxes。"""
    from PIL import Image, ImageDraw
    records = view_manifest if isinstance(view_manifest, dict) else json.loads(Path(view_manifest).read_text(encoding="utf-8"))
    anchor_to_model = np.linalg.inv(np.asarray(records["model_to_anchor"]))
    regions = np.asarray(metadata["regions_anchor"])
    rows = []
    for record in records["views"]:
        width, height = int(record["width"]), int(record["height"])
        original_mask = Image.open(record["mask_path"]).convert("L")
        mask = np.asarray(original_mask.resize((width, height), Image.Resampling.NEAREST)) > 0
        assert mask.shape == (height, width)
        canvas = Image.new("1", (width, height), 0)
        drawing = ImageDraw.Draw(canvas)
        visible_boxes = 0
        near_clipped_boxes = 0
        for bounds in regions:
            corners = np.array(list(itertools.product(*bounds.T)))
            model = corners@anchor_to_model[:3, :3].T+anchor_to_model[:3, 3]
            homogeneous = np.column_stack((model, np.ones(8)))
            depth = (homogeneous@np.asarray(record["world_view"]))[:, 2]
            near = helper.MIN_VIEW_DEPTH+1e-5
            valid = depth > near
            if not valid.any():
                continue
            vertices = list(homogeneous[valid])
            if not valid.all():
                near_clipped_boxes += 1
                for first in range(8):
                    for bit in [1, 2, 4]:
                        second = first ^ bit
                        if first < second and valid[first] != valid[second]:
                            fraction = (near-depth[first])/(depth[second]-depth[first])
                            vertices.append(homogeneous[first]+fraction*(homogeneous[second]-homogeneous[first]))
            clip = np.asarray(vertices)@np.asarray(record["full_proj"])
            ndc = clip[:, :2]/clip[:, 3:4]
            xy = ((ndc+1)*[width, height]-1)*.5
            x0, y0 = np.floor(xy.min(axis=0)).astype(int)
            x1, y1 = np.ceil(xy.max(axis=0)).astype(int)
            if x1 < 0 or y1 < 0 or x0 >= width or y0 >= height:
                continue
            drawing.polygon(_convex_hull_2d(xy), fill=1)
            visible_boxes += 1
        projected = np.asarray(canvas, dtype=bool)
        union, intersection = mask | projected, mask & projected
        rows.append({"id": record["id"], "mask_pixels": int(mask.sum()),
                     "hd_projected_envelope_pixels": int(projected.sum()), "intersection_pixels": int(intersection.sum()),
                     "mask_coverage_by_hd_envelope": float(intersection.sum()/mask.sum()) if mask.any() else None,
                     "hd_envelope_to_mask_area_ratio": float(projected.sum()/mask.sum()) if mask.any() else None,
                     "iou": float(intersection.sum()/union.sum()) if union.any() else None,
                     "visible_regions": visible_boxes, "near_clipped_regions": near_clipped_boxes,
                     "input_mask_size": list(original_mask.size), "render_size": [width, height]})
    return {"rows": rows, "scope": "near-clipped projected temporal-Hd AABB convex silhouette vs actual input 2D dynamic masks",
            "mask_resize": "nearest-neighbor1920x1280 to actual scale4 rendering480x320",
            "limitation": "projected region overlap does not certify occlusion, per-point contribution or correct road assignment"}


def _convex_hull_2d(points):
    points = sorted(set(map(tuple, points)))
    def cross(first, second, third):
        return (second[0]-first[0])*(third[1]-first[1])-(second[1]-first[1])*(third[0]-first[0])
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1]+upper[:-1]


def actual_region_view_records(source, metadata):
    """从实际 calib/pose 重建 Camera 小矩阵，独立 pinhole 投影作 CPU 交叉核对。"""
    tree = ast.parse((source/"utils/graphics_utils.py").read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "getProjectionMatrixCenterShift")
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "actual_projection_cpu", "exec"), namespace)
    root = source/"region_audit"
    transform = np.asarray(metadata["world_to_model"])
    first_pose = np.loadtxt(root/"pose/0017000.txt")
    scale = float(metadata["meter_to_model_scale"])
    records = {"model_to_anchor": np.linalg.inv(transform@first_pose).tolist(), "views": []}
    crosschecks = []
    region = np.asarray(metadata["regions_anchor"][0])
    anchor_corners = np.array(list(itertools.product(*region.T)))
    anchor_homo = np.column_stack((anchor_corners, np.ones(8)))
    model_homo = anchor_homo@(transform@first_pose).T
    for mask_path in sorted((root/"mask").glob("*.png")):
        frame = mask_path.stem
        calibration = {}
        for line in (root/"calib"/(frame+".txt")).read_text(encoding="utf-8").splitlines():
            key, values = line.split(":", 1)
            calibration[key] = np.fromstring(values, sep=" ")
        pinhole = calibration["P0"].reshape(3, 4)[:, :3]/4
        pinhole[2, 2] = 1
        lidar_to_camera = np.eye(4)
        lidar_to_camera[:3] = calibration["Tr_velo_to_cam_0"].reshape(3, 4)
        pose = np.loadtxt(root/"pose"/(frame+".txt"))
        model_c2w = transform@pose@np.linalg.inv(lidar_to_camera)
        model_c2w[:3, :3] /= scale
        worldview = torch.tensor(np.linalg.inv(model_c2w).astype(np.float32)).T
        projection = namespace["getProjectionMatrixCenterShift"](.01, 1000., pinhole[0, 2], pinhole[1, 2],
                                                                  pinhole[0, 0], pinhole[1, 1], 480, 320).T
        full_proj = worldview@projection
        clip = model_homo@full_proj.numpy()
        cpu_pixels = ((clip[:, :2]/clip[:, 3:4]+1)*[480, 320]-1)*.5
        physical_camera = anchor_homo@(lidar_to_camera@np.linalg.inv(pose)@first_pose).T
        pinhole_pixels = physical_camera[:, :3]@pinhole.T
        pinhole_pixels = pinhole_pixels[:, :2]/pinhole_pixels[:, 2:3]-.5
        error = float(np.max(np.abs(cpu_pixels-pinhole_pixels)))
        assert error < .002, (frame, error)
        records["views"].append({"id": "image_0/"+frame, "mask_path": str(mask_path), "width": 480, "height": 320,
                                 "world_view": worldview.tolist(), "full_proj": full_proj.tolist()})
        crosschecks.append({"id": "image_0/"+frame, "camera_vs_calibration_projection_max_abs_px": error})
    assert len(records["views"]) == 3
    return records, crosschecks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--region-views", type=Path)
    args = parser.parse_args()
    report, module = regressions(args.source)
    report["actual_geometry"] = actual_geometry_measurement(args.source, module)
    if args.region_views:
        metadata = json.loads((args.source/"sgcm_runtime_geometry.json").read_text(encoding="utf-8"))
        report["actual_region_mask_overlap"] = region_mask_measurement(args.region_views, metadata)
    elif (args.source/"region_audit").is_dir():
        metadata = json.loads((args.source/"sgcm_runtime_geometry.json").read_text(encoding="utf-8"))
        records, crosschecks = actual_region_view_records(args.source, metadata)
        report["actual_region_mask_overlap"] = region_mask_measurement(records, metadata)
        report["actual_region_mask_overlap"]["camera_projection_crosschecks"] = crosschecks
    assert not torch.cuda.is_initialized()
    result = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(result+"\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
