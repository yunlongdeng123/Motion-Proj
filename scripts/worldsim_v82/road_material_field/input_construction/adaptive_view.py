"""Deterministic camera-view candidates for automatic diffusion input building.

No RGB, target truth, scene IDs, frame IDs, or stratum labels are accepted.
Physical unknown/core cannot supply geometry observations. Effects are observed
but material-contaminated: their actual road/line geometry may remain available,
while their RGB never counts as reliable material support. Clean GT is forbidden.

Matrices use pixel-corner coordinates: ``native_from_input @ [u+.5,v+.5,1]``
maps an input pixel center to a native camera pixel center. Local material and
full-frame geometry branches therefore share one exact camera coordinate system.
The module plans inputs; it does not claim that a downstream network consumes
either branch. The caller must render the returned plans and wire those tensors.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import label


FEATURE_SCHEMA = "adaptive_view_features.v1"
FEATURE_NAMES = (
    "input_pixels_over_cap", "crop_native_area_fraction", "sampling_scale", "content_fraction",
    "information_density", "target_latent_min_over64", "target_latent_area_over4096",
    "target_model_fraction", "effect_model_fraction", "visible_road_crop_fraction",
    "near_road_retention", "largest_road_component_fraction", "road_anchor_input_pixels_over4096",
    "lane_anchor_retention", "lane_input_mass_over256", "lane_available",
    "relevant_keep_retention", "relevant_keep_present", "margin_left", "margin_right",
    "margin_top", "margin_bottom", "required_domain_fraction", "frame_edge_fraction",
)


@dataclass(frozen=True)
class ViewConfig:
    alignment: int = 8
    vae_stride: int = 8
    model_min_axis: int = 64
    global_long_edge: int = 384
    max_local_pixels: int = 1024 * 512
    max_total_pixels: int | None = None
    min_target_latent_axis: float = 12.0
    min_target_latent_area: float = 64.0
    min_visible_road_input_pixels: float = 1024.0
    min_connected_road_input_pixels: float = 256.0
    min_near_road_retention: float = .40
    min_lane_anchor_retention: float = .65
    min_lane_native_mass: float = 16.0
    max_letterbox_fraction: float = .15
    anchor_context_scale: float = 2.8
    relevant_keep_context_scale: float = 1.4
    road_touch_context_scale: float = 1.5
    require_road_context: bool = True
    preserve_relevant_keep: bool = True
    trim_unused_bucket: bool = True
    orient_buckets_to_portrait: bool = True
    utility_tolerance: float = 0.0


# Four oracle actions maximum. Buckets are caps; small native crops are not
# upscaled to fill them. Optional aligned active-canvas trimming avoids gray fill.
DEFAULT_CANDIDATES = (
    ("compact", (768, 384), 1.2, False),
    ("balanced", (768, 512), 1.6, False),
    ("context", (1024, 512), 2.2, False),
    ("full_frame", (768, 512), 1.0, True),
)


def _mask(masks: Mapping[str, Any], names: Sequence[str], shape, default=None, ignore=None):
    for name in names:
        if name in masks and masks[name] is not None:
            value = np.asarray(masks[name])
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} differs from imageHW {shape}")
            if ignore is not None:
                value = value.copy()
                value[ignore] = 0
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"{name} contains nonfinite values")
            return value > 0
    return np.zeros(shape, bool) if default is None else np.asarray(default, bool).copy()


def _bbox(mask):
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]


def _box_union(*boxes):
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def _expand_box(box, factor, image_hw):
    h, w = image_hw
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    dx, dy = (box[2] - box[0]) * factor / 2, (box[3] - box[1]) * factor / 2
    return [max(0, int(math.floor(cx - dx))), max(0, int(math.floor(cy - dy))),
            min(w, int(math.ceil(cx + dx))), min(h, int(math.ceil(cy + dy)))]


def _slice(box):
    return np.s_[box[1]:box[3], box[0]:box[2]]


def _count(values, box):
    return float(np.sum(values[_slice(box)], dtype=np.float64))


def _intersection_box(a, b):
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def _safe_count(values, box):
    return _count(values, box) if box[2] > box[0] and box[3] > box[1] else 0.


def _make_window(required_box, image_hw, bucket_wh, context_scale, full_frame):
    h, w = image_hw
    if full_frame:
        return [0, 0, w, h]
    ratio = bucket_wh[0] / bucket_wh[1]
    rw, rh = required_box[2] - required_box[0], required_box[3] - required_box[1]
    desired_w = max(rw * context_scale, rh * context_scale * ratio)
    desired_h = desired_w / ratio
    crop_w, crop_h = min(w, int(math.ceil(desired_w))), min(h, int(math.ceil(desired_h)))
    crop_w, crop_h = max(crop_w, rw), max(crop_h, rh)
    cx, cy = (required_box[0] + required_box[2]) / 2, (required_box[1] + required_box[3]) / 2
    x0 = int(math.floor(cx - crop_w / 2))
    y0 = int(math.floor(cy - crop_h / 2))
    x0, y0 = max(0, min(w - crop_w, x0)), max(0, min(h - crop_h, y0))
    box = [x0, y0, x0 + crop_w, y0 + crop_h]
    assert box[0] <= required_box[0] and box[1] <= required_box[1]
    assert box[2] >= required_box[2] and box[3] >= required_box[3]
    return box


def _render_plan(box, bucket_wh, alignment, trim=True, min_axis=64):
    cw, ch = box[2] - box[0], box[3] - box[1]
    # Hard no-upscale: extra model pixels cannot manufacture native evidence.
    scale = min(1., bucket_wh[0] / cw, bucket_wh[1] / ch)
    content_w, content_h = cw * scale, ch * scale
    if trim:
        minimum = max(alignment, int(math.ceil(min_axis / alignment)) * alignment)
        ow = min(bucket_wh[0], max(minimum, int(math.ceil(content_w / alignment)) * alignment))
        oh = min(bucket_wh[1], max(minimum, int(math.ceil(content_h / alignment)) * alignment))
    else:
        ow, oh = bucket_wh
    px, py = (ow - content_w) / 2, (oh - content_h) / 2
    if px < -1e-8 or py < -1e-8:
        raise ValueError("Bucket dimensions must be positive alignment multiples")
    native_from_input = np.array([[1 / scale, 0., box[0] - px / scale],
                                  [0., 1 / scale, box[1] - py / scale], [0., 0., 1.]], np.float64)
    return {"window_xyxy": list(box), "native_crop_xyxy": list(box), "bucket_wh": list(bucket_wh),
            "output_wh": [ow, oh], "input_wh": [ow, oh], "isotropic_scale": scale,
            "content_box_xyxy": [px, py, px + content_w, py + content_h], "padding_xy": [px, py],
            "content_fraction": content_w * content_h / (ow * oh),
            "native_from_input": native_from_input.tolist(),
            "input_from_native": np.linalg.inv(native_from_input).tolist(),
            "pixel_coordinate_convention": "pixel corners; apply matrix to [u+.5,v+.5,1] for input centers",
            "upscaled": False, "sizing_policy": "aligned active canvas within bucket cap" if trim else "fixed bucket, isotropic letterbox"}


def _global_plan(image_hw, cfg):
    h, w = image_hw
    scale = min(1., cfg.global_long_edge / max(h, w))
    minimum = max(cfg.alignment, int(math.ceil(cfg.model_min_axis / cfg.alignment)) * cfg.alignment)
    ow = max(minimum, int(math.ceil(w * scale / cfg.alignment)) * cfg.alignment)
    oh = max(minimum, int(math.ceil(h * scale / cfg.alignment)) * cfg.alignment)
    result = _render_plan([0, 0, w, h], (ow, oh), cfg.alignment, trim=True, min_axis=cfg.model_min_axis)
    result.update(id="global_geometry", role="full native camera geometry context; preserve all retained objects",
                  source_coverage_fraction=1., protected_source_coverage_fraction=1.)
    return result


def _connected_road(road, crop, touch_box):
    cropped = road[_slice(crop)]
    if not cropped.any():
        return 0., 0.
    ids, _ = label(cropped, structure=np.ones((3, 3), np.uint8))
    areas = np.bincount(ids.ravel())
    areas[0] = 0
    largest = int(areas.max())
    touched = _intersection_box(crop, touch_box)
    if touched[2] <= touched[0] or touched[3] <= touched[1]:
        return largest / int(cropped.sum()), 0.
    local = [touched[0] - crop[0], touched[1] - crop[1], touched[2] - crop[0], touched[3] - crop[1]]
    labels_at_target = np.unique(ids[_slice(local)])
    labels_at_target = labels_at_target[labels_at_target > 0]
    anchor_area = int(areas[labels_at_target].max()) if len(labels_at_target) else 0
    return largest / int(cropped.sum()), float(anchor_area)


def _keep_near_required(keep, required_box, image_hw, factor):
    ids, _ = label(keep, structure=np.ones((3, 3), np.uint8))
    nearby = _expand_box(required_box, factor, image_hw)
    selected = np.unique(ids[_slice(nearby)])
    selected = selected[selected > 0]
    if not len(selected):
        return np.zeros_like(keep), []
    relevant = np.isin(ids, selected)
    return relevant, selected.tolist()


def _failure(code, actual, required):
    return {"code": code, "actual": float(actual), "required": float(required)}


def select_candidate(candidates, utility_scores=None, utility_tolerance=0.0):
    """Select among construction-valid views using only visible-input features.

    Utilities can be predicted once from the returned fixed feature matrix.
    This function never runs diffusion or accepts a GT/oracle callback.
    Empirical quality thresholds do not make an oracle action ineligible.
    The deterministic fallback prefers quality-feasible, then minimum quality
    deficit. A missing construction-valid view is reported without authorization.
    """
    valid = [c for c in candidates if c.get("hard_valid", c.get("feasible", False))]
    if valid:
        quality = [c for c in valid if c.get("quality_feasible", c.get("feasible", False))]
        shortlist = quality or valid
        reason = "fastest_quality_feasible_pixel_proxy" if quality else "hard_valid_min_quality_deficit_fallback"
        if utility_scores is not None:
            scores = np.asarray(utility_scores, np.float64)
            if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
                raise ValueError("utility_scores must contain one finite predicted score per candidate")
            by_id = {c["id"]: float(score) for c, score in zip(candidates, scores)}
            maximum = max(by_id[c["id"]] for c in valid)
            shortlist = [c for c in valid if by_id[c["id"]] >= maximum - utility_tolerance]
            reason = "predicted_utility_then_fastest_hard_valid"
        elif not quality:
            best_deficit = min(c.get("quality_deficit", c["constraint_deficit"]) for c in valid)
            shortlist = [c for c in valid if c.get("quality_deficit", c["constraint_deficit"]) <= best_deficit + 1e-12]
        chosen = min(shortlist, key=lambda c: (c["estimated_pixel_cost"], -c["stats"]["near_road_retention"], c["id"]))
        return chosen, reason
    if not candidates:
        return None, "no_candidates"
    # Do not silently call an infeasible view valid. Maximize source information
    # after minimizing normalized constraint deficits, then break ties by cost.
    chosen = min(candidates, key=lambda c: (c["critical_failure"], c.get("hard_deficit", c["constraint_deficit"]),
                                           c.get("quality_deficit", c["constraint_deficit"]),
                                           -c["stats"]["target_effective_latent_min_axis"],
                                           c["estimated_pixel_cost"], c["id"]))
    return chosen, "no_hard_valid_view_explicit_min_violation_fallback"


def plan_views(image_hw, masks: Mapping[str, Any], lane_support=None,
               config: ViewConfig | Mapping[str, Any] | None = None, utility_scores=None):
    """Return <=4 local view candidates, selection, and a global geometry plan.

    Mask aliases match InputBuilder NativeLayers: target_core, target_unknown/F,
    rgb_unknown_core/physical_unknown, effect_inferred, K/retained_core, C_mat,
    geometry_road_visible/road_geometry_observed/road_observed, foreground_core. Unknown
    metadata (including any source/stratum fields) is ignored, never a feature.
    C_mat is authoritative certified appearance support; uncertain proposals or
    bounding boxes are not reintroduced as unconditional exclusion rectangles.
    """
    started = time.perf_counter()
    cfg = ViewConfig() if config is None else config if isinstance(config, ViewConfig) else ViewConfig(**dict(config))
    h, w = map(int, image_hw)
    if h <= 0 or w <= 0 or cfg.alignment <= 0 or cfg.vae_stride <= 0 or cfg.model_min_axis <= 0 or cfg.global_long_edge <= 0 or cfg.max_local_pixels <= 0:
        raise ValueError("Positive image dimensions, alignment, and VAE stride are required")
    if not (0 <= cfg.max_letterbox_fraction <= 1 and 0 <= cfg.min_near_road_retention <= 1 and 0 <= cfg.min_lane_anchor_retention <= 1):
        raise ValueError("Fractional view constraints must be in [0,1]")
    shape = (h, w)
    core = _mask(masks, ("target_core",), shape)
    uncertainty = _mask(masks, ("uncertainty", "target_uncertainty"), shape)
    unknown = _mask(masks, ("target_unknown", "F"), shape, core | uncertainty) | core
    effect = _mask(masks, ("effect_inferred", "effect"), shape)
    keep = _mask(masks, ("K", "retained_core", "protectedK"), shape)
    foreground = _mask(masks, ("foreground_core",), shape)
    # F is the edit domain and can include observed shadows/effects. It is not
    # the physical visibility domain. An explicit physical mask wins; fallback
    # separates effect from F, with target_core always physically unknown.
    physical_unknown = _mask(masks, ("physical_unknown", "rgb_unknown_core", "core_unknown"),
                             shape, (unknown & ~effect) | core) | core
    geometry_forbidden = physical_unknown | keep | foreground
    appearance_forbidden = geometry_forbidden | effect
    road_appearance = _mask(masks, ("C_mat", "road_observed", "visibleRoad", "visible_road"),
                            shape, ignore=appearance_forbidden) & ~appearance_forbidden
    road = _mask(masks, ("road_geometry_observed", "geometry_road_visible", "road_observed",
                         "visibleRoad", "visible_road", "C_mat"),
                 shape, ignore=geometry_forbidden) & ~geometry_forbidden
    if lane_support is None:
        lane_support = masks.get("lane_support", masks.get("laneSupport"))
    lane = np.zeros(shape, np.float32) if lane_support is None else np.asarray(lane_support, np.float32).copy()
    if lane.shape != shape:
        raise ValueError("lane_support must share native HxW")
    lane[geometry_forbidden | ~road] = 0
    if not np.isfinite(lane).all() or (lane < 0).any() or (lane > 1).any():
        raise ValueError("lane_support must be finite HxW confidence in [0,1]")
    safe_effect = effect & ~keep
    required = core | ((unknown | safe_effect) & ~keep)
    # InputBuilder can trim K around the target to render a disjoint mask. Its
    # original ownership conflict must not disappear merely due to that priority.
    retained_identity = _mask(masks, ("retained_core",), shape)
    ownership_conflict = _mask(masks, ("ownership_conflict", "target_retained_conflict"), shape)
    critical_conflict = int(((core & (keep | retained_identity)) | ownership_conflict).sum())
    core_box, required_box = _bbox(core), _bbox(required)
    global_geometry = _global_plan(shape, cfg)
    if core_box is None:
        return {"schema": "adaptive_view_plan.v1", "feature_schema": FEATURE_SCHEMA, "feature_names": list(FEATURE_NAMES),
                "image_hw": [h, w], "candidates": [], "feature_matrix": [], "selected": None,
                "global_geometry": global_geometry, "selection_reason": "empty_target_core", "can_infer": False,
                "hard_valid": False, "quality_constraints_satisfied": False,
                "hard_failures": [_failure("empty_target_core", 0, 1)],
                "planning_cpu_ms": (time.perf_counter() - started) * 1000}
    relevant_keep, keep_components = _keep_near_required(keep, required_box, shape, cfg.relevant_keep_context_scale)
    keep_box = _bbox(relevant_keep)
    context_required_box = _box_union(required_box, keep_box) if cfg.preserve_relevant_keep else required_box
    anchor_box = _expand_box(core_box, cfg.anchor_context_scale, shape)
    touch_box = _expand_box(core_box, cfg.road_touch_context_scale, shape)
    near_road_total = _count(road, anchor_box)
    near_lane_total = _count(lane, anchor_box)
    lane_available = near_lane_total >= cfg.min_lane_native_mass
    core_count, required_count, effect_count = int(core.sum()), int(required.sum()), int(safe_effect.sum())
    relevant_count = int(relevant_keep.sum())
    global_pixels = int(np.prod(global_geometry["output_wh"]))
    total_budget = cfg.max_total_pixels if cfg.max_total_pixels is not None else cfg.max_local_pixels + global_pixels
    portrait = cfg.orient_buckets_to_portrait and (core_box[3] - core_box[1]) > (core_box[2] - core_box[0]) * 1.25
    candidates, seen = [], set()
    for index, (name, bucket, factor, full_frame) in enumerate(DEFAULT_CANDIDATES):
        bucket = tuple(reversed(bucket)) if portrait else bucket
        crop = _make_window(context_required_box, shape, bucket, factor, full_frame)
        view = _render_plan(crop, bucket, cfg.alignment, cfg.trim_unused_bucket, cfg.model_min_axis)
        signature = (tuple(crop), tuple(view["output_wh"]), round(view["isotropic_scale"], 12))
        if signature in seen:
            continue
        seen.add(signature)
        scale = view["isotropic_scale"]
        information_scale = min(scale, 1.)
        pixels = int(np.prod(view["output_wh"]))
        crop_pixels = (crop[2] - crop[0]) * (crop[3] - crop[1])
        core_coverage = _count(core, crop) / core_count
        required_coverage = _count(required, crop) / required_count
        keep_retention = _count(relevant_keep, crop) / relevant_count if relevant_count else 1.
        near_box = _intersection_box(crop, anchor_box)
        road_pixels = _count(road, crop)
        appearance_pixels = _count(road_appearance, crop)
        road_retention = _safe_count(road, near_box) / near_road_total if near_road_total else 0.
        lane_retention = _safe_count(lane, near_box) / near_lane_total if near_lane_total else 1.
        largest_road_fraction, road_anchor_area = _connected_road(road, crop, touch_box)
        latent_axis = min(core_box[2] - core_box[0], core_box[3] - core_box[1]) * information_scale / cfg.vae_stride
        latent_area = core_count * information_scale**2 / cfg.vae_stride**2
        lane_input_mass = _count(lane, crop) * scale**2
        stat = {
            "target_coverage": core_coverage, "required_coverage": required_coverage,
            "target_effective_latent_min_axis": latent_axis, "target_effective_latent_area": latent_area,
            "target_native_short_axis": min(core_box[2] - core_box[0], core_box[3] - core_box[1]),
            "road_native_pixels": road_pixels, "road_input_pixel_equivalent": road_pixels * scale**2,
            "geometry_road_native_pixels": road_pixels,
            "material_road_native_pixels": appearance_pixels,
            "material_road_input_pixel_equivalent": appearance_pixels * scale**2,
            "observed_effect_geometry_native_pixels": _count(road & effect, crop),
            "near_road_retention": road_retention, "largest_road_component_fraction": largest_road_fraction,
            "road_anchor_input_pixel_equivalent": road_anchor_area * scale**2,
            "lane_anchor_retention": lane_retention, "lane_input_mass": lane_input_mass,
            "lane_available": lane_available, "relevant_keep_retention": keep_retention,
            "letterbox_fraction": 1 - view["content_fraction"], "local_pixels": pixels, "global_pixels": global_pixels,
        }
        hard_failures, quality_failures = [], []
        def minimum(code, actual, minimum, *, hard=False):
            if actual + 1e-9 < minimum:
                (hard_failures if hard else quality_failures).append(_failure(code, actual, minimum))
        def maximum(code, actual, maximum, *, hard=False):
            if actual > maximum + 1e-9:
                (hard_failures if hard else quality_failures).append(_failure(code, actual, maximum))
        minimum("target_not_fully_covered", core_coverage, 1., hard=True)
        minimum("required_unknown_effect_cropped", required_coverage, 1., hard=True)
        minimum("model_input_axis_too_small", min(view["output_wh"]), cfg.model_min_axis, hard=True)
        minimum("target_information_axis_too_small", latent_axis, cfg.min_target_latent_axis)
        minimum("target_information_area_too_small", latent_area, cfg.min_target_latent_area)
        if cfg.require_road_context:
            minimum("reliable_material_road_too_small", appearance_pixels * scale**2, cfg.min_visible_road_input_pixels)
            minimum("connected_road_anchor_too_small", road_anchor_area * scale**2, cfg.min_connected_road_input_pixels)
            minimum("near_road_context_cropped", road_retention, cfg.min_near_road_retention)
        if lane_available:
            minimum("observed_lane_anchor_cropped", lane_retention, cfg.min_lane_anchor_retention)
        if cfg.preserve_relevant_keep:
            minimum("relevant_retained_geometry_cropped", keep_retention, 1., hard=True)
        maximum("excess_letterbox", 1 - view["content_fraction"], cfg.max_letterbox_fraction)
        maximum("local_pixel_budget_exceeded", pixels, cfg.max_local_pixels, hard=True)
        maximum("total_pixel_budget_exceeded", pixels + global_pixels, total_budget, hard=True)
        if critical_conflict:
            hard_failures.append(_failure("target_core_protected_identity_conflict", critical_conflict, 0))
        raw_features = [
            pixels / cfg.max_local_pixels, crop_pixels / (h * w), scale, view["content_fraction"], information_scale**2,
            min(latent_axis, 64.) / 64., min(latent_area, 4096.) / 4096., core_count * scale**2 / pixels,
            effect_count * scale**2 / pixels, appearance_pixels / crop_pixels, road_retention, largest_road_fraction,
            min(road_anchor_area * scale**2, 4096.) / 4096., lane_retention, min(lane_input_mass, 256.) / 256., float(lane_available),
            keep_retention, float(relevant_count > 0), (core_box[0] - crop[0]) / (crop[2] - crop[0]),
            (crop[2] - core_box[2]) / (crop[2] - crop[0]), (core_box[1] - crop[1]) / (crop[3] - crop[1]),
            (crop[3] - core_box[3]) / (crop[3] - crop[1]), required_count / crop_pixels,
            sum((crop[0] == 0, crop[1] == 0, crop[2] == w, crop[3] == h)) / 4.,
        ]
        features = np.clip(np.asarray(raw_features, np.float64), 0, 1).tolist()
        assert len(features) == len(FEATURE_NAMES) and np.isfinite(features).all()
        local_from_native = np.asarray(view["input_from_native"])
        local_from_global = local_from_native @ np.asarray(global_geometry["native_from_input"])
        # Separate construction validity from empirical quality preferences.
        # Oracle runs every hard-valid candidate, even low-latent full-frame.
        def deficit(failures):
            return sum(abs(f["actual"] - f["required"]) / max(abs(f["required"]), 1.) for f in failures)
        hard_deficit, quality_deficit = deficit(hard_failures), deficit(quality_failures)
        view.update(id=f"v{index}_{name}", role="local material and aligned local geometry crop",
                    context_scale=factor, full_native_frame=full_frame,
                    feature_schema=FEATURE_SCHEMA, feature_names=list(FEATURE_NAMES), features=features,
                    feature_values=dict(zip(FEATURE_NAMES, features)), stats=stat,
                    estimated_pixel_cost=pixels + global_pixels, estimated_local_pixels=pixels,
                    construction_valid=not hard_failures, hard_valid=not hard_failures,
                    rank_eligible=not hard_failures, quality_feasible=not quality_failures,
                    feasible=not quality_failures, hard_failures=hard_failures,
                    quality_failures=quality_failures, hard_deficit=hard_deficit,
                    quality_deficit=quality_deficit, constraint_deficit=hard_deficit + quality_deficit,
                    critical_failure=bool(critical_conflict),
                    input_from_global=local_from_global.tolist(), global_from_input=np.linalg.inv(local_from_global).tolist())
        candidates.append(view)
    selected, reason = select_candidate(candidates, utility_scores, cfg.utility_tolerance)
    can_infer = bool(selected and selected["hard_valid"])
    return {"schema": "adaptive_view_plan.v1", "feature_schema": FEATURE_SCHEMA, "feature_names": list(FEATURE_NAMES),
            "image_hw": [h, w], "config": asdict(cfg), "candidates": candidates,
            "feature_matrix": [c["features"] for c in candidates], "selected": selected,
            "selected_id": selected["id"] if selected else None, "selection_reason": reason, "can_infer": can_infer,
            "hard_valid": can_infer,
            "quality_constraints_satisfied": bool(selected and selected["quality_feasible"]),
            "global_geometry": global_geometry, "target_core_bbox_native": core_box,
            "required_bbox_native": required_box, "local_required_context_bbox_native": context_required_box,
            "relevant_keep_component_labels": keep_components,
            "observed_feature_sanitation": {"target_unknown_pixels": int(unknown.sum()), "effect_pixels": int(effect.sum()),
                                            "physical_unknown_pixels": int(physical_unknown.sum()),
                                            "blocked_geometry_observation_pixels": int(geometry_forbidden.sum()),
                                            "blocked_material_observation_pixels": int(appearance_forbidden.sum()),
                                            "blocked_lane_observation_pixels": int((geometry_forbidden | ~road).sum()),
                                            "effect_geometry_observations_allowed": True,
                                            "clean_supervision_observations_allowed": False,
                                            "hidden_rgb_read": False, "hidden_semantic_or_lane_observation_used": False},
            "critical_identity_conflict_pixels": critical_conflict,
            "constraint_policy": "hard_valid/rank_eligible permits generation; feasible aliases quality_feasible only; empirical latent/road/lane/padding thresholds never exclude oracle ranking",
            "cost_definition": "actual local input pixels + fixed global geometry pixels; a deterministic proxy, not a measured GPU latency",
            "max_oracle_actions": 4, "planning_cpu_ms": (time.perf_counter() - started) * 1000}


# Explicit, scene-agnostic aliases for InputBuilder callers.
enumerate_views = plan_views


def feature_matrix(plan):
    """Return N×24 float32 features in the frozen schema order."""
    if plan["feature_schema"] != FEATURE_SCHEMA or tuple(plan["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Incompatible adaptive-view feature schema")
    return np.asarray(plan["feature_matrix"], np.float32).reshape(-1, len(FEATURE_NAMES))


__all__ = ["ViewConfig", "FEATURE_SCHEMA", "FEATURE_NAMES", "DEFAULT_CANDIDATES", "plan_views", "enumerate_views", "select_candidate", "feature_matrix"]
