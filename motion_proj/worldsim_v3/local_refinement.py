"""WorldSim V3 A3 局部精修的冻结语义与 fail-closed 审计。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor


TASK_ID = "WS-V3-A3-LOCAL-REFINE-01"
AUDIT_VERSION = "A3-LOCAL-REFINE-PROTOCOL-v1"
STRATA = ("S-A-observed", "S-B-geometric", "S-C-unsupported")
VARIANTS = (
    "r0-no-refine-exact-alias",
    "r1-reactivate",
    "r2-appearance",
    "r3-hole-seed",
    "r4-temporal",
)
HELDOUT_FRAMES = list(range(10, 200, 10))
FORBIDDEN_DEPENDENCIES = {
    "configs/dynamic_editing_v2/m5_protocol_v1.yaml",
    "motion_proj/dynamic_editing_v2/stress_metrics.py",
    "scripts/run_dr_v2_m5_stress_scene.py",
}
FORBIDDEN_SCOPE = {
    "formal_training",
    "whole_scene_retraining",
    "D3_depth_normal_densification",
    "D4_provenance_pruning",
    "S-C_pseudo_truth",
    "single_view_RGB_to_3D_seed",
    "large_video_diffusion",
    "untracked_V2_M5_dependency",
}
MUTABLE_FIELDS = {
    "r0-no-refine-exact-alias": set(),
    "r1-reactivate": {"Background._opacities", "Background._scales"},
    "r2-appearance": {
        "Background._opacities",
        "Background._scales",
        "Background._features_dc",
        "Background._features_rest",
    },
    "r4-temporal": {
        "Background._opacities",
        "Background._scales",
        "Background._features_dc",
        "Background._features_rest",
    },
}


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def _sha256(value: Any, detail: str) -> None:
    text = str(value or "")
    _require(
        len(text) == 64
        and all(character in "0123456789abcdef" for character in text),
        detail,
    )


def validate_a3_protocol(contract: Mapping[str, Any]) -> None:
    """在工程 smoke 前锁死 A3 的证据边界与资产依赖。"""

    _require(contract.get("schema_version") == 1, "unsupported A3 schema")
    _require(contract.get("task_id") == TASK_ID, "A3 task ID drift")
    _require(contract.get("audit_version") == AUDIT_VERSION, "A3 audit drift")
    _require(
        contract.get("protocol_status") == "frozen_before_engineering_smoke",
        "A3 protocol status drift",
    )
    _require(
        contract.get("formal_training_authorized") is False,
        "A3 formal training is not authorized",
    )

    depends_on = contract.get("depends_on") or {}
    _require(
        depends_on.get("a2_closeout_commit")
        == "224669317418a2c7664b7381ebaa64134d83367c",
        "A2 closeout dependency drift",
    )
    _require(
        depends_on.get("a2_decision") == "tradeoff_non_dominated",
        "A2 decision drift",
    )
    for name in (
        "selected_summary_sha256",
        "selected_manifest_sha256",
        "selected_checkpoint_sha256",
        "selected_checkpoint_config_sha256",
        "selected_actor_registry_sha256",
        "d1_fallback_checkpoint_sha256",
    ):
        _sha256(depends_on.get(name), f"missing A3 dependency SHA: {name}")
    _require(
        depends_on.get("selected_checkpoint_sha256")
        == "1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c",
        "selected D2 checkpoint drift",
    )

    committed = contract.get("committed_inputs") or {}
    for name in (
        "edit_runner",
        "paired_mask_metrics",
        "typed_depth",
        "drivestudio_depth_adapter",
    ):
        value = committed.get(name) or {}
        _require(bool(value.get("path")), f"missing committed input: {name}")
        _sha256(value.get("sha256"), f"missing committed input SHA: {name}")
    _require(
        set(committed.get("forbidden_dependencies") or ())
        == FORBIDDEN_DEPENDENCIES,
        "untracked V2 dependency boundary drift",
    )

    design = contract.get("paired_design") or {}
    _require(design.get("scene") == "scene-0230", "A3 scene drift")
    _require(design.get("scene_index") == 179, "A3 scene index drift")
    _require(design.get("seed") == 0, "A3 seed drift")
    _require(design.get("cameras") == [0, 1, 2], "A3 cameras drift")
    _require(design.get("edits") == ["lateral", "delete"], "A3 edits drift")
    _require(
        design.get("actor_roles") == ["high-support", "boundary-support"],
        "A3 actor roles drift",
    )
    actors = design.get("actors") or {}
    _require(
        (actors.get("high-support") or {}).get("instance_token")
        == "af663976db5e412e83db033d309c5c29",
        "high-support actor drift",
    )
    _require(
        (actors.get("boundary-support") or {}).get("instance_token")
        == "18c7f0c5fa6b49449f71c9dbae5c31d4",
        "boundary-support actor drift",
    )
    heldout = design.get("heldout") or {}
    _require(heldout.get("frames") == HELDOUT_FRAMES, "held-out frames drift")
    _require(
        heldout.get("use")
        == "evaluation_only_excluded_from_all_optimization_and_support_selection",
        "held-out use drift",
    )
    _require(
        tuple(design.get("execution_order") or ()) == VARIANTS,
        "A3 variant order drift",
    )
    _require(
        design.get("first_engineering_gate")
        == ["r0-no-refine-exact-alias", "r1-reactivate"],
        "first engineering gate drift",
    )

    edit = contract.get("edit_contract") or {}
    _require(
        (edit.get("lateral") or {}).get("actor_local_y_m") == 1.0,
        "lateral edit magnitude drift",
    )
    _require(
        edit.get("target_actor_parameters") == "context_only_frozen"
        and edit.get("target_actor_trajectory")
        == "frozen_after_deterministic_edit",
        "target actor freeze boundary drift",
    )

    affected = contract.get("affected_set") or {}
    footprint = affected.get("paired_footprint") or {}
    _require(
        footprint.get("difference_threshold_uint8") == 2
        and footprint.get("counterfactual_mask_dilation_pixels") == 2
        and footprint.get("affected_union_dilation_pixels") == 3,
        "affected footprint morphology drift",
    )
    gaussian = affected.get("gaussian_membership") or {}
    _require(
        float(gaussian.get("depth_order_tolerance_m", -1)) == 0.05,
        "depth-order tolerance drift",
    )
    outside = affected.get("outside_contract") or {}
    _require(
        outside.get("gaussian_parameters") == "exact_bitwise_frozen"
        and outside.get("optimizer_state") == "exact_bitwise_frozen",
        "outside exact-freeze contract drift",
    )

    strata = contract.get("support_strata") or {}
    _require(tuple(strata.get("precedence") or ()) == STRATA, "strata drift")
    _require(
        (strata.get("S-A-observed") or {}).get("rgb_loss_allowed") is True,
        "S-A RGB contract drift",
    )
    _require(
        (strata.get("S-B-geometric") or {}).get("rgb_loss_allowed") is False,
        "S-B must not use RGB loss",
    )
    unsupported = strata.get("S-C-unsupported") or {}
    _require(
        unsupported.get("parameter_update_allowed") is False
        and unsupported.get("gaussian_seed_allowed") is False
        and unsupported.get("loss_allowed") is False,
        "S-C must remain unsupported",
    )
    _require(
        strata.get("ancestry_nearest_lidar_distance_is_provenance_not_metric_depth")
        is True,
        "ancestry must not become metric depth",
    )

    depths = contract.get("typed_depth_contract") or {}
    _require(
        (depths.get("depth_render_expected") or {}).get("truth_tier")
        == "diagnostic",
        "expected depth truth tier drift",
    )
    _require(
        (depths.get("depth_surface_first_hit") or {}).get("truth_tier") == "T1",
        "first-hit truth tier drift",
    )
    _require(
        (depths.get("depth_lidar_measured") or {}).get("truth_tier") == "T0",
        "LiDAR truth tier drift",
    )

    variants = contract.get("variants") or {}
    _require(set(variants) == set(VARIANTS), "A3 variants drift")
    _require(
        (variants["r0-no-refine-exact-alias"]).get("optimizer_steps") == 0
        and (variants["r0-no-refine-exact-alias"]).get("new_checkpoint_keys")
        is False,
        "R0 exact-alias boundary drift",
    )
    _require(
        set(variants["r1-reactivate"].get("mutable_parameters") or ())
        == MUTABLE_FIELDS["r1-reactivate"],
        "R1 mutable fields drift",
    )
    _require(
        variants["r1-reactivate"].get("position_updates") is False
        and variants["r1-reactivate"].get("color_updates") is False
        and variants["r1-reactivate"].get("gaussian_seed") is False,
        "R1 must add only opacity/scale",
    )

    gate = contract.get("engineering_gate") or {}
    _require(
        gate.get("module_off_exact_alias_required") is True
        and gate.get("outside_parameter_and_optimizer_state_exact_required")
        is True,
        "A3 engineering exactness gate drift",
    )
    _require(
        gate.get("formal_training_authorized") is False,
        "A3 engineering gate cannot authorize formal training",
    )
    _require(
        set((contract.get("scope_boundary") or {}).get("forbidden") or ())
        == FORBIDDEN_SCOPE,
        "A3 scope boundary drift",
    )


def classify_support(
    *,
    alternate_observed_rgb: bool,
    target_excluded: bool,
    calibrated_reprojection: bool,
    lidar_measured_valid: bool,
    multiview_geometry_count: int,
) -> str:
    """按 S-A→S-B→S-C 优先级对单个支持单元分类。"""

    if multiview_geometry_count < 0:
        raise ValueError("multiview geometry count cannot be negative")
    if alternate_observed_rgb and not target_excluded:
        raise ValueError("observed RGB support must exclude the target view")
    if alternate_observed_rgb and calibrated_reprojection and target_excluded:
        return "S-A-observed"
    if lidar_measured_valid or multiview_geometry_count >= 2:
        return "S-B-geometric"
    return "S-C-unsupported"


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("affected masks must be two-dimensional")
    if radius < 0:
        raise ValueError("dilation radius cannot be negative")
    if radius == 0:
        return value.copy()
    padded = np.pad(value, radius, mode="constant")
    output = np.zeros_like(value)
    height, width = value.shape
    for row in range(2 * radius + 1):
        for column in range(2 * radius + 1):
            output |= padded[row : row + height, column : column + width]
    return output


def affected_pixel_mask(
    source_footprint: np.ndarray,
    edited_footprint: np.ndarray,
    *,
    supported_hole: np.ndarray | None = None,
    depth_order_conflict: np.ndarray | None = None,
    dilation_radius: int = 3,
) -> np.ndarray:
    """构造冻结的 footprint/hole/depth-conflict 并集。"""

    masks = [
        np.asarray(source_footprint, dtype=bool),
        np.asarray(edited_footprint, dtype=bool),
    ]
    if supported_hole is not None:
        masks.append(np.asarray(supported_hole, dtype=bool))
    if depth_order_conflict is not None:
        masks.append(np.asarray(depth_order_conflict, dtype=bool))
    shape = masks[0].shape
    if any(value.shape != shape for value in masks):
        raise ValueError("affected masks must share one shape")
    union = np.logical_or.reduce(masks)
    return _binary_dilate(union, dilation_radius)


def mutable_parameter_fields(variant: str) -> set[str]:
    if variant == "r3-hole-seed":
        raise ValueError("R3 changes tensor cardinality and needs a seed audit")
    if variant not in MUTABLE_FIELDS:
        raise ValueError(f"unknown A3 variant: {variant}")
    return set(MUTABLE_FIELDS[variant])


def auditable_update_rows(
    affected_rows: Tensor,
    strata: list[str] | tuple[str, ...],
) -> Tensor:
    """只允许 affected 且属于 S-A/S-B 的既有 Background 行更新。"""

    mask = torch.as_tensor(affected_rows, dtype=torch.bool).flatten()
    if len(strata) != mask.numel():
        raise ValueError("strata and affected rows must align")
    unknown = sorted(set(strata).difference(STRATA))
    if unknown:
        raise ValueError(f"unknown support strata: {unknown}")
    supported = torch.tensor(
        [value != "S-C-unsupported" for value in strata],
        dtype=torch.bool,
        device=mask.device,
    )
    return mask & supported


def audit_frozen_rows(
    before: Mapping[str, Tensor],
    after: Mapping[str, Tensor],
    *,
    mutable_rows: Tensor,
    mutable_fields: set[str],
) -> dict[str, Any]:
    """逐 tensor 精确检查 outside、actor 和非授权字段没有漂移。"""

    if set(before) != set(after):
        raise ValueError("state fields changed")
    rows = torch.as_tensor(mutable_rows, dtype=torch.bool).flatten()
    checks: dict[str, dict[str, Any]] = {}
    for name in sorted(before):
        left = torch.as_tensor(before[name])
        right = torch.as_tensor(after[name])
        if left.shape != right.shape or left.dtype != right.dtype:
            checks[name] = {
                "shape_and_dtype_exact": False,
                "outside_exact": False,
                "authorized_field": name in mutable_fields,
            }
            continue
        authorized = name in mutable_fields
        if authorized:
            if left.ndim == 0 or left.shape[0] != rows.numel():
                raise ValueError(f"mutable field is not row-aligned: {name}")
            field_rows = rows.to(device=left.device)
            outside_exact = torch.equal(left[~field_rows], right[~field_rows])
        else:
            outside_exact = torch.equal(left, right)
        checks[name] = {
            "shape_and_dtype_exact": True,
            "outside_exact": bool(outside_exact),
            "authorized_field": authorized,
        }
    return {
        "checks": checks,
        "pass": all(
            value["shape_and_dtype_exact"] and value["outside_exact"]
            for value in checks.values()
        ),
    }
