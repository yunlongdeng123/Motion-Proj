"""WorldSim V3 A3 局部精修的冻结语义与 fail-closed 审计。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


TASK_ID = "WS-V3-A3-LOCAL-REFINE-01"
AUDIT_VERSION = "A3-LOCAL-REFINE-PROTOCOL-v1"
SIDECAR_AUDIT_VERSION = "A3-R1-SIDECAR-v1"
R1_NUMERIC_AUDIT_VERSION = "A3-R1-NUMERIC-FREEZE-v1"
R1_EVAL_AUDIT_VERSION = "A3-R1-HELDOUT-EVAL-PROTOCOL-v1"
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


def validate_a3_r1_numeric_freeze(contract: Mapping[str, Any]) -> None:
    """Validate the post-smoke R1 engineering replay budget."""

    _require(contract.get("schema_version") == 1, "unsupported A3 R1 numeric schema")
    _require(contract.get("task_id") == TASK_ID, "A3 R1 numeric task drift")
    _require(
        contract.get("audit_version") == R1_NUMERIC_AUDIT_VERSION,
        "A3 R1 numeric audit drift",
    )
    _require(
        contract.get("status") == "frozen_after_real_paired_engineering_smoke",
        "A3 R1 numeric status drift",
    )
    _require(
        contract.get("formal_training_authorized") is False
        and contract.get("quality_claim_authorized") is False,
        "A3 R1 numeric freeze cannot authorize claims or formal training",
    )
    evidence = contract.get("evidence") or {}
    for name in (
        "summary_sha256",
        "manifest_sha256",
        "resolved_config_sha256",
        "output_checkpoint_sha256",
        "controller_log_sha256",
        "exit_code_sha256",
        "sidecar_manifest_sha256",
        "sidecar_arrays_sha256",
    ):
        _sha256(evidence.get(name), f"missing A3 R1 evidence SHA: {name}")
    immutable = contract.get("immutable_inputs") or {}
    for name in (
        "protocol_sha256",
        "source_config_sha256",
        "source_checkpoint_sha256",
        "actor_registry_sha256",
    ):
        _sha256(immutable.get(name), f"missing A3 R1 immutable SHA: {name}")
    _require(
        immutable.get("source_checkpoint_sha256")
        == "1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c",
        "A3 R1 numeric source checkpoint drift",
    )
    execution = contract.get("execution") or {}
    _require(
        execution.get("scene") == "scene-0230"
        and execution.get("scene_index") == 179
        and execution.get("seed") == 0,
        "A3 R1 numeric scene/seed drift",
    )
    _require(
        execution.get("source_checkpoint_step") == 30_000
        and execution.get("optimizer_steps") == 4
        and execution.get("output_checkpoint_step") == 30_004,
        "A3 R1 numeric step budget drift",
    )
    _require(
        execution.get("unit_order")
        == [
            "high-support::lateral",
            "high-support::delete",
            "boundary-support::lateral",
            "boundary-support::delete",
        ],
        "A3 R1 unit order drift",
    )
    learning_rates = execution.get("learning_rates") or {}
    _require(
        set(learning_rates) == MUTABLE_FIELDS["r1-reactivate"],
        "A3 R1 learning-rate fields drift",
    )
    _require(
        float(learning_rates["Background._opacities"].get("value", -1)) == 0.05
        and float(learning_rates["Background._scales"].get("value", -1)) == 0.005,
        "A3 R1 learning-rate value drift",
    )
    _require(
        float(execution.get("first_hit_alpha_threshold", -1)) == 0.5
        and execution.get("maximum_seed_gaussians") == 0,
        "A3 R1 alpha/seed budget drift",
    )
    caps = contract.get("authorization_caps") or {}
    _require(
        caps
        == {
            "maximum_affected_background_gaussians": 16_502,
            "maximum_mutable_background_gaussians": 51,
            "s_a_rgb_loss_pixels": 0,
            "s_b_t0_geometry_loss_pixels": 8,
            "s_c_abstain_background_gaussians": 16_451,
        },
        "A3 R1 authorization caps drift",
    )
    observed = contract.get("observed_resources") or {}
    ceilings = contract.get("replay_resource_ceilings") or {}
    for name in (
        "wall_time_seconds",
        "peak_gpu_memory_mib",
        "peak_cgroup_memory_bytes",
        "run_bytes",
        "checkpoint_bytes",
    ):
        observed_name = (
            "peak_cgroup_memory_bytes_sampled"
            if name == "peak_cgroup_memory_bytes"
            else name
        )
        _require(
            float(observed.get(observed_name, -1)) <= float(ceilings.get(name, -2)),
            f"A3 R1 observed resource exceeds ceiling: {name}",
        )
    _require(
        observed.get("oom_events") == ceilings.get("oom_events") == 0
        and observed.get("oom_kill_events") == ceilings.get("oom_kill_events") == 0,
        "A3 R1 OOM budget drift",
    )
    required = contract.get("required_audits") or {}
    _require(required and all(value is True for value in required.values()), "A3 R1 audit gate drift")
    scope = contract.get("scope") or {}
    _require(
        scope.get("valid_for") == "engineering_replay_only"
        and scope.get("support_mode") == "conservative_S_B_T0_only"
        and scope.get("s_a_status") == "ABSTAIN_NOT_MATERIALIZED"
        and scope.get("r2_r3_r4_authorized") is False
        and scope.get("formal_training_authorized") is False,
        "A3 R1 numeric scope drift",
    )


def validate_a3_r1_eval_protocol(contract: Mapping[str, Any]) -> None:
    """Validate the result-blind, read-only R0/R1 heldout evaluation contract."""

    _require(contract.get("schema_version") == 1, "unsupported A3 R1 eval schema")
    _require(contract.get("task_id") == TASK_ID, "A3 R1 eval task drift")
    _require(
        contract.get("audit_version") == R1_EVAL_AUDIT_VERSION,
        "A3 R1 eval audit drift",
    )
    _require(
        contract.get("protocol_status") == "frozen_before_reading_heldout_results",
        "A3 R1 eval protocol is not result-blind frozen",
    )
    _require(
        contract.get("formal_training_authorized") is False
        and contract.get("quality_claim_authorized") is False,
        "A3 R1 eval cannot authorize formal training or quality claims",
    )
    dependencies = contract.get("depends_on") or {}
    for name in (
        "main_protocol_sha256",
        "numeric_freeze_sha256",
        "sidecar_manifest_sha256",
        "r0_checkpoint_sha256",
        "r1_checkpoint_sha256",
    ):
        _sha256(dependencies.get(name), f"missing A3 R1 eval dependency SHA: {name}")
    _require(
        dependencies.get("main_protocol_sha256")
        == "03fbf632645326692bbcf18ab18a08b5440c7733c709f925945c78018bb272d0"
        and dependencies.get("numeric_freeze_sha256")
        == "d9289df0b2ac7df7a7c408b5cb1601bc5f874e2922ebc9cb87961aacee43b3e3"
        and dependencies.get("r0_checkpoint_sha256")
        == "1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c"
        and dependencies.get("r1_checkpoint_sha256")
        == "e995e7c266d9fed4e64c86813718e46ab4576bbfdf60500a637bdaeaaba78cd1",
        "A3 R1 eval checkpoint/numeric dependency drift",
    )
    matrix = contract.get("matrix") or {}
    _require(
        matrix.get("scene") == "scene-0230"
        and matrix.get("scene_index") == 179
        and matrix.get("seed") == 0,
        "A3 R1 eval scene/seed drift",
    )
    _require(
        matrix.get("variants")
        == ["r0-no-refine-exact-alias", "r1-reactivate"]
        and matrix.get("actor_roles") == ["high-support", "boundary-support"]
        and matrix.get("edits") == ["lateral", "delete"]
        and matrix.get("cameras") == [0, 1, 2]
        and matrix.get("heldout_frames") == HELDOUT_FRAMES,
        "A3 R1 eval matrix drift",
    )
    _require(
        matrix.get("actor_frame_eligibility")
        == "source_checkpoint_instances_fv_true"
        and matrix.get("invalid_actor_frame_policy")
        == "exclude_and_report_count",
        "A3 R1 eval actor eligibility drift",
    )
    masks = contract.get("fixed_masks") or {}
    source = masks.get("source_actor_footprint") or {}
    edited = masks.get("edited_actor_footprint") or {}
    affected = masks.get("affected_pixel_mask") or {}
    _require(
        masks.get("source_variant") == "r0-no-refine-exact-alias"
        and source.get("max_uint8_difference_threshold") == 2
        and source.get("dilation_pixels") == 2
        and edited.get("max_uint8_difference_threshold") == 2
        and edited.get("dilation_pixels") == 2
        and affected.get("dilation_pixels") == 3,
        "A3 R1 eval fixed-mask morphology drift",
    )
    _require(
        (masks.get("s_b_t0_mask") or {}).get("variant_independent") is True
        and (masks.get("non_target_mask") or {}).get("variant_independent") is True
        and masks.get("empty_mask_policy")
        == "exclude_metric_unit_and_report_coverage",
        "A3 R1 eval mask independence/missing policy drift",
    )
    depth = contract.get("typed_depth") or {}
    _require(
        (depth.get("depth_lidar_measured") or {}).get("truth_tier") == "T0"
        and (depth.get("depth_surface_first_hit") or {}).get("truth_tier") == "T1"
        and float((depth.get("depth_surface_first_hit") or {}).get("alpha_threshold", -1)) == 0.5
        and (depth.get("depth_render_expected") or {}).get("truth_tier") == "diagnostic"
        and float(depth.get("depth_order_tolerance_m", -1)) == 0.05,
        "A3 R1 eval typed-depth drift",
    )
    endpoints = contract.get("endpoints") or {}
    _require(
        endpoints.get("rgb_comparison_encoding")
        == "round_clip_0_1_times_255_uint8"
        and endpoints.get("rgb_mse_normalization")
        == "squared_uint8_error_divided_by_65025",
        "A3 R1 eval RGB comparison precision drift",
    )
    directions = {
        "s_b_first_hit_valid_coverage": "higher",
        "s_b_depth_order_violation_rate": "lower",
        "s_b_t0_first_hit_mae_m": "lower",
        "non_target_observed_rgb_mse": "lower",
        "original_global_observed_rgb_mse": "lower",
    }
    _require(
        all((endpoints.get(name) or {}).get("direction") == direction for name, direction in directions.items()),
        "A3 R1 eval endpoint direction drift",
    )
    aggregation = contract.get("aggregation") or {}
    _require(
        aggregation.get("raw_key_order") == ["role", "edit", "frame", "camera", "variant"]
        and aggregation.get("missing_values") == "null_never_zero"
        and aggregation.get("no_posthoc_camera_actor_or_frame_reweighting") is True,
        "A3 R1 eval aggregation drift",
    )
    decision = contract.get("decision") or {}
    _require(
        decision.get("comparator") == "exact_no_numeric_tolerance"
        and decision.get("pass_rule")
        == "all_primary_axes_non_worse_and_at_least_one_strictly_better"
        and decision.get("tradeoff_rule") == "neither_variant_dominates"
        and decision.get("zero_s_b_coverage")
        == "insufficient_evidence_not_success",
        "A3 R1 eval decision drift",
    )
    audits = contract.get("required_audits") or {}
    _require(audits and all(value is True for value in audits.values()), "A3 R1 eval audit gate drift")
    ceilings = contract.get("resource_ceilings") or {}
    _require(
        ceilings.get("wall_time_seconds") == 900
        and ceilings.get("peak_gpu_memory_mib") == 12_288
        and ceilings.get("peak_cgroup_memory_bytes") == 34_359_738_368
        and ceilings.get("oom_events_delta") == 0
        and ceilings.get("oom_kill_events_delta") == 0,
        "A3 R1 eval resource ceiling drift",
    )
    claims = contract.get("claim_boundary") or {}
    _require(
        claims.get("s_a_status") == "ABSTAIN_NOT_MATERIALIZED"
        and claims.get("s_a_rgb_quality_claim") == "forbidden"
        and claims.get("s_b_rgb_psnr_ssim_lpips_claim") == "forbidden"
        and claims.get("s_c_update_seed_loss_or_success_claim") == "forbidden"
        and claims.get("formal_training_authorized") is False
        and claims.get("r2_r3_r4_authorized") is False,
        "A3 R1 eval claim boundary drift",
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


def measured_background_support_mask(
    *,
    affected_mask: np.ndarray,
    source_actor_footprint: np.ndarray,
    edited_actor_footprint: np.ndarray,
    depth_lidar_measured: np.ndarray,
) -> np.ndarray:
    """Authorize conservative S-B pixels with direct non-actor LiDAR support."""

    affected = np.asarray(affected_mask, dtype=bool)
    source = np.asarray(source_actor_footprint, dtype=bool)
    edited = np.asarray(edited_actor_footprint, dtype=bool)
    measured = np.asarray(depth_lidar_measured, dtype=np.float32)
    if not (
        affected.shape == source.shape == edited.shape == measured.shape
        and affected.ndim == 2
    ):
        raise ValueError("A3 measured-support arrays must share shape [H, W]")
    valid = np.isfinite(measured) & (measured > 0)
    return affected & ~source & ~edited & valid


def projected_background_rows(
    *,
    means2d: Tensor,
    radii: Tensor,
    pixel_mask: Tensor | np.ndarray,
    background_point_count: int,
) -> Tensor:
    """Map an evidence-view pixel mask to stable Background checkpoint rows."""

    if means2d.ndim == 3:
        if means2d.shape[0] != 1:
            raise ValueError("A3 projection currently requires one camera")
        means2d = means2d[0]
    if radii.ndim == 2:
        if radii.shape[0] != 1:
            raise ValueError("A3 radii currently require one camera")
        radii = radii[0]
    if means2d.ndim != 2 or means2d.shape[-1] != 2:
        raise ValueError("A3 means2d must have shape [N, 2]")
    if radii.ndim != 1 or radii.shape[0] != means2d.shape[0]:
        raise ValueError("A3 radii must align with means2d")
    if not 0 < background_point_count <= means2d.shape[0]:
        raise ValueError("invalid A3 Background point count")
    mask = torch.as_tensor(pixel_mask, dtype=torch.bool, device=means2d.device)
    if mask.ndim != 2:
        raise ValueError("A3 projected evidence mask must have shape [H, W]")
    height, width = mask.shape
    centers = means2d[:background_point_count]
    visible_radii = radii[:background_point_count]
    x, y = centers[:, 0], centers[:, 1]
    visible = (
        (visible_radii > 0)
        & torch.isfinite(x)
        & torch.isfinite(y)
        & (x >= 0)
        & (x <= width - 1)
        & (y >= 0)
        & (y <= height - 1)
    )
    indices = torch.where(visible)[0]
    rows = torch.zeros(
        background_point_count, dtype=torch.bool, device=means2d.device
    )
    if indices.numel() == 0:
        return rows
    pixel_x = torch.floor(x[indices] + 0.5).to(torch.long)
    pixel_y = torch.floor(y[indices] + 0.5).to(torch.long)
    rows[indices] = mask[pixel_y, pixel_x]
    return rows


def merge_s_b_row_observations(
    observations: list[tuple[Tensor, Tensor]],
    *,
    background_point_count: int,
) -> tuple[Tensor, Tensor]:
    """Merge view-local affected/S-B rows; all remaining affected rows are S-C."""

    if not observations:
        raise ValueError("A3 row observations cannot be empty")
    affected = torch.zeros(background_point_count, dtype=torch.bool)
    supported = torch.zeros_like(affected)
    for affected_rows, supported_rows in observations:
        left = torch.as_tensor(affected_rows, dtype=torch.bool).cpu().flatten()
        right = torch.as_tensor(supported_rows, dtype=torch.bool).cpu().flatten()
        if left.shape != affected.shape or right.shape != affected.shape:
            raise ValueError("A3 row observation count drift")
        if torch.any(right & ~left):
            raise ValueError("A3 supported rows must be affected")
        affected |= left
        supported |= right
    strata = torch.full((background_point_count,), 2, dtype=torch.uint8)
    strata[supported] = 1
    if not bool(supported.any()):
        raise ValueError("A3 S-B sidecar authorizes no rows")
    return affected, strata


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


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for an immutable A3 input."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_a3_sidecar_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol_sha256: str,
    checkpoint_sha256: str,
) -> None:
    """Validate the compact R1 row/provenance manifest without loading arrays."""

    _require(manifest.get("schema_version") == 1, "unsupported A3 sidecar schema")
    _require(manifest.get("task_id") == TASK_ID, "A3 sidecar task drift")
    _require(
        manifest.get("audit_version") == SIDECAR_AUDIT_VERSION,
        "A3 sidecar audit drift",
    )
    _require(manifest.get("variant") == "r1-reactivate", "A3 sidecar variant drift")
    _require(
        manifest.get("formal_training_authorized") is False,
        "A3 sidecar cannot authorize formal training",
    )
    _sha256(protocol_sha256, "invalid expected A3 protocol SHA")
    _sha256(checkpoint_sha256, "invalid expected A3 checkpoint SHA")
    _require(
        manifest.get("protocol_sha256") == protocol_sha256,
        "A3 sidecar protocol SHA drift",
    )
    _require(
        manifest.get("checkpoint_sha256") == checkpoint_sha256,
        "A3 sidecar checkpoint SHA drift",
    )
    count = manifest.get("background_point_count")
    _require(isinstance(count, int) and count > 0, "invalid background point count")
    arrays = manifest.get("arrays") or {}
    _require(bool(arrays.get("path")), "missing A3 sidecar arrays path")
    _sha256(arrays.get("sha256"), "missing A3 sidecar arrays SHA")
    _require(
        arrays.get("affected_rows_key") == "affected_background_rows"
        and arrays.get("support_strata_key") == "support_strata_codes",
        "A3 sidecar array key drift",
    )
    evidence = manifest.get("evidence") or {}
    _require(
        evidence.get("support_provenance_complete") is True,
        "A3 support provenance is incomplete",
    )
    _require(
        evidence.get("heldout_frames") == HELDOUT_FRAMES
        and evidence.get("heldout_excluded_from_support") is True,
        "A3 held-out leakage boundary drift",
    )
    _require(
        evidence.get("typed_depth_truth_tiers")
        == {
            "depth_render_expected": "diagnostic",
            "depth_surface_first_hit": "T1",
            "depth_lidar_measured": "T0",
        },
        "A3 typed-depth evidence drift",
    )


@dataclass(frozen=True)
class LoadedRefinementSidecar:
    """Validated row authorization loaded from an immutable NPZ sidecar."""

    manifest_path: Path
    arrays_path: Path
    arrays_sha256: str
    affected_rows: Tensor
    support_strata_codes: Tensor
    mutable_rows: Tensor


def load_a3_refinement_sidecar(
    manifest_path: str | Path,
    *,
    protocol_sha256: str,
    checkpoint_sha256: str,
    device: torch.device | str | None = None,
) -> LoadedRefinementSidecar:
    """Load R1 row authorization and fail closed on any provenance drift."""

    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_a3_sidecar_manifest(
        manifest,
        protocol_sha256=protocol_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    arrays_path = Path(manifest["arrays"]["path"])
    if not arrays_path.is_absolute():
        arrays_path = (path.parent / arrays_path).resolve()
    actual_sha256 = sha256_file(arrays_path)
    _require(
        actual_sha256 == manifest["arrays"]["sha256"],
        "A3 sidecar arrays SHA drift",
    )
    with np.load(arrays_path, allow_pickle=False) as arrays:
        _require(
            set(arrays.files)
            == {"affected_background_rows", "support_strata_codes"},
            "A3 sidecar arrays changed",
        )
        affected = np.asarray(arrays["affected_background_rows"])
        strata_codes = np.asarray(arrays["support_strata_codes"])
    count = int(manifest["background_point_count"])
    _require(
        affected.shape == (count,) and affected.dtype == np.bool_,
        "affected Background rows must be a bool vector",
    )
    _require(
        strata_codes.shape == (count,)
        and np.issubdtype(strata_codes.dtype, np.integer),
        "support strata codes must be an integer vector",
    )
    _require(
        bool(np.isin(strata_codes, [0, 1, 2]).all()),
        "unknown A3 support stratum code",
    )
    affected_tensor = torch.as_tensor(affected, dtype=torch.bool, device=device)
    strata_tensor = torch.as_tensor(
        strata_codes, dtype=torch.uint8, device=device
    )
    mutable = affected_tensor & (strata_tensor != 2)
    _require(bool(mutable.any()), "A3 R1 sidecar authorizes no mutable rows")
    return LoadedRefinementSidecar(
        manifest_path=path,
        arrays_path=arrays_path,
        arrays_sha256=actual_sha256,
        affected_rows=affected_tensor,
        support_strata_codes=strata_tensor,
        mutable_rows=mutable,
    )


def _clone_optimizer_state_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().clone()
    return value


def snapshot_optimizer_state(
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, Tensor],
) -> dict[str, dict[str, Any]]:
    """Clone per-parameter optimizer state using stable protocol field names."""

    return {
        name: {
            key: _clone_optimizer_state_value(value)
            for key, value in optimizer.state.get(parameter, {}).items()
        }
        for name, parameter in parameters.items()
    }


def audit_optimizer_frozen_rows(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    *,
    parameter_shapes: Mapping[str, torch.Size],
    mutable_rows: Tensor,
    mutable_fields: set[str],
) -> dict[str, Any]:
    """Audit Adam moments exactly outside authorized Background rows."""

    if set(before) != set(after) or set(before) != set(parameter_shapes):
        raise ValueError("optimizer parameter fields changed")
    rows = torch.as_tensor(mutable_rows, dtype=torch.bool).flatten()
    checks: dict[str, dict[str, Any]] = {}
    for name in sorted(before):
        left = before[name]
        right = after[name]
        authorized = name in mutable_fields
        keys_exact = set(left) == set(right)
        state_exact = keys_exact
        if keys_exact:
            for key in left:
                left_value = left[key]
                right_value = right[key]
                if isinstance(left_value, Tensor) != isinstance(right_value, Tensor):
                    state_exact = False
                    break
                if not isinstance(left_value, Tensor):
                    if left_value != right_value:
                        state_exact = False
                        break
                    continue
                if left_value.shape != right_value.shape or left_value.dtype != right_value.dtype:
                    state_exact = False
                    break
                if (
                    authorized
                    and tuple(left_value.shape) == tuple(parameter_shapes[name])
                ):
                    field_rows = rows.to(device=left_value.device)
                    if left_value.ndim == 0 or left_value.shape[0] != rows.numel():
                        state_exact = False
                        break
                    if not torch.equal(
                        left_value[~field_rows], right_value[~field_rows]
                    ):
                        state_exact = False
                        break
                elif authorized and key == "step":
                    # Adam owns one scalar step for the authorized tensor.
                    continue
                elif not torch.equal(left_value, right_value):
                    state_exact = False
                    break
        checks[name] = {
            "keys_exact": keys_exact,
            "outside_exact": state_exact,
            "authorized_field": authorized,
        }
    return {
        "checks": checks,
        "pass": all(
            value["keys_exact"] and value["outside_exact"]
            for value in checks.values()
        ),
    }


class LocalRefinementGuard:
    """Fail-closed R1 gradient mask and post-step exactness auditor."""

    def __init__(
        self,
        *,
        parameters: Mapping[str, torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        mutable_rows: Tensor,
        variant: str = "r1-reactivate",
    ) -> None:
        if type(optimizer) is not torch.optim.Adam:
            raise ValueError("A3 R1 currently requires exact torch.optim.Adam")
        self.parameters = dict(parameters)
        self.optimizer = optimizer
        self.mutable_fields = mutable_parameter_fields(variant)
        self.mutable_rows = torch.as_tensor(
            mutable_rows, dtype=torch.bool
        ).flatten()
        _require(bool(self.mutable_rows.any()), "A3 R1 authorizes no rows")
        _require(
            self.mutable_fields.issubset(self.parameters),
            "A3 R1 mutable parameters are missing",
        )
        self.parameter_order = tuple(self.parameters)
        self.parameter_ids = {
            name: id(parameter) for name, parameter in self.parameters.items()
        }
        self.parameter_shapes = {
            name: parameter.shape for name, parameter in self.parameters.items()
        }
        self.parameter_dtypes = {
            name: parameter.dtype for name, parameter in self.parameters.items()
        }
        for name in self.mutable_fields:
            parameter = self.parameters[name]
            _require(
                parameter.ndim > 0
                and parameter.shape[0] == self.mutable_rows.numel(),
                f"A3 mutable row count mismatch: {name}",
            )
        optimizer_parameters = {
            id(parameter): group
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        _require(
            set(self.parameter_ids.values()).issubset(optimizer_parameters),
            "A3 parameter is missing from optimizer",
        )
        for name in self.mutable_fields:
            group = optimizer_parameters[self.parameter_ids[name]]
            _require(
                float(group.get("weight_decay", 0.0)) == 0.0,
                f"A3 mutable field cannot use weight decay: {name}",
            )
            self._initialize_adam_state(self.parameters[name], group)
        self._parameter_before: dict[str, Tensor] | None = None
        self._optimizer_before: dict[str, dict[str, Any]] | None = None

    def _initialize_adam_state(
        self, parameter: torch.nn.Parameter, group: Mapping[str, Any]
    ) -> None:
        state = self.optimizer.state[parameter]
        if state:
            _require(
                {"step", "exp_avg", "exp_avg_sq"}.issubset(state),
                "unexpected pre-existing Adam state",
            )
            return
        capturable = bool(group.get("capturable", False))
        fused = bool(group.get("fused", False))
        step_device = parameter.device if capturable or fused else torch.device("cpu")
        state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)
        state["exp_avg"] = torch.zeros_like(
            parameter, memory_format=torch.preserve_format
        )
        state["exp_avg_sq"] = torch.zeros_like(
            parameter, memory_format=torch.preserve_format
        )
        if bool(group.get("amsgrad", False)):
            state["max_exp_avg_sq"] = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )

    def _assert_structure(self) -> None:
        _require(tuple(self.parameters) == self.parameter_order, "parameter order drift")
        for name, parameter in self.parameters.items():
            _require(id(parameter) == self.parameter_ids[name], f"parameter identity drift: {name}")
            _require(parameter.shape == self.parameter_shapes[name], f"parameter shape drift: {name}")
            _require(parameter.dtype == self.parameter_dtypes[name], f"parameter dtype drift: {name}")

    def before_optimizer_step(self) -> dict[str, Any]:
        """Snapshot exact state, then leave gradients only on authorized rows."""

        _require(self._parameter_before is None, "A3 optimizer step is already open")
        self._assert_structure()
        self._parameter_before = {
            name: parameter.detach().clone()
            for name, parameter in self.parameters.items()
        }
        self._optimizer_before = snapshot_optimizer_state(
            self.optimizer, self.parameters
        )
        gradient_checks: dict[str, dict[str, Any]] = {}
        for name, parameter in self.parameters.items():
            authorized = name in self.mutable_fields
            if not authorized:
                parameter.grad = None
                gradient_checks[name] = {
                    "authorized_field": False,
                    "finite": True,
                    "outside_zero": True,
                }
                continue
            _require(parameter.grad is not None, f"missing A3 gradient: {name}")
            gradient = parameter.grad
            _require(gradient.shape == parameter.shape, f"gradient shape drift: {name}")
            finite = bool(torch.isfinite(gradient).all())
            _require(finite, f"non-finite A3 gradient: {name}")
            rows = self.mutable_rows.to(device=gradient.device)
            view = rows.reshape(rows.shape[0], *([1] * (gradient.ndim - 1)))
            gradient.mul_(view)
            outside_zero = bool(torch.count_nonzero(gradient[~rows]) == 0)
            _require(outside_zero, f"outside gradient mask failed: {name}")
            inside_nonzero = int(torch.count_nonzero(gradient[rows]).item())
            inside_l2 = float(torch.linalg.vector_norm(gradient[rows]).item())
            gradient_checks[name] = {
                "authorized_field": True,
                "finite": finite,
                "outside_zero": outside_zero,
                "inside_nonzero": inside_nonzero,
                "inside_l2": inside_l2,
            }
        return {"pass": True, "checks": gradient_checks}

    def after_optimizer_step(self) -> dict[str, Any]:
        """Audit parameters and Adam moments before allowing another step."""

        _require(
            self._parameter_before is not None
            and self._optimizer_before is not None,
            "A3 optimizer step was not opened",
        )
        self._assert_structure()
        after = {
            name: parameter.detach().clone()
            for name, parameter in self.parameters.items()
        }
        parameter_audit = audit_frozen_rows(
            self._parameter_before,
            after,
            mutable_rows=self.mutable_rows,
            mutable_fields=self.mutable_fields,
        )
        optimizer_audit = audit_optimizer_frozen_rows(
            self._optimizer_before,
            snapshot_optimizer_state(self.optimizer, self.parameters),
            parameter_shapes=self.parameter_shapes,
            mutable_rows=self.mutable_rows,
            mutable_fields=self.mutable_fields,
        )
        changed_inside: dict[str, int] = {}
        rows = self.mutable_rows
        for name in sorted(self.mutable_fields):
            before_value = self._parameter_before[name]
            after_value = after[name]
            device_rows = rows.to(device=before_value.device)
            changed_inside[name] = int(
                torch.count_nonzero(
                    before_value[device_rows] != after_value[device_rows]
                ).item()
            )
        result = {
            "pass": bool(parameter_audit["pass"] and optimizer_audit["pass"]),
            "parameter_audit": parameter_audit,
            "optimizer_audit": optimizer_audit,
            "changed_inside_elements": changed_inside,
            "parameter_order": list(self.parameter_order),
        }
        self._parameter_before = None
        self._optimizer_before = None
        if not result["pass"]:
            raise RuntimeError("A3 outside exactness audit failed")
        return result
