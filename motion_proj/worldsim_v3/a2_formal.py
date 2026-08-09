"""Frozen comparison helpers for the WorldSim V3 A2-D1 formal pair."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


TASK_ID = "WS-V3-A2-ACTOR-DENSIFY-01"
AUDIT_VERSION = "A2-D1-FORMAL-v1"
VARIANTS = ("d0-native", "d1-actor-quota")
QUALITY_DIRECTIONS = {
    "global.psnr": "max",
    "global.ssim": "max",
    "global.lpips": "min",
    "high.actor.psnr": "max",
    "high.actor.ssim": "max",
    "high.actor.lpips": "min",
    "high.boundary.psnr": "max",
    "high.boundary.ssim": "max",
    "high.boundary.lpips": "min",
    "boundary.actor.psnr": "max",
    "boundary.actor.ssim": "max",
    "boundary.actor.lpips": "min",
    "boundary.boundary.psnr": "max",
    "boundary.boundary.ssim": "max",
    "boundary.boundary.lpips": "min",
    "non_target.psnr": "max",
    "non_target.ssim": "max",
    "non_target.lpips": "min",
    "non_target.mae": "min",
}
COST_DIRECTIONS = {
    "cost.rigid_gaussians": "min",
    "cost.total_gaussians": "min",
    "cost.train_seconds": "min",
    "cost.peak_gpu_mib": "min",
    "cost.peak_cgroup_bytes": "min",
}


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def validate_a2_d1_formal_contract(contract: Mapping[str, Any]) -> None:
    """Fail closed if the preregistered formal protocol drifts."""

    _require(contract.get("schema_version") == 1, "unsupported formal schema")
    _require(contract.get("task_id") == TASK_ID, "formal task ID drift")
    _require(contract.get("audit_version") == AUDIT_VERSION, "audit version drift")
    design = contract.get("paired_design") or {}
    _require(design.get("scene") == "scene-0230", "formal scene drift")
    _require(design.get("scene_index") == 179, "formal scene index drift")
    _require(design.get("seed") == 0, "formal seed drift")
    _require(tuple(design.get("order") or ()) == VARIANTS, "formal arm order drift")
    _require(design.get("num_iters") == 30_000, "formal fixed-step budget drift")
    _require(
        design.get("checkpoint_interval") == 5_000,
        "formal checkpoint interval drift",
    )
    _require(design.get("test_image_stride") == 10, "held-out split drift")

    matched = contract.get("matched_gaussian_budget") or {}
    _require(matched.get("scope") == "RigidNodes", "matched budget scope drift")
    _require(
        matched.get("target") == "d0_final_30000_rigid_gaussians",
        "matched target drift",
    )
    _require(
        matched.get("candidate_steps")
        == [5_000, 10_000, 15_000, 20_000, 25_000, 30_000],
        "matched candidate grid drift",
    )
    _require(
        matched.get("selection") == "minimum_absolute_gap_then_earliest_step",
        "matched selection drift",
    )
    _require(
        float(matched.get("maximum_relative_gap", -1.0)) == 0.02,
        "matched relative-gap gate drift",
    )
    _require(
        matched.get("failure_status") == "ABSTAIN_BUDGET_NOT_MATCHED",
        "matched failure status drift",
    )
    _require(
        matched.get("posthoc_pruning") is False
        and matched.get("retraining") is False
        and matched.get("quota_retuning") is False,
        "matched view must remain read-only",
    )

    evaluation = contract.get("evaluation") or {}
    _require(evaluation.get("formal_full_split") is True, "formal split must be full")
    _require(
        tuple(evaluation.get("actor_roles") or ())
        == ("high-support", "boundary-support"),
        "actor role drift",
    )
    _require(
        evaluation.get("non_target_definition")
        == "complement_of_selected_actor_counterfactual_union",
        "non-target definition drift",
    )
    _require(
        evaluation.get("quality_directions") == QUALITY_DIRECTIONS,
        "quality-axis contract drift",
    )

    resources = contract.get("resource_contract") or {}
    _require(resources.get("execution") == "sequential", "execution order drift")
    _require(resources.get("gpu_count") == 1, "GPU count drift")
    _require(resources.get("minimum_free_disk_gib") == 40, "disk gate drift")
    _require(resources.get("memory_stop_fraction") == 0.90, "memory gate drift")
    _require(resources.get("memory_stop_consecutive_samples") == 2, "memory samples drift")
    _require(resources.get("monitoring_interval_seconds") == 10, "monitor interval drift")
    _require(resources.get("gpu_idle_max_mib") == 2048, "GPU idle gate drift")

    forbidden = set((contract.get("scope_boundary") or {}).get("forbidden") or ())
    _require(
        forbidden
        == {
            "boundary_weighting",
            "photometric_residual_weighting",
            "depth_residual_weighting",
            "normal_residual_weighting",
            "gaussian_scale_cap",
            "lidar_distance_weighting",
            "visibility_weighting",
        },
        "D1 forbidden-factor boundary drift",
    )


def select_matched_checkpoint(
    target_rigid_gaussians: int,
    candidates: Sequence[Mapping[str, Any]],
    maximum_relative_gap: float,
) -> dict[str, Any]:
    """Select the preregistered closest D1 checkpoint without modifying it."""

    if target_rigid_gaussians <= 0:
        raise ValueError("matched target must be positive")
    if not candidates:
        raise ValueError("matched selection requires candidates")
    normalized: list[dict[str, Any]] = []
    for row in candidates:
        step = int(row["step"])
        count = int(row["rigid_gaussians"])
        if step <= 0 or count <= 0:
            raise ValueError("candidate step and Gaussian count must be positive")
        absolute_gap = abs(count - target_rigid_gaussians)
        normalized.append(
            {
                **dict(row),
                "step": step,
                "rigid_gaussians": count,
                "absolute_gap": absolute_gap,
                "relative_gap": absolute_gap / target_rigid_gaussians,
            }
        )
    selected = min(
        normalized,
        key=lambda row: (
            int(row["absolute_gap"]),
            int(row["step"]),
            str(row.get("checkpoint", "")),
        ),
    )
    matched = float(selected["relative_gap"]) <= float(maximum_relative_gap)
    return {
        "status": "done" if matched else "ABSTAIN_BUDGET_NOT_MATCHED",
        "scope": "RigidNodes",
        "target_rigid_gaussians": target_rigid_gaussians,
        "maximum_relative_gap": float(maximum_relative_gap),
        "selection": "minimum_absolute_gap_then_earliest_step",
        "selected": selected,
        "candidates": sorted(normalized, key=lambda row: int(row["step"])),
        "read_only": True,
    }


def _float(value: Any, label: str) -> float:
    if value is None:
        raise ValueError(f"missing metric: {label}")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite metric: {label}")
    return result


def quality_vector(evaluation: Mapping[str, Any]) -> dict[str, float]:
    heldout = evaluation["heldout_metrics"]
    actor = evaluation["actor_metrics"]
    roles = actor["roles"]
    high = roles["high-support"]
    boundary = roles["boundary-support"]
    non_target = actor["non_target"]["quality"]
    values: dict[str, float] = {
        "global.psnr": _float(heldout["image_metrics/test/psnr"], "global.psnr"),
        "global.ssim": _float(heldout["image_metrics/test/ssim"], "global.ssim"),
        "global.lpips": _float(heldout["image_metrics/test/lpips"], "global.lpips"),
        "non_target.psnr": _float(non_target["psnr"], "non_target.psnr"),
        "non_target.ssim": _float(non_target["ssim"], "non_target.ssim"),
        "non_target.lpips": _float(
            non_target["masked_lpips_alex_tight_crop_256px"], "non_target.lpips"
        ),
        "non_target.mae": _float(
            non_target["mean_absolute_error"], "non_target.mae"
        ),
    }
    for prefix, payload in (("high", high), ("boundary", boundary)):
        if payload.get("status") != "done":
            raise ValueError(f"formal actor role did not complete: {prefix}")
        for region_name, output_name in (
            ("actor_region", "actor"),
            ("boundary_band", "boundary"),
        ):
            region = payload[region_name]
            values[f"{prefix}.{output_name}.psnr"] = _float(
                region["psnr"], f"{prefix}.{output_name}.psnr"
            )
            values[f"{prefix}.{output_name}.ssim"] = _float(
                region["ssim"], f"{prefix}.{output_name}.ssim"
            )
            values[f"{prefix}.{output_name}.lpips"] = _float(
                region["masked_lpips_alex_tight_crop_256px"],
                f"{prefix}.{output_name}.lpips",
            )
    if set(values) != set(QUALITY_DIRECTIONS):
        raise ValueError("quality vector does not match the frozen axes")
    return values


def cost_vector(
    checkpoint_audit: Mapping[str, Any], train_resources: Mapping[str, Any]
) -> dict[str, float]:
    rigid = _float(checkpoint_audit["rigid_total"], "cost.rigid_gaussians")
    background = _float(
        checkpoint_audit["background_total"], "cost.background_gaussians"
    )
    return {
        "cost.rigid_gaussians": rigid,
        "cost.total_gaussians": rigid + background,
        "cost.train_seconds": _float(
            train_resources["duration_seconds"], "cost.train_seconds"
        ),
        "cost.peak_gpu_mib": _float(
            train_resources["peak_gpu_memory_mib_sampled"], "cost.peak_gpu_mib"
        ),
        "cost.peak_cgroup_bytes": _float(
            train_resources["peak_cgroup_memory_bytes"], "cost.peak_cgroup_bytes"
        ),
    }


def compare_vectors(
    d0: Mapping[str, float],
    d1: Mapping[str, float],
    directions: Mapping[str, str],
) -> dict[str, Any]:
    """Perform an exact, no-tolerance pairwise Pareto comparison."""

    if set(d0) != set(directions) or set(d1) != set(directions):
        raise ValueError("comparison vector axes do not match the contract")
    deltas: dict[str, dict[str, Any]] = {}
    d1_better = 0
    d0_better = 0
    for key, direction in directions.items():
        left = float(d0[key])
        right = float(d1[key])
        if direction not in {"min", "max"}:
            raise ValueError(f"unknown direction for {key}: {direction}")
        signed = right - left
        improvement = signed if direction == "max" else -signed
        relation = "equal"
        if improvement > 0:
            relation = "d1_better"
            d1_better += 1
        elif improvement < 0:
            relation = "d0_better"
            d0_better += 1
        deltas[key] = {
            "d0": left,
            "d1": right,
            "raw_delta_d1_minus_d0": signed,
            "direction": direction,
            "relation": relation,
        }
    if d1_better and not d0_better:
        verdict = "d1_strictly_dominates_d0"
    elif d0_better and not d1_better:
        verdict = "d0_strictly_dominates_d1"
    elif d0_better and d1_better:
        verdict = "tradeoff_non_dominated"
    else:
        verdict = "identical"
    return {
        "verdict": verdict,
        "d1_better_axis_count": d1_better,
        "d0_better_axis_count": d0_better,
        "equal_axis_count": len(directions) - d1_better - d0_better,
        "no_numeric_tolerance": True,
        "axes": deltas,
    }


def compare_view(
    d0_evaluation: Mapping[str, Any],
    d1_evaluation: Mapping[str, Any],
    d0_checkpoint: Mapping[str, Any],
    d1_checkpoint: Mapping[str, Any],
    d0_resources: Mapping[str, Any],
    d1_resources: Mapping[str, Any],
) -> dict[str, Any]:
    d0_quality = quality_vector(d0_evaluation)
    d1_quality = quality_vector(d1_evaluation)
    d0_cost = cost_vector(d0_checkpoint, d0_resources)
    d1_cost = cost_vector(d1_checkpoint, d1_resources)
    return {
        "quality": compare_vectors(d0_quality, d1_quality, QUALITY_DIRECTIONS),
        "quality_cost_pareto": compare_vectors(
            {**d0_quality, **d0_cost},
            {**d1_quality, **d1_cost},
            {**QUALITY_DIRECTIONS, **COST_DIRECTIONS},
        ),
    }
