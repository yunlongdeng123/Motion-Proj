#!/usr/bin/env python3
"""Freeze the nuScenes M2 development router and matched repair table."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "WS-V4-M2-REPAIR-ROUTER-01"
RISK_KEYS = ("photo", "geometry", "temporal", "uncertainty", "compute_cost")
CANDIDATE_RISK_KEYS = {
    "photo": "photo_risk",
    "geometry": "geometry_risk",
    "temporal": "temporal_risk",
    "uncertainty": "uncertainty",
    "compute_cost": "compute_cost",
}
QUALITY_METRICS = (
    "global_valid_psnr_db",
    "global_valid_ssim",
    "global_valid_lpips_alex",
    "hole_cross_view_psnr_db",
    "hole_geometry_mae_m",
    "hole_coverage",
    "static_lidar_depth_mae_m",
    "edit_error",
)


class M2DevelopmentAggregationError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M2DevelopmentAggregationError(f"JSON root is not a mapping: {path}")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M2DevelopmentAggregationError(f"YAML root is not a mapping: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"], text=True
        ).strip()
    )


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        raise M2DevelopmentAggregationError("mean requires at least one value")
    return float(math.fsum(rows) / len(rows))


def _metric_aggregate(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, Any]:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _finite(row["metrics"].get(metric))
        if value is not None:
            by_scene[str(row["scene"])].append(value)
    scene_means = {
        scene: _mean(values) for scene, values in sorted(by_scene.items())
    }
    request_values = [value for values in by_scene.values() for value in values]
    return {
        "request_mean": _mean(request_values) if request_values else None,
        "request_count": len(request_values),
        "scene_balanced_mean": _mean(scene_means.values()) if scene_means else None,
        "scene_count": len(scene_means),
        "scene_means": scene_means,
    }


def _aggregate_quality(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {metric: _metric_aggregate(rows, metric) for metric in QUALITY_METRICS}


def _abstain_metrics(request: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        row
        for row in request["matched_arms"]
        if row.get("arm") == "ABSTAIN" and row.get("status") == "atomic_noop"
    ]
    if len(matches) != 1 or matches[0].get("metrics", {}).get("atomic_noop") is not True:
        raise M2DevelopmentAggregationError(
            f"request lacks one atomic ABSTAIN row: {request.get('request_id')}"
        )
    return matches[0]["metrics"]


def _score_candidate(
    candidate_record: Mapping[str, Any], weights: Mapping[str, float]
) -> float:
    candidate = candidate_record["candidate"]
    return float(
        math.fsum(
            float(weights[key]) * float(candidate[CANDIDATE_RISK_KEYS[key]])
            for key in RISK_KEYS
        )
    )


def _route_request(
    request: Mapping[str, Any],
    *,
    weights: Mapping[str, float],
    threshold: float,
    tie_priority: Sequence[str],
) -> dict[str, Any]:
    priority = {method: index for index, method in enumerate(tie_priority)}
    candidates = list(request["candidates"])
    if not candidates:
        r0 = dict(_abstain_metrics(request))
        return {
            "request_id": request["request_id"],
            "accepted": False,
            "selected_arm": None,
            "selected_method": None,
            "selected_candidate_id": None,
            "selected_score": None,
            "normalized_selected_risk": 1.0,
            "counterfactual_edit_error": float(r0["edit_error"]),
            "policy_arm": "ABSTAIN",
            "metrics": r0,
            "scores": [],
        }
    scored = [
        {
            "record": row,
            "score": _score_candidate(row, weights),
            "priority": priority[str(row["candidate"]["method"])],
        }
        for row in candidates
    ]
    scored.sort(
        key=lambda row: (
            row["score"],
            row["priority"],
            str(row["record"]["candidate"]["candidate_id"]),
        )
    )
    selected = scored[0]
    record = selected["record"]
    score = float(selected["score"])
    accepted = score <= float(threshold)
    maximum_score = float(math.fsum(float(weights[key]) for key in RISK_KEYS))
    if maximum_score <= 0.0:
        raise M2DevelopmentAggregationError("risk weights have zero total")
    metrics = dict(record["metrics"] if accepted else _abstain_metrics(request))
    return {
        "request_id": request["request_id"],
        "accepted": accepted,
        "selected_arm": record["arm"],
        "selected_method": record["candidate"]["method"],
        "selected_candidate_id": record["candidate"]["candidate_id"],
        "selected_score": score,
        "normalized_selected_risk": float(min(score / maximum_score, 1.0)),
        "counterfactual_edit_error": float(record["metrics"]["edit_error"]),
        "policy_arm": record["arm"] if accepted else "ABSTAIN",
        "metrics": metrics,
        "scores": [
            {
                "arm": row["record"]["arm"],
                "candidate_id": row["record"]["candidate"]["candidate_id"],
                "method": row["record"]["candidate"]["method"],
                "score": float(row["score"]),
            }
            for row in scored
        ],
    }


def _selection_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["accepted"]]
    abstained = [row for row in rows if not row["accepted"]]
    accepted_error = (
        _mean(float(row["counterfactual_edit_error"]) for row in accepted)
        if accepted
        else None
    )
    abstained_error = (
        _mean(float(row["counterfactual_edit_error"]) for row in abstained)
        if abstained
        else None
    )
    separation = (
        float(abstained_error - accepted_error)
        if accepted_error is not None and abstained_error is not None
        else None
    )
    policy_error = _metric_aggregate(rows, "edit_error")
    return {
        "request_count": len(rows),
        "accepted_count": len(accepted),
        "abstain_count": len(abstained),
        "request_coverage": len(accepted) / len(rows),
        "accepted_counterfactual_mean_error": accepted_error,
        "abstained_counterfactual_mean_error": abstained_error,
        "abstain_minus_accepted_counterfactual_error": separation,
        "meaningful_abstention_gate": separation is not None and separation > 0.0,
        "request_mean_policy_edit_error": policy_error["request_mean"],
        "scene_balanced_policy_edit_error": policy_error["scene_balanced_mean"],
        "accepted_arm_counts": dict(
            sorted(Counter(row["policy_arm"] for row in accepted).items())
        ),
    }


def select_development_operating_point(
    requests: Sequence[Mapping[str, Any]],
    *,
    weight_grid: Sequence[Mapping[str, Any]],
    threshold_grid: Sequence[float],
    tie_priority: Sequence[str],
    require_meaningful_abstention: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid: list[dict[str, Any]] = []
    for weight_index, spec in enumerate(weight_grid):
        weights = {key: float(spec[key]) for key in RISK_KEYS}
        for threshold_index, threshold in enumerate(threshold_grid):
            decisions = []
            for request in requests:
                row = _route_request(
                    request,
                    weights=weights,
                    threshold=float(threshold),
                    tie_priority=tie_priority,
                )
                row["scene"] = request["scene"]
                decisions.append(row)
            statistics = _selection_statistics(decisions)
            valid = statistics["accepted_count"] > 0 and statistics["abstain_count"] > 0
            if require_meaningful_abstention:
                valid = valid and statistics["meaningful_abstention_gate"]
            grid.append(
                {
                    "weight_name": spec["name"],
                    "weights": weights,
                    "threshold": float(threshold),
                    "weight_grid_index": weight_index,
                    "threshold_grid_index": threshold_index,
                    "valid": valid,
                    "statistics": statistics,
                    "decisions": decisions,
                }
            )
    valid_rows = [row for row in grid if row["valid"]]
    if not valid_rows:
        raise M2DevelopmentAggregationError(
            "no development grid point has both groups and meaningful abstention"
        )
    selected = min(
        valid_rows,
        key=lambda row: (
            row["statistics"]["scene_balanced_policy_edit_error"],
            row["statistics"]["request_mean_policy_edit_error"],
            -row["statistics"]["request_coverage"],
            row["weight_grid_index"],
            row["threshold_grid_index"],
        ),
    )
    return selected, grid


def _fixed_arm_rows(
    requests: Sequence[Mapping[str, Any]], arm: str
) -> list[dict[str, Any]]:
    output = []
    for request in requests:
        match = next((row for row in request["candidates"] if row["arm"] == arm), None)
        if arm == "ABSTAIN" or match is None:
            metrics = dict(_abstain_metrics(request))
            policy_arm = "ABSTAIN"
            available = arm == "ABSTAIN"
        else:
            metrics = dict(match["metrics"])
            policy_arm = arm
            available = True
        output.append(
            {
                "scene": request["scene"],
                "request_id": request["request_id"],
                "available": available,
                "policy_arm": policy_arm,
                "metrics": metrics,
            }
        )
    return output


def _router_policy_rows(decisions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scene": row["scene"],
            "request_id": row["request_id"],
            "available": row["accepted"],
            "policy_arm": row["policy_arm"],
            "metrics": dict(row["metrics"]),
        }
        for row in decisions
    ]


def _matched_table(
    requests: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[str],
    router_decisions: Sequence[Mapping[str, Any]],
    cohort_scene_count: int,
) -> list[dict[str, Any]]:
    output = []
    for arm in arms:
        rows = (
            _router_policy_rows(router_decisions)
            if arm == "RISK_ROUTER"
            else _fixed_arm_rows(requests, arm)
        )
        output.append(
            {
                "arm": arm,
                "requested_count": len(rows),
                "available_or_accepted_count": sum(bool(row["available"]) for row in rows),
                "atomic_abstain_count": sum(row["policy_arm"] == "ABSTAIN" for row in rows),
                "request_coverage": sum(row["policy_arm"] != "ABSTAIN" for row in rows)
                / len(rows),
                "evaluable_scene_count": len({row["scene"] for row in rows}),
                "cohort_scene_count": cohort_scene_count,
                "evaluable_scene_fraction": len({row["scene"] for row in rows})
                / cohort_scene_count,
                "quality": _aggregate_quality(rows),
            }
        )
    return output


def evaluate_acceptance_gates(
    *,
    router: Mapping[str, Any],
    baseline: Mapping[str, Any],
    selective: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    router_quality = router["quality"]
    baseline_quality = baseline["quality"]

    def scene(metric: str, source: Mapping[str, Any]) -> float:
        value = source[metric]["scene_balanced_mean"]
        if value is None:
            raise M2DevelopmentAggregationError(f"gate metric is undefined: {metric}")
        return float(value)

    psnr_delta = scene("global_valid_psnr_db", router_quality) - scene(
        "global_valid_psnr_db", baseline_quality
    )
    ssim_delta = scene("global_valid_ssim", router_quality) - scene(
        "global_valid_ssim", baseline_quality
    )
    lpips_delta = scene("global_valid_lpips_alex", router_quality) - scene(
        "global_valid_lpips_alex", baseline_quality
    )
    static_delta = scene("static_lidar_depth_mae_m", router_quality) - scene(
        "static_lidar_depth_mae_m", baseline_quality
    )
    hole_rows = []
    for metric, direction in gates["hole_endpoints"]["metrics"].items():
        router_value = scene(metric, router_quality)
        baseline_value = scene(metric, baseline_quality)
        improvement = (
            router_value - baseline_value
            if direction == "maximize"
            else baseline_value - router_value
        )
        hole_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "router": router_value,
                "baseline": baseline_value,
                "signed_improvement": improvement,
                "strictly_improved": improvement > 0.0,
            }
        )
    checks = {
        "global_psnr": {
            "value": psnr_delta,
            "threshold": float(gates["global_valid"]["psnr_delta_db_min"]),
            "passed": psnr_delta >= float(gates["global_valid"]["psnr_delta_db_min"]),
        },
        "global_ssim": {
            "value": ssim_delta,
            "threshold": float(gates["global_valid"]["ssim_delta_min"]),
            "passed": ssim_delta >= float(gates["global_valid"]["ssim_delta_min"]),
        },
        "global_lpips": {
            "value": lpips_delta,
            "threshold": float(gates["global_valid"]["lpips_delta_max"]),
            "passed": lpips_delta <= float(gates["global_valid"]["lpips_delta_max"]),
        },
        "hole_any_endpoint": {
            "endpoints": hole_rows,
            "passed": any(row["strictly_improved"] for row in hole_rows),
        },
        "static_lidar": {
            "value": static_delta,
            "threshold": float(gates["static_lidar_depth_mae_degradation_m_max"]),
            "passed": static_delta
            <= float(gates["static_lidar_depth_mae_degradation_m_max"]),
        },
        "meaningful_abstention": {
            "value": selective["abstain_minus_accepted_counterfactual_error"],
            "threshold": 0.0,
            "passed": bool(selective["meaningful_abstention_gate"]),
        },
    }
    return {
        "baseline_arm": baseline["arm"],
        "checks": checks,
        "all_gates_passed": all(bool(row["passed"]) for row in checks.values()),
        "same_view_background_gt_metrics": gates["hole_endpoints"][
            "same_view_background_gt_metrics"
        ],
    }


def _selective_curve(
    decisions: Sequence[Mapping[str, Any]], requested_coverages: Sequence[float]
) -> list[dict[str, Any]]:
    ordered = sorted(
        decisions,
        key=lambda row: (row["normalized_selected_risk"], row["request_id"]),
    )
    output = []
    for requested in requested_coverages:
        retained = max(1, min(len(ordered), math.ceil(float(requested) * len(ordered))))
        rows = ordered[:retained]
        output.append(
            {
                "requested_coverage": float(requested),
                "retained_count": retained,
                "coverage": retained / len(ordered),
                "mean_counterfactual_edit_error": _mean(
                    float(row["counterfactual_edit_error"]) for row in rows
                ),
                "maximum_normalized_selected_risk": rows[-1][
                    "normalized_selected_risk"
                ],
            }
        )
    return output


def _verify_binding(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise M2DevelopmentAggregationError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise M2DevelopmentAggregationError(
            f"{label} SHA drift: expected={expected} actual={actual}"
        )


def _load_verified_inputs(
    config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol_binding = config["candidate_protocol"]
    protocol_path = Path(protocol_binding["path"])
    _verify_binding(protocol_path, protocol_binding["sha256"], "candidate protocol")
    protocol = _yaml(protocol_path)
    if protocol.get("task_id") != TASK_ID or protocol.get("status") != "pending":
        raise M2DevelopmentAggregationError("candidate protocol identity drift")
    expected_scenes = list(protocol["protocol"]["development_scenes"])
    if list(config["scene_runs"]) != expected_scenes:
        raise M2DevelopmentAggregationError("selection scene order is not frozen cohort")
    summaries: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for scene in expected_scenes:
        binding = config["scene_runs"][scene]
        summary_path = Path(binding["path"]) / "summary.json"
        _verify_binding(summary_path, binding["summary_sha256"], f"{scene} summary")
        summary = _json(summary_path)
        if (
            summary.get("task_id") != TASK_ID
            or summary.get("scene") != scene
            or summary.get("heldout_content_read") is not False
            or summary.get("test_quality_read") is not False
            or summary.get("project_git_head") != protocol_binding["project_git_head"]
        ):
            raise M2DevelopmentAggregationError(f"{scene} terminal contract drift")
        scene_config_binding = summary.get("input_scene_config", {})
        snapshot = Path(scene_config_binding.get("snapshot_path", ""))
        _verify_binding(snapshot, scene_config_binding.get("sha256", ""), f"{scene} input config")
        scene_config = _yaml(snapshot)
        if scene_config.get("source_config", {}).get("sha256") != protocol_binding["sha256"]:
            raise M2DevelopmentAggregationError(f"{scene} candidate protocol binding drift")
        if summary["status"] == "done":
            if (
                summary.get("phase") != "formal_development"
                or summary.get("project_git_dirty") is not False
                or summary.get("checkpoint_immutable") is not True
                or summary.get("checkpoint_sha256_before")
                != summary.get("checkpoint_sha256_after")
                or summary.get("request_count") != summary.get("frozen_request_count")
            ):
                raise M2DevelopmentAggregationError(f"{scene} formal renderer contract drift")
            for request in summary["requests"]:
                if [row["arm"] for row in request["matched_arms"]] != protocol[
                    "ablations"
                ]["matched_repair_arms"]:
                    raise M2DevelopmentAggregationError(
                        f"{request['request_id']} matched arm order drift"
                    )
                for candidate in request["candidates"]:
                    asset = candidate["candidate"]["gaussians"]
                    _verify_binding(
                        Path(asset["path"]), asset["sha256"], f"{candidate['arm']} asset"
                    )
                    if Path(asset["path"]).stat().st_size != int(asset["bytes"]):
                        raise M2DevelopmentAggregationError("candidate asset byte drift")
                requests.append({"scene": scene, **request})
        elif summary["status"] != "abstain" or summary.get("retained_in_denominator") is not True:
            raise M2DevelopmentAggregationError(f"{scene} is neither done nor retained abstain")
        summaries.append(summary)
    if not requests:
        raise M2DevelopmentAggregationError("development has no evaluable repair requests")
    if len({row["request_id"] for row in requests}) != len(requests):
        raise M2DevelopmentAggregationError("duplicate development request ID")
    return protocol, summaries, requests


def aggregate(config_path: Path, run_dir: Path) -> dict[str, Any]:
    if _git_dirty():
        raise M2DevelopmentAggregationError("formal development aggregation requires clean git")
    config = _yaml(config_path)
    if (
        config.get("schema_version") != "worldsim_v4_m2_development_selection_v1"
        or config.get("task_id") != TASK_ID
        or config.get("partition") != "development"
        or config.get("dataset") != "nuScenes"
        or config.get("reporting", {}).get("heldout_content_read") is not False
        or config.get("reporting", {}).get("test_quality_read") is not False
    ):
        raise M2DevelopmentAggregationError("development selection root contract drift")
    protocol, summaries, requests = _load_verified_inputs(config)
    selected, grid = select_development_operating_point(
        requests,
        weight_grid=protocol["risk"]["development_weight_candidates"],
        threshold_grid=protocol["risk"]["development_threshold_candidates"],
        tie_priority=protocol["risk"]["tie_priority"],
        require_meaningful_abstention=bool(
            config["selection"]["require_meaningful_abstention"]
        ),
    )
    arms = list(protocol["ablations"]["matched_repair_arms"])
    table = _matched_table(
        requests,
        arms=arms,
        router_decisions=selected["decisions"],
        cohort_scene_count=len(summaries),
    )
    non_router = [row for row in table if row["arm"] != "RISK_ROUTER"]
    baseline = min(
        non_router,
        key=lambda row: (
            row["quality"]["edit_error"]["scene_balanced_mean"],
            arms.index(row["arm"]),
        ),
    )
    router = next(row for row in table if row["arm"] == "RISK_ROUTER")
    gate_report = evaluate_acceptance_gates(
        router=router,
        baseline=baseline,
        selective=selected["statistics"],
        gates=config["acceptance_gates"],
    )
    curve = _selective_curve(
        selected["decisions"], config["reporting"]["requested_selective_coverages"]
    )
    run_dir.mkdir(parents=True)
    source_snapshot = run_dir / "source_snapshot"
    source_snapshot.mkdir()
    shutil.copy2(config_path, source_snapshot / config_path.name)
    shutil.copy2(Path(config["candidate_protocol"]["path"]), source_snapshot / "m2_router_v1.yaml")
    compact_grid = [
        {key: value for key, value in row.items() if key != "decisions"} for row in grid
    ]
    _write_json(run_dir / "artifacts/development_grid.json", compact_grid)
    _write_json(run_dir / "artifacts/router_decisions.json", selected["decisions"])
    _write_json(run_dir / "artifacts/matched_repair_table.json", table)
    _write_json(run_dir / "artifacts/selective_risk_curve.json", curve)
    _write_json(run_dir / "artifacts/gate_report.json", gate_report)
    composite_delta = float(
        router["quality"]["edit_error"]["scene_balanced_mean"]
        - baseline["quality"]["edit_error"]["scene_balanced_mean"]
    )
    summary = {
        "schema_version": "worldsim_v4_m2_development_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "phase": "six_scene_development_selection",
        "partition": "development",
        "dataset": "nuScenes",
        "scene_count": len(summaries),
        "evaluable_scene_count": sum(row["status"] == "done" for row in summaries),
        "retained_abstain_scene_count": sum(row["status"] == "abstain" for row in summaries),
        "request_count": len(requests),
        "candidate_count": sum(len(row["candidates"]) for row in requests),
        "frozen_router": {
            "weight_name": selected["weight_name"],
            "weights": selected["weights"],
            "threshold": selected["threshold"],
            "tie_priority": protocol["risk"]["tie_priority"],
            "selection_unit": config["selection"]["unit"],
            "selection_objective": config["selection"]["objective"],
        },
        "selection_statistics": selected["statistics"],
        "best_matched_non_router": baseline["arm"],
        "router_minus_baseline_scene_balanced_composite_edit_error": composite_delta,
        "composite_edit_error_is_acceptance_gate": False,
        "development_gate": gate_report,
        "development_gate_passed": gate_report["all_gates_passed"],
        "validation_authorized": gate_report["all_gates_passed"],
        "candidate_protocol": config["candidate_protocol"],
        "scene_summary_bindings": config["scene_runs"],
        "project_git_head": _git_head(),
        "project_git_dirty": _git_dirty(),
        "development_content_read": True,
        "development_optimization_read": True,
        "heldout_content_read": False,
        "test_quality_read": False,
    }
    _write_json(run_dir / "summary.json", summary)
    inventory = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        inventory.append(
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": "worldsim_v4_m2_development_manifest_v1",
        "task_id": TASK_ID,
        "inventory": inventory,
        "heldout_content_read": False,
        "test_quality_read": False,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(
        run_dir / "status.json",
        {
            "task_id": TASK_ID,
            "status": "done",
            "summary_sha256": sha256_file(run_dir / "summary.json"),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    try:
        summary = aggregate(args.config.resolve(), args.run_dir.resolve())
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "request_count": summary["request_count"],
                    "selected_weight": summary["frozen_router"]["weight_name"],
                    "threshold": summary["frozen_router"]["threshold"],
                    "best_non_router": summary["best_matched_non_router"],
                    "development_gate_passed": summary["development_gate_passed"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            args.run_dir / "status.json",
            {
                "task_id": TASK_ID,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "heldout_content_read": False,
                "test_quality_read": False,
            },
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
