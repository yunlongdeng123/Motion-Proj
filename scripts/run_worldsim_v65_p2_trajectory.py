"""Evaluate the frozen P1R task-risk candidate once on fresh P2 scenes."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

from motion_proj.worldsim_v61.occupancy import FREE
from motion_proj.worldsim_v64.conditional_state_bake import _target_free_boundary
from motion_proj.worldsim_v64.gaussian_route_consumer import _future_route_in_target_lidar
from motion_proj.worldsim_v64.native_voxel_uq import _evidence_on_native_grid, _native_unit_dir, _unit_dirs
from motion_proj.worldsim_v65.conditional_validity import MonotoneFitResult, MonotoneTaskRiskResidual, score_monotone_task_risk
from motion_proj.worldsim_v65.task_contract import continuous_trajectory_features


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _q0_embedding(model: object, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda")
    values = (features.astype(np.float32) - model.mean) / model.scale
    network = model.model.to(device).eval()
    hidden_parts, logit_parts = [], []
    with torch.inference_mode():
        for offset in range(0, values.shape[0], 131072):
            batch = torch.from_numpy(values[offset : offset + 131072]).to(device)
            with torch.cuda.amp.autocast():
                hidden = network.layers[:5](batch)
                logits = network.layers[5](hidden).squeeze(1)
            hidden_parts.append(hidden.half().cpu().numpy())
            logit_parts.append(logits.float().cpu().numpy())
    return np.concatenate(hidden_parts), np.concatenate(logit_parts).astype(np.float32)


def _load_task_model(path: Path) -> MonotoneFitResult:
    artifact = torch.load(path, map_location="cpu")
    model = MonotoneTaskRiskResidual(
        int(artifact["state_dict"]["film.weight"].shape[0] // 2),
        int(np.asarray(artifact["trajectory_mean"]).shape[0]),
    )
    model.load_state_dict(artifact["state_dict"])
    model.to("cuda").eval()
    return MonotoneFitResult(
        model=model,
        trajectory_mean=np.asarray(artifact["trajectory_mean"], dtype=np.float32),
        trajectory_scale=np.asarray(artifact["trajectory_scale"], dtype=np.float32),
        epoch_losses=[],
    )


def _arm(scores: np.ndarray, labels: np.ndarray, route: np.ndarray, coverage: float) -> dict[str, object]:
    count = max(1, int(math.floor(float(coverage) * scores.size)))
    selected = np.argsort(scores, kind="stable")[:count]
    selected_route = route[selected]
    selected_nonroute = ~selected_route
    route_eligible = int(route.sum())
    route_conflicts = int(np.count_nonzero(labels[selected] & selected_route))
    nonroute_count = int(selected_nonroute.sum())
    nonroute_conflicts = int(np.count_nonzero(labels[selected] & selected_nonroute))
    return {
        "eligible_count": int(scores.size),
        "selected_count": count,
        "realized_coverage": float(count / scores.size),
        "route_eligible_count": route_eligible,
        "route_conflict_count": route_conflicts,
        "fixed_route_conflict_density": float(route_conflicts / route_eligible if route_eligible else 0.0),
        "nonroute_selected_count": nonroute_count,
        "nonroute_conflict_count": nonroute_conflicts,
        "nonroute_emitted_conflict_rate": float(nonroute_conflicts / nonroute_count if nonroute_count else 0.0),
    }


def _summarize(rows: list[dict[str, object]], name: str, tail_fraction: float) -> dict[str, object]:
    arms = [row["arms"][name] for row in rows]
    route_denominator = sum(row["route_eligible_count"] for row in arms)
    route_conflicts = sum(row["route_conflict_count"] for row in arms)
    nonroute_count = sum(row["nonroute_selected_count"] for row in arms)
    nonroute_conflicts = sum(row["nonroute_conflict_count"] for row in arms)
    densities = np.asarray([row["fixed_route_conflict_density"] for row in arms])
    tail_count = max(1, int(math.ceil(float(tail_fraction) * densities.size)))
    scene_rows = []
    for scene in sorted({row["scene"] for row in rows}):
        scene_arms = [row["arms"][name] for row in rows if row["scene"] == scene]
        denominator = sum(row["route_eligible_count"] for row in scene_arms)
        conflicts = sum(row["route_conflict_count"] for row in scene_arms)
        scene_rows.append({"scene": scene, "route_eligible_count": denominator, "route_conflict_count": conflicts, "fixed_route_conflict_density": float(conflicts / denominator if denominator else 0.0)})
    return {
        "mean_realized_coverage": float(np.mean([row["realized_coverage"] for row in arms])),
        "route_eligible_count": route_denominator,
        "route_conflict_count": route_conflicts,
        "pooled_fixed_route_conflict_density": float(route_conflicts / route_denominator if route_denominator else 0.0),
        "worst_tail_fixed_route_cvar": float(np.sort(densities)[::-1][:tail_count].mean()),
        "tail_count": tail_count,
        "nonroute_selected_count": nonroute_count,
        "nonroute_conflict_count": nonroute_conflicts,
        "nonroute_emitted_conflict_rate": float(nonroute_conflicts / nonroute_count if nonroute_count else 0.0),
        "scene_rows": scene_rows,
    }


def run(config_path: Path, runs_root: Path, processed_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v65" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    inputs = config["inputs"]
    evidence_root = runs_root / inputs["evidence_run"]
    native_root = runs_root / inputs["native_run"]
    q0 = joblib.load(runs_root / inputs["risk_run"] / inputs["risk_model_relative_path"])
    task_model = _load_task_model(runs_root / inputs["task_model_run"] / inputs["task_model_relative_path"])
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    coverage = float(config["evaluation"]["matched_total_coverage"])
    radius = float(config["evaluation"]["route_corridor_radius_m"])
    partition = str(inputs["native_partition"])
    rows = []
    monotone_violations = 0
    for scene in config["scenes"]:
        name = str(scene["name"])
        for evidence_unit in _unit_dirs(evidence_root, name):
            native_unit = _native_unit_dir(native_root, name, evidence_unit.name, {name: partition})
            indices, centers, features = _target_free_boundary(evidence_unit, native_unit, native_origin_m=origin, native_voxel_size_m=voxel_size)
            with np.load(evidence_unit / "TARGET_EVIDENCE.npz", allow_pickle=False) as source:
                target = {key: np.asarray(source[key]) for key in source.files}
            shape = tuple(int(value) for value in np.load(native_unit / "ARGMAX.npy", mmap_mode="r").shape)
            target_state, target_valid = _evidence_on_native_grid(target, native_shape=shape, native_origin_m=origin, native_voxel_size_m=voxel_size)
            x, y, z = indices.T
            valid = target_valid[x, y, z]
            centers, features, labels = centers[valid], features[valid], target_state[x, y, z][valid] == FREE
            frame = int(evidence_unit.name.removeprefix("f"))
            route_xy = _future_route_in_target_lidar(processed_root / f"{int(scene['processed_index']):03d}", frame, int(config["trajectory"]["future_frame_count"]))
            trajectory = continuous_trajectory_features(centers, route_xy, torch.device("cuda")).cpu().numpy()
            hidden, q0_logit = _q0_embedding(q0, features)
            q0_scores = torch.sigmoid(torch.from_numpy(q0_logit)).numpy()
            task_scores = score_monotone_task_risk(task_model, hidden, q0_logit, trajectory)
            monotone_violations += int(np.count_nonzero(task_scores + 1e-7 < q0_scores))
            route = trajectory[:, 0] <= radius
            rows.append({"scene": name, "unit": evidence_unit.name, "arms": {"q0": _arm(q0_scores, labels, route, coverage), "task": _arm(task_scores, labels, route, coverage)}})
    q0_summary = _summarize(rows, "q0", float(config["evaluation"]["tail_fraction"]))
    task_summary = _summarize(rows, "task", float(config["evaluation"]["tail_fraction"]))
    q0_density = float(q0_summary["pooled_fixed_route_conflict_density"])
    task_density = float(task_summary["pooled_fixed_route_conflict_density"])
    relative_reduction = float((q0_density - task_density) / q0_density if q0_density > 0 else 0.0)
    q0_scenes = {row["scene"]: row for row in q0_summary["scene_rows"]}
    task_scenes = {row["scene"]: row for row in task_summary["scene_rows"]}
    deltas = {scene: task_scenes[scene]["fixed_route_conflict_density"] - q0_scenes[scene]["fixed_route_conflict_density"] for scene in q0_scenes}
    relative_regressions = []
    for scene, delta in deltas.items():
        baseline = q0_scenes[scene]["fixed_route_conflict_density"]
        relative_regressions.append(delta / baseline if baseline > 0 else (math.inf if delta > 0 else 0.0))
    q0_nonroute = float(q0_summary["nonroute_emitted_conflict_rate"])
    task_nonroute = float(task_summary["nonroute_emitted_conflict_rate"])
    nonroute_change = float((task_nonroute - q0_nonroute) / q0_nonroute if q0_nonroute > 0 else 0.0)
    comparison = {
        "relative_fixed_route_risk_reduction": relative_reduction,
        "scene_lower_count": sum(value < 0 for value in deltas.values()),
        "scene_equal_count": sum(value == 0 for value in deltas.values()),
        "scene_higher_count": sum(value > 0 for value in deltas.values()),
        "maximum_scene_relative_risk_increase": max(relative_regressions),
        "nonroute_relative_risk_change": nonroute_change,
        "monotone_score_violation_count": monotone_violations,
        "scene_deltas": deltas,
    }
    gates = {
        "minimum_fixed_route_risk_reduction": relative_reduction >= float(config["gates"]["minimum_fixed_route_risk_reduction"]),
        "minimum_scene_support": comparison["scene_lower_count"] >= int(config["gates"]["minimum_scene_support"]),
        "maximum_scene_regression": comparison["maximum_scene_relative_risk_increase"] <= float(config["gates"]["maximum_scene_relative_risk_increase"]),
        "nonroute_risk_not_worse": nonroute_change <= float(config["gates"]["maximum_nonroute_relative_risk_increase"]),
        "coverage_matched": abs(float(task_summary["mean_realized_coverage"]) - float(q0_summary["mean_realized_coverage"])) <= 1e-6,
        "monotone_semantics": monotone_violations == 0,
    }
    verdict = "supported_fresh_trajectory_condition" if all(gates.values()) else "rejected_fresh_trajectory_condition"
    with (run_dir / "CASE_METRICS.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": "worldsim_v65.p2_trajectory_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "scene_count": len(config["scenes"]),
        "case_count": len(rows),
        "arms": {"q0": q0_summary, "task": task_summary},
        "comparison": comparison,
        "gate_results": gates,
        "formal_v65_selection_read": True,
        "model_refit": False,
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3), "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2), "wall_seconds": time.monotonic() - started},
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates, "comparison": comparison, "resources": summary["resources"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.processed_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
