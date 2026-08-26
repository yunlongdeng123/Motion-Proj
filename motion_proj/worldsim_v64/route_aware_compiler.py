"""Calibrate a route-aware constrained selector without changing total coverage."""

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
from motion_proj.worldsim_v64.native_voxel_uq import (
    _evidence_on_native_grid,
    _native_unit_dir,
    _unit_dirs,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _constrained_order(scores: np.ndarray, in_route: np.ndarray, route_cap: int) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    chosen = []
    route_chosen = 0
    for index in order:
        if bool(in_route[index]):
            if route_chosen >= route_cap:
                continue
            route_chosen += 1
        chosen.append(int(index))
    return np.asarray(chosen, dtype=np.int64)


def _empirical_cvar(values: list[float], fraction: float) -> tuple[float, int]:
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.mean(np.sort(np.asarray(values))[::-1][:count])), count


def run(config_path: Path, runs_root: Path, processed_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v64" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()

    inputs = config["inputs"]
    evidence_root = runs_root / inputs["evidence_run"]
    native_root = runs_root / inputs["native_run"]
    model = joblib.load(runs_root / inputs["risk_run"] / inputs["risk_model_relative_path"])
    scenes = [str(row["name"]) for row in config["scenes"]]
    strata = {str(row["name"]): str(row["stratum"]) for row in config["scenes"]}
    processed_indices = {str(row["name"]): int(row["processed_index"]) for row in config["scenes"]}
    partition_by_scene = {scene: str(inputs["native_partition"]) for scene in scenes}
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    shape = tuple(int(value) for value in config["native_grid"]["shape"])
    conditional_coverages = {
        str(key): float(value)
        for key, value in config["policy"]["m0_conditional_nominal_coverages"].items()
    }
    route_cap_coverage = float(config["policy"]["m1_route_nominal_coverage_cap"])
    conflict_threshold = float(config["risk"]["hidden_free_conflict_threshold"])
    future_frame_count = int(config["route"]["future_frame_count"])
    corridor_radius = float(config["route"]["corridor_radius_m"])
    x = origin[0] + (np.arange(shape[0], dtype=np.float32) + 0.5) * voxel_size
    y = origin[1] + (np.arange(shape[1], dtype=np.float32) + 0.5) * voxel_size
    xx, yy = np.meshgrid(x, y, indexing="ij")
    grid_xy = torch.from_numpy(np.stack((xx, yy), axis=-1)).to("cuda")

    rows = []
    rows_path = run_dir / "CASE_METRICS.jsonl"
    with torch.inference_mode():
        for scene in scenes:
            for evidence_unit in _unit_dirs(evidence_root, scene):
                native_unit = _native_unit_dir(
                    native_root, scene, evidence_unit.name, partition_by_scene
                )
                indices, _, features = _target_free_boundary(
                    evidence_unit,
                    native_unit,
                    native_origin_m=origin,
                    native_voxel_size_m=voxel_size,
                )
                logits = features[:, :17]
                scores = np.asarray(model.score(features, logits), dtype=np.float32)
                target_frame = int(evidence_unit.name.removeprefix("f"))
                route_xy = _future_route_in_target_lidar(
                    processed_root / f"{processed_indices[scene]:03d}",
                    target_frame,
                    future_frame_count,
                )
                route_tensor = torch.from_numpy(route_xy).to("cuda")
                corridor = (
                    (grid_xy[None] - route_tensor[:, None, None]).square().sum(dim=-1).amin(dim=0)
                    <= corridor_radius**2
                ).cpu().numpy()
                ix, iy, iz = indices.T
                in_route = corridor[ix, iy]

                with np.load(evidence_unit / "TARGET_EVIDENCE.npz", allow_pickle=False) as source:
                    target = {name: np.asarray(source[name]) for name in source.files}
                target_state, target_valid = _evidence_on_native_grid(
                    target,
                    native_shape=shape,
                    native_origin_m=origin,
                    native_voxel_size_m=voxel_size,
                )
                hidden_free = (target_state[ix, iy, iz] == FREE) & target_valid[ix, iy, iz]

                nominal = conditional_coverages[strata[scene]]
                selected_count = max(1, int(np.floor(nominal * scores.size)))
                m0_order = np.argsort(scores, kind="stable")[:selected_count]
                route_count = int(np.count_nonzero(in_route))
                route_cap = int(np.floor(route_cap_coverage * route_count))
                m1_order = _constrained_order(scores, in_route, route_cap)[:selected_count]
                if m1_order.size != selected_count:
                    raise RuntimeError(f"insufficient non-route capacity: {scene}/{evidence_unit.name}")

                arms = {}
                for name, selected in (("m0_conditional", m0_order), ("m1_route_aware", m1_order)):
                    selected_route = selected[in_route[selected]]
                    overall_conflict = float(hidden_free[selected].mean())
                    route_conflict = (
                        float(hidden_free[selected_route].mean()) if selected_route.size else 0.0
                    )
                    arms[name] = {
                        "selected_count": int(selected.size),
                        "realized_coverage": float(selected.size / scores.size),
                        "route_eligible_count": route_count,
                        "route_selected_count": int(selected_route.size),
                        "route_realized_coverage": float(selected_route.size / route_count) if route_count else 0.0,
                        "overall_hidden_free_conflict": overall_conflict,
                        "overall_case_loss": overall_conflict > conflict_threshold,
                        "route_hidden_free_conflict": route_conflict,
                        "route_case_loss": route_conflict > conflict_threshold,
                    }
                row = {
                    "scene": scene,
                    "stratum": strata[scene],
                    "unit": evidence_unit.name,
                    "eligible_count": int(scores.size),
                    "arms": arms,
                }
                rows.append(row)
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fraction = float(config["tail"]["fraction"])
    summaries = {}
    for arm in ("m0_conditional", "m1_route_aware"):
        arm_rows = [row["arms"][arm] for row in rows]
        cvar, tail_count = _empirical_cvar(
            [float(row["route_hidden_free_conflict"]) for row in arm_rows], fraction
        )
        route_conflicts = [
            int(round(float(row["route_hidden_free_conflict"]) * int(row["route_selected_count"])))
            for row in arm_rows
        ]
        fixed_densities = [
            float(conflicts / int(row["route_eligible_count"]))
            if int(row["route_eligible_count"]) else 0.0
            for conflicts, row in zip(route_conflicts, arm_rows)
        ]
        fixed_cvar, fixed_tail_count = _empirical_cvar(fixed_densities, fraction)
        route_eligible_count = sum(int(row["route_eligible_count"]) for row in arm_rows)
        route_conflict_count = sum(route_conflicts)
        summaries[arm] = {
            "mean_realized_coverage": float(np.mean([row["realized_coverage"] for row in arm_rows])),
            "mean_route_realized_coverage": float(
                np.mean([row["route_realized_coverage"] for row in arm_rows])
            ),
            "route_eligible_count": route_eligible_count,
            "route_selected_count": sum(int(row["route_selected_count"]) for row in arm_rows),
            "route_hidden_free_conflict_count": route_conflict_count,
            "overall_case_failure_count": sum(bool(row["overall_case_loss"]) for row in arm_rows),
            "route_case_failure_count": sum(bool(row["route_case_loss"]) for row in arm_rows),
            "route_empirical_cvar": cvar,
            "tail_count": tail_count,
            "pooled_fixed_denominator_conflict_density": float(
                route_conflict_count / route_eligible_count
            ),
            "fixed_denominator_empirical_cvar": fixed_cvar,
            "fixed_denominator_tail_count": fixed_tail_count,
        }
    coverage_delta = float(
        summaries["m1_route_aware"]["mean_realized_coverage"]
        - summaries["m0_conditional"]["mean_realized_coverage"]
    )
    fixed_cvar_delta = float(
        summaries["m1_route_aware"]["fixed_denominator_empirical_cvar"]
        - summaries["m0_conditional"]["fixed_denominator_empirical_cvar"]
    )
    pooled_fixed_density_delta = float(
        summaries["m1_route_aware"]["pooled_fixed_denominator_conflict_density"]
        - summaries["m0_conditional"]["pooled_fixed_denominator_conflict_density"]
    )
    case_fixed_deltas = []
    for row in rows:
        fixed = {}
        for arm in ("m0_conditional", "m1_route_aware"):
            arm_row = row["arms"][arm]
            conflict = int(
                round(
                    float(arm_row["route_hidden_free_conflict"])
                    * int(arm_row["route_selected_count"])
                )
            )
            eligible = int(arm_row["route_eligible_count"])
            fixed[arm] = float(conflict / eligible) if eligible else 0.0
        case_fixed_deltas.append(fixed["m1_route_aware"] - fixed["m0_conditional"])
    lower = sum(value < 0.0 for value in case_fixed_deltas)
    equal = sum(value == 0.0 for value in case_fixed_deltas)
    higher = sum(value > 0.0 for value in case_fixed_deltas)
    fixed_comparison = {
        "m1_minus_m0_cvar": fixed_cvar_delta,
        "m1_minus_m0_pooled_density": pooled_fixed_density_delta,
        "paired_case_m1_lower_count": lower,
        "paired_case_equal_count": equal,
        "paired_case_m1_higher_count": higher,
        "paired_probability_m1_improves_with_half_ties": float(
            (lower + 0.5 * equal) / len(case_fixed_deltas)
        ),
    }
    if str(config.get("evaluation", {}).get("primary", "selected_route_cvar")) == "fixed_route_denominator":
        gates = {
            "total_coverage_preserved": abs(coverage_delta)
            <= float(config["gates"]["maximum_absolute_mean_coverage_delta"]),
            "fixed_denominator_cvar_not_worse": fixed_cvar_delta
            <= float(config["gates"]["maximum_m1_minus_m0_fixed_denominator_cvar"]),
            "pooled_fixed_denominator_density_not_worse": pooled_fixed_density_delta
            <= float(config["gates"]["maximum_m1_minus_m0_pooled_fixed_denominator_density"]),
        }
    else:
        gates = {
            "maximum_m1_route_empirical_cvar": summaries["m1_route_aware"]["route_empirical_cvar"]
            <= float(config["gates"]["maximum_m1_route_empirical_cvar"]),
            "total_coverage_preserved": abs(coverage_delta)
            <= float(config["gates"]["maximum_absolute_mean_coverage_delta"]),
        }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "case_count": len(rows),
        "arms": summaries,
        "m1_minus_m0_mean_coverage": coverage_delta,
        "fixed_denominator_comparison": fixed_comparison,
        "route_nominal_coverage_cap": route_cap_coverage,
        "model_refit": False,
        "policy_selection_during_run": False,
        "new_confirmation_read": bool(config.get("new_confirmation_read", False)),
        "gate_results": gates,
        "resources": {
            "gpu_used": True,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        },
        "failure_ledger_refs": config["failure_ledger_refs"],
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "resource.json", summary["resources"])
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {
        "run_dir": str(run_dir),
        "verdict": verdict,
        "m1_route_empirical_cvar": summaries["m1_route_aware"]["route_empirical_cvar"],
        "m1_minus_m0_fixed_denominator_cvar": fixed_cvar_delta,
        "m1_minus_m0_mean_coverage": coverage_delta,
        "gate_results": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config.resolve(),
                args.runs_root.resolve(),
                args.processed_root.resolve(),
                args.run_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
