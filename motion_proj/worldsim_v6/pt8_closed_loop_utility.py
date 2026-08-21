"""V6 PT8 冻结风险策略的有限时域纵向闭环效用实验。"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from motion_proj.worldsim_v6.pt2_risk_policy import (
    _balanced_accuracy,
    _canonical,
    _git,
    _sha256,
    _write_json,
    _write_jsonl,
)
from motion_proj.worldsim_v6.pt6_compositional_risk import _predict_actor


TASK_ID = "WS-V6-PT8-CLOSED-LOOP-UTILITY-01"


class PT8ClosedLoopError(RuntimeError):
    """PT8 closed-loop experiment 正式合同失败。"""


def _geometry(
    relative_forward_m: float,
    lateral_m: float,
    size_m: tuple[float, float],
    yaw_deg: float,
    ego_half: tuple[float, float],
) -> tuple[float, float, float, float, float, float, float]:
    yaw = math.radians(float(yaw_deg))
    actor_hx, actor_hy = 0.5 * float(size_m[0]), 0.5 * float(size_m[1])
    projected_hx = abs(math.cos(yaw)) * actor_hx + abs(math.sin(yaw)) * actor_hy
    projected_hy = abs(math.sin(yaw)) * actor_hx + abs(math.cos(yaw)) * actor_hy
    forward = abs(float(relative_forward_m))
    lateral = abs(float(lateral_m))
    dx = forward - (ego_half[0] + projected_hx)
    dy = lateral - (ego_half[1] + projected_hy)
    return max(dx, dy), forward, lateral, projected_hx, projected_hy, dx, dy


def _is_collision(geometry: tuple[float, ...], decimals: int) -> bool:
    return round(geometry[5], decimals) <= 0.0 and round(geometry[6], decimals) <= 0.0


def _uncontrolled_hazard(scenario: Mapping[str, Any], dynamics: Mapping[str, Any]) -> bool:
    ego_half = (0.5 * float(dynamics["ego_length_m"]), 0.5 * float(dynamics["ego_width_m"]))
    steps = int(round(float(dynamics["horizon_seconds"]) / float(dynamics["dt_seconds"])))
    decimals = int(dynamics["label_canonicalization_decimals"])
    for step in range(steps + 1):
        ego_x = float(scenario["initial_speed_mps"]) * float(dynamics["dt_seconds"]) * step
        geometry = _geometry(
            float(scenario["actor_forward_m"]) - ego_x,
            float(scenario["actor_lateral_m"]),
            tuple(scenario["actor_size_m"]),
            float(scenario["actor_yaw_deg"]),
            ego_half,
        )
        if _is_collision(geometry, decimals):
            return True
    return False


def _rollout(
    model: Mapping[str, Any],
    scenario: Mapping[str, Any],
    dynamics: Mapping[str, Any],
    force_brake: bool = False,
) -> dict[str, Any]:
    dt = float(dynamics["dt_seconds"])
    steps = int(round(float(dynamics["horizon_seconds"]) / dt))
    preview_steps = int(round(float(dynamics["preview_seconds"]) / dt))
    ego_half = (0.5 * float(dynamics["ego_length_m"]), 0.5 * float(dynamics["ego_width_m"]))
    decimals = int(dynamics["label_canonicalization_decimals"])
    ego_x = 0.0
    velocity = float(scenario["initial_speed_mps"])
    acceleration = 0.0
    collided = False
    max_abs_jerk = 0.0
    brake_steps = 0
    for _ in range(steps + 1):
        current_geometry = _geometry(
            float(scenario["actor_forward_m"]) - ego_x,
            float(scenario["actor_lateral_m"]),
            tuple(scenario["actor_size_m"]),
            float(scenario["actor_yaw_deg"]),
            ego_half,
        )
        if _is_collision(current_geometry, decimals):
            collided = True
            break
        predicted_hazard = force_brake
        if not force_brake:
            for preview_step in range(1, preview_steps + 1):
                preview_x = ego_x + velocity * dt * preview_step
                preview_geometry = _geometry(
                    float(scenario["actor_forward_m"]) - preview_x,
                    float(scenario["actor_lateral_m"]),
                    tuple(scenario["actor_size_m"]),
                    float(scenario["actor_yaw_deg"]),
                    ego_half,
                )
                if _predict_actor(model, preview_geometry):
                    predicted_hazard = True
                    break
        desired_acceleration = -float(dynamics["maximum_deceleration_mps2"]) if predicted_hazard else 0.0
        jerk_limit = float(dynamics["jerk_limit_mps3"])
        delta_limit = jerk_limit * dt
        next_acceleration = float(
            np.clip(desired_acceleration, acceleration - delta_limit, acceleration + delta_limit)
        )
        jerk = (next_acceleration - acceleration) / dt
        max_abs_jerk = max(max_abs_jerk, abs(jerk))
        acceleration = next_acceleration
        if predicted_hazard:
            brake_steps += 1
        velocity = max(0.0, velocity + acceleration * dt)
        ego_x += velocity * dt
    reference_progress = float(scenario["initial_speed_mps"]) * float(dynamics["horizon_seconds"])
    safe_progress = ego_x >= float(dynamics["safe_progress_fraction"]) * reference_progress
    comfortable = max_abs_jerk <= float(dynamics["comfort_jerk_limit_mps3"])
    return {
        "collided": collided,
        "progress_m": ego_x,
        "reference_progress_m": reference_progress,
        "safe_progress": safe_progress,
        "final_speed_mps": velocity,
        "brake_fraction": brake_steps / (steps + 1),
        "max_abs_jerk_mps3": max_abs_jerk,
        "comfortable": comfortable,
    }


def _metrics(rows: list[dict[str, Any]], hazard_denominator: str) -> dict[str, Any]:
    if hazard_denominator == "oracle_avoidable_only":
        hazards = [row for row in rows if row["oracle_avoidable_hazard"]]
    elif hazard_denominator == "all_uncontrolled_hazards":
        hazards = [row for row in rows if row["uncontrolled_hazard"]]
    else:
        raise PT8ClosedLoopError(f"未知 hazard denominator: {hazard_denominator}")
    safe = [row for row in rows if not row["uncontrolled_hazard"]]
    unavoidable = [row for row in rows if row.get("unavoidable_hazard", False)]
    hazard_success = np.asarray([not row["rollout"]["collided"] for row in hazards], dtype=int)
    safe_success = np.asarray(
        [not row["rollout"]["collided"] and row["rollout"]["safe_progress"] for row in safe],
        dtype=int,
    )
    labels = np.asarray([1] * len(hazards) + [0] * len(safe), dtype=int)
    predictions = np.asarray(
        [int(value) for value in hazard_success] + [int(not value) for value in safe_success],
        dtype=int,
    )
    return {
        "scenario_count": len(rows),
        "hazard_scenario_count": len(hazards),
        "unavoidable_hazard_count": len(unavoidable),
        "safe_scenario_count": len(safe),
        "collision_rate_on_hazards": float(1.0 - hazard_success.mean()),
        "hazard_avoidance_rate": float(hazard_success.mean()),
        "safe_route_completion": float(safe_success.mean()),
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "safe_stuck_rate": float(np.mean([not row["rollout"]["safe_progress"] for row in safe])),
        "comfort_rate": float(np.mean([row["rollout"]["comfortable"] for row in rows])),
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    repo_root = repo_root.resolve()
    config_path = (repo_root / config_path).resolve() if not config_path.is_absolute() else config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["task_id"] != TASK_ID:
        raise PT8ClosedLoopError("task_id 不匹配")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{stamp}__closed-loop-utility-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
            raise PT8ClosedLoopError("磁盘资源不足")
        source_commit = _git(repo_root, "rev-parse", "HEAD")
        if _git(repo_root, "status", "--short"):
            raise PT8ClosedLoopError("正式运行要求 clean worktree")
        policy_path = Path(config["frozen_policy"]["policy_arms_path"])
        if _sha256(policy_path) != config["frozen_policy"]["policy_arms_sha256"]:
            raise PT8ClosedLoopError("冻结 policy hash 漂移")
        policy_arms = json.loads(policy_path.read_text(encoding="utf-8"))
        frozen_paths = [config_path, policy_path]
        hashes_before = {str(path): _sha256(path) for path in frozen_paths}
        scenarios = []
        for speed, forward, lateral, size, yaw in product(
            config["scenario_grid"]["initial_speeds_mps"],
            config["scenario_grid"]["actor_forward_m"],
            config["scenario_grid"]["actor_lateral_m"],
            config["scenario_grid"]["actor_sizes_m"],
            config["scenario_grid"]["actor_yaw_deg"],
        ):
            scenario = {
                "initial_speed_mps": float(speed),
                "actor_forward_m": float(forward),
                "actor_lateral_m": float(lateral),
                "actor_size_m": [float(value) for value in size],
                "actor_yaw_deg": float(yaw),
            }
            scenario["uncontrolled_hazard"] = _uncontrolled_hazard(scenario, config["dynamics"])
            oracle_rollout = _rollout(
                next(iter(policy_arms.values()))["model"],
                scenario,
                config["dynamics"],
                force_brake=True,
            )
            scenario["oracle_avoidable_hazard"] = bool(
                scenario["uncontrolled_hazard"] and not oracle_rollout["collided"]
            )
            scenario["unavoidable_hazard"] = bool(
                scenario["uncontrolled_hazard"] and oracle_rollout["collided"]
            )
            scenarios.append(scenario)
        arm_rows1 = {
            arm: [
                {**scenario, "rollout": _rollout(value["model"], scenario, config["dynamics"])}
                for scenario in scenarios
            ]
            for arm, value in policy_arms.items()
        }
        arm_rows2 = {
            arm: [
                {**scenario, "rollout": _rollout(value["model"], scenario, config["dynamics"])}
                for scenario in scenarios
            ]
            for arm, value in policy_arms.items()
        }
        hazard_denominator = str(config.get("hazard_denominator", "all_uncontrolled_hazards"))
        metrics1 = {arm: _metrics(rows, hazard_denominator) for arm, rows in arm_rows1.items()}
        metrics2 = {arm: _metrics(rows, hazard_denominator) for arm, rows in arm_rows2.items()}
        _write_json(run_dir / "CLOSED_LOOP_ARMS.json", metrics1)
        _write_jsonl(
            run_dir / "CLOSED_LOOP_EPISODES.jsonl",
            [dict(row, arm=arm) for arm, rows in arm_rows1.items() for row in rows],
        )
        _write_json(run_dir / "SOURCE_AUDIT.json", {"source_sha256": hashes_before})
        v6 = metrics1["real_plus_v6_verified_compiled"]
        real = metrics1["real_only"]
        naive = metrics1["real_plus_naive_synthetic"]
        cfg = config["gate"]
        wall_seconds = time.monotonic() - started
        checks = {
            "frozen_policy_exact": True,
            "scenario_denominator_exact": len(scenarios) == int(config["expected_scenario_count"]),
            "both_outcomes_present": v6["hazard_scenario_count"] > 0 and v6["safe_scenario_count"] > 0,
            "avoidable_hazard_fraction": v6["hazard_scenario_count"] / (v6["hazard_scenario_count"] + v6["unavoidable_hazard_count"]) >= float(cfg.get("require_avoidable_hazard_fraction_at_least", 0.0)),
            "v6_balanced_accuracy": v6["balanced_accuracy"] >= float(cfg["require_v6_balanced_accuracy_at_least"]),
            "v6_collision_rate": v6["collision_rate_on_hazards"] <= float(cfg["require_v6_collision_rate_at_most"]),
            "v6_safe_route_completion": v6["safe_route_completion"] >= float(cfg["require_v6_safe_route_completion_at_least"]),
            "v6_comfort_rate": v6["comfort_rate"] >= float(cfg["require_v6_comfort_rate_at_least"]),
            "v6_collision_no_worse_than_each_baseline": v6["collision_rate_on_hazards"] <= min(real["collision_rate_on_hazards"], naive["collision_rate_on_hazards"]),
            "v6_completion_no_worse_than_each_baseline": v6["safe_route_completion"] >= max(real["safe_route_completion"], naive["safe_route_completion"]),
            "v6_balanced_accuracy_gain_vs_real_only": v6["balanced_accuracy"] - real["balanced_accuracy"] >= float(cfg["require_balanced_accuracy_gain_vs_each_at_least"]),
            "v6_balanced_accuracy_gain_vs_naive": v6["balanced_accuracy"] - naive["balanced_accuracy"] >= float(cfg["require_balanced_accuracy_gain_vs_each_at_least"]),
            "rollout_repeat_exact": _canonical(arm_rows1) == _canonical(arm_rows2),
            "source_immutable": hashes_before == {str(path): _sha256(path) for path in frozen_paths},
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.pt8_closed_loop_utility_gate.v1",
            "checks": checks,
            "decision": "accept_preview_controller_closed_loop_utility" if checks["passed"] else "reject_preview_controller_closed_loop_utility",
        }
        _write_json(run_dir / "PT8_CLOSED_LOOP_GATE.json", gate)
        summary = {
            "schema_version": "worldsim_v6.pt8_closed_loop_utility_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "source_commit": source_commit,
            "method_arms": metrics1,
            "wall_seconds": wall_seconds,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["CLOSED_LOOP_ARMS.json", "CLOSED_LOOP_EPISODES.jsonl", "SOURCE_AUDIT.json", "PT8_CLOSED_LOOP_GATE.json", "SUMMARY.json"]
        _write_json(
            run_dir / "MANIFEST.json",
            {"schema_version": "worldsim_v6.pt8_closed_loop_utility_manifest.v1", "source_commit": source_commit, "config": str(config_path), "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}},
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {"schema_version": "worldsim_v6.terminal.v1", "status": summary["status"], "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"], "manifest_sha256": _sha256(run_dir / "MANIFEST.json")},
        )
        return run_dir
    except Exception as error:
        _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": "blocked", "task_id": TASK_ID, "error_type": type(error).__name__, "error": str(error)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    print(run_experiment(args.repo_root, args.config, args.run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
