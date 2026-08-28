"""运行V6.7 ray-terminated Actor-local physical surface repair。"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from motion_proj.worldsim_v66.sensor_surface_repair import repair_surface


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()
    evaluation, artifacts = repair_surface(config, runs_root)
    (run_dir / "ACTOR_REPAIR_METRICS.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in artifacts["state_rows"]),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "REPAIRED_ACTOR_BOUNDARY.npz", **artifacts["repaired"])
    gate_config = config["gates"]
    gates = {
        "minimum_conflict_point_reduction": evaluation["conflict_point_reduction"]
        >= float(gate_config["minimum_conflict_point_reduction"]),
        "minimum_actor_retention": evaluation["actor_retention"]
        >= float(gate_config["minimum_actor_retention"]),
        "minimum_collision_shell_retention": evaluation["collision_shell_retention"]
        >= float(gate_config["minimum_collision_shell_retention"]),
        "minimum_actor_id_track_trajectory_retention": evaluation[
            "actor_id_track_trajectory_retention"
        ]
        >= float(gate_config["minimum_actor_id_track_trajectory_retention"]),
        "maximum_actor_removed_count": evaluation["actor_removed_count"]
        <= int(gate_config["maximum_actor_removed_count"]),
        "maximum_hazard_proxy_distribution_shift": evaluation[
            "maximum_hazard_proxy_distribution_shift"
        ]
        <= float(gate_config["maximum_hazard_proxy_distribution_shift"]),
        "minimum_overall_boundary_retention": evaluation["overall_boundary_retention"]
        >= float(gate_config["minimum_overall_boundary_retention"]),
        "minimum_clean_boundary_retention": evaluation["clean_boundary_retention"]
        >= float(gate_config["minimum_clean_boundary_retention"]),
        "minimum_world_scene_yield": evaluation["world_scene_yield"]
        >= float(gate_config["minimum_world_scene_yield"]),
    }
    verdict = (
        str(config.get("verdict_on_pass", "supported_task_untouched_ray_terminated_surface_repair"))
        if all(gates.values())
        else str(config.get("verdict_on_failure", "rejected_task_untouched_ray_terminated_surface_repair"))
    )
    summary = {
        "schema_version": "worldsim_v67.p4_ray_surface_repair_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "claim_boundary": config["claim_boundary"],
        "evaluation": evaluation,
        "gate_results": gates,
        "failure_ledger_delta": "pending_result",
        "resources": {
            "gpu_used": False,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2),
            "wall_seconds": time.monotonic() - started,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
