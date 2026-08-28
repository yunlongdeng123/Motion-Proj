"""运行V6.6 P8固定参数响应式Actor capability audit。"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from motion_proj.worldsim_v66.reactive_actor import run_capability


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v66" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()
    package_dir = runs_root / str(config["source"]["package_run"]) / "package"
    result = run_capability(config, package_dir)
    _write_jsonl(run_dir / "SCENE_METRICS.jsonl", result["scene_metrics"])
    _write_jsonl(run_dir / "TRAJECTORIES.jsonl", result["trajectory_rows"])
    selected_rows = [
        {
            "scene": row["scene"],
            "actor_key": row["actor_key"],
            "track_id": row["track_id"],
            "class": row["class"],
            "lifecycle": row["lifecycle"],
            "sample_count": len(row["trajectory"]),
            "median_logged_speed_mps": row["median_logged_speed_mps"],
            "logged_path_length_m": row["logged_path_length_m"],
        }
        for row in result["selected_actors"]
    ]
    _write_jsonl(run_dir / "SELECTED_ACTORS.jsonl", selected_rows)
    selected_scene_count = len(result["scene_metrics"])
    supported_scene_count = int(result["supported_scene_count"])
    gates = {
        "all_scenes_selected": selected_scene_count == len(config["scenes"]),
        "minimum_supported_scenes": supported_scene_count
        >= int(config["gates"]["minimum_supported_scenes"]),
        "identity_lifecycle_retained": all(
            bool(row["gates"]["identity_lifecycle_exact"])
            for row in result["scene_metrics"]
        ),
        "logged_path_retained": all(
            bool(row["gates"]["logged_path_deviation_zero"])
            for row in result["scene_metrics"]
        ),
    }
    verdict = (
        "supported_synthetic_lead_brake_reactive_actor_capability"
        if all(gates.values())
        else "rejected_synthetic_lead_brake_reactive_actor_capability"
    )
    summary = {
        "schema_version": "worldsim_v66.p8_reactive_actor_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "claim_boundary": config["claim_boundary"],
        "metrics": {
            "selected_scene_count": selected_scene_count,
            "supported_scene_count": supported_scene_count,
            "total_x0_collision_steps": sum(
                int(row["x0_collision_steps"]) for row in result["scene_metrics"]
            ),
            "total_x1_collision_steps": sum(
                int(row["x1_collision_steps"]) for row in result["scene_metrics"]
            ),
            "minimum_x1_gap_m": min(
                float(row["x1_minimum_gap_m"]) for row in result["scene_metrics"]
            ),
            "maximum_absolute_command_jerk_mps3": max(
                float(row["maximum_absolute_command_jerk_mps3"])
                for row in result["scene_metrics"]
            ),
        },
        "gate_results": gates,
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": "pending_result",
        "resources": {
            "gpu_used": False,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
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
