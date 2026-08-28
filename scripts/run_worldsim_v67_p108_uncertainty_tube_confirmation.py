"""Evaluate the frozen P107 Actor uncertainty tube on one unread scene cohort."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import materialize_actor_query_rows
from scripts.run_worldsim_v67_p107_actor_uncertainty_tube import (
    ActorQuantileTube, _evaluate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["runs_root"] = str(args.runs_root)
    run_dir = args.runs_root / "worldsim_v67" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.monotonic()

    data = config["evaluation_data"]
    metadata = Path(data["metadata_root"]) / "v1.0-trainval"
    scene_table = json.loads((metadata / "scene.json").read_text(encoding="utf-8"))
    index_by_name = {str(row["name"]): index for index, row in enumerate(scene_table)}
    scene_dirs = [
        Path(data["processed_root"]) / f"{index_by_name[str(name)]:03d}"
        for name in data["scene_names"]
    ]
    deadline = time.monotonic() + float(data["readiness_timeout_seconds"])
    while not all(
        (scene / "instances" / "instances_info.json").is_file() and (scene / "lidar_pose").is_dir()
        for scene in scene_dirs
    ):
        if time.monotonic() >= deadline:
            raise TimeoutError("P108 processed scenes not ready")
        time.sleep(10.0)
    raw = materialize_actor_query_rows(scene_dirs, data["horizons_seconds"], data)
    row_partial = run_dir / "P108_CONFIRMATION_ACTOR_UNCERTAINTY_ROWS.partial.npz"
    np.savez_compressed(row_partial, **raw)
    row_partial.replace(run_dir / "P108_CONFIRMATION_ACTOR_UNCERTAINTY_ROWS.npz")

    checkpoint = torch.load(
        args.runs_root / config["frozen_p107"]["run"] / config["frozen_p107"]["artifact"],
        map_location="cuda",
    )
    model = ActorQuantileTube(20, checkpoint["hidden_dimensions"]).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    torch.cuda.reset_peak_memory_stats()
    metrics = _evaluate(
        raw, model,
        np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        np.asarray(checkpoint["feature_scale"], dtype=np.float32), config,
    )
    decision_checks = {
        "fewer_events_than_actor_only": metrics["query_selected_occupancy_flip_events"]
        < metrics["actor_selected_occupancy_flip_events"],
        "no_more_events_than_frozen_p75": metrics["query_selected_occupancy_flip_events"]
        <= metrics["frozen_p75_selected_occupancy_flip_events"],
    }
    verdict = (
        config["verdict_on_pass"] if all(decision_checks.values()) else config["verdict_on_failure"]
    )
    summary = {
        "schema_version": config["output_schema_version"],
        "task_id": config["task_id"], "hypothesis_id": config["hypothesis_id"],
        "status": "done", "verdict": verdict, "role": config["role"],
        "fresh_confirmation_evaluation": metrics,
        "decision_checks": decision_checks,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started,
        },
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict,
                      "fresh_confirmation_evaluation": metrics,
                      "decision_checks": decision_checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
