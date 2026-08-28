"""Evaluate the frozen no-learning clearance baseline on P108 rows."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import binary_auroc
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p87_deepset_trajectory_reliability import _build_sets
from scripts.run_worldsim_v67_p104_temporal_flip_supervision import _aligned_group_max


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.runs_root / "worldsim_v67" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.monotonic()
    row_path = args.runs_root / config["confirmation_rows"]["run"] / config["confirmation_rows"]["artifact"]
    deadline = time.monotonic() + float(config["confirmation_rows"]["readiness_timeout_seconds"])
    while not row_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"P111 confirmation rows not ready: {row_path}")
        time.sleep(10.0)
    arrays = dict(np.load(row_path, allow_pickle=False))
    target = dict(arrays)
    target["raw_actor_state_error_m"] = arrays["occupancy_decision_flip"].astype(np.float32)
    evaluation = _build_sets(
        target, float(config["evaluation"]["visited_region_radius_m"]),
        float(config["evaluation"]["unreliable_actor_state_error_m"]),
        int(config["evaluation"]["maximum_visited_actors"]),
    )
    signed = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32) - np.asarray(
        arrays["occupancy_interaction_radius_m"], dtype=np.float32,
    )[:, None]
    row_score = np.max(
        1.0 / np.maximum(np.abs(signed), float(config["clearance_floor_m"])), axis=1,
    )
    row_keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    score = _aligned_group_max(row_keys, row_score, evaluation["identity"])
    selected = _select_by_scene(
        score, evaluation["scene_index"], float(config["selection"]["coverage_fraction"]),
    )
    events = evaluation["events"]
    metrics = {
        "trajectory_count": int(len(events)), "all_occupancy_flip_events": int(np.count_nonzero(events)),
        "selected_trajectory_count": int(len(selected)),
        "clearance_only_selected_occupancy_flip_events": int(np.count_nonzero(events[selected])),
        "clearance_only_event_auroc": binary_auroc(events, score),
    }
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "status": "done", "role": config["role"], "fresh_confirmation_evaluation": metrics,
        "resources": {"wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
