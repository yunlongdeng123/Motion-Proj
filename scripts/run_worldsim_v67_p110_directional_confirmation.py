"""Evaluate a frozen P109 directional Gaussian on the P108 confirmation rows."""

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

from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian, _evaluate,
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
    row_path = args.runs_root / config["confirmation_rows"]["run"] / config["confirmation_rows"]["artifact"]
    deadline = time.monotonic() + float(config["confirmation_rows"]["readiness_timeout_seconds"])
    while not row_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"P110 confirmation rows not ready: {row_path}")
        time.sleep(10.0)
    raw = dict(np.load(row_path, allow_pickle=False))
    checkpoint = torch.load(
        args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"],
        map_location="cuda",
    )
    model = DirectionalActorGaussian(20, checkpoint["hidden_dimensions"]).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    torch.cuda.reset_peak_memory_stats()
    metrics = _evaluate(
        raw, model,
        np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        np.asarray(checkpoint["feature_scale"], dtype=np.float32),
        np.asarray(checkpoint["target_mean"], dtype=np.float32),
        np.asarray(checkpoint["target_scale"], dtype=np.float32), config,
    )
    decision_checks = {
        "fewer_events_than_actor_only": metrics["query_selected_occupancy_flip_events"]
        < metrics["actor_selected_occupancy_flip_events"],
        "no_more_events_than_frozen_p75": metrics["query_selected_occupancy_flip_events"]
        <= metrics["frozen_p75_selected_occupancy_flip_events"],
    }
    verdict = config["verdict_on_pass"] if all(decision_checks.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "fresh_confirmation_evaluation": metrics,
        "decision_checks": decision_checks,
        "resources": {"gpu": torch.cuda.get_device_name(0),
                      "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict,
                      "fresh_confirmation_evaluation": metrics,
                      "decision_checks": decision_checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
