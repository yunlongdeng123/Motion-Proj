"""Run the frozen V7 P1 AV2 Actor-surface factorial atlas."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v7.av2_canonical_surface import extract_log, summarize


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs_root = Path(config["runs_root"])
    run_dir = runs_root / "worldsim_v7" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = run_dir / "logs"
    surfaces_dir = run_dir / "surfaces"
    logs_dir.mkdir()
    surfaces_dir.mkdir()
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    cohort_path = repo_root / str(config["cohort_config"])
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    selected = [
        row for row in cohort["logs"] if str(row["role"]) == str(config["cohort_role"])
    ]
    data_root = Path(config["dataset_root"])
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V7 P1 is frozen to CUDA, but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    actor_rows: list[dict[str, Any]] = []
    atlas_rows: list[dict[str, Any]] = []
    log_rows = []
    for log_index, cohort_row in enumerate(selected):
        log_id = str(cohort_row["log_id"])
        log_started = time.monotonic()
        result = extract_log(data_root / log_id, config, device)
        for row in result["actor_rows"]:
            row["log_id"] = log_id
        for row in result["atlas_rows"]:
            row["log_id"] = log_id
        actor_rows.extend(result["actor_rows"])
        atlas_rows.extend(result["atlas_rows"])
        surface_payload = result["surfaces"]
        np.savez_compressed(
            surfaces_dir / f"{log_id}.npz",
            points=surface_payload["points"],
            temporal_support=surface_payload["temporal_support"],
            view_support=surface_payload["view_support"],
            offsets=surface_payload["offsets"],
        )
        _write_json(
            surfaces_dir / f"{log_id}.actors.json",
            {"actor_ids": surface_payload["actor_ids"]},
        )
        log_summary = {
            "log_id": log_id,
            "cohort_index": int(cohort_row["index"]),
            "metadata_tracks": int(result["eligible_metadata_tracks"]),
            "eligible_surface_actors": len(result["actor_rows"]),
            "hazard_actors": int(sum(row["hazardous"] for row in result["actor_rows"])),
            "stable_surfels": int(len(surface_payload["points"])),
            "atlas_probes": len(result["atlas_rows"]),
            "sweeps_read": int(result["sweeps_read"]),
            "wall_seconds": time.monotonic() - log_started,
        }
        log_rows.append(log_summary)
        _write_json(logs_dir / f"{log_id}.json", log_summary)
        print(
            json.dumps(
                {
                    "progress": f"{log_index + 1}/{len(selected)}",
                    **log_summary,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if not actor_rows or not atlas_rows:
        raise RuntimeError("frozen AV2 cohort produced no evaluable Actor surfaces")
    evaluation = summarize(actor_rows, atlas_rows)
    gates_config = config["gates"]
    gates = {
        "minimum_eligible_actor_count": evaluation["eligible_actor_count"]
        >= int(gates_config["minimum_eligible_actor_count"]),
        "minimum_hazard_actor_count": evaluation["hazard_actor_count"]
        >= int(gates_config["minimum_hazard_actor_count"]),
        "minimum_artifact_detection_recall": evaluation["artifact_detection_recall"]
        >= float(gates_config["minimum_artifact_detection_recall"]),
        "maximum_clean_hazard_false_artifact_rate": evaluation[
            "clean_hazard_false_artifact_rate"
        ]
        <= float(gates_config["maximum_clean_hazard_false_artifact_rate"]),
        "minimum_action_accuracy": evaluation["action_accuracy"]
        >= float(gates_config["minimum_action_accuracy"]),
        "minimum_actor_hazard_retention": min(
            evaluation["actor_identity_trajectory_size_retention"],
            evaluation["hazard_label_retention"],
        )
        >= float(gates_config["minimum_actor_hazard_retention"]),
        "minimum_ghost_free_space_violation_reduction": evaluation[
            "ghost_free_space_violation_reduction"
        ]
        >= float(gates_config["minimum_ghost_free_space_violation_reduction"]),
        "maximum_validity_hazard_leakage": max(
            evaluation["safe_to_hazard_artifact_score_shift"],
            evaluation["clean_to_artifact_hazard_score_shift"],
        )
        <= float(gates_config["maximum_validity_hazard_leakage"]),
        "maximum_fused_target_distance_ratio": evaluation[
            "fused_target_distance_ratio"
        ]
        <= float(gates_config["maximum_fused_target_distance_ratio"]),
        "minimum_mean_recall_delta": evaluation["mean_recall_delta"]
        >= float(gates_config["minimum_mean_recall_delta"]),
    }
    verdict = (
        str(config["verdict_on_pass"])
        if all(gates.values())
        else str(config["verdict_on_failure"])
    )
    _write_jsonl(run_dir / "ACTOR_METRICS.jsonl", actor_rows)
    _write_jsonl(run_dir / "FACTORIAL_ATLAS.jsonl", atlas_rows)
    _write_jsonl(run_dir / "LOG_SUMMARIES.jsonl", log_rows)
    resources = {
        "gpu_used": device.type == "cuda",
        "device": str(device),
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        "wall_seconds": time.monotonic() - started,
    }
    summary = {
        "schema_version": "worldsim_v7.p1_av2_factorial_atlas_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "cohort": {
            "role": config["cohort_role"],
            "log_count": len(selected),
            "quality_read_before_freeze": False,
            "fine_tuning": False,
            "threshold_selection": False,
        },
        "evaluation": evaluation,
        "gate_results": gates,
        "resources": resources,
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
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config.resolve(), args.repo_root.resolve(), args.run_id),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
