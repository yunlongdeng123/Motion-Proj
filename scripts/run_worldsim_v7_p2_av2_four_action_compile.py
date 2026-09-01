"""Run the frozen V7 P2 real-coordinate four-action AV2 compiler."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v7.av2_four_action_compiler import compile_log, summarize_role


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


def _gates(summary: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, bool]:
    actions = summary["action_counts"]
    return {
        "minimum_actor_count": summary["actor_count"] >= int(config["minimum_actor_count"]),
        "minimum_hazard_actor_count": summary["hazard_actor_count"]
        >= int(config["minimum_hazard_actor_count"]),
        "minimum_each_action_count": min(
            int(actions.get(action, 0)) for action in ("KEEP", "PROJECT", "COMPLETE", "UNKNOWN")
        )
        >= int(config["minimum_each_action_count"]),
        "minimum_clean_keep_rate": summary["clean_keep_rate"]
        >= float(config["minimum_clean_keep_rate"]),
        "minimum_ghost_project_rate": summary["ghost_project_rate"]
        >= float(config["minimum_ghost_project_rate"]),
        "minimum_unknown_artifact_rate": summary["unknown_artifact_rate"]
        >= float(config["minimum_unknown_artifact_rate"]),
        "minimum_completion_decision_rate": summary["completion_decision_rate"]
        >= float(config["minimum_completion_decision_rate"]),
        "minimum_completion_target_support": summary["completion_target_support"]
        >= float(config["minimum_completion_target_support"]),
        "minimum_artifact_repair_or_reject_rate": summary["artifact_repair_or_reject_rate"]
        >= float(config["minimum_artifact_repair_or_reject_rate"]),
        "minimum_target_recall_delta": summary["mean_target_recall_delta"]
        >= float(config["minimum_target_recall_delta"]),
        "minimum_surface_precision_delta": summary["mean_surface_precision_delta"]
        >= float(config["minimum_surface_precision_delta"]),
        "maximum_chamfer_ratio": summary["mean_chamfer_ratio"]
        <= float(config["maximum_chamfer_ratio"]),
        "minimum_actor_hazard_retention": min(
            summary["actor_identity_trajectory_size_retention"],
            summary["hazard_label_retention"],
        )
        >= float(config["minimum_actor_hazard_retention"]),
    }


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = (
        Path(config["runs_root"]) / "worldsim_v7" / str(config["task_id"]) / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    (run_dir / "compiled_surfaces").mkdir()
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    cohort = json.loads(
        (repo_root / str(config["cohort_config"])).read_text(encoding="utf-8")
    )
    roles = set(str(value) for value in config["cohort_roles"])
    selected = [row for row in cohort["logs"] if str(row["role"]) in roles]
    data_root = Path(config["dataset_root"])
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V7 P2 is frozen to CUDA, but CUDA is unavailable")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    actor_rows: list[dict[str, Any]] = []
    log_rows = []
    for position, cohort_row in enumerate(selected):
        log_id = str(cohort_row["log_id"])
        role = str(cohort_row["role"])
        log_started = time.monotonic()
        result = compile_log(data_root / log_id, config, device)
        for row in result["actor_rows"]:
            row["log_id"] = log_id
            row["role"] = role
        actor_rows.extend(result["actor_rows"])
        compiled = result["compiled"]
        np.savez_compressed(
            run_dir / "compiled_surfaces" / f"{log_id}.npz",
            points=compiled["points"],
            offsets=compiled["offsets"],
        )
        _write_json(
            run_dir / "compiled_surfaces" / f"{log_id}.actors.json",
            {"actor_ids": compiled["actor_ids"]},
        )
        log_summary = {
            "log_id": log_id,
            "role": role,
            "cohort_index": int(cohort_row["index"]),
            "metadata_tracks": int(result["metadata_tracks"]),
            "compiled_actors": len(result["actor_rows"]),
            "hazard_actors": int(sum(row["hazardous"] for row in result["actor_rows"])),
            "compiled_points": int(len(compiled["points"])),
            "sweeps_read": int(result["sweeps_read"]),
            "wall_seconds": time.monotonic() - log_started,
        }
        log_rows.append(log_summary)
        _write_json(run_dir / "logs" / f"{log_id}.json", log_summary)
        print(
            json.dumps(
                {"progress": f"{position + 1}/{len(selected)}", **log_summary},
                ensure_ascii=False,
            ),
            flush=True,
        )
    summaries = {
        role: summarize_role([row for row in actor_rows if row["role"] == role])
        for role in sorted(roles)
    }
    gate_results = {
        role: _gates(summaries[role], config["gates"][role]) for role in sorted(roles)
    }
    passed = all(all(values.values()) for values in gate_results.values())
    verdict = str(config["verdict_on_pass"] if passed else config["verdict_on_failure"])
    _write_jsonl(run_dir / "ACTOR_COMPILE_METRICS.jsonl", actor_rows)
    _write_jsonl(run_dir / "LOG_SUMMARIES.jsonl", log_rows)
    resources = {
        "gpu_used": True,
        "device": str(device),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        "wall_seconds": time.monotonic() - started,
    }
    summary = {
        "schema_version": "worldsim_v7.p2_av2_four_action_compile_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "cohort": {
            "roles": sorted(roles),
            "log_count": len(selected),
            "quality_read_before_freeze": False,
            "fine_tuning": False,
            "threshold_selection": False,
            "failed_log_deletion": False,
        },
        "role_evaluation": summaries,
        "gate_results": gate_results,
        "resources": resources,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gate_results}


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
