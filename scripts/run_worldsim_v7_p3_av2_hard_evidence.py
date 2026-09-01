"""Run frozen V7 P3 hard-geometry evidence and deterministic visual extraction."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from motion_proj.worldsim_v7.av2_four_action_compiler import compile_log
from motion_proj.worldsim_v7.av2_p3_hard_evidence import (
    compile_hard_evidence,
    render_actor_panel,
    summarize_hard_evidence,
)


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
    maximum_state_shift = max(
        float(summary["maximum_trajectory_displacement_m"]),
        float(summary["maximum_speed_shift_mps"]),
        float(summary["maximum_acceleration_shift_mps2"]),
        float(summary["maximum_ttc_shift_s"]),
    )
    return {
        "maximum_free_space_violation_rate_after": summary[
            "free_space_violation_rate_after"
        ]
        <= float(config["maximum_free_space_violation_rate_after"]),
        "nondecreasing_ray_termination_consistency": summary[
            "ray_termination_consistency_after"
        ]
        >= summary["ray_termination_consistency_before"],
        "nonincreasing_lidar_depth_error": summary["lidar_depth_error_after_m"]
        <= summary["lidar_depth_error_before_m"],
        "maximum_sdf_zero_level_ratio": summary["sdf_zero_level_ratio"]
        <= float(config["maximum_sdf_zero_level_ratio"]),
        "maximum_ghost_component_ratio": summary["ghost_connected_component_ratio"]
        <= float(config["maximum_ghost_component_ratio"]),
        "maximum_chamfer_ratio": summary["symmetric_chamfer_ratio"]
        <= float(config["maximum_chamfer_ratio"]),
        "maximum_mean_temporal_surface_jitter": summary[
            "mean_temporal_surface_jitter_m"
        ]
        <= float(config["maximum_mean_temporal_surface_jitter_m"]),
        "exact_actor_hazard_state_retention": min(
            float(summary["actor_retention"]),
            float(summary["id_lifecycle_retention"]),
            float(summary["hazard_label_retention"]),
        )
        >= 1.0
        and maximum_state_shift <= float(config["maximum_state_shift"])
        and int(summary["hazard_event_count_change"]) == 0,
    }


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    p2_config = yaml.safe_load(
        (repo_root / str(config["p2_config"])).read_text(encoding="utf-8")
    )
    p2_config["compiler_geometry"].update(config.get("compiler_overrides", {}))
    run_dir = (
        Path(config["runs_root"]) / "worldsim_v7" / str(config["task_id"]) / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "visuals").mkdir()
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    cohort = json.loads(
        (repo_root / str(p2_config["cohort_config"])).read_text(encoding="utf-8")
    )
    roles = set(str(value) for value in config["cohort_roles"])
    selected = [row for row in cohort["logs"] if str(row["role"]) in roles]
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V7 P3 is frozen to CUDA, but CUDA is unavailable")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    visual_cases: list[dict[str, Any]] = []
    qualitative_position = 0

    try:
        for position, cohort_row in enumerate(selected):
            log_id = str(cohort_row["log_id"])
            role = str(cohort_row["role"])
            result = compile_log(
                Path(p2_config["dataset_root"]) / log_id,
                p2_config,
                device,
                include_diagnostics=True,
            )
            diagnostics = result["compiled"]["diagnostics"]
            log_evidence: dict[str, dict[str, Any]] = {}
            for actor_row in result["actor_rows"]:
                track_id = str(actor_row["track_id"])
                evidence = compile_hard_evidence(
                    actor_row,
                    diagnostics[track_id],
                    config["hard_evidence"],
                    device,
                )
                evidence["log_id"] = log_id
                evidence["role"] = role
                rows.append(evidence)
                log_evidence[track_id] = evidence

            if role == "qualitative":
                actor_ids = result["compiled"]["actor_ids"][: int(
                    config["visuals"]["actors_per_log"]
                )]
                for actor_rank, track_id in enumerate(actor_ids):
                    filename = (
                        f"q{qualitative_position:02d}_a{actor_rank}_"
                        f"{track_id[:8]}.png"
                    )
                    is_main = (
                        qualitative_position < int(config["visuals"]["main_log_count"])
                        and actor_rank == 0
                    )
                    render_actor_panel(
                        diagnostics[track_id],
                        log_evidence[track_id],
                        run_dir / "visuals" / filename,
                        int(config["visuals"]["point_limit_per_layer"]),
                    )
                    visual_cases.append(
                        {
                            "log_id": log_id,
                            "cohort_index": int(cohort_row["index"]),
                            "qualitative_position": qualitative_position,
                            "actor_rank": actor_rank,
                            "track_id": track_id,
                            "category": log_evidence[track_id]["category"],
                            "hazardous": log_evidence[track_id]["hazardous"],
                            "main": is_main,
                            "supplement": True,
                            "selection_rule": "first eligible Actor IDs in lexical order; no metric ranking",
                            "path": f"visuals/{filename}",
                        }
                    )
                qualitative_position += 1
            print(
                json.dumps(
                    {
                        "progress": f"{position + 1}/{len(selected)}",
                        "log_id": log_id,
                        "role": role,
                        "actors": len(result["actor_rows"]),
                        "visual_cases": len(visual_cases),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "quality_read_complete": False,
            },
        )
        raise

    summaries = {
        role: summarize_hard_evidence([row for row in rows if row["role"] == role])
        for role in sorted(roles)
    }
    gate_results = {
        role: _gates(summaries[role], config["gates"][role])
        for role in sorted(roles)
    }
    main_count = sum(bool(row["main"]) for row in visual_cases)
    supplement_count = sum(bool(row["supplement"]) for row in visual_cases)
    visual_gate_results = {
        "minimum_main_cases": main_count
        >= int(config["visuals"]["minimum_main_cases"]),
        "minimum_supplement_cases": supplement_count
        >= int(config["visuals"]["minimum_supplement_cases"]),
        "all_qualitative_logs_represented": len(
            {row["log_id"] for row in visual_cases}
        )
        == qualitative_position,
    }
    passed = (
        all(all(values.values()) for values in gate_results.values())
        and all(visual_gate_results.values())
    )
    verdict = str(config["verdict_on_pass"] if passed else config["verdict_on_failure"])
    resources = {
        "gpu_used": True,
        "device": str(device),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        "wall_seconds": time.monotonic() - started,
    }
    _write_jsonl(run_dir / "ACTOR_HARD_EVIDENCE.jsonl", rows)
    _write_jsonl(run_dir / "VISUAL_CASES.jsonl", visual_cases)
    _write_json(
        run_dir / "MAIN_PANELS.json",
        [row for row in visual_cases if row["main"]],
    )
    _write_json(run_dir / "SUPPLEMENT_PANELS.json", visual_cases)
    summary = {
        "schema_version": "worldsim_v7.p3_av2_hard_evidence.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "source_p2_run": config["source_p2_run"],
        "cohort": {
            "roles": sorted(roles),
            "log_count": len(selected),
            "quality_read_before_freeze": False,
            "failed_log_deletion": False,
            "visual_selection": "metadata-frozen logs plus lexical eligible Actor order",
        },
        "role_evaluation": summaries,
        "gate_results": gate_results,
        "visuals": {
            "main_case_count": main_count,
            "supplement_case_count": supplement_count,
            "qualitative_log_count": qualitative_position,
            "gate_results": visual_gate_results,
        },
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
