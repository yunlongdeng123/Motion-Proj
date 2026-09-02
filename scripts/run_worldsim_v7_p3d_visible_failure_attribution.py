"""Run frozen all-Actor output-provenance attribution for V7 P3-C failures."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from motion_proj.worldsim_v7.av2_four_action_compiler import compile_log
from motion_proj.worldsim_v7.visible_failure_attribution import (
    attribute_actor,
    summarize_attributions,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    p2_config = yaml.safe_load(
        (repo_root / str(config["p2_config"])).read_text(encoding="utf-8")
    )
    p2_config["compiler_geometry"].update(config.get("compiler_overrides", {}))
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    cohort = json.loads(
        (repo_root / str(config["cohort_config"])).read_text(encoding="utf-8")
    )
    roles = set(config["cohort_roles"])
    selected = [row for row in cohort["logs"] if row["role"] in roles]
    if len(selected) != int(config["expected_log_count"]):
        raise RuntimeError("frozen P3-D cohort count changed")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V7 P3-D is frozen to CUDA, but CUDA is unavailable")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []

    try:
        for position, cohort_row in enumerate(selected):
            log_id = str(cohort_row["log_id"])
            result = compile_log(
                Path(p2_config["dataset_root"]) / log_id,
                p2_config,
                device,
                include_diagnostics=True,
            )
            diagnostics = result["compiled"]["diagnostics"]
            for actor_row in result["actor_rows"]:
                track_id = str(actor_row["track_id"])
                row = attribute_actor(
                    actor_row,
                    diagnostics[track_id],
                    config["attribution"],
                    float(p2_config["compiler_geometry"]["output_voxel_size_m"]),
                    device,
                )
                row["log_id"] = log_id
                rows.append(row)
            print(
                json.dumps(
                    {"progress": f"{position + 1}/{len(selected)}", "log_id": log_id},
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
            },
        )
        raise

    summary_metrics = summarize_attributions(rows)
    (run_dir / "ACTOR_PROVENANCE_ATTRIBUTION.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "worldsim_v7.p3d_visible_failure_attribution.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": config["verdict"],
        "claim_boundary": config["claim_boundary"],
        "cohort": {"log_count": len(selected), "failed_log_deletion": False},
        "attribution": summary_metrics,
        "resources": {
            "gpu_used": True,
            "device": str(device),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "wall_seconds": time.monotonic() - started,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {"run_dir": str(run_dir), "verdict": config["verdict"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.repo_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
