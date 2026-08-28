"""运行 V6.6 P1-D Actor validity×hazard development atlas。"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from motion_proj.worldsim_v66.actor_factorial import (
    build_factorial_rows,
    evaluate_atlas,
    materialize_base_rows,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
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
    torch.cuda.reset_peak_memory_stats()
    base_rows, materialization = materialize_base_rows(config, runs_root)
    rows = build_factorial_rows(
        base_rows,
        config["artifact_families"],
        config["certificate"],
    )
    metrics = evaluate_atlas(rows)
    family_signal_count = sum(
        value >= 0.9 for value in metrics["certificate_family_recall"].values()
    )
    gates = {
        "minimum_actor_unit_base_rows": metrics["base_actor_unit_count"]
        >= int(config["evaluation"]["minimum_actor_unit_base_rows"]),
        "minimum_artifact_signal_family_count": family_signal_count
        >= int(config["evaluation"]["minimum_artifact_signal_family_count"]),
        "q0_leaves_artifact_signal_unresolved": metrics["q0_artifact_ranking"]["auroc"]
        <= 0.55,
        "hazard_pair_certificate_invariant": metrics[
            "mean_absolute_hazard_pair_certificate_delta"
        ]
        <= 0.03,
    }
    verdict = (
        "supported_development_factorial_separation_proceed_to_p2"
        if all(gates.values())
        else "rejected_development_factorial_separation"
    )
    (run_dir / "ACTOR_FACTORIAL_ROWS.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "worldsim_v66.p1_factorial_atlas_dev_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "claim_boundary": config["claim_boundary"],
        **materialization,
        **metrics,
        "gate_results": gates,
        "fresh_v66_quality_read": False,
        "corruption_metadata_used_by_inference": False,
        "physical_hazard_edit_claim": False,
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": "none",
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "wall_seconds": time.monotonic() - started,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {
        "run_dir": str(run_dir),
        "verdict": verdict,
        "gate_results": gates,
        "resources": summary["resources"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config.resolve(), args.runs_root.resolve(), args.run_id),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
