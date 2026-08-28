"""运行 V6.6 P4-D matched DROP / ABSTAIN / REPAIR。"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from motion_proj.worldsim_v66.physical_repair import compile_repair_arms, evaluate_repair_arms


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
    source_path = (
        runs_root / str(config["inputs"]["p1_run"]) / str(config["inputs"]["rows_relative_path"])
    )
    source_rows = [json.loads(line) for line in source_path.read_text().splitlines() if line]
    rows = compile_repair_arms(source_rows, config["certificate"])
    arm_metrics = evaluate_repair_arms(rows)
    candidate = arm_metrics["R2_REPAIR"]
    gates_config = config["gates"]
    gates = {
        "minimum_artifact_violation_reduction": candidate["artifact_violation_reduction"]
        >= float(gates_config["minimum_artifact_violation_reduction"]),
        "minimum_clean_hazard_actor_retention": candidate["clean_hazard_actor_retention"]
        >= float(gates_config["minimum_clean_hazard_actor_retention"]),
        "minimum_actor_id_track_trajectory_exactness": candidate[
            "actor_id_track_trajectory_exact_for_retained"
        ]
        >= float(gates_config["minimum_actor_id_track_trajectory_exactness"]),
        "maximum_hazard_event_count_shift": candidate["hazard_event_count_shift"]
        <= float(gates_config["maximum_hazard_event_count_shift"]),
        "maximum_nonartifact_regression": candidate["nonartifact_regression_rate"]
        <= float(gates_config["maximum_nonartifact_regression"]),
        "maximum_hard_observed_evidence_violations": candidate[
            "hard_observed_evidence_violations_after"
        ]
        <= int(gates_config["maximum_hard_observed_evidence_violations"]),
    }
    verdict = (
        "supported_development_repair_first_compiler"
        if all(gates.values())
        else "rejected_development_repair_first_compiler"
    )
    (run_dir / "REPAIR_ARM_ROWS.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "worldsim_v66.p4_repair_dev_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "claim_boundary": config["claim_boundary"],
        "source_row_count": len(source_rows),
        "compiled_row_count": len(rows),
        "arm_metrics": arm_metrics,
        "gate_results": gates,
        "learned_model_trained": False,
        "fresh_v66_quality_read": False,
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": "none",
        "resources": {
            "gpu": "none",
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
