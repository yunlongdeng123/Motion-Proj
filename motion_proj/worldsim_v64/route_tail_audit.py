"""Compute a frozen empirical CVaR audit from route-local conflict rows."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _empirical_upper_tail(values: list[float], fraction: float) -> dict[str, object]:
    tail_count = max(1, int(math.ceil(len(values) * fraction)))
    order = np.argsort(np.asarray(values), kind="stable")[::-1][:tail_count]
    tail = [float(values[int(index)]) for index in order]
    return {
        "tail_fraction": fraction,
        "tail_count": tail_count,
        "cvar": float(np.mean(tail)),
        "maximum": float(max(values)),
        "tail_values": tail,
    }


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v64" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()

    source_run = runs_root / config["inputs"]["route_conflict_run"]
    rows = [
        json.loads(line)
        for line in (source_run / "ROUTE_CONFLICT_ROWS.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_count = int(config["gates"]["expected_case_count"])
    if len(rows) != expected_count:
        raise RuntimeError(f"expected {expected_count} route rows, found {len(rows)}")
    c0_values = [float(row["c0_route_hidden_free_conflict_rate"] or 0.0) for row in rows]
    m0_values = [float(row["m0_route_hidden_free_conflict_rate"] or 0.0) for row in rows]
    fraction = float(config["tail"]["fraction"])
    c0_tail = _empirical_upper_tail(c0_values, fraction)
    m0_tail = _empirical_upper_tail(m0_values, fraction)
    threshold = float(config["gates"]["maximum_m0_empirical_cvar"])
    gates = {"maximum_m0_empirical_cvar": float(m0_tail["cvar"]) <= threshold}
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "case_count": len(rows),
        "c0": c0_tail,
        "m0": m0_tail,
        "m0_minus_c0_cvar": float(m0_tail["cvar"] - c0_tail["cvar"]),
        "m0_case_count_above_pointwise_threshold": sum(value > threshold for value in m0_values),
        "pointwise_threshold": threshold,
        "target_evidence_reread": False,
        "model_or_policy_change": False,
        "gate_results": gates,
        "resources": {
            "gpu_used": False,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        },
        "failure_ledger_refs": config["failure_ledger_refs"],
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "resource.json", summary["resources"])
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {
        "run_dir": str(run_dir),
        "verdict": verdict,
        "m0_empirical_cvar": m0_tail["cvar"],
        "m0_minus_c0_cvar": summary["m0_minus_c0_cvar"],
        "gate_results": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
