"""Diagnose sparse route conflicts with a fixed route-eligible denominator."""

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


def _tail(values: list[float], fraction: float) -> dict[str, object]:
    count = max(1, int(math.ceil(len(values) * fraction)))
    order = np.argsort(np.asarray(values), kind="stable")[::-1][:count]
    selected = [float(values[int(index)]) for index in order]
    return {
        "tail_count": count,
        "cvar": float(np.mean(selected)),
        "maximum": float(max(values)),
        "tail_values": selected,
    }


def _arm(rows: list[dict[str, object]], name: str, fraction: float) -> dict[str, object]:
    densities = []
    conflicts = 0
    selected = 0
    eligible = 0
    for row in rows:
        arm = row["arms"][name]
        route_selected = int(arm["route_selected_count"])
        route_eligible = int(arm["route_eligible_count"])
        conflict = int(round(float(arm["route_hidden_free_conflict"]) * route_selected))
        conflicts += conflict
        selected += route_selected
        eligible += route_eligible
        densities.append(float(conflict / route_eligible) if route_eligible else 0.0)
    return {
        "route_eligible_count": eligible,
        "route_selected_count": selected,
        "route_hidden_free_conflict_count": conflicts,
        "pooled_fixed_denominator_conflict_density": float(conflicts / eligible),
        "fixed_denominator_tail": _tail(densities, fraction),
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
    fraction = float(config["tail"]["fraction"])
    expected = int(config["denominator"]["expected_case_count_per_cohort"])
    cohorts = {}
    for cohort, relative in config["inputs"].items():
        source = runs_root / str(relative)
        rows = [
            json.loads(line)
            for line in (source / "CASE_METRICS.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != expected:
            raise RuntimeError(f"expected {expected} rows for {cohort}, found {len(rows)}")
        m0 = _arm(rows, "m0_conditional", fraction)
        m1 = _arm(rows, "m1_route_aware", fraction)
        cohorts[str(cohort)] = {
            "case_count": len(rows),
            "m0": m0,
            "m1": m1,
            "m1_minus_m0_fixed_denominator_cvar": float(
                m1["fixed_denominator_tail"]["cvar"]
                - m0["fixed_denominator_tail"]["cvar"]
            ),
            "m1_minus_m0_pooled_fixed_denominator_density": float(
                m1["pooled_fixed_denominator_conflict_density"]
                - m0["pooled_fixed_denominator_conflict_density"]
            ),
        }
    directional = all(
        float(row["m1_minus_m0_fixed_denominator_cvar"]) <= 0.0
        for row in cohorts.values()
    )
    verdict = (
        "diagnosed_fixed_denominator_direction_consistent"
        if directional
        else "diagnosed_fixed_denominator_direction_not_consistent"
    )
    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "cohorts": cohorts,
        "directional_diagnostic_consistent": directional,
        "target_evidence_reread": False,
        "model_or_policy_change": False,
        "post_hoc_exploratory_diagnostic": True,
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
        "directional_diagnostic_consistent": directional,
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
