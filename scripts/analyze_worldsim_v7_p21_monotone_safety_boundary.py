"""Derive the P21 monotone first-return safety boundary from P20."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0.0 else None


def _format(value: float | None, digits: int = 3) -> str:
    return "undefined" if value is None else f"{value:.{digits}f}"


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = json.loads((Path(config["input_run"]) / "summary.json").read_text(encoding="utf-8"))
    if source["ray_operator"] != config["ray_operator"]:
        raise RuntimeError("P20/P21 ray operator changed")
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "analysis"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    started = time.monotonic()
    rows = []
    for name in config["policies"]:
        policy = source["policies"][name]
        baseline = policy["baseline"]
        candidate = policy["p16"]
        hazard_baseline = policy["hazard"]["baseline"]
        hazard_candidate = policy["hazard"]["p16"]
        hazard_removed = int(
            hazard_baseline["new_early_count"] - hazard_candidate["new_early_count"]
        )
        total_removed = int(
            baseline["new_early_count"] - candidate["new_early_count"]
        )
        new_hits_lost = int(baseline["new_hit_count"] - candidate["new_hit_count"])
        chamfer_penalty_m = float(
            candidate["mean_chamfer_m"] - baseline["mean_chamfer_m"]
        )
        rows.append(
            {
                "policy": name,
                "completion_coverage": float(policy["completion_coverage"]),
                "hazard_events_removed": hazard_removed,
                "total_events_removed": total_removed,
                "new_hits_lost": new_hits_lost,
                "mean_chamfer_penalty_m": chamfer_penalty_m,
                "hazard_events_removed_per_new_hit_lost": _ratio(
                    hazard_removed, new_hits_lost
                ),
                "hazard_events_removed_per_chamfer_millimeter": _ratio(
                    hazard_removed, 1000.0 * chamfer_penalty_m
                ),
            }
        )
    theorem = {
        "definition": "d_S(r)=minimum positive projected depth within the frozen lateral tolerance, or infinity",
        "premise": "S_prime is a subset of S and query/target ray geometry is fixed",
        "result": "d_S_prime(r) >= d_S(r), so early(S_prime,r) <= early(S,r)",
        "new_early_corollary": "with a fixed query surface, new-early is monotone non-increasing under deletion",
        "not_guaranteed": ["hit retention", "symmetric Chamfer", "collision freedom"],
    }
    summary = {
        "schema_version": "worldsim_v7.p21_monotone_safety_boundary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": "supported_monotone_first_return_safety_boundary",
        "input_run": config["input_run"],
        "input_evidence_status": source["source_evidence_status"],
        "theorem": theorem,
        "policy_frontier": rows,
        "fresh_target_data_read": False,
        "resources": {"device": "cpu", "wall_seconds": time.monotonic() - started},
    }
    _write_json(run_dir / "summary.json", summary)
    lines = [
        "# Monotone First-Return Safety Boundary",
        "",
        "For a fixed ray and `S' subset S`, the minimum valid positive depth obeys `d_S' >= d_S`; therefore deleting points cannot increase the literal first-return early predicate. Hit retention, symmetric Chamfer, and collision freedom are not implied.",
        "",
        "| Policy | Coverage | Hazard early removed | Total early removed | New hits lost | Chamfer penalty (mm) | Hazard removed / hit lost | Hazard removed / Chamfer mm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {100.0 * row['completion_coverage']:.2f}% | "
            f"{row['hazard_events_removed']} | {row['total_events_removed']} | "
            f"{row['new_hits_lost']} | {1000.0 * row['mean_chamfer_penalty_m']:.4f} | "
            f"{_format(row['hazard_events_removed_per_new_hit_lost'])} | "
            f"{_format(row['hazard_events_removed_per_chamfer_millimeter'])} |"
        )
    lines.extend(
        [
            "",
            "All ratios are descriptive for the consumed nuScenes source-development cohort. No ratio is a formal road-safety guarantee or fresh transfer result.",
            "",
        ]
    )
    (run_dir / "BOUNDARY.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json(
        run_dir / "status.json",
        {
            "status": "done",
            "phase": "analysis",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {"run_dir": str(run_dir), "verdict": summary["verdict"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
