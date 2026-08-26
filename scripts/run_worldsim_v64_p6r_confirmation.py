"""Apply the frozen 40% P6R policy once on the exact-once confirmation cohort."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import yaml

from motion_proj.worldsim_v64.native_voxel_uq import _native_boundary_chunk, _native_unit_dir, _unit_dirs


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v64" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    evidence_root = runs_root / config["inputs"]["evidence_run"]
    native_root = runs_root / config["inputs"]["native_run"]
    risk_root = runs_root / config["inputs"]["risk_run"]
    model = joblib.load(risk_root / config["inputs"]["risk_model_relative_path"])
    scenes = [str(row["name"]) for row in config["confirmation_scenes"]]
    strata = {str(row["name"]): str(row["stratum"]) for row in config["confirmation_scenes"]}
    partition_by_scene = {scene: str(config["inputs"]["native_partition"]) for scene in scenes}
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    coverage = float(config["policy"]["nominal_coverage"])
    conflict_threshold = float(config["policy"]["hidden_free_conflict_threshold"])
    cases = []
    for scene in scenes:
        for evidence_unit in _unit_dirs(evidence_root, scene):
            native_unit = _native_unit_dir(native_root, scene, evidence_unit.name, partition_by_scene)
            chunk = _native_boundary_chunk(
                evidence_unit, native_unit, native_origin_m=origin, native_voxel_size_m=voxel_size
            )
            scores = model.score(chunk.features, chunk.logits)
            order = np.argsort(scores, kind="stable")
            selected_count = max(1, int(np.floor(coverage * order.size)))
            conflict = float(chunk.hidden_free[order[:selected_count]].mean())
            cases.append(
                {
                    "scene": scene,
                    "stratum": strata[scene],
                    "unit": evidence_unit.name,
                    "eligible_count": int(order.size),
                    "selected_count": selected_count,
                    "realized_coverage": float(selected_count / order.size),
                    "hidden_free_conflict": conflict,
                    "case_loss": bool(conflict > conflict_threshold),
                }
            )
    stratum_results = {}
    for stratum in sorted(set(strata.values())):
        rows = [row for row in cases if row["stratum"] == stratum]
        failures = sum(bool(row["case_loss"]) for row in rows)
        stratum_results[stratum] = {
            "case_count": len(rows),
            "failure_count": failures,
            "empirical_case_risk": float(failures / len(rows)),
        }
    failures = sum(bool(row["case_loss"]) for row in cases)
    gates = {
        "overall_case_risk_at_most_target": failures <= int(config["gates"]["maximum_overall_failures"]),
        "each_stratum_case_risk_at_most_target": all(
            row["failure_count"] <= int(config["gates"]["maximum_failures_per_stratum"])
            for row in stratum_results.values()
        ),
    }
    verdict = "supported_exact_once_confirmation" if all(gates.values()) else "rejected_exact_once_confirmation"
    with (run_dir / "CASE_METRICS.jsonl").open("w", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "scene_count": len(scenes),
        "case_count": len(cases),
        "nominal_coverage": coverage,
        "mean_realized_coverage": float(np.mean([row["realized_coverage"] for row in cases])),
        "failure_count": failures,
        "empirical_case_risk": float(failures / len(cases)),
        "strata": stratum_results,
        "gate_results": gates,
        "model_refit": False,
        "policy_selection": False,
        "confirmation_read": True,
        "resources": {
            "gpu_used": True,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        },
        "failure_ledger_refs": config["failure_ledger_refs"],
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "resource.json", summary["resources"])
    _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"run_dir": str(run_dir), "verdict": verdict, "failure_count": failures, "gate_results": gates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
