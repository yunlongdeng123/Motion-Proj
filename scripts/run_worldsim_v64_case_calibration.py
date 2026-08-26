"""在独立 calibration case 上冻结 U3 selective coverage policy。"""

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
from scipy.stats import beta

from motion_proj.worldsim_v64.native_voxel_uq import (
    _native_boundary_chunk,
    _native_unit_dir,
    _unit_dirs,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _upper_bound(failures: int, count: int, confidence: float) -> float:
    if failures >= count:
        return 1.0
    return float(beta.ppf(confidence, failures + 1, count - failures))


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v64" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": _utc_now()})
    started = time.monotonic()

    evidence_root = runs_root / config["inputs"]["evidence_run"]
    native_root = runs_root / config["inputs"]["native_run"]
    risk_root = runs_root / config["inputs"]["risk_run"]
    risk_model = joblib.load(risk_root / config["inputs"]["risk_model_relative_path"])
    partition = str(config["inputs"]["native_partition"])
    partition_by_scene = {
        str(row["name"]): partition for row in config["calibration_scenes"]
    }
    stratum_by_scene = {
        str(row["name"]): str(row["stratum"])
        for row in config["calibration_scenes"]
    }
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    coverages = [float(value) for value in config["policy"]["candidate_coverages"]]
    conflict_threshold = float(config["policy"]["hidden_free_conflict_threshold"])
    cases = []
    for scene in partition_by_scene:
        for evidence_unit in _unit_dirs(evidence_root, scene):
            native_unit = _native_unit_dir(
                native_root, scene, evidence_unit.name, partition_by_scene
            )
            chunk = _native_boundary_chunk(
                evidence_unit,
                native_unit,
                native_origin_m=origin,
                native_voxel_size_m=voxel_size,
            )
            scores = risk_model.score(chunk.features, chunk.logits)
            order = np.argsort(scores, kind="stable")
            policies = {}
            for coverage in coverages:
                selected_count = max(1, int(np.floor(coverage * order.size)))
                selected = order[:selected_count]
                conflict = float(chunk.hidden_free[selected].mean())
                policies[str(coverage)] = {
                    "selected_count": selected_count,
                    "coverage": float(selected_count / order.size),
                    "hidden_free_conflict": conflict,
                    "case_loss": bool(conflict > conflict_threshold),
                }
            cases.append(
                {
                    "scene": scene,
                    "stratum": stratum_by_scene[scene],
                    "unit": evidence_unit.name,
                    "eligible_count": int(order.size),
                    "policies": policies,
                }
            )

    target_risk = float(config["policy"]["target_case_risk"])
    confidence = float(config["policy"]["confidence"])
    adjusted_confidence = 1.0 - (1.0 - confidence) / len(coverages)
    candidates = []
    for coverage in coverages:
        key = str(coverage)
        failures = sum(bool(row["policies"][key]["case_loss"]) for row in cases)
        stratum_rows = {}
        for stratum in sorted(set(stratum_by_scene.values())):
            selected_cases = [row for row in cases if row["stratum"] == stratum]
            stratum_rows[stratum] = {
                "case_count": len(selected_cases),
                "failure_count": sum(
                    bool(row["policies"][key]["case_loss"])
                    for row in selected_cases
                ),
            }
        candidates.append(
            {
                "nominal_coverage": coverage,
                "mean_realized_coverage": float(
                    np.mean([row["policies"][key]["coverage"] for row in cases])
                ),
                "case_count": len(cases),
                "failure_count": failures,
                "empirical_case_risk": float(failures / len(cases)),
                "simultaneous_upper_bound": _upper_bound(
                    failures, len(cases), adjusted_confidence
                ),
                "strata": stratum_rows,
            }
        )
    passing = [
        row for row in candidates if row["simultaneous_upper_bound"] <= target_risk
    ]
    selected = max(passing, key=lambda row: row["nominal_coverage"]) if passing else None
    policy = {
        "schema_version": "worldsim_v64.p6_case_policy.v1",
        "score": "u3_supervised_hidden_free",
        "selection_unit": "case_target",
        "selected_nominal_coverage": (
            None if selected is None else selected["nominal_coverage"]
        ),
        "hidden_free_conflict_threshold": conflict_threshold,
        "target_case_risk": target_risk,
        "confidence": confidence,
        "simultaneous_selection": config["policy"]["simultaneous_selection"],
        "candidate_coverages": coverages,
        "confirmation_read": False,
    }
    _write_json(run_dir / "CALIBRATION_POLICY.json", policy)
    with (run_dir / "CASE_METRICS.jsonl").open("w", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": "supported_selective_policy" if selected is not None else "rejected_no_positive_coverage",
        "claim_boundary": config["claim_boundary"],
        "case_count": len(cases),
        "scene_count": len(partition_by_scene),
        "stratum_count": len(set(stratum_by_scene.values())),
        "candidate_results": candidates,
        "selected_policy": policy,
        "gate_results": {
            "finite_sample_upper_bound_at_most_target": selected is not None,
            "positive_coverage": selected is not None
            and float(selected["mean_realized_coverage"])
            >= float(config["policy"]["minimum_positive_coverage"]),
        },
        "calibration_target_read": True,
        "confirmation_read": False,
        "exact_once_test_read": False,
        "resources": {
            "gpu_used": False,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        },
        "failure_ledger_refs": config["failure_ledger_refs"],
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "resource.json", summary["resources"])
    _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": _utc_now()})
    return {
        "run_dir": str(run_dir),
        "verdict": summary["verdict"],
        "selected_policy": policy,
        "gate_results": summary["gate_results"],
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
