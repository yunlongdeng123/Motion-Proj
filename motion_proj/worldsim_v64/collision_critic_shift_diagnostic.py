"""Rows-only post-hoc calibration-shift diagnostic for the closed P11 critic family."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0)
    return {f"q{int(level * 100):02d}": float(np.quantile(values, level)) for level in levels}


def _arm_metrics(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    labels = np.asarray([row["actual_unsafe"] for row in selected], dtype=bool)
    scores = np.asarray([row["unsafe_probability"] for row in selected], dtype=np.float64)
    unsafe = scores[labels]
    safe = scores[~labels]
    return {
        "action_count": len(selected),
        "unsafe_action_count": int(labels.sum()),
        "unsafe_prior": float(labels.mean()),
        "average_precision": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
        "unsafe_score_quantiles": _quantiles(unsafe),
        "safe_score_quantiles": _quantiles(safe),
    }


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v64" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()
    source = runs_root / str(config["inputs"]["source_run"])
    calibration_rows = _read_jsonl(source / str(config["inputs"]["calibration_rows"]))
    evaluation_rows = _read_jsonl(source / str(config["inputs"]["evaluation_rows"]))
    thresholds = json.loads(
        (source / str(config["inputs"]["thresholds"])).read_text(encoding="utf-8")
    )["thresholds"]
    arms = [str(value) for value in config["arms"]]
    calibration = {arm: _arm_metrics(calibration_rows, arm) for arm in arms}
    evaluation = {arm: _arm_metrics(evaluation_rows, arm) for arm in arms}
    shifts = {}
    for arm in arms:
        cal = calibration[arm]
        test = evaluation[arm]
        shifts[arm] = {
            "unsafe_prior_delta": float(test["unsafe_prior"] - cal["unsafe_prior"]),
            "unsafe_q20_delta": float(
                test["unsafe_score_quantiles"]["q20"]
                - cal["unsafe_score_quantiles"]["q20"]
            ),
            "unsafe_median_delta": float(
                test["unsafe_score_quantiles"]["q50"]
                - cal["unsafe_score_quantiles"]["q50"]
            ),
            "safe_median_delta": float(
                test["safe_score_quantiles"]["q50"]
                - cal["safe_score_quantiles"]["q50"]
            ),
            "average_precision_delta": float(test["average_precision"] - cal["average_precision"]),
            "auroc_delta": float(test["auroc"] - cal["auroc"]),
            "frozen_threshold": float(thresholds[arm]),
        }
    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": "diagnosed_p11_cross_cohort_score_and_prior_shift",
        "claim_boundary": config["claim_boundary"],
        "post_hoc_rows_only": True,
        "confirmatory_gate": None,
        "calibration": calibration,
        "evaluation": evaluation,
        "shifts": shifts,
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
        "verdict": summary["verdict"],
        "verified_shift": shifts["real_plus_unc_verified"],
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
