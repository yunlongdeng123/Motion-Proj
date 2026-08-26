"""Independent operating-point calibration for frozen bounded collision critics."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import torch
import yaml

from motion_proj.worldsim_v64.bounded_collision_critic import (
    _cohort_rows,
    _evaluate,
    _predict,
    _write_json,
    _write_jsonl,
)


def _unsafe_recall_threshold(
    probabilities: np.ndarray, labels: np.ndarray, target_recall: float
) -> float:
    unsafe = np.asarray(probabilities, dtype=np.float64)[np.asarray(labels, dtype=bool)]
    if not unsafe.size:
        raise RuntimeError("calibration cohort has no unsafe actions")
    return float(np.quantile(unsafe, 1.0 - float(target_recall), method="higher"))


def _public_metrics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evaluation.items() if key != "selected_actions"}


def run(config_path: Path, runs_root: Path, processed_root: Path, run_id: str) -> dict[str, Any]:
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
    if not torch.cuda.is_available():
        raise RuntimeError("calibrated collision critic requires the configured single CUDA GPU")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    risk_model = joblib.load(
        runs_root
        / str(config["inputs"]["risk_run"])
        / str(config["inputs"]["risk_model_relative_path"])
    )
    critic_payload = json.loads(
        (
            runs_root
            / str(config["inputs"]["critic_run"])
            / str(config["inputs"]["critic_models_relative_path"])
        ).read_text(encoding="utf-8")
    )
    critics = critic_payload["arms"]

    calibration_rows = _cohort_rows(
        config["inputs"]["calibration"],
        config=config,
        runs_root=runs_root,
        processed_root=processed_root,
        model=risk_model,
        device=device,
    )
    labels = np.asarray([row["actual_unsafe"] for row in calibration_rows], dtype=bool)
    target_recall = float(config["calibration"]["target_unsafe_action_recall"])
    thresholds = {}
    calibration_metrics = {}
    calibration_rows_out = []
    for arm, critic in critics.items():
        probabilities = _predict(critic, calibration_rows)
        threshold = _unsafe_recall_threshold(probabilities, labels, target_recall)
        thresholds[arm] = threshold
        evaluation = _evaluate(
            calibration_rows,
            probabilities,
            threshold=threshold,
            stuck_threshold=float(config["metrics"]["stuck_progress_ratio"]),
            calibration_bins=int(config["metrics"]["calibration_bins"]),
        )
        calibration_metrics[arm] = _public_metrics(evaluation)
        for row, probability in zip(calibration_rows, probabilities):
            calibration_rows_out.append(
                {
                    **{key: value for key, value in row.items() if key != "feature_values"},
                    "arm": arm,
                    "unsafe_probability": float(probability),
                    "calibrated_threshold": threshold,
                }
            )
    _write_jsonl(run_dir / "CALIBRATION_ACTION_ROWS.jsonl", calibration_rows_out)
    _write_json(
        run_dir / "CALIBRATED_THRESHOLDS.json",
        {
            "selection_rule": "unsafe_probability_quantile",
            "target_unsafe_action_recall": target_recall,
            "thresholds": thresholds,
            "calibration_metrics": calibration_metrics,
            "evaluation_action_labels_read": False,
        },
    )

    evaluation_rows = _cohort_rows(
        config["inputs"]["evaluation"],
        config=config,
        runs_root=runs_root,
        processed_root=processed_root,
        model=risk_model,
        device=device,
    )
    metrics = {}
    evaluation_rows_out = []
    for arm, critic in critics.items():
        probabilities = _predict(critic, evaluation_rows)
        evaluation = _evaluate(
            evaluation_rows,
            probabilities,
            threshold=float(thresholds[arm]),
            stuck_threshold=float(config["metrics"]["stuck_progress_ratio"]),
            calibration_bins=int(config["metrics"]["calibration_bins"]),
        )
        metrics[arm] = _public_metrics(evaluation)
        _write_jsonl(
            run_dir / f"SELECTED_ACTIONS_{arm.upper()}.jsonl", evaluation["selected_actions"]
        )
        for row, probability in zip(evaluation_rows, probabilities):
            evaluation_rows_out.append(
                {
                    **{key: value for key, value in row.items() if key != "feature_values"},
                    "arm": arm,
                    "unsafe_probability": float(probability),
                    "calibrated_threshold": float(thresholds[arm]),
                    "predicted_unsafe": bool(probability >= float(thresholds[arm])),
                }
            )
    _write_jsonl(run_dir / "EVALUATION_ACTION_ROWS.jsonl", evaluation_rows_out)

    real = metrics["real_only"]
    naive = metrics["real_plus_naive_generated"]
    verified = metrics["real_plus_unc_verified"]
    gates = {
        "verified_unsafe_action_recall": float(verified["unsafe_action_recall"])
        >= float(config["gates"]["minimum_verified_unsafe_action_recall"]),
        "verified_collision_false_safe_not_worse": int(
            verified["policy"]["collision_false_safe_count"]
        )
        <= min(
            int(real["policy"]["collision_false_safe_count"]),
            int(naive["policy"]["collision_false_safe_count"]),
        ),
        "verified_mean_progress_nontrivial": float(verified["policy"]["mean_progress_ratio"])
        >= float(config["gates"]["minimum_verified_mean_progress_ratio"]),
        "verified_stuck_rate_bounded": float(verified["policy"]["stuck_rate"])
        <= float(config["gates"]["maximum_verified_stuck_rate"]),
    }
    if not all(gates.values()):
        verdict = str(config["verdict_on_failure"])
    elif int(verified["policy"]["collision_false_safe_count"]) < int(
        real["policy"]["collision_false_safe_count"]
    ):
        verdict = str(config["verdict_on_pass"])
    else:
        verdict = str(config["verdict_on_no_increment"])
    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "source_critics_retrained": False,
        "calibration_case_count": len({row["case_id"] for row in calibration_rows}),
        "calibration_action_count": len(calibration_rows),
        "calibration_unsafe_action_count": int(labels.sum()),
        "calibrated_thresholds": thresholds,
        "calibration_metrics": calibration_metrics,
        "evaluation_case_count": len({row["case_id"] for row in evaluation_rows}),
        "evaluation_action_count": len(evaluation_rows),
        "evaluation_unsafe_action_count": sum(
            bool(row["actual_unsafe"]) for row in evaluation_rows
        ),
        "arms": metrics,
        "gate_results": gates,
        "large_nwm_trained": False,
        "evaluation_action_labels_read_after_threshold_freeze": True,
        "model_or_threshold_selection_during_evaluation": False,
        "resources": {
            "gpu_used": True,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        },
        "references": config["references"],
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
        "calibrated_thresholds": thresholds,
        "gate_results": gates,
        "policy_metrics": {arm: values["policy"] for arm, values in metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config.resolve(),
                args.runs_root.resolve(),
                args.processed_root.resolve(),
                args.run_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
