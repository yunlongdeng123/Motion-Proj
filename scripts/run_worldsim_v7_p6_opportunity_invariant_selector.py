"""Fit on nuScenes, then exactly-once evaluate the V7-F11 recovery on fresh AV2."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from scipy.stats import wasserstein_distance

from motion_proj.worldsim_v7.av2_four_action_compiler import compile_log
from motion_proj.worldsim_v7.opportunity_invariant_selector import (
    OPPORTUNITY_VALIDITY_FEATURE_NAMES,
    maximum_opportunity_feature_shift,
    predict_validity,
    rows_to_opportunity_arrays,
    train_validity_model,
)
from motion_proj.worldsim_v7.selective_validity_hazard import (
    FactorizedTwoHead,
    HAZARD_FEATURE_NAMES,
    SmallMLP,
    Standardizer,
    VALIDITY_FEATURE_NAMES,
    calibrate_crc_threshold,
    evaluate_scores,
    predict,
    rows_to_arrays,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _standardizer(payload: Mapping[str, Any]) -> Standardizer:
    return Standardizer(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        scale=np.asarray(payload["scale"], dtype=np.float32),
    )


def _source_model(
    source: Path, device: torch.device
) -> tuple[dict[str, Any], FactorizedTwoHead, Standardizer, Standardizer]:
    artifact = torch.load(source / "MODEL.pt", map_location=device, weights_only=False)
    model = FactorizedTwoHead(
        len(VALIDITY_FEATURE_NAMES), len(HAZARD_FEATURE_NAMES), 32
    ).to(device)
    model.load_state_dict(artifact["factorized_state"])
    return (
        artifact,
        model,
        _standardizer(artifact["validity_standardizer"]),
        _standardizer(artifact["hazard_standardizer"]),
    )


def _source_arrays(
    rows: list[Mapping[str, Any]],
    validity_standardizer: Standardizer,
    hazard_standardizer: Standardizer,
) -> dict[str, np.ndarray]:
    arrays = rows_to_arrays(rows)
    return {
        **arrays,
        "validity": validity_standardizer.transform(arrays["validity"]),
        "hazard": hazard_standardizer.transform(arrays["hazard"]),
    }


def _candidate_arrays(
    rows: list[Mapping[str, Any]],
    validity_standardizer: Standardizer,
    hazard_standardizer: Standardizer,
) -> dict[str, np.ndarray]:
    arrays = rows_to_opportunity_arrays(rows)
    return {
        **arrays,
        "validity": validity_standardizer.transform(arrays["validity"]),
        "hazard": hazard_standardizer.transform(arrays["hazard"]),
    }


def _score_distribution(scores: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "quantiles_05_25_50_75_95": np.quantile(
            scores, [0.05, 0.25, 0.50, 0.75, 0.95]
        ).tolist(),
    }


def fit(config: Mapping[str, Any], run_dir: Path, device: torch.device) -> dict[str, Any]:
    source = Path(str(config["source_p4_run"]))
    role_rows = {
        role: _read_jsonl(source / f"NUSCENES_{role.upper()}_ACTORS.jsonl")
        for role in ("train", "calibration", "test")
    }
    raw = {role: rows_to_opportunity_arrays(rows) for role, rows in role_rows.items()}
    validity_standardizer = Standardizer.fit(raw["train"]["validity"])
    source_artifact, source_model, source_validity_std, source_hazard_std = _source_model(
        source, device
    )
    arrays = {
        role: _candidate_arrays(rows, validity_standardizer, source_hazard_std)
        for role, rows in role_rows.items()
    }
    torch.manual_seed(int(config["model"]["seed"]))
    candidate = SmallMLP(
        len(OPPORTUNITY_VALIDITY_FEATURE_NAMES), int(config["model"]["hidden_dim"])
    ).to(device)
    history = train_validity_model(candidate, arrays["train"], config["model"], device)
    calibration_scores = predict_validity(candidate, arrays["calibration"], device)
    calibration = calibrate_crc_threshold(
        calibration_scores,
        arrays["calibration"]["repairable"],
        float(config["selective_risk"]["false_repair_alpha"]),
    )
    test_scores = predict_validity(candidate, arrays["test"], device)
    source_test_arrays = _source_arrays(
        role_rows["test"], source_validity_std, source_hazard_std
    )
    _, source_test_hazard = predict(source_model, source_test_arrays, device)
    test_evaluation = evaluate_scores(
        arrays["test"], test_scores, source_test_hazard, float(calibration["threshold"])
    )
    source_summary = _read_json(source / "summary.json")
    source_test_auroc = float(
        source_summary["nuscenes_evaluation"]["test"]["factorized"]["repairability"]["auroc"]
    )
    opportunity_shift = maximum_opportunity_feature_shift(
        role_rows["test"], config["opportunity_intervention"]["scale_factors"]
    )
    fit_gates = {
        "nuscenes_test_repair_auroc_noninferior": float(
            test_evaluation["repairability"]["auroc"]
        )
        >= source_test_auroc
        - float(config["gates"]["maximum_nuscenes_test_auroc_degradation_from_p4"]),
        "opportunity_transform_invariant": opportunity_shift
        <= float(
            config["opportunity_intervention"]["maximum_transformed_feature_shift"]
        ),
    }
    torch.save(
        {
            "candidate_state": candidate.state_dict(),
            "validity_standardizer": validity_standardizer.payload(),
            "validity_feature_names": OPPORTUNITY_VALIDITY_FEATURE_NAMES,
            "threshold": float(calibration["threshold"]),
            "calibration_scores": calibration_scores,
            "training_dataset": "nuScenes only",
            "source_p4_threshold": float(source_artifact["thresholds"]["factorized"]),
        },
        run_dir / "MODEL.pt",
    )
    result = {
        "status": "model_frozen_waiting_fresh_av2",
        "training": {
            "dataset": "nuScenes only",
            "actor_count": len(role_rows["train"]),
            "history": history,
            "model_selection_sweep": False,
        },
        "calibration": calibration,
        "nuscenes_test": test_evaluation,
        "source_p4_nuscenes_test_repair_auroc": source_test_auroc,
        "opportunity_intervention_max_feature_shift": opportunity_shift,
        "fit_gates": fit_gates,
    }
    _write_json(run_dir / "FIT_SUMMARY.json", result)
    _write_json(
        run_dir / "status.json",
        {
            "status": "model_frozen_waiting_fresh_av2",
            "completed_fit_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return result


def external(
    config: Mapping[str, Any], repo_root: Path, run_dir: Path, device: torch.device
) -> dict[str, Any]:
    source = Path(str(config["source_p4_run"]))
    fit_summary = _read_json(run_dir / "FIT_SUMMARY.json")
    candidate_artifact = torch.load(
        run_dir / "MODEL.pt", map_location=device, weights_only=False
    )
    candidate = SmallMLP(
        len(OPPORTUNITY_VALIDITY_FEATURE_NAMES), int(config["model"]["hidden_dim"])
    ).to(device)
    candidate.load_state_dict(candidate_artifact["candidate_state"])
    candidate_std = _standardizer(candidate_artifact["validity_standardizer"])
    _, source_model, source_validity_std, source_hazard_std = _source_model(source, device)

    p4_config = yaml.safe_load(
        (repo_root / str(config["p4_config"])).read_text(encoding="utf-8")
    )
    p3_config = yaml.safe_load(
        (repo_root / str(p4_config["p3_config"])).read_text(encoding="utf-8")
    )
    compiler_config = yaml.safe_load(
        (repo_root / str(p3_config["p2_config"])).read_text(encoding="utf-8")
    )
    compiler_config["compiler_geometry"].update(p3_config.get("compiler_overrides", {}))
    cohort = _read_json(repo_root / str(config["fresh_av2_cohort"]))
    rows: list[dict[str, Any]] = []
    for position, cohort_row in enumerate(cohort["logs"]):
        log_id = str(cohort_row["log_id"])
        compiled = compile_log(
            Path(str(compiler_config["dataset_root"])) / log_id,
            compiler_config,
            device,
            include_diagnostics=False,
        )
        for row in compiled["actor_rows"]:
            row["dataset"] = "Argoverse2"
            row["log_id"] = log_id
            row["role"] = "fresh_external_confirmation"
            row["target_supported_repairable"] = bool(
                float(row["after"]["symmetric_chamfer_m"])
                <= float(row["query_only"]["symmetric_chamfer_m"])
            )
            rows.append(row)
        print(
            json.dumps(
                {
                    "stage": "fresh_AV2",
                    "progress": f"{position + 1}/{len(cohort['logs'])}",
                    "log_id": log_id,
                    "actors": len(rows),
                }
            ),
            flush=True,
        )
    _write_jsonl(run_dir / "FRESH_AV2_ACTORS.jsonl", rows)

    candidate_arrays = _candidate_arrays(rows, candidate_std, source_hazard_std)
    source_arrays = _source_arrays(rows, source_validity_std, source_hazard_std)
    candidate_scores = predict_validity(candidate, candidate_arrays, device)
    source_scores, source_hazard = predict(source_model, source_arrays, device)
    candidate_eval = evaluate_scores(
        candidate_arrays,
        candidate_scores,
        source_hazard,
        float(candidate_artifact["threshold"]),
    )
    source_eval = evaluate_scores(
        source_arrays,
        source_scores,
        source_hazard,
        float(candidate_artifact["source_p4_threshold"]),
    )
    calibration_candidate = np.asarray(
        candidate_artifact["calibration_scores"], dtype=np.float32
    )
    calibration_rows = _read_jsonl(source / "NUSCENES_CALIBRATION_ACTORS.jsonl")
    source_calibration_arrays = _source_arrays(
        calibration_rows, source_validity_std, source_hazard_std
    )
    source_calibration_scores, _ = predict(
        source_model, source_calibration_arrays, device
    )
    shift = {
        "candidate_wasserstein": float(
            wasserstein_distance(calibration_candidate, candidate_scores)
        ),
        "source_p4_wasserstein": float(
            wasserstein_distance(source_calibration_scores, source_scores)
        ),
    }
    always_failure = 1.0 - (
        candidate_eval["repairable_count"] / max(candidate_eval["actor_count"], 1)
    )
    gates = {
        **fit_summary["fit_gates"],
        "fresh_av2_coverage_nontrivial": candidate_eval["coverage"]
        >= float(config["gates"]["minimum_fresh_av2_coverage"]),
        "fresh_av2_false_repair_below_always_repair": candidate_eval["false_repair_rate"]
        < always_failure,
        "fresh_av2_selective_chamfer_nonworse_than_query": candidate_eval[
            "mean_selective_surface_chamfer_m"
        ]
        <= candidate_eval["mean_query_chamfer_m"],
        "fresh_av2_score_shift_below_p4": shift["candidate_wasserstein"]
        < shift["source_p4_wasserstein"],
    }
    scored_rows = [
        {
            "log_id": str(row["log_id"]),
            "track_id": str(row["track_id"]),
            "category": str(row["category"]),
            "hazardous": bool(row["hazardous"]),
            "repairable": bool(row["target_supported_repairable"]),
            "query_chamfer_m": float(row["query_only"]["symmetric_chamfer_m"]),
            "compiled_chamfer_m": float(row["after"]["symmetric_chamfer_m"]),
            "candidate_repair_score": float(candidate_scores[index]),
            "p4_repair_score": float(source_scores[index]),
            "hazard_score": float(source_hazard[index]),
        }
        for index, row in enumerate(rows)
    ]
    _write_jsonl(run_dir / "FRESH_AV2_SCORES.jsonl", scored_rows)
    passed = all(bool(value) for value in gates.values())
    verdict = str(config["verdict_on_pass"] if passed else config["verdict_on_failure"])
    summary = {
        "schema_version": "worldsim_v7.p6_opportunity_invariant_selector.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "fresh_log_count": len(cohort["logs"]),
        "fresh_actor_count": len(rows),
        "candidate_evaluation": candidate_eval,
        "source_p4_evaluation_on_fresh": source_eval,
        "candidate_score_distribution": _score_distribution(candidate_scores),
        "source_p4_score_distribution": _score_distribution(source_scores),
        "score_shift_from_nuscenes_calibration": shift,
        "fit_summary": fit_summary,
        "gates": gates,
        "resources": {
            "gpu_used": True,
            "device": str(device),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return summary


def run(config_path: Path, repo_root: Path, run_id: str, phase: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / str(config["task_id"]) / run_id
    if phase == "fit":
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        _write_json(
            run_dir / "status.json",
            {"status": "fitting", "started_at_utc": datetime.now(timezone.utc).isoformat()},
        )
    elif not run_dir.is_dir():
        raise FileNotFoundError(f"frozen fit run does not exist: {run_dir}")
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V7 P6 is frozen to CUDA, but CUDA is unavailable")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        result = (
            fit(config, run_dir, device)
            if phase == "fit"
            else external(config, repo_root, run_dir, device)
        )
        result["run_dir"] = str(run_dir)
        result["phase_wall_seconds"] = time.monotonic() - started
        return result
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": phase,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=("fit", "external"), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config.resolve(),
                args.repo_root.resolve(),
                args.run_id,
                args.phase,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
