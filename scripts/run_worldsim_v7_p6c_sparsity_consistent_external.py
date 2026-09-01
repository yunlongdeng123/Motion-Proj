"""Exactly-once fresh-AV2 phase for the frozen V7 P6-C selector."""

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
from motion_proj.worldsim_v7.selective_validity_hazard import (
    FactorizedTwoHead,
    HAZARD_FEATURE_NAMES,
    SmallMLP,
    Standardizer,
    VALIDITY_FEATURE_NAMES,
    evaluate_scores,
    predict,
    rows_to_arrays,
)
from motion_proj.worldsim_v7.sparsity_consistent_selector import predict_validity


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


def _standardized_arrays(
    rows: list[Mapping[str, Any]],
    validity: Standardizer,
    hazard: Standardizer,
) -> dict[str, np.ndarray]:
    arrays = rows_to_arrays(rows)
    return {
        **arrays,
        "validity": validity.transform(arrays["validity"]),
        "hazard": hazard.transform(arrays["hazard"]),
    }


def _distribution(scores: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(scores)),
        "standard_deviation": float(np.std(scores)),
        "quantiles_05_25_50_75_95": np.quantile(
            scores, [0.05, 0.25, 0.50, 0.75, 0.95]
        ).tolist(),
    }


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = Path(str(config["source_p4_run"]))
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / str(config["task_id"]) / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"frozen P6-C fit run does not exist: {run_dir}")
    fit_summary = _read_json(run_dir / "FIT_SUMMARY.json")
    if not all(bool(value) for value in fit_summary["fit_gates"].values()):
        raise RuntimeError("P6-C fit gates did not pass; fresh external read is prohibited")
    cohort = _read_json(repo_root / str(config["fresh_av2_cohort"]))
    state_root = Path(str(config["fresh_av2_download_state"]))
    missing_markers = [
        str(row["log_id"])
        for row in cohort["logs"]
        if not (state_root / f"{row['log_id']}.complete").is_file()
    ]
    if missing_markers:
        raise RuntimeError(
            f"fresh AV2 download incomplete: {len(missing_markers)}/{len(cohort['logs'])} markers missing"
        )

    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V7 P6-C external phase is frozen to CUDA, but CUDA is unavailable")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    _write_json(
        run_dir / "status.json",
        {
            "status": "fresh_external_running",
            "started_external_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        candidate_artifact = torch.load(
            run_dir / "MODEL.pt", map_location=device, weights_only=False
        )
        candidate = SmallMLP(
            len(VALIDITY_FEATURE_NAMES), int(config["model"]["hidden_dim"])
        ).to(device)
        candidate.load_state_dict(candidate_artifact["candidate_state"])
        candidate_std = _standardizer(candidate_artifact["validity_standardizer"])

        source_artifact = torch.load(
            source / "MODEL.pt", map_location=device, weights_only=False
        )
        source_model = FactorizedTwoHead(
            len(VALIDITY_FEATURE_NAMES), len(HAZARD_FEATURE_NAMES), 32
        ).to(device)
        source_model.load_state_dict(source_artifact["factorized_state"])
        source_validity_std = _standardizer(source_artifact["validity_standardizer"])
        source_hazard_std = _standardizer(source_artifact["hazard_standardizer"])

        p4_config = yaml.safe_load(
            (repo_root / str(config["p4_config"])).read_text(encoding="utf-8")
        )
        p3_config = yaml.safe_load(
            (repo_root / str(p4_config["p3_config"])).read_text(encoding="utf-8")
        )
        compiler_config = yaml.safe_load(
            (repo_root / str(p3_config["p2_config"])).read_text(encoding="utf-8")
        )
        compiler_config["compiler_geometry"].update(
            p3_config.get("compiler_overrides", {})
        )
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

        candidate_arrays = _standardized_arrays(
            rows, candidate_std, source_hazard_std
        )
        source_arrays = _standardized_arrays(
            rows, source_validity_std, source_hazard_std
        )
        candidate_scores = predict_validity(candidate, candidate_arrays, device)
        source_scores, hazard_scores = predict(source_model, source_arrays, device)
        candidate_evaluation = evaluate_scores(
            candidate_arrays,
            candidate_scores,
            hazard_scores,
            float(candidate_artifact["threshold"]),
        )
        source_evaluation = evaluate_scores(
            source_arrays,
            source_scores,
            hazard_scores,
            float(source_artifact["thresholds"]["factorized"]),
        )

        calibration_rows = _read_jsonl(source / "NUSCENES_CALIBRATION_ACTORS.jsonl")
        source_calibration_arrays = _standardized_arrays(
            calibration_rows, source_validity_std, source_hazard_std
        )
        source_calibration_scores, _ = predict(
            source_model, source_calibration_arrays, device
        )
        candidate_calibration_scores = np.asarray(
            candidate_artifact["calibration_scores"], dtype=np.float32
        )
        shift = {
            "candidate_wasserstein": float(
                wasserstein_distance(candidate_calibration_scores, candidate_scores)
            ),
            "source_p4_wasserstein": float(
                wasserstein_distance(source_calibration_scores, source_scores)
            ),
        }
        always_failure = 1.0 - (
            candidate_evaluation["repairable_count"]
            / max(candidate_evaluation["actor_count"], 1)
        )
        gates = {
            **fit_summary["fit_gates"],
            "fresh_av2_coverage_nontrivial": candidate_evaluation["coverage"]
            >= float(config["external_gates"]["minimum_fresh_av2_coverage"]),
            "fresh_av2_false_repair_below_always_repair": candidate_evaluation[
                "false_repair_rate"
            ]
            < always_failure,
            "fresh_av2_selective_chamfer_nonworse_than_query": candidate_evaluation[
                "mean_selective_surface_chamfer_m"
            ]
            <= candidate_evaluation["mean_query_chamfer_m"],
            "fresh_av2_score_shift_below_p4": shift["candidate_wasserstein"]
            < shift["source_p4_wasserstein"],
        }
        passed = all(bool(value) for value in gates.values())
        verdict = str(
            config["verdict_on_pass"] if passed else config["verdict_on_failure"]
        )
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
                "hazard_score": float(hazard_scores[index]),
                "candidate_selected": bool(
                    candidate_scores[index] >= float(candidate_artifact["threshold"])
                ),
                "p4_selected": bool(
                    source_scores[index]
                    >= float(source_artifact["thresholds"]["factorized"])
                ),
            }
            for index, row in enumerate(rows)
        ]
        _write_jsonl(run_dir / "FRESH_AV2_SCORES.jsonl", scored_rows)
        summary = {
            "schema_version": "worldsim_v7.p6c_sparsity_consistent_external.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": verdict,
            "claim_boundary": config["claim_boundary"],
            "fresh_log_count": len(cohort["logs"]),
            "fresh_actor_count": len(rows),
            "candidate_evaluation": candidate_evaluation,
            "source_p4_evaluation_on_fresh": source_evaluation,
            "candidate_score_distribution": _distribution(candidate_scores),
            "source_p4_score_distribution": _distribution(source_scores),
            "score_shift_from_nuscenes_calibration": shift,
            "fit_summary": fit_summary,
            "gates": gates,
            "resources": {
                "gpu_used": True,
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
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
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config.resolve(), args.repo_root.resolve(), args.run_id),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
