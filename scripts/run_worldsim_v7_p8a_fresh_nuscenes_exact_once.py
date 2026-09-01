"""Exactly-once fresh-nuScenes evaluation of the frozen P6-C selector."""

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

from motion_proj.worldsim_v7.nuscenes_actor_surface import (
    build_selected_index,
    compile_nuscenes_scene,
)
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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
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
    p4_config = yaml.safe_load(
        (repo_root / str(config["p4_config"])).read_text(encoding="utf-8")
    )
    p6c_config = yaml.safe_load(
        (repo_root / str(config["p6c_config"])).read_text(encoding="utf-8")
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
    cohort = _read_json(repo_root / str(config["cohort"]))
    scene_names = [str(row["scene_name"]) for row in cohort["scenes"]]
    run_dir = (
        Path(str(config["runs_root"]))
        / "worldsim_v7"
        / str(config["task_id"])
        / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V7 P8-A is frozen to CUDA, but CUDA is unavailable")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    source_p4 = Path(str(config["source_p4_run"]))
    source_p6c = Path(str(config["source_p6c_run"]))

    try:
        candidate_artifact = torch.load(
            source_p6c / "MODEL.pt", map_location=device, weights_only=False
        )
        candidate = SmallMLP(
            len(VALIDITY_FEATURE_NAMES), int(p6c_config["model"]["hidden_dim"])
        ).to(device)
        candidate.load_state_dict(candidate_artifact["candidate_state"])
        candidate.eval()
        candidate_std = _standardizer(candidate_artifact["validity_standardizer"])

        p4_artifact = torch.load(
            source_p4 / "MODEL.pt", map_location=device, weights_only=False
        )
        p4_model = FactorizedTwoHead(
            len(VALIDITY_FEATURE_NAMES),
            len(HAZARD_FEATURE_NAMES),
            int(p4_config["model"]["hidden_dim"]),
        ).to(device)
        p4_model.load_state_dict(p4_artifact["factorized_state"])
        p4_model.eval()
        p4_validity_std = _standardizer(p4_artifact["validity_standardizer"])
        p4_hazard_std = _standardizer(p4_artifact["hazard_standardizer"])

        index = build_selected_index(
            Path(str(p4_config["nuscenes"]["dataset_root"])),
            {"fresh_final_test": scene_names},
            p4_config["nuscenes"]["allowed_category_prefixes"],
        )
        rows: list[dict[str, Any]] = []
        scene_actor_counts: dict[str, int] = {}
        for position, scene_name in enumerate(scene_names):
            scene_rows = compile_nuscenes_scene(
                scene_name,
                index["scenes"][scene_name],
                Path(str(p4_config["nuscenes"]["dataset_root"])),
                p4_config["nuscenes"]["actors"],
                compiler_config,
                device,
            )
            rows.extend(scene_rows)
            scene_actor_counts[scene_name] = len(scene_rows)
            print(
                json.dumps(
                    {
                        "stage": "fresh_nuScenes_exact_once",
                        "progress": f"{position + 1}/{len(scene_names)}",
                        "scene": scene_name,
                        "scene_actors": len(scene_rows),
                        "total_actors": len(rows),
                    }
                ),
                flush=True,
            )
        if not rows:
            raise RuntimeError("frozen fresh nuScenes cohort produced no eligible Actors")
        _write_jsonl(run_dir / "FRESH_NUSCENES_ACTORS.jsonl", rows)

        candidate_arrays = _standardized_arrays(rows, candidate_std, p4_hazard_std)
        p4_arrays = _standardized_arrays(rows, p4_validity_std, p4_hazard_std)
        candidate_scores = predict_validity(candidate, candidate_arrays, device)
        p4_scores, hazard_scores = predict(p4_model, p4_arrays, device)
        candidate_evaluation = evaluate_scores(
            candidate_arrays,
            candidate_scores,
            hazard_scores,
            float(candidate_artifact["threshold"]),
        )
        p4_evaluation = evaluate_scores(
            p4_arrays,
            p4_scores,
            hazard_scores,
            float(p4_artifact["thresholds"]["factorized"]),
        )

        calibration_rows = _read_jsonl(
            source_p4 / "NUSCENES_CALIBRATION_ACTORS.jsonl"
        )
        p4_calibration_arrays = _standardized_arrays(
            calibration_rows, p4_validity_std, p4_hazard_std
        )
        p4_calibration_scores, _ = predict(
            p4_model, p4_calibration_arrays, device
        )
        candidate_calibration_scores = np.asarray(
            candidate_artifact["calibration_scores"], dtype=np.float32
        )
        score_shift = {
            "candidate_wasserstein": float(
                wasserstein_distance(candidate_calibration_scores, candidate_scores)
            ),
            "p4_wasserstein": float(
                wasserstein_distance(p4_calibration_scores, p4_scores)
            ),
        }

        always_failure = 1.0 - (
            candidate_evaluation["repairable_count"]
            / max(candidate_evaluation["actor_count"], 1)
        )
        gates = {
            "candidate_repair_auroc_noninferior_to_p4": candidate_evaluation[
                "repairability"
            ]["auroc"]
            >= p4_evaluation["repairability"]["auroc"]
            - float(config["gates"]["maximum_candidate_auroc_degradation_from_p4"]),
            "candidate_coverage_nontrivial": candidate_evaluation["coverage"]
            >= float(config["gates"]["minimum_candidate_coverage"]),
            "candidate_false_repair_below_always_repair": candidate_evaluation[
                "false_repair_rate"
            ]
            < always_failure,
            "candidate_selective_chamfer_nonworse_than_query": candidate_evaluation[
                "mean_selective_surface_chamfer_m"
            ]
            <= candidate_evaluation["mean_query_chamfer_m"],
        }
        passed = all(bool(value) for value in gates.values())
        verdict = str(
            config["verdict_on_pass"] if passed else config["verdict_on_failure"]
        )
        scored_rows = [
            {
                "scene_name": str(row["scene_name"]),
                "track_id": str(row["track_id"]),
                "category": str(row["category"]),
                "hazardous": bool(row["hazardous"]),
                "repairable": bool(row["target_supported_repairable"]),
                "query_chamfer_m": float(row["query_only"]["symmetric_chamfer_m"]),
                "compiled_chamfer_m": float(row["after"]["symmetric_chamfer_m"]),
                "candidate_repair_score": float(candidate_scores[index]),
                "p4_repair_score": float(p4_scores[index]),
                "hazard_score": float(hazard_scores[index]),
                "candidate_selected": bool(
                    candidate_scores[index] >= float(candidate_artifact["threshold"])
                ),
                "p4_selected": bool(
                    p4_scores[index]
                    >= float(p4_artifact["thresholds"]["factorized"])
                ),
            }
            for index, row in enumerate(rows)
        ]
        _write_jsonl(run_dir / "FRESH_NUSCENES_SCORES.jsonl", scored_rows)
        summary = {
            "schema_version": "worldsim_v7.p8a_fresh_nuscenes_exact_once.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": verdict,
            "claim_boundary": config["claim_boundary"],
            "exact_once_protocol": {
                "quality_reads": 1,
                "frozen_scene_count": len(scene_names),
                "scene_replacement": False,
                "model_or_threshold_update_after_read": False,
                "metadata_only_selection": True,
            },
            "fresh_scene_count": len(scene_names),
            "fresh_actor_count": len(rows),
            "scene_actor_counts": scene_actor_counts,
            "candidate_evaluation": candidate_evaluation,
            "p4_evaluation": p4_evaluation,
            "candidate_score_distribution": _distribution(candidate_scores),
            "p4_score_distribution": _distribution(p4_scores),
            "score_shift_from_nuscenes_calibration": score_shift,
            "frozen_thresholds": {
                "candidate": float(candidate_artifact["threshold"]),
                "p4": float(p4_artifact["thresholds"]["factorized"]),
            },
            "gates": gates,
            "resources": {
                "gpu_used": True,
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device)
                / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return {"run_dir": str(run_dir), "verdict": verdict, "gates": gates}
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
