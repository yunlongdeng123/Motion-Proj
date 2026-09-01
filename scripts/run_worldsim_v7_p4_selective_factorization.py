"""Train on nuScenes and zero-shot evaluate V7 selective validity--hazard factorization."""

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

from motion_proj.worldsim_v7.av2_four_action_compiler import compile_log
from motion_proj.worldsim_v7.nuscenes_actor_surface import (
    build_selected_index,
    compile_nuscenes_scene,
)
from motion_proj.worldsim_v7.selective_validity_hazard import (
    FactorizedTwoHead,
    HAZARD_FEATURE_NAMES,
    SharedTwoHead,
    Standardizer,
    VALIDITY_FEATURE_NAMES,
    calibrate_crc_threshold,
    evaluate_scores,
    paired_input_leakage,
    predict,
    rows_to_arrays,
    train_model,
)


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


def _standardize(
    arrays: Mapping[str, np.ndarray],
    validity_standardizer: Standardizer,
    hazard_standardizer: Standardizer,
) -> dict[str, np.ndarray]:
    return {
        **arrays,
        "validity": validity_standardizer.transform(arrays["validity"]),
        "hazard": hazard_standardizer.transform(arrays["hazard"]),
    }


def _score_rows(
    rows: list[Mapping[str, Any]],
    shared_scores: tuple[np.ndarray, np.ndarray],
    factorized_scores: tuple[np.ndarray, np.ndarray],
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        output.append(
            {
                "dataset": str(row["dataset"]),
                "role": str(row["role"]),
                "scene_or_log": str(row.get("scene_name", row.get("log_id"))),
                "track_id": str(row["track_id"]),
                "category": str(row["category"]),
                "hazardous": bool(row["hazardous"]),
                "repairable": bool(row["target_supported_repairable"]),
                "query_chamfer_m": float(row["query_only"]["symmetric_chamfer_m"]),
                "compiled_chamfer_m": float(row["after"]["symmetric_chamfer_m"]),
                "shared_repair_score": float(shared_scores[0][index]),
                "shared_hazard_score": float(shared_scores[1][index]),
                "factorized_repair_score": float(factorized_scores[0][index]),
                "factorized_hazard_score": float(factorized_scores[1][index]),
                "shared_selected": bool(shared_scores[0][index] >= thresholds["shared"]),
                "factorized_selected": bool(
                    factorized_scores[0][index] >= thresholds["factorized"]
                ),
            }
        )
    return output


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    p3_config = yaml.safe_load(
        (repo_root / str(config["p3_config"])).read_text(encoding="utf-8")
    )
    compiler_config = yaml.safe_load(
        (repo_root / str(p3_config["p2_config"])).read_text(encoding="utf-8")
    )
    compiler_config["compiler_geometry"].update(p3_config.get("compiler_overrides", {}))
    run_dir = (
        Path(config["runs_root"]) / "worldsim_v7" / str(config["task_id"]) / run_id
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
        raise RuntimeError("V7 P4 is frozen to CUDA, but CUDA is unavailable")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    try:
        print(json.dumps({"stage": "nuScenes_index", "status": "starting"}), flush=True)
        index = build_selected_index(
            Path(str(config["nuscenes"]["dataset_root"])),
            config["nuscenes"]["role_scenes"],
            config["nuscenes"]["allowed_category_prefixes"],
        )
        role_rows: dict[str, list[dict[str, Any]]] = {
            role: [] for role in config["nuscenes"]["role_scenes"]
        }
        scene_order = [
            (role, scene_name)
            for role, scenes in config["nuscenes"]["role_scenes"].items()
            for scene_name in scenes
        ]
        for position, (role, scene_name) in enumerate(scene_order):
            rows = compile_nuscenes_scene(
                scene_name,
                index["scenes"][scene_name],
                Path(str(config["nuscenes"]["dataset_root"])),
                config["nuscenes"]["actors"],
                compiler_config,
                device,
            )
            role_rows[role].extend(rows)
            print(
                json.dumps(
                    {
                        "stage": "nuScenes_corpus",
                        "progress": f"{position + 1}/{len(scene_order)}",
                        "role": role,
                        "scene": scene_name,
                        "scene_actors": len(rows),
                        "role_actors": len(role_rows[role]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        for role, rows in role_rows.items():
            if not rows:
                raise RuntimeError(f"nuScenes role has no eligible actors: {role}")
            _write_jsonl(run_dir / f"NUSCENES_{role.upper()}_ACTORS.jsonl", rows)

        raw_arrays = {role: rows_to_arrays(rows) for role, rows in role_rows.items()}
        validity_standardizer = Standardizer.fit(raw_arrays["train"]["validity"])
        hazard_standardizer = Standardizer.fit(raw_arrays["train"]["hazard"])
        arrays = {
            role: _standardize(values, validity_standardizer, hazard_standardizer)
            for role, values in raw_arrays.items()
        }

        model_config = config["model"]
        torch.manual_seed(int(model_config["seed"]))
        shared = SharedTwoHead(
            len(VALIDITY_FEATURE_NAMES),
            len(HAZARD_FEATURE_NAMES),
            int(model_config["hidden_dim"]),
        ).to(device)
        shared_history = train_model(shared, arrays["train"], model_config, device)
        torch.manual_seed(int(model_config["seed"]))
        factorized = FactorizedTwoHead(
            len(VALIDITY_FEATURE_NAMES),
            len(HAZARD_FEATURE_NAMES),
            int(model_config["hidden_dim"]),
        ).to(device)
        factorized_history = train_model(factorized, arrays["train"], model_config, device)

        calibration_predictions = {
            "shared": predict(shared, arrays["calibration"], device),
            "factorized": predict(factorized, arrays["calibration"], device),
        }
        calibrations = {
            name: calibrate_crc_threshold(
                scores[0],
                arrays["calibration"]["repairable"],
                float(config["selective_risk"]["false_repair_alpha"]),
            )
            for name, scores in calibration_predictions.items()
        }
        thresholds = {
            name: float(row["threshold"]) for name, row in calibrations.items()
        }
        nuscenes_evaluation: dict[str, Any] = {}
        scored_rows: list[dict[str, Any]] = []
        for role in ("calibration", "test"):
            shared_scores = (
                calibration_predictions["shared"]
                if role == "calibration"
                else predict(shared, arrays[role], device)
            )
            factorized_scores = (
                calibration_predictions["factorized"]
                if role == "calibration"
                else predict(factorized, arrays[role], device)
            )
            nuscenes_evaluation[role] = {
                "shared": evaluate_scores(
                    arrays[role], *shared_scores, thresholds["shared"]
                ),
                "factorized": evaluate_scores(
                    arrays[role], *factorized_scores, thresholds["factorized"]
                ),
                "shared_paired_leakage": paired_input_leakage(
                    shared, arrays[role], device
                ),
                "factorized_paired_leakage": paired_input_leakage(
                    factorized, arrays[role], device
                ),
            }
            scored_rows.extend(
                _score_rows(
                    role_rows[role],
                    shared_scores,
                    factorized_scores,
                    thresholds,
                )
            )

        print(
            json.dumps(
                {"stage": "model_frozen", "calibrations": calibrations},
                ensure_ascii=False,
            ),
            flush=True,
        )
        av2_cohort = json.loads(
            (repo_root / str(compiler_config["cohort_config"])).read_text(encoding="utf-8")
        )
        av2_rows: list[dict[str, Any]] = []
        for position, cohort_row in enumerate(av2_cohort["logs"]):
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
                row["role"] = str(cohort_row["role"])
                row["target_supported_repairable"] = bool(
                    float(row["after"]["symmetric_chamfer_m"])
                    <= float(row["query_only"]["symmetric_chamfer_m"])
                )
                row["clean_query_to_compiled_chamfer_delta_m"] = float(
                    row["after"]["symmetric_chamfer_m"]
                    - row["query_only"]["symmetric_chamfer_m"]
                )
                av2_rows.append(row)
            print(
                json.dumps(
                    {
                        "stage": "AV2_zero_shot",
                        "progress": f"{position + 1}/{len(av2_cohort['logs'])}",
                        "log_id": log_id,
                        "actors": len(av2_rows),
                    }
                ),
                flush=True,
            )
        _write_jsonl(run_dir / "AV2_ZERO_SHOT_ACTORS.jsonl", av2_rows)
        av2_raw = rows_to_arrays(av2_rows)
        av2_arrays = _standardize(
            av2_raw, validity_standardizer, hazard_standardizer
        )
        av2_shared = predict(shared, av2_arrays, device)
        av2_factorized = predict(factorized, av2_arrays, device)
        av2_evaluation = {
            "shared": evaluate_scores(
                av2_arrays, *av2_shared, thresholds["shared"]
            ),
            "factorized": evaluate_scores(
                av2_arrays, *av2_factorized, thresholds["factorized"]
            ),
            "shared_paired_leakage": paired_input_leakage(
                shared, av2_arrays, device
            ),
            "factorized_paired_leakage": paired_input_leakage(
                factorized, av2_arrays, device
            ),
            "threshold_source": "nuScenes calibration only",
            "exchangeability_guarantee": False,
        }
        scored_rows.extend(
            _score_rows(
                av2_rows, av2_shared, av2_factorized, thresholds
            )
        )
        _write_jsonl(run_dir / "SELECTIVE_SCORES.jsonl", scored_rows)

        shared_test = nuscenes_evaluation["test"]["shared"]
        factorized_test = nuscenes_evaluation["test"]["factorized"]
        av2_main = av2_evaluation["factorized"]
        av2_always_failure = 1.0 - (
            av2_main["repairable_count"] / max(av2_main["actor_count"], 1)
        )
        gates = {
            "nuscenes_repairability_noninferior_to_shared": factorized_test[
                "repairability"
            ]["auroc"]
            >= shared_test["repairability"]["auroc"]
            - float(config["gates"]["maximum_main_task_auroc_degradation"]),
            "nuscenes_hazard_noninferior_to_shared": factorized_test["hazard"][
                "auroc"
            ]
            >= shared_test["hazard"]["auroc"]
            - float(config["gates"]["maximum_main_task_auroc_degradation"]),
            "factorized_cross_input_leakage_zero": max(
                nuscenes_evaluation["test"]["factorized_paired_leakage"].values()
            )
            <= float(config["gates"]["maximum_factorized_score_shift"]),
            "av2_zero_shot_coverage_nontrivial": av2_main["coverage"]
            >= float(config["gates"]["minimum_av2_coverage"]),
            "av2_false_repair_below_always_repair": av2_main["false_repair_rate"]
            < av2_always_failure,
            "av2_selective_surface_nonworse_than_query": av2_main[
                "mean_selective_surface_chamfer_m"
            ]
            <= av2_main["mean_query_chamfer_m"],
            "actor_hazard_state_retained": min(
                av2_main["actor_retention"], av2_main["hazard_label_retention"]
            )
            >= 1.0,
        }
        passed = all(gates.values())
        verdict = str(
            config["verdict_on_pass"] if passed else config["verdict_on_failure"]
        )
        torch.save(
            {
                "shared_state": shared.state_dict(),
                "factorized_state": factorized.state_dict(),
                "validity_standardizer": validity_standardizer.payload(),
                "hazard_standardizer": hazard_standardizer.payload(),
                "validity_feature_names": VALIDITY_FEATURE_NAMES,
                "hazard_feature_names": HAZARD_FEATURE_NAMES,
                "thresholds": thresholds,
                "training_dataset": "nuScenes only",
            },
            run_dir / "MODEL.pt",
        )
        summary = {
            "schema_version": "worldsim_v7.p4_selective_factorization.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": verdict,
            "claim_boundary": config["claim_boundary"],
            "nuscenes_scene_counts": {
                role: len(scenes)
                for role, scenes in config["nuscenes"]["role_scenes"].items()
            },
            "nuscenes_actor_counts": {
                role: len(rows) for role, rows in role_rows.items()
            },
            "training": {
                "dataset": "nuScenes only",
                "shared_history": shared_history,
                "factorized_history": factorized_history,
                "fixed_epochs": int(model_config["epochs"]),
                "model_selection_sweep": False,
            },
            "calibrations": calibrations,
            "nuscenes_evaluation": nuscenes_evaluation,
            "av2_zero_shot_evaluation": av2_evaluation,
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
