"""Run the nuScenes-only fit gate for V7 P6-C before any fresh AV2 read."""

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
from motion_proj.worldsim_v7.sparsity_consistent_selector import (
    fit_source_view_standardizer_values,
    mean_intervention_score_shift,
    opportunity_view_arrays,
    predict_validity,
    train_sparsity_consistent_model,
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


def _standardizer(payload: Mapping[str, Any]) -> Standardizer:
    return Standardizer(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        scale=np.asarray(payload["scale"], dtype=np.float32),
    )


def _standardize(
    arrays: Mapping[str, np.ndarray], validity: Standardizer, hazard: Standardizer
) -> dict[str, np.ndarray]:
    return {
        **arrays,
        "validity": validity.transform(arrays["validity"]),
        "hazard": hazard.transform(arrays["hazard"]),
    }


def _load_source(
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


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = Path(str(config["source_p4_run"]))
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "fitting", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V7 P6-C is frozen to CUDA, but CUDA is unavailable")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        role_rows = {
            role: _read_jsonl(source / f"NUSCENES_{role.upper()}_ACTORS.jsonl")
            for role in ("train", "calibration", "test")
        }
        scales = [float(value) for value in config["opportunity_augmentation"]["scale_factors"]]
        source_artifact, source_model, source_validity_std, source_hazard_std = _load_source(
            source, device
        )
        validity_std = Standardizer.fit(
            fit_source_view_standardizer_values(role_rows["train"], scales)
        )
        raw_original = {role: rows_to_arrays(rows) for role, rows in role_rows.items()}
        original = {
            role: _standardize(arrays, validity_std, source_hazard_std)
            for role, arrays in raw_original.items()
        }
        augmented = {
            role: [
                _standardize(
                    opportunity_view_arrays(role_rows[role], scale),
                    validity_std,
                    source_hazard_std,
                )
                for scale in scales
            ]
            for role in role_rows
        }
        torch.manual_seed(int(config["model"]["seed"]))
        candidate = SmallMLP(
            len(VALIDITY_FEATURE_NAMES), int(config["model"]["hidden_dim"])
        ).to(device)
        history = train_sparsity_consistent_model(
            candidate,
            original["train"],
            augmented["train"],
            config["model"],
            float(config["opportunity_augmentation"]["consistency_weight"]),
            device,
        )
        calibration_scores = predict_validity(candidate, original["calibration"], device)
        calibration = calibrate_crc_threshold(
            calibration_scores,
            original["calibration"]["repairable"],
            float(config["selective_risk"]["false_repair_alpha"]),
        )
        test_scores = predict_validity(candidate, original["test"], device)
        source_test = _standardize(
            raw_original["test"], source_validity_std, source_hazard_std
        )
        _, source_hazard_scores = predict(source_model, source_test, device)
        evaluation = evaluate_scores(
            original["test"],
            test_scores,
            source_hazard_scores,
            float(calibration["threshold"]),
        )
        candidate_shift = mean_intervention_score_shift(
            candidate, original["test"], augmented["test"], device
        )
        source_augmented = [
            _standardize(
                opportunity_view_arrays(role_rows["test"], scale),
                source_validity_std,
                source_hazard_std,
            )
            for scale in scales
        ]
        source_reference_scores, _ = predict(source_model, source_test, device)
        source_shift = float(
            np.mean(
                [
                    np.mean(
                        np.abs(predict(source_model, view, device)[0] - source_reference_scores)
                    )
                    for view in source_augmented
                ]
            )
        )
        shift_ratio = candidate_shift / max(source_shift, 1e-12)
        source_summary = _read_json(source / "summary.json")
        source_auroc = float(
            source_summary["nuscenes_evaluation"]["test"]["factorized"]["repairability"]["auroc"]
        )
        gates = {
            "nuscenes_test_repair_auroc_noninferior": float(
                evaluation["repairability"]["auroc"]
            )
            >= source_auroc
            - float(
                config["fit_gates"]["maximum_nuscenes_test_auroc_degradation_from_p4"]
            ),
            "intervention_score_shift_reduced": shift_ratio
            <= float(config["fit_gates"]["maximum_intervention_shift_ratio_to_p4"]),
        }
        passed = all(bool(value) for value in gates.values())
        status = (
            "model_frozen_waiting_fresh_av2"
            if passed
            else "fit_rejected_external_not_read"
        )
        torch.save(
            {
                "candidate_state": candidate.state_dict(),
                "validity_standardizer": validity_std.payload(),
                "validity_feature_names": VALIDITY_FEATURE_NAMES,
                "threshold": float(calibration["threshold"]),
                "calibration_scores": calibration_scores,
                "source_p4_threshold": float(source_artifact["thresholds"]["factorized"]),
                "training_dataset": "nuScenes only",
            },
            run_dir / "MODEL.pt",
        )
        summary = {
            "schema_version": "worldsim_v7.p6c_sparsity_consistent_fit.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": status,
            "fit_verdict": "passed" if passed else "rejected",
            "claim_boundary": config["claim_boundary"],
            "actor_counts": {role: len(rows) for role, rows in role_rows.items()},
            "training": {"history": history, "model_selection_sweep": False},
            "calibration": calibration,
            "nuscenes_test": evaluation,
            "source_p4_nuscenes_test_repair_auroc": source_auroc,
            "intervention": {
                "scale_factors": scales,
                "candidate_mean_score_shift": candidate_shift,
                "source_p4_mean_score_shift": source_shift,
                "candidate_to_p4_shift_ratio": shift_ratio,
            },
            "fit_gates": gates,
            "resources": {
                "gpu_used": True,
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "FIT_SUMMARY.json", summary)
        _write_json(
            run_dir / "status.json",
            {"status": status, "completed_fit_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        summary["run_dir"] = str(run_dir)
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
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
