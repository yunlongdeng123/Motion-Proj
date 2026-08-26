"""运行 V6.4 fit-only hidden-FREE 监督风险头。"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml

from motion_proj.worldsim_v64.native_voxel_uq import (
    NativeBoundarySupervisedRisk,
    evaluate_scene_native_scores,
    sample_training_points_native,
)
from motion_proj.worldsim_v64.retrospective_uq import pooled_metrics


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v64" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "RISK_MODEL").mkdir()
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": _utc_now()},
    )
    started = time.monotonic()

    evidence_root = runs_root / config["inputs"]["evidence_run"]
    native_root = runs_root / config["inputs"]["native_run"]
    density_run = runs_root / config["inputs"]["density_run"]
    density_model = joblib.load(
        density_run / config["inputs"]["density_model_relative_path"]
    )
    partition_by_scene = config["inputs"]["native_partition_by_scene"]
    native_grid = config["native_grid"]
    train = sample_training_points_native(
        evidence_root,
        native_root,
        config["partitions"]["fit_scenes"],
        partition_by_scene=partition_by_scene,
        maximum_points_per_scene=int(config["sampling"]["fit_points_per_scene"]),
        seed=int(config["seed"]),
        native_origin_m=native_grid["origin_m"],
        native_voxel_size_m=float(native_grid["voxel_size_m"]),
    )
    model = NativeBoundarySupervisedRisk(
        representation=density_model,
        regularization_c=float(config["model"]["regularization_c"]),
        maximum_iterations=int(config["model"]["maximum_iterations"]),
        seed=int(config["seed"]),
    ).fit(train.features, train.logits, train.hidden_free)
    joblib.dump(model, run_dir / "RISK_MODEL" / "hidden_free_logistic.joblib")

    scene_metrics = []
    scene_arrays = []
    for scene in config["partitions"]["evaluation_scenes"]:
        metrics, arrays = evaluate_scene_native_scores(
            {
                "u2_feature_density": density_model,
                "u3_supervised_hidden_free": model,
            },
            evidence_root,
            native_root,
            scene,
            partition_by_scene=partition_by_scene,
            native_origin_m=native_grid["origin_m"],
            native_voxel_size_m=float(native_grid["voxel_size_m"]),
        )
        scene_metrics.append(metrics)
        scene_arrays.append(arrays)
    pooled = pooled_metrics(scene_arrays)
    score_name = "u3_supervised_hidden_free"
    pooled_auroc = float(pooled["scores"][score_name]["auroc"])
    scene_aurocs = {
        row["scene"]: float(row["scores"][score_name]["auroc"])
        for row in scene_metrics
    }
    gate_results = {
        "pooled_auroc": pooled_auroc
        >= float(config["gates"]["minimum_pooled_auroc"]),
        "all_scene_auroc": min(scene_aurocs.values())
        >= float(config["gates"]["minimum_scene_auroc"]),
    }
    wall = time.monotonic() - started
    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": "supported" if all(gate_results.values()) else "rejected",
        "scope": "v64_fresh_native_boundary_fit_only_supervised_risk",
        "denominator": "identical_to_p4n_unique_native_boundary_voxels",
        "claim_boundary": config["claim_boundary"],
        "fit_scenes": config["partitions"]["fit_scenes"],
        "evaluation_scenes": config["partitions"]["evaluation_scenes"],
        "fit_point_count": int(train.features.shape[0]),
        "fit_hidden_free_count": int(train.hidden_free.sum()),
        "fit_hidden_free_prevalence": float(train.hidden_free.mean()),
        "scene_metrics": scene_metrics,
        "pooled_metrics": pooled,
        "comparison": {
            "u2_feature_density_auroc": float(
                pooled["scores"]["u2_feature_density"]["auroc"]
            ),
            "u3_supervised_hidden_free_auroc": pooled_auroc,
            "u3_absolute_auroc_gain_over_u2": pooled_auroc
            - float(pooled["scores"]["u2_feature_density"]["auroc"]),
            "scene_u3_aurocs": scene_aurocs,
            "gate_results": gate_results,
        },
        "resources": {
            "gpu_used": False,
            "wall_seconds": wall,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2),
        },
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": "none",
    }
    _write_json(run_dir / "summary.json", summary)
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in scene_metrics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.write(json.dumps({"scene": "pooled", **pooled}, ensure_ascii=False) + "\n")
    _write_json(run_dir / "resource.json", summary["resources"])
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": _utc_now()},
    )
    return {
        "run_dir": str(run_dir),
        "verdict": summary["verdict"],
        **summary["comparison"],
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
