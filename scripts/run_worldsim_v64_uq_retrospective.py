"""运行 WorldSim V6.4 原生 UQ 机制诊断。"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml

from motion_proj.worldsim_v64.retrospective_uq import (
    NativeFeatureDensityUQ,
    evaluate_scene,
    pooled_metrics,
    sample_training_points,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v64" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "GMM_MODEL").mkdir()
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": _utc_now()},
    )
    started = time.monotonic()
    native_root = runs_root / config["inputs"]["native_run"]
    surface_root = runs_root / config["inputs"]["surface_run"]
    native_partition_by_scene = config["inputs"].get(
        "native_partition_by_scene", {}
    )
    train = sample_training_points(
        surface_root,
        native_root,
        config["partitions"]["fit_scenes"],
        maximum_points_per_scene=int(config["sampling"]["fit_points_per_scene"]),
        seed=int(config["seed"]),
        native_partition_by_scene=native_partition_by_scene,
    )
    model = NativeFeatureDensityUQ(
        pca_dimension=int(config["model"]["pca_dimension"]),
        component_count=int(config["model"]["gmm_components"]),
        seed=int(config["seed"]),
    ).fit(train.features, train.logits)
    joblib.dump(model, run_dir / "GMM_MODEL" / "native_density.joblib")

    scene_metrics = []
    scene_arrays = []
    for scene in config["partitions"]["evaluation_scenes"]:
        metrics, arrays = evaluate_scene(
            model,
            surface_root,
            native_root,
            scene,
            chunk_size=int(config["sampling"]["evaluation_chunk_size"]),
            native_partition_by_scene=native_partition_by_scene,
        )
        scene_metrics.append(metrics)
        scene_arrays.append(arrays)
    pooled = pooled_metrics(scene_arrays)
    best_u0 = max(
        pooled["scores"][name]["auroc"]
        for name in ("u0_max_probability", "u0_entropy", "u0_inverse_margin")
    )
    u2_auroc = pooled["scores"]["u2_feature_density"]["auroc"]
    best_u0_auprc = max(
        pooled["scores"][name]["auprc"]
        for name in ("u0_max_probability", "u0_entropy", "u0_inverse_margin")
    )
    u2_auprc = pooled["scores"]["u2_feature_density"]["auprc"]
    support = sum(
        row["scores"]["u2_feature_density"]["auroc"]
        > max(
            row["scores"][name]["auroc"]
            for name in ("u0_max_probability", "u0_entropy", "u0_inverse_margin")
        )
        for row in scene_metrics
    )
    wall = time.monotonic() - started
    gates = config.get("gates")
    gate_results = None
    verdict = "diagnostic"
    if gates is not None:
        gate_results = {
            "pooled_auroc_gain": float(u2_auroc - best_u0)
            >= float(gates["minimum_pooled_auroc_gain"]),
            "scene_support": int(support)
            >= int(gates["minimum_scene_support"]),
        }
        verdict = "supported" if all(gate_results.values()) else "rejected"
    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "scope": config.get("scope", "v63_retrospective_mechanism_only"),
        "fresh_v64_claim_allowed": bool(
            config.get("locks", {}).get("fresh_v64_claim_allowed", False)
        ),
        "claim_boundary": config.get("claim_boundary"),
        "verdict": verdict,
        "fit_scenes": config["partitions"]["fit_scenes"],
        "evaluation_scenes": config["partitions"]["evaluation_scenes"],
        "fit_point_count": int(train.features.shape[0]),
        "fit_hidden_free_prevalence": float(train.hidden_free.mean()),
        "scene_metrics": scene_metrics,
        "pooled_metrics": pooled,
        "comparison": {
            "best_u0_auroc": float(best_u0),
            "u2_feature_density_auroc": float(u2_auroc),
            "u2_absolute_auroc_gain": float(u2_auroc - best_u0),
            "best_u0_auprc": float(best_u0_auprc),
            "u2_feature_density_auprc": float(u2_auprc),
            "u2_absolute_auprc_gain": float(u2_auprc - best_u0_auprc),
            "u2_scene_support": int(support),
            "scene_denominator": len(scene_metrics),
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
        handle.write(
            json.dumps({"scene": "pooled", **pooled}, ensure_ascii=False) + "\n"
        )
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"event": "run_started", "at_utc": _utc_now()}) + "\n"
        )
        handle.write(
            json.dumps({"event": "run_done", "at_utc": _utc_now()}) + "\n"
        )
    _write_json(
        run_dir / "resource.json",
        {"gpu_used": False, "wall_seconds": wall, "peak_rss_gib": summary["resources"]["peak_rss_gib"]},
    )
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": _utc_now()},
    )
    return {"run_dir": str(run_dir), **summary["comparison"]}


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
