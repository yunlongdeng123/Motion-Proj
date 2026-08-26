"""Train the frozen P6R full-native selective MLP on consumed development scenes."""

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
from sklearn.metrics import roc_auc_score

from motion_proj.worldsim_v64.native_voxel_uq import sample_training_points_native
from motion_proj.worldsim_v64.selective_mlp import NativeBoundarySelectiveMLP


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    scenes = [str(row["name"]) for row in config["development_scenes"]]
    partition_by_scene = {
        scene: str(config["inputs"]["native_partition"]) for scene in scenes
    }
    train = sample_training_points_native(
        evidence_root,
        native_root,
        scenes,
        partition_by_scene=partition_by_scene,
        maximum_points_per_scene=int(config["sampling"]["maximum_points_per_scene"]),
        seed=int(config["seed"]),
        native_origin_m=config["native_grid"]["origin_m"],
        native_voxel_size_m=float(config["native_grid"]["voxel_size_m"]),
    )
    model_config = config["model"]
    model = NativeBoundarySelectiveMLP(
        hidden_dimensions=tuple(model_config["hidden_dimensions"]),
        dropout=float(model_config["dropout"]),
        focal_gamma=float(model_config["focal_gamma"]),
        focal_alpha=float(model_config["focal_alpha"]),
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
        epochs=int(model_config["epochs"]),
        batch_size=int(model_config["batch_size"]),
        seed=int(config["seed"]),
    ).fit(train.features, train.hidden_free)
    development_scores = model.score(train.features, train.logits)
    development_auroc = float(roc_auc_score(train.hidden_free, development_scores))
    model_dir = run_dir / "RISK_MODEL"
    model_dir.mkdir()
    joblib.dump(model, model_dir / "full_native_selective_mlp.joblib", compress=0)

    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": "development_training_complete",
        "claim_boundary": config["claim_boundary"],
        "development_scene_count": len(scenes),
        "fit": model.fit_summary,
        "development_auroc_descriptive_only": development_auroc,
        "independent_calibration_read": False,
        "new_confirmation_read": False,
        "parameter_sweep": False,
        "resources": {
            "gpu_used": True,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        },
        "failure_ledger_refs": config["failure_ledger_refs"],
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "resource.json", summary["resources"])
    _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": _utc_now()})
    return {"run_dir": str(run_dir), "verdict": summary["verdict"], "fit": summary["fit"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
