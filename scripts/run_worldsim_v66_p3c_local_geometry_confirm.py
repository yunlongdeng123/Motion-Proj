"""Exact-once确认冻结P3L local geometry head。"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v66.local_geometry_head import (
    FEATURE_NAMES,
    LocalGeometryHead,
    feature_matrix,
    labels,
    ranking_metrics,
    scene_support,
    score_head,
)
from motion_proj.worldsim_v66.natural_actor_conflict import materialize_rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v66" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    rows, materialization = materialize_rows(
        {
            "seed": config["seed"],
            "inputs": config["inputs"],
            "scenes": config["scenes"],
            "native_grid": config["native_grid"],
            "sampling": config["sampling"],
            "certificate": config["certificate"],
        },
        runs_root,
    )
    checkpoint_path = (
        runs_root / str(config["model"]["run"]) / str(config["model"]["artifact"])
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("P3C feature contract differs from frozen P3L checkpoint")
    model = LocalGeometryHead(len(FEATURE_NAMES), checkpoint["hidden_dimensions"])
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda().eval()
    values = feature_matrix(rows)
    target = labels(rows)
    scores = score_head(model, values, checkpoint["mean"], checkpoint["scale"])
    ranking = ranking_metrics(target, scores)
    q0_ranking = ranking_metrics(
        target, np.asarray([float(row["q0_mean"]) for row in rows])
    )
    deterministic = {"auroc": 0.5, "auprc": float(np.mean(target))}
    improvements = {
        "auroc_over_deterministic": ranking["auroc"] - deterministic["auroc"],
        "auprc_over_deterministic": ranking["auprc"] - deterministic["auprc"],
        "auroc_over_q0": ranking["auroc"] - q0_ranking["auroc"],
        "auprc_over_q0": ranking["auprc"] - q0_ranking["auprc"],
    }
    support = scene_support(rows, target, scores)
    evaluation = config["evaluation"]
    gates = {
        "minimum_auroc_improvement_over_deterministic": improvements[
            "auroc_over_deterministic"
        ]
        >= float(evaluation["minimum_auroc_improvement_over_deterministic"]),
        "minimum_auprc_improvement_over_deterministic": improvements[
            "auprc_over_deterministic"
        ]
        >= float(evaluation["minimum_auprc_improvement_over_deterministic"]),
        "minimum_above_chance_scene_count": support["above_chance_scene_count"]
        >= int(evaluation["minimum_above_chance_scene_count"]),
        "actor_existence_authority_remains_disabled": True,
    }
    verdict = (
        "supported_independent_legacy_local_geometry_confirmation"
        if all(gates.values())
        else "rejected_independent_legacy_local_geometry_confirmation"
    )
    (run_dir / "CONFIRMATION_SCORES.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "base_id": str(row["base_id"]),
                    "scene": str(row["scene"]),
                    "local_geometry_conflict": bool(row["local_geometry_conflict"]),
                    "q0_mean": float(row["q0_mean"]),
                    "p_local_conflict": float(score),
                },
                sort_keys=True,
            )
            + "\n"
            for row, score in zip(rows, scores)
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "worldsim_v66.p3c_local_geometry_confirmation_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "claim_boundary": config["claim_boundary"],
        "materialization": materialization,
        "row_count": len(rows),
        "class_count": {
            "conflict": int(np.count_nonzero(target)),
            "clean": int(np.count_nonzero(~target)),
        },
        "confirmation_ranking": ranking,
        "q0_ranking": q0_ranking,
        "deterministic_baseline": deterministic,
        "improvements": improvements,
        "scene_support": support,
        "gate_results": gates,
        "model_refit": False,
        "normalization_refit": False,
        "selection_threshold": None,
        "actor_existence_authority": False,
        "fresh_v66_quality_read": False,
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": "pending_result",
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "wall_seconds": time.monotonic() - started,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
