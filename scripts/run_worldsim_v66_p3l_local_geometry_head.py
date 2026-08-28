"""训练并单次选择 V6.6 P3L instance-evidence local geometry head。"""

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
    feature_matrix,
    labels,
    ranking_metrics,
    scene_support,
    score_head,
    train_head,
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
    materialize_config = {
        "seed": config["seed"],
        "inputs": config["train"]["inputs"],
        "scenes": config["train"]["scenes"],
        "native_grid": config["native_grid"],
        "sampling": config["sampling"],
        "certificate": config["certificate"],
    }
    train_rows, train_materialization = materialize_rows(materialize_config, runs_root)
    selection_path = (
        runs_root
        / str(config["selection"]["p2n_run"])
        / str(config["selection"]["rows_relative_path"])
    )
    selection_rows = [
        json.loads(line) for line in selection_path.read_text().splitlines() if line
    ]
    train_x, selection_x = feature_matrix(train_rows), feature_matrix(selection_rows)
    train_y, selection_y = labels(train_rows), labels(selection_rows)
    model, mean, scale, training = train_head(
        train_x, train_y, config["model"], int(config["seed"])
    )
    train_scores = score_head(model, train_x, mean, scale)
    selection_scores = score_head(model, selection_x, mean, scale)
    train_metrics = ranking_metrics(train_y, train_scores)
    selection_metrics = ranking_metrics(selection_y, selection_scores)
    q0_selection = ranking_metrics(
        selection_y,
        np.asarray([float(row["q0_mean"]) for row in selection_rows]),
    )
    deterministic = {"auroc": 0.5, "auprc": float(np.mean(selection_y))}
    improvements = {
        "auroc_over_deterministic": selection_metrics["auroc"] - deterministic["auroc"],
        "auprc_over_deterministic": selection_metrics["auprc"] - deterministic["auprc"],
        "auroc_over_q0": selection_metrics["auroc"] - q0_selection["auroc"],
        "auprc_over_q0": selection_metrics["auprc"] - q0_selection["auprc"],
    }
    support = scene_support(selection_rows, selection_y, selection_scores)
    gates_config = config["evaluation"]
    gates = {
        "minimum_auroc_improvement_over_deterministic": improvements[
            "auroc_over_deterministic"
        ]
        >= float(gates_config["minimum_auroc_improvement_over_deterministic"]),
        "minimum_auprc_improvement_over_deterministic": improvements[
            "auprc_over_deterministic"
        ]
        >= float(gates_config["minimum_auprc_improvement_over_deterministic"]),
        "minimum_above_chance_scene_count": support["above_chance_scene_count"]
        >= int(gates_config["minimum_above_chance_scene_count"]),
        "actor_existence_authority_remains_disabled": True,
    }
    verdict = (
        "supported_legacy_selection_local_geometry_head"
        if all(gates.values())
        else "rejected_legacy_selection_local_geometry_head"
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": list(FEATURE_NAMES),
            "hidden_dimensions": list(config["model"]["hidden_dimensions"]),
            "mean": mean,
            "scale": scale,
        },
        run_dir / "LOCAL_GEOMETRY_HEAD.pt",
    )
    (run_dir / "TRAIN_ACTOR_ROWS.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    (run_dir / "SELECTION_SCORES.jsonl").write_text(
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
            for row, score in zip(selection_rows, selection_scores)
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "worldsim_v66.p3l_local_geometry_head_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "claim_boundary": config["claim_boundary"],
        "train_materialization": train_materialization,
        "train_row_count": len(train_rows),
        "selection_row_count": len(selection_rows),
        "training": training,
        "train_ranking": train_metrics,
        "selection_ranking": selection_metrics,
        "selection_q0_ranking": q0_selection,
        "selection_deterministic_baseline": deterministic,
        "selection_improvements": improvements,
        "selection_scene_support": support,
        "gate_results": gates,
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
