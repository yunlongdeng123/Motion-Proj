"""Train the V6.7 differentiable residual directional-surface authority."""

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

from motion_proj.worldsim_v67.directional_surface_head import (
    GEOMETRY_FEATURE_NAMES,
    evaluate_support,
    materialize_points,
    score_residual_head,
    train_residual_head,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / str(config["task_id"]) / run_id
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
    train = materialize_points(
        config["train"],
        seed=int(config["seed"]),
        native_grid=config["native_grid"],
        sampling=config["sampling"],
        support_radius_m=float(config["support_radius_m"]),
        runs_root=runs_root,
    )
    selection = materialize_points(
        config["selection"],
        seed=int(config["seed"]),
        native_grid=config["native_grid"],
        sampling=config["sampling"],
        support_radius_m=float(config["support_radius_m"]),
        runs_root=runs_root,
    )
    model, mean, scale, threshold, training = train_residual_head(
        train["values"],
        train["conflicts"],
        train["analytic_support"],
        config["model"],
        int(config["seed"]),
    )
    selection_scores = score_residual_head(
        model, selection["values"], mean, scale
    )
    evaluation = evaluate_support(
        selection,
        selection_scores,
        threshold,
        runs_root
        / str(config["selection"]["action_run"])
        / str(config["selection"]["action_rows"]),
        str(config["selection"]["action_arm"]),
    )
    gate_config = config["evaluation"]
    learned = evaluation["learned_residual"]
    gates = {
        "minimum_conflict_point_reduction": learned["conflict_point_reduction"]
        >= float(gate_config["minimum_conflict_point_reduction"]),
        "minimum_overall_boundary_retention": learned["overall_boundary_retention"]
        >= float(gate_config["minimum_overall_boundary_retention"]),
        "minimum_clean_boundary_retention": learned["clean_boundary_retention"]
        >= float(gate_config["minimum_clean_boundary_retention"]),
        "minimum_clean_retention_improvement_over_analytic": evaluation[
            "clean_retention_improvement_over_analytic"
        ]
        >= float(gate_config["minimum_clean_retention_improvement_over_analytic"]),
        "minimum_rescued_clean_point_count": evaluation["rescued_clean_point_count"]
        >= int(gate_config["minimum_rescued_clean_point_count"]),
        "analytic_core_and_actor_existence_preserved": (
            not evaluation["analytic_core_removed"]
            and not evaluation["actor_existence_mutated"]
        ),
    }
    verdict = (
        "supported_learned_residual_directional_surface_selection"
        if all(gates.values())
        else "rejected_learned_residual_directional_surface_selection"
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dimension": int(train["values"].shape[1]),
            "native_feature_dimension": int(
                train["values"].shape[1] - len(GEOMETRY_FEATURE_NAMES)
            ),
            "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
            "hidden_dimensions": list(config["model"]["hidden_dimensions"]),
            "mean": mean,
            "scale": scale,
            "probability_threshold": float(threshold),
            "support_radius_m": float(config["support_radius_m"]),
        },
        run_dir / "DIRECTIONAL_SURFACE_HEAD.pt",
    )
    np.savez_compressed(
        run_dir / "SELECTION_POINT_SCORES.npz",
        base_ids=np.asarray(selection["base_ids"]),
        scenes=np.asarray(selection["scenes"]),
        conflict=np.asarray(selection["conflicts"], dtype=bool),
        analytic_support=np.asarray(selection["analytic_support"], dtype=bool),
        residual_clean_probability=selection_scores.astype(np.float32),
    )
    summary = {
        "schema_version": "worldsim_v67.p14_directional_surface_train_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "claim_boundary": config["claim_boundary"],
        "model": {
            "input_dimension": int(train["values"].shape[1]),
            "native_feature_dimension": int(
                train["values"].shape[1] - len(GEOMETRY_FEATURE_NAMES)
            ),
            "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
            "hidden_dimensions": list(config["model"]["hidden_dimensions"]),
            "analytic_core_frozen": True,
            "residual_rescue_only": True,
        },
        "training": training,
        "selection": evaluation,
        "gate_results": gates,
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
