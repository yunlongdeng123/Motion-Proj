"""Train Actor reliability across three horizons while overlapping IO and GPU work."""

from __future__ import annotations

import argparse
import json
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import (
    ACTOR_FEATURE_NAMES, FEATURE_NAMES, ReliabilityMLP, evaluate_reliability,
    materialize_actor_query_rows, predict_reliability,
)


def _train_epochs(
    query_model: ReliabilityMLP,
    actor_model: ReliabilityMLP,
    optimizer: torch.optim.Optimizer,
    raw_features: np.ndarray,
    target_cost: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    epochs: int,
    huber_beta: float,
    label: str,
) -> tuple[float, float]:
    normalized = (np.asarray(raw_features, dtype=np.float32) - mean) / scale
    features = torch.from_numpy(normalized).cuda()
    actor_features = features[:, :len(ACTOR_FEATURE_NAMES)]
    target = torch.from_numpy(np.log1p(np.asarray(target_cost, dtype=np.float32))).cuda()
    final_query = final_actor = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        query_prediction = query_model(features)
        actor_prediction = actor_model(actor_features)
        query_loss = torch.nn.functional.smooth_l1_loss(
            query_prediction, target, beta=huber_beta
        )
        actor_loss = torch.nn.functional.smooth_l1_loss(
            actor_prediction, target, beta=huber_beta
        )
        (query_loss + actor_loss).backward()
        optimizer.step()
        final_query = float(query_loss.detach().cpu())
        final_actor = float(actor_loss.detach().cpu())
        if epoch % 250 == 0 or epoch + 1 == epochs:
            print(
                f"multi-horizon {label} epoch={epoch + 1} "
                f"query={final_query:.6f} actor={final_actor:.6f}", flush=True,
            )
    del features, actor_features, target
    torch.cuda.empty_cache()
    return final_query, final_actor


def _select_by_scene(score: np.ndarray, scenes: np.ndarray, fraction: float) -> np.ndarray:
    selected: list[int] = []
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        count = max(1, int(np.floor(len(members) * fraction)))
        selected.extend(members[np.argsort(score[members], kind="mergesort")[:count]].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.runs_root / "worldsim_v67" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.monotonic()
    torch.manual_seed(int(config["seed"]))

    source = args.runs_root / config["source"]["run"]
    base = dict(np.load(source / config["source"]["training_rows"], allow_pickle=False))
    raw_base_features = np.asarray(base["features"], dtype=np.float32)
    mean = raw_base_features.mean(axis=0)
    scale = raw_base_features.std(axis=0).clip(min=1e-4)
    query_model = ReliabilityMLP(raw_base_features.shape[1], config["model"]["hidden_dimensions"]).cuda()
    actor_model = ReliabilityMLP(len(ACTOR_FEATURE_NAMES), config["model"]["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        list(query_model.parameters()) + list(actor_model.parameters()),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )

    source_root = Path(config["extension_data"]["processed_root"])
    all_source_scenes = sorted(
        path for path in source_root.iterdir()
        if path.is_dir() and (path / "instances" / "instances_info.json").exists()
        and (path / "lidar_pose").exists()
    )
    divisor = int(config["extension_data"]["source_scene_modulus"])
    remainder = int(config["extension_data"]["excluded_scene_remainder"])
    source_train_scenes = [path for path in all_source_scenes if int(path.name) % divisor != remainder]
    evaluation_root = Path(config["evaluation_data"]["processed_root"])
    evaluation_scenes = [
        evaluation_root / f"{int(scene):03d}" for scene in config["evaluation_data"]["scene_indices"]
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        extension_future = pool.submit(
            materialize_actor_query_rows, source_train_scenes,
            [float(config["extension_data"]["horizon_seconds"])], config["extension_data"],
        )
        warm_query, warm_actor = _train_epochs(
            query_model, actor_model, optimizer, base["features"], base["target_cost"],
            mean, scale, int(config["model"]["warmup_epochs"]),
            float(config["model"]["huber_beta"]), "warmup",
        )
        extension = extension_future.result()
        extension_save = pool.submit(
            np.savez_compressed, run_dir / "SOURCE_H2P5_ACTOR_QUERY_ROWS.npz", **extension
        )
        evaluation_future = pool.submit(
            materialize_actor_query_rows, evaluation_scenes,
            [float(config["evaluation_data"]["horizon_seconds"])], config["evaluation_data"],
        )
        combined_features = np.concatenate((base["features"], extension["features"]), axis=0)
        combined_target = np.concatenate((base["target_cost"], extension["target_cost"]), axis=0)
        joint_query, joint_actor = _train_epochs(
            query_model, actor_model, optimizer, combined_features, combined_target,
            mean, scale, int(config["model"]["joint_epochs"]),
            float(config["model"]["huber_beta"]), "joint",
        )
        evaluation = evaluation_future.result()
        extension_save.result()
    np.savez_compressed(run_dir / "EVALUATION_H3_ACTOR_QUERY_ROWS.npz", **evaluation)

    query_prediction = predict_reliability(query_model.eval(), evaluation["features"], mean, scale)
    actor_prediction = predict_reliability(
        actor_model.eval(), evaluation["features"], mean, scale, actor_only=True
    )
    trained_evaluation = evaluate_reliability(
        evaluation, query_prediction, actor_prediction, config["evaluation"]
    )
    frozen_artifact = torch.load(source / config["source"]["artifact"], map_location="cuda")
    frozen_query_model = ReliabilityMLP(len(FEATURE_NAMES), frozen_artifact["hidden_dimensions"]).cuda()
    frozen_query_model.load_state_dict(frozen_artifact["query_model_state_dict"])
    frozen_actor_model = ReliabilityMLP(len(ACTOR_FEATURE_NAMES), frozen_artifact["hidden_dimensions"]).cuda()
    frozen_actor_model.load_state_dict(frozen_artifact["actor_only_model_state_dict"])
    frozen_mean = np.asarray(frozen_artifact["feature_mean"], dtype=np.float32)
    frozen_scale = np.asarray(frozen_artifact["feature_scale"], dtype=np.float32)
    frozen_query = predict_reliability(
        frozen_query_model.eval(), evaluation["features"], frozen_mean, frozen_scale
    )
    frozen_actor = predict_reliability(
        frozen_actor_model.eval(), evaluation["features"], frozen_mean, frozen_scale, actor_only=True
    )
    frozen_evaluation = evaluate_reliability(
        evaluation, frozen_query, frozen_actor, config["evaluation"]
    )

    target = np.asarray(evaluation["target_cost"], dtype=np.float64)
    scenes = np.asarray(evaluation["scene_index"])
    unreliable = (
        np.asarray(evaluation["raw_actor_state_error_m"]) > float(config["evaluation"]["unreliable_actor_state_error_m"])
    ) & (
        np.asarray(evaluation["predicted_minimum_separation_m"]) <= float(config["evaluation"]["unreliable_exposure_radius_m"])
    )
    fraction = float(config["selection"]["coverage_fraction"])
    query_selected = _select_by_scene(query_prediction, scenes, fraction)
    actor_selected = _select_by_scene(actor_prediction, scenes, fraction)
    frozen_selected = _select_by_scene(frozen_query, scenes, fraction)
    all_cost = float(target.mean())
    query_cost = float(target[query_selected].mean())
    selection = {
        "achieved_coverage": float(len(query_selected) / len(target)),
        "all_mean_cost": all_cost,
        "multi_horizon_query_selected_mean_cost": query_cost,
        "multi_horizon_actor_selected_mean_cost": float(target[actor_selected].mean()),
        "frozen_p66_query_selected_mean_cost": float(target[frozen_selected].mean()),
        "multi_horizon_query_cost_reduction": (all_cost - query_cost) / max(all_cost, 1e-12),
        "all_unreliable_prevalence": float(unreliable.mean()),
        "multi_horizon_query_selected_unreliable_prevalence": float(unreliable[query_selected].mean()),
    }
    query_mae_improvement_over_frozen = (
        frozen_evaluation["query_conditioned_mae"] - trained_evaluation["query_conditioned_mae"]
    ) / max(frozen_evaluation["query_conditioned_mae"], 1e-12)
    gates = {
        "minimum_mae_reduction_over_actor_only": (
            trained_evaluation["mae_reduction_over_actor_only"]
            >= float(config["gates"]["minimum_mae_reduction_over_actor_only"])
        ),
        "minimum_query_mae_improvement_over_frozen_p66": (
            query_mae_improvement_over_frozen
            >= float(config["gates"]["minimum_query_mae_improvement_over_frozen_p66"])
        ),
        "minimum_selective_cost_reduction": (
            selection["multi_horizon_query_cost_reduction"]
            >= float(config["gates"]["minimum_selective_cost_reduction"])
        ),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    torch.save({
        "feature_names": FEATURE_NAMES, "feature_mean": mean, "feature_scale": scale,
        "hidden_dimensions": config["model"]["hidden_dimensions"],
        "query_model_state_dict": query_model.state_dict(),
        "actor_only_model_state_dict": actor_model.state_dict(),
    }, run_dir / "MULTI_HORIZON_ACTOR_RELIABILITY.pt")
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"base_row_count": int(len(base["target_cost"])),
            "extension_row_count": int(len(extension["target_cost"])),
            "joint_row_count": int(len(combined_target)),
            "warmup_final_query_loss": warm_query, "warmup_final_actor_loss": warm_actor,
            "joint_final_query_loss": joint_query, "joint_final_actor_loss": joint_actor},
        "multi_horizon_evaluation": trained_evaluation,
        "frozen_p66_evaluation": frozen_evaluation,
        "query_mae_improvement_over_frozen_p66": query_mae_improvement_over_frozen,
        "selection": selection, "gate_results": gates,
        "resources": {"gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}, indent=2))


if __name__ == "__main__":
    main()
