"""Train fixed-coverage Actor-state admission heads across multiple horizons."""

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
    ACTOR_FEATURE_NAMES, BinaryReliabilityMLP, ReliabilityMLP, binary_auroc,
    materialize_actor_query_rows, predict_reliability,
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))


def _admission_labels(arrays: dict[str, np.ndarray]) -> np.ndarray:
    target = np.asarray(arrays["target_cost"], dtype=np.float64)
    scenes = np.asarray(arrays["scene_index"])
    horizons = np.asarray(arrays["horizon_seconds"])
    labels = np.zeros(len(target), dtype=np.float32)
    for scene in np.unique(scenes):
        for horizon in np.unique(horizons[scenes == scene]):
            members = np.flatnonzero((scenes == scene) & (horizons == horizon))
            labels[members] = target[members] <= np.median(target[members])
    return labels


def _train_epochs(
    query_model: BinaryReliabilityMLP,
    actor_model: BinaryReliabilityMLP,
    optimizer: torch.optim.Optimizer,
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    epochs: int,
    label: str,
) -> tuple[float, float]:
    normalized = (np.asarray(arrays["features"], dtype=np.float32) - mean) / scale
    features = torch.from_numpy(normalized).cuda()
    actor_features = features[:, :len(ACTOR_FEATURE_NAMES)]
    targets = torch.from_numpy(_admission_labels(arrays)).cuda()
    final_query = final_actor = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        query_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            query_model(features), targets
        )
        actor_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            actor_model(actor_features), targets
        )
        (query_loss + actor_loss).backward()
        optimizer.step()
        final_query = float(query_loss.detach().cpu())
        final_actor = float(actor_loss.detach().cpu())
        if epoch % 250 == 0 or epoch + 1 == epochs:
            print(
                f"fixed-coverage admission {label} epoch={epoch + 1} "
                f"query={final_query:.6f} actor={final_actor:.6f}", flush=True,
            )
    del features, actor_features, targets
    torch.cuda.empty_cache()
    return final_query, final_actor


def _predict(model: BinaryReliabilityMLP, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(features).cuda())).cpu().numpy()


def _select_by_scene(reliability: np.ndarray, scenes: np.ndarray, fraction: float) -> np.ndarray:
    selected: list[int] = []
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        count = max(1, int(np.floor(len(members) * fraction)))
        selected.extend(members[np.argsort(-reliability[members], kind="mergesort")[:count]].tolist())
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

    base_path = args.runs_root / config["source"]["base_run"] / config["source"]["base_rows"]
    base = _load_npz(base_path)
    raw_base = np.asarray(base["features"], dtype=np.float32)
    mean = raw_base.mean(axis=0)
    scale = raw_base.std(axis=0).clip(min=1e-4)
    query_model = BinaryReliabilityMLP(raw_base.shape[1], config["model"]["hidden_dimensions"]).cuda()
    actor_model = BinaryReliabilityMLP(len(ACTOR_FEATURE_NAMES), config["model"]["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        list(query_model.parameters()) + list(actor_model.parameters()),
        lr=float(config["model"]["learning_rate"]), weight_decay=float(config["model"]["weight_decay"]),
    )
    evaluation_root = Path(config["evaluation_data"]["processed_root"])
    evaluation_scenes = [
        evaluation_root / f"{int(scene):03d}" for scene in config["evaluation_data"]["scene_indices"]
    ]
    extension_path = (
        args.runs_root / config["source"]["extension_run"] / config["source"]["extension_rows"]
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        extension_future = pool.submit(_load_npz, extension_path)
        evaluation_future = pool.submit(
            materialize_actor_query_rows, evaluation_scenes,
            [float(config["evaluation_data"]["horizon_seconds"])], config["evaluation_data"],
        )
        warm_query, warm_actor = _train_epochs(
            query_model, actor_model, optimizer, base, mean, scale,
            int(config["model"]["warmup_epochs"]), "warmup",
        )
        extension = extension_future.result()
        joint = {
            key: np.concatenate((base[key], extension[key]), axis=0) for key in base
        }
        joint_query, joint_actor = _train_epochs(
            query_model, actor_model, optimizer, joint, mean, scale,
            int(config["model"]["joint_epochs"]), "joint",
        )
        evaluation = evaluation_future.result()
    np.savez_compressed(run_dir / "EVALUATION_H3P5_ACTOR_QUERY_ROWS.npz", **evaluation)

    normalized = (np.asarray(evaluation["features"], dtype=np.float32) - mean) / scale
    query_reliability = _predict(query_model.eval(), normalized)
    actor_reliability = _predict(actor_model.eval(), normalized[:, :len(ACTOR_FEATURE_NAMES)])
    target = np.asarray(evaluation["target_cost"], dtype=np.float64)
    scenes = np.asarray(evaluation["scene_index"])
    fraction = float(config["selection"]["coverage_fraction"])
    query_selected = _select_by_scene(query_reliability, scenes, fraction)
    actor_selected = _select_by_scene(actor_reliability, scenes, fraction)

    continuous_artifact = torch.load(
        args.runs_root / config["continuous_baseline"]["run"] / config["continuous_baseline"]["artifact"],
        map_location="cuda",
    )
    continuous_model = ReliabilityMLP(
        len(continuous_artifact["feature_names"]), continuous_artifact["hidden_dimensions"]
    ).cuda()
    continuous_model.load_state_dict(continuous_artifact["query_model_state_dict"])
    continuous_score = predict_reliability(
        continuous_model.eval(), evaluation["features"],
        np.asarray(continuous_artifact["feature_mean"], dtype=np.float32),
        np.asarray(continuous_artifact["feature_scale"], dtype=np.float32),
    )
    continuous_selected = _select_by_scene(-continuous_score, scenes, fraction)
    unreliable = (
        np.asarray(evaluation["raw_actor_state_error_m"]) > float(config["evaluation"]["unreliable_actor_state_error_m"])
    ) & (
        np.asarray(evaluation["predicted_minimum_separation_m"]) <= float(config["evaluation"]["unreliable_exposure_radius_m"])
    )
    query_cost = float(target[query_selected].mean())
    actor_cost = float(target[actor_selected].mean())
    continuous_cost = float(target[continuous_selected].mean())
    all_cost = float(target.mean())
    labels = _admission_labels(evaluation).astype(bool)
    scene_nonincreasing = 0
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        chosen = query_selected[np.isin(query_selected, members)]
        scene_nonincreasing += int(target[chosen].mean() <= target[members].mean())
    metrics = {
        "row_count": int(len(target)), "selected_row_count": int(len(query_selected)),
        "achieved_coverage": float(len(query_selected) / len(target)),
        "all_mean_cost": all_cost, "query_admission_selected_mean_cost": query_cost,
        "actor_admission_selected_mean_cost": actor_cost,
        "continuous_p73_selected_mean_cost": continuous_cost,
        "query_cost_reduction": (all_cost - query_cost) / max(all_cost, 1e-12),
        "query_cost_reduction_over_actor_admission": (actor_cost - query_cost) / max(actor_cost, 1e-12),
        "query_cost_reduction_over_continuous_p73": (continuous_cost - query_cost) / max(continuous_cost, 1e-12),
        "query_admission_auroc": binary_auroc(labels, query_reliability),
        "actor_admission_auroc": binary_auroc(labels, actor_reliability),
        "all_unreliable_prevalence": float(unreliable.mean()),
        "query_selected_unreliable_prevalence": float(unreliable[query_selected].mean()),
        "actor_selected_unreliable_prevalence": float(unreliable[actor_selected].mean()),
        "continuous_selected_unreliable_prevalence": float(unreliable[continuous_selected].mean()),
        "scene_nonincreasing_count": int(scene_nonincreasing),
        "scene_count": int(len(np.unique(scenes))),
    }
    gates = {
        "minimum_cost_reduction_over_actor_admission": (
            metrics["query_cost_reduction_over_actor_admission"]
            >= float(config["gates"]["minimum_cost_reduction_over_actor_admission"])
        ),
        "minimum_cost_reduction_over_continuous_p73": (
            metrics["query_cost_reduction_over_continuous_p73"]
            >= float(config["gates"]["minimum_cost_reduction_over_continuous_p73"])
        ),
        "minimum_absolute_cost_reduction": (
            metrics["query_cost_reduction"] >= float(config["gates"]["minimum_absolute_cost_reduction"])
        ),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    torch.save({
        "feature_mean": mean, "feature_scale": scale,
        "hidden_dimensions": config["model"]["hidden_dimensions"],
        "query_admission_state_dict": query_model.state_dict(),
        "actor_admission_state_dict": actor_model.state_dict(),
    }, run_dir / "FIXED_COVERAGE_ACTOR_ADMISSION.pt")
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"base_row_count": int(len(base["target_cost"])),
            "extension_row_count": int(len(extension["target_cost"])),
            "joint_row_count": int(len(joint["target_cost"])),
            "warmup_final_query_loss": warm_query, "warmup_final_actor_loss": warm_actor,
            "joint_final_query_loss": joint_query, "joint_final_actor_loss": joint_actor},
        "selection": metrics, "gate_results": gates,
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
