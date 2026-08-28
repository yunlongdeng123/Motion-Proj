"""Train four-horizon continuous reliability and read a new validation cohort once."""

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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))


def _combine(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.concatenate((left[key], right[key]), axis=0) for key in left}


def _train_epochs(
    query_model: ReliabilityMLP,
    actor_model: ReliabilityMLP,
    optimizer: torch.optim.Optimizer,
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    epochs: int,
    beta: float,
    label: str,
) -> tuple[float, float]:
    normalized = (np.asarray(arrays["features"], dtype=np.float32) - mean) / scale
    features = torch.from_numpy(normalized).cuda()
    actor_features = features[:, :len(ACTOR_FEATURE_NAMES)]
    target = torch.from_numpy(np.log1p(
        np.asarray(arrays["target_cost"], dtype=np.float32)
    )).cuda()
    final_query = final_actor = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        query_loss = torch.nn.functional.smooth_l1_loss(
            query_model(features), target, beta=beta
        )
        actor_loss = torch.nn.functional.smooth_l1_loss(
            actor_model(actor_features), target, beta=beta
        )
        (query_loss + actor_loss).backward()
        optimizer.step()
        final_query = float(query_loss.detach().cpu())
        final_actor = float(actor_loss.detach().cpu())
        if epoch % 250 == 0 or epoch + 1 == epochs:
            print(
                f"fresh-validation {label} epoch={epoch + 1} "
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


def _scene_identities(metadata: Path, names: list[str]) -> list[tuple[str, int]]:
    scenes = json.loads((metadata / "scene.json").read_text(encoding="utf-8"))
    index_by_name = {str(row["name"]): index for index, row in enumerate(scenes)}
    return [(name, int(index_by_name[name])) for name in names]


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

    base = _load_npz(
        args.runs_root / config["source"]["base_run"] / config["source"]["base_rows"]
    )
    h2p5 = _load_npz(
        args.runs_root / config["source"]["h2p5_run"] / config["source"]["h2p5_rows"]
    )
    warmup = _combine(base, h2p5)
    raw_warmup = np.asarray(warmup["features"], dtype=np.float32)
    mean = raw_warmup.mean(axis=0)
    scale = raw_warmup.std(axis=0).clip(min=1e-4)
    query_model = ReliabilityMLP(
        raw_warmup.shape[1], config["model"]["hidden_dimensions"]
    ).cuda()
    actor_model = ReliabilityMLP(
        len(ACTOR_FEATURE_NAMES), config["model"]["hidden_dimensions"]
    ).cuda()
    optimizer = torch.optim.AdamW(
        list(query_model.parameters()) + list(actor_model.parameters()),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )

    source_root = Path(config["source_h3_data"]["processed_root"])
    all_source = sorted(
        path for path in source_root.iterdir()
        if path.is_dir() and (path / "instances" / "instances_info.json").exists()
        and (path / "lidar_pose").exists()
    )
    divisor = int(config["source_h3_data"]["source_scene_modulus"])
    remainder = int(config["source_h3_data"]["excluded_scene_remainder"])
    source_scenes = [path for path in all_source if int(path.name) % divisor != remainder]
    with ThreadPoolExecutor(max_workers=1) as pool:
        if "source_h3_cache" in config:
            h3_future = pool.submit(
                _load_npz,
                args.runs_root / config["source_h3_cache"]["run"]
                / config["source_h3_cache"]["artifact"],
            )
        else:
            h3_future = pool.submit(
                materialize_actor_query_rows, source_scenes,
                [float(config["source_h3_data"]["horizon_seconds"])], config["source_h3_data"],
            )
        warm_query, warm_actor = _train_epochs(
            query_model, actor_model, optimizer, warmup, mean, scale,
            int(config["model"]["warmup_epochs"]), float(config["model"]["huber_beta"]),
            "warmup",
        )
        h3 = h3_future.result()
    four_horizon = _combine(warmup, h3)
    with ThreadPoolExecutor(max_workers=1) as pool:
        h3_save = pool.submit(
            np.savez_compressed, run_dir / "SOURCE_H3_ACTOR_QUERY_ROWS.npz", **h3
        )
        joint_query, joint_actor = _train_epochs(
            query_model, actor_model, optimizer, four_horizon, mean, scale,
            int(config["model"]["joint_epochs"]), float(config["model"]["huber_beta"]),
            "joint",
        )
        h3_save.result()

    model_artifact = {
        "feature_names": FEATURE_NAMES, "feature_mean": mean, "feature_scale": scale,
        "hidden_dimensions": config["model"]["hidden_dimensions"],
        "query_model_state_dict": query_model.state_dict(),
        "actor_only_model_state_dict": actor_model.state_dict(),
    }
    torch.save(model_artifact, run_dir / "FOUR_HORIZON_ACTOR_RELIABILITY.pt")

    evaluation_data = config["evaluation_data"]
    metadata = Path(evaluation_data["metadata_root"]) / "v1.0-trainval"
    names = [str(name) for name in evaluation_data["scene_names"]]
    identities = _scene_identities(metadata, names)
    processed_root = Path(evaluation_data["processed_root"])
    deadline = time.monotonic() + float(evaluation_data["readiness_timeout_seconds"])
    while True:
        ready = [
            (processed_root / f"{index:03d}" / "instances" / "instances_info.json").is_file()
            and (processed_root / f"{index:03d}" / "lidar_pose").is_dir()
            for _, index in identities
        ]
        if all(ready):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"fresh validation Actor scenes not ready: {sum(ready)}/{len(ready)}")
        print(f"waiting for fresh validation Actor scenes ready={sum(ready)}/{len(ready)}", flush=True)
        time.sleep(10.0)
    evaluation_scenes = [processed_root / f"{index:03d}" for _, index in identities]
    evaluation = materialize_actor_query_rows(
        evaluation_scenes, [float(evaluation_data["horizon_seconds"])], evaluation_data
    )
    np.savez_compressed(run_dir / "FRESH_VALIDATION_H3P5_ACTOR_QUERY_ROWS.npz", **evaluation)

    query_prediction = predict_reliability(query_model.eval(), evaluation["features"], mean, scale)
    actor_prediction = predict_reliability(
        actor_model.eval(), evaluation["features"], mean, scale, actor_only=True
    )
    trained_evaluation = evaluate_reliability(
        evaluation, query_prediction, actor_prediction, config["evaluation"]
    )
    p73_artifact = torch.load(
        args.runs_root / config["p73_baseline"]["run"] / config["p73_baseline"]["artifact"],
        map_location="cuda",
    )
    p73_query_model = ReliabilityMLP(len(FEATURE_NAMES), p73_artifact["hidden_dimensions"]).cuda()
    p73_query_model.load_state_dict(p73_artifact["query_model_state_dict"])
    p73_score = predict_reliability(
        p73_query_model.eval(), evaluation["features"],
        np.asarray(p73_artifact["feature_mean"], dtype=np.float32),
        np.asarray(p73_artifact["feature_scale"], dtype=np.float32),
    )

    target = np.asarray(evaluation["target_cost"], dtype=np.float64)
    scenes = np.asarray(evaluation["scene_index"])
    fraction = float(config["selection"]["coverage_fraction"])
    query_selected = _select_by_scene(query_prediction, scenes, fraction)
    actor_selected = _select_by_scene(actor_prediction, scenes, fraction)
    p73_selected = _select_by_scene(p73_score, scenes, fraction)
    unreliable = (
        np.asarray(evaluation["raw_actor_state_error_m"]) > float(config["evaluation"]["unreliable_actor_state_error_m"])
    ) & (
        np.asarray(evaluation["predicted_minimum_separation_m"]) <= float(config["evaluation"]["unreliable_exposure_radius_m"])
    )
    all_cost = float(target.mean())
    query_cost = float(target[query_selected].mean())
    actor_cost = float(target[actor_selected].mean())
    p73_cost = float(target[p73_selected].mean())
    scene_nonincreasing = 0
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        chosen = query_selected[np.isin(query_selected, members)]
        scene_nonincreasing += int(target[chosen].mean() <= target[members].mean())
    selection = {
        "row_count": int(len(target)), "selected_row_count": int(len(query_selected)),
        "achieved_coverage": float(len(query_selected) / len(target)),
        "all_mean_cost": all_cost, "query_selected_mean_cost": query_cost,
        "actor_only_selected_mean_cost": actor_cost, "p73_selected_mean_cost": p73_cost,
        "query_cost_reduction": (all_cost - query_cost) / max(all_cost, 1e-12),
        "query_cost_reduction_over_actor_only": (actor_cost - query_cost) / max(actor_cost, 1e-12),
        "query_cost_reduction_over_p73": (p73_cost - query_cost) / max(p73_cost, 1e-12),
        "all_unreliable_prevalence": float(unreliable.mean()),
        "query_selected_unreliable_prevalence": float(unreliable[query_selected].mean()),
        "actor_selected_unreliable_prevalence": float(unreliable[actor_selected].mean()),
        "p73_selected_unreliable_prevalence": float(unreliable[p73_selected].mean()),
        "scene_nonincreasing_count": int(scene_nonincreasing),
        "scene_count": int(len(np.unique(scenes))),
    }
    gates = {
        "minimum_cost_reduction_over_actor_only": (
            selection["query_cost_reduction_over_actor_only"]
            >= float(config["gates"]["minimum_cost_reduction_over_actor_only"])
        ),
        "minimum_cost_reduction_over_p73": (
            selection["query_cost_reduction_over_p73"]
            >= float(config["gates"]["minimum_cost_reduction_over_p73"])
        ),
        "minimum_absolute_cost_reduction": (
            selection["query_cost_reduction"]
            >= float(config["gates"]["minimum_absolute_cost_reduction"])
        ),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"warmup_row_count": int(len(warmup["target_cost"])),
            "h3_extension_row_count": int(len(h3["target_cost"])),
            "four_horizon_row_count": int(len(four_horizon["target_cost"])),
            "warmup_final_query_loss": warm_query, "warmup_final_actor_loss": warm_actor,
            "joint_final_query_loss": joint_query, "joint_final_actor_loss": joint_actor},
        "fresh_scene_names": names,
        "fresh_scene_indices": [index for _, index in identities],
        "fresh_validation_evaluation": trained_evaluation,
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
