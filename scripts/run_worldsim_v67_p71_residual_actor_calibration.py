"""Train target-domain residual adapters and evaluate on fresh Actor scenes."""

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

from motion_proj.worldsim_v67.actor_state_reliability import (
    ACTOR_FEATURE_NAMES,
    ReliabilityMLP,
    evaluate_reliability,
    materialize_actor_query_rows,
    predict_reliability,
)


def _select_by_scene(score: np.ndarray, scenes: np.ndarray, fraction: float) -> np.ndarray:
    selected: list[int] = []
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        count = max(1, int(np.floor(len(members) * fraction)))
        selected.extend(members[np.argsort(score[members], kind="mergesort")[:count]].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _normalized_tensor(
    raw_features: np.ndarray, mean: np.ndarray, scale: np.ndarray, actor_only: bool
) -> torch.Tensor:
    normalized = (np.asarray(raw_features, dtype=np.float32) - mean) / scale
    if actor_only:
        normalized = normalized[:, :len(ACTOR_FEATURE_NAMES)]
    return torch.from_numpy(normalized).cuda()


def _adapted_prediction(
    base: ReliabilityMLP,
    adapter: torch.nn.Linear,
    features: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        base_log = base(features)
        residual = adapter(base.encode(features)).squeeze(-1)
        prediction_log = torch.clamp(base_log + residual, min=0.0)
    return np.expm1(prediction_log.cpu().numpy()).clip(min=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.runs_root / "worldsim_v67" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    started = time.monotonic()

    source = args.runs_root / config["source"]["run"]
    artifact = torch.load(source / config["source"]["artifact"], map_location="cuda")
    hidden = artifact["hidden_dimensions"]
    query_base = ReliabilityMLP(len(artifact["feature_names"]), hidden).cuda()
    query_base.load_state_dict(artifact["query_model_state_dict"])
    actor_base = ReliabilityMLP(len(ACTOR_FEATURE_NAMES), hidden).cuda()
    actor_base.load_state_dict(artifact["actor_only_model_state_dict"])
    query_base.eval().requires_grad_(False)
    actor_base.eval().requires_grad_(False)
    mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
    scale = np.asarray(artifact["feature_scale"], dtype=np.float32)

    processed_root = Path(config["calibration_data"]["processed_root"])
    calibration_scenes = [
        processed_root / f"{int(scene):03d}"
        for scene in config["calibration_data"]["scene_indices"]
    ]
    calibration = materialize_actor_query_rows(
        calibration_scenes,
        [float(config["calibration_data"]["horizon_seconds"])],
        config["calibration_data"],
    )
    np.savez_compressed(run_dir / "CALIBRATION_ACTOR_QUERY_ROWS.npz", **calibration)
    fresh = dict(np.load(
        args.runs_root / config["evaluation_data"]["run"]
        / config["evaluation_data"]["rows"], allow_pickle=False
    ))

    query_features = _normalized_tensor(calibration["features"], mean, scale, actor_only=False)
    actor_features = _normalized_tensor(calibration["features"], mean, scale, actor_only=True)
    target = torch.from_numpy(np.log1p(
        np.asarray(calibration["target_cost"], dtype=np.float32)
    )).cuda()
    query_adapter = torch.nn.Linear(int(hidden[-1]), 1).cuda()
    actor_adapter = torch.nn.Linear(int(hidden[-1]), 1).cuda()
    torch.nn.init.zeros_(query_adapter.weight)
    torch.nn.init.zeros_(query_adapter.bias)
    torch.nn.init.zeros_(actor_adapter.weight)
    torch.nn.init.zeros_(actor_adapter.bias)
    optimizer = torch.optim.AdamW(
        list(query_adapter.parameters()) + list(actor_adapter.parameters()),
        lr=float(config["adapter"]["learning_rate"]),
        weight_decay=float(config["adapter"]["weight_decay"]),
    )
    with torch.no_grad():
        query_base_log = query_base(query_features)
        actor_base_log = actor_base(actor_features)
        query_embedding = query_base.encode(query_features)
        actor_embedding = actor_base.encode(actor_features)
    final_query_loss = final_actor_loss = 0.0
    for epoch in range(int(config["adapter"]["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        query_prediction = torch.clamp(
            query_base_log + query_adapter(query_embedding).squeeze(-1), min=0.0
        )
        actor_prediction = torch.clamp(
            actor_base_log + actor_adapter(actor_embedding).squeeze(-1), min=0.0
        )
        query_loss = torch.nn.functional.smooth_l1_loss(
            query_prediction, target, beta=float(config["adapter"]["huber_beta"])
        )
        actor_loss = torch.nn.functional.smooth_l1_loss(
            actor_prediction, target, beta=float(config["adapter"]["huber_beta"])
        )
        (query_loss + actor_loss).backward()
        optimizer.step()
        final_query_loss = float(query_loss.detach().cpu())
        final_actor_loss = float(actor_loss.detach().cpu())
        if epoch % 250 == 0 or epoch + 1 == int(config["adapter"]["epochs"]):
            print(
                f"residual actor calibration epoch={epoch + 1} "
                f"query={final_query_loss:.6f} actor={final_actor_loss:.6f}", flush=True
            )

    fresh_query_features = _normalized_tensor(fresh["features"], mean, scale, actor_only=False)
    fresh_actor_features = _normalized_tensor(fresh["features"], mean, scale, actor_only=True)
    adapted_query = _adapted_prediction(query_base, query_adapter, fresh_query_features)
    adapted_actor = _adapted_prediction(actor_base, actor_adapter, fresh_actor_features)
    frozen_query = predict_reliability(
        query_base, fresh["features"], mean, scale, actor_only=False
    )
    frozen_actor = predict_reliability(
        actor_base, fresh["features"], mean, scale, actor_only=True
    )
    adapted_evaluation = evaluate_reliability(
        fresh, adapted_query, adapted_actor, config["evaluation"]
    )
    frozen_evaluation = evaluate_reliability(
        fresh, frozen_query, frozen_actor, config["evaluation"]
    )

    target_fresh = np.asarray(fresh["target_cost"], dtype=np.float64)
    unreliable = (
        np.asarray(fresh["raw_actor_state_error_m"])
        > float(config["evaluation"]["unreliable_actor_state_error_m"])
    ) & (
        np.asarray(fresh["predicted_minimum_separation_m"])
        <= float(config["evaluation"]["unreliable_exposure_radius_m"])
    )
    scenes = np.asarray(fresh["scene_index"])
    coverage = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(adapted_query, scenes, coverage)
    selected_cost = float(target_fresh[selected].mean())
    all_cost = float(target_fresh.mean())
    selected_prevalence = float(unreliable[selected].mean())
    all_prevalence = float(unreliable.mean())
    scene_nonincreasing = 0
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        chosen = selected[np.isin(selected, members)]
        scene_nonincreasing += int(
            target_fresh[chosen].mean() <= target_fresh[members].mean()
        )
    selection = {
        "selected_row_count": int(len(selected)),
        "achieved_coverage": float(len(selected) / len(target_fresh)),
        "all_mean_cost": all_cost,
        "adapted_query_selected_mean_cost": selected_cost,
        "adapted_query_cost_reduction": (all_cost - selected_cost) / max(all_cost, 1e-12),
        "all_unreliable_prevalence": all_prevalence,
        "adapted_query_selected_unreliable_prevalence": selected_prevalence,
        "adapted_query_unreliable_prevalence_reduction": (
            (all_prevalence - selected_prevalence) / max(all_prevalence, 1e-12)
        ),
        "scene_nonincreasing_count": int(scene_nonincreasing),
        "scene_count": int(len(np.unique(scenes))),
    }
    query_mae_improvement_over_frozen = (
        frozen_evaluation["query_conditioned_mae"] - adapted_evaluation["query_conditioned_mae"]
    ) / max(frozen_evaluation["query_conditioned_mae"], 1e-12)
    gates = {
        "minimum_mae_reduction_over_adapted_actor_only": (
            adapted_evaluation["mae_reduction_over_actor_only"]
            >= float(config["gates"]["minimum_mae_reduction_over_adapted_actor_only"])
        ),
        "minimum_query_mae_improvement_over_frozen": (
            query_mae_improvement_over_frozen
            >= float(config["gates"]["minimum_query_mae_improvement_over_frozen"])
        ),
        "maximum_spearman_drop_from_frozen_query": (
            adapted_evaluation["query_conditioned_spearman"]
            >= frozen_evaluation["query_conditioned_spearman"]
            - float(config["gates"]["maximum_spearman_drop_from_frozen_query"])
        ),
        "minimum_selective_cost_reduction": (
            selection["adapted_query_cost_reduction"]
            >= float(config["gates"]["minimum_selective_cost_reduction"])
        ),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    torch.save({
        "query_adapter_state_dict": query_adapter.state_dict(),
        "actor_adapter_state_dict": actor_adapter.state_dict(),
        "hidden_dimension": int(hidden[-1]),
    }, run_dir / "TARGET_DOMAIN_RESIDUAL_ADAPTERS.pt")
    summary = {
        "schema_version": config["output_schema_version"],
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "training": {
            "calibration_scene_indices": [
                int(scene) for scene in config["calibration_data"]["scene_indices"]
            ],
            "calibration_row_count": int(len(target)),
            "query_final_loss": final_query_loss,
            "actor_only_final_loss": final_actor_loss,
        },
        "frozen_fresh_evaluation": frozen_evaluation,
        "adapted_fresh_evaluation": adapted_evaluation,
        "query_mae_improvement_over_frozen": query_mae_improvement_over_frozen,
        "selection": selection,
        "gate_results": gates,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started,
        },
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        json.dumps({
            "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()
        }, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "run_dir": str(run_dir), "verdict": verdict, "gate_results": gates
    }, indent=2))


if __name__ == "__main__":
    main()
