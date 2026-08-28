"""Rank forecast failure only among Actor states visited by the candidate Ego trajectory."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import (
    ACTOR_FEATURE_NAMES, FEATURE_NAMES, ReliabilityMLP, binary_auroc, predict_reliability,
)


class ActorFailureMLP(torch.nn.Module):
    def __init__(self, hidden_dimensions: Sequence[int]) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = len(ACTOR_FEATURE_NAMES)
        for hidden in hidden_dimensions:
            layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        self.encoder = torch.nn.Sequential(*layers)
        self.event_head = torch.nn.Linear(width, 1)
        self.error_head = torch.nn.Linear(width, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        return self.event_head(encoded).squeeze(-1), torch.nn.functional.softplus(
            self.error_head(encoded).squeeze(-1)
        )


def _load(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))


def _combine(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}


def _select_visited_by_scene(
    score: np.ndarray, scenes: np.ndarray, visited: np.ndarray, fraction: float,
) -> np.ndarray:
    selected: list[int] = []
    for scene in np.unique(scenes):
        members = np.flatnonzero((scenes == scene) & visited)
        if not len(members):
            continue
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
    source = config["source"]
    arrays = _combine([
        _load(args.runs_root / source["base_run"] / source["base_rows"]),
        _load(args.runs_root / source["h2p5_run"] / source["h2p5_rows"]),
        _load(args.runs_root / source["h3_run"] / source["h3_rows"]),
    ])
    actor_raw_all = np.asarray(arrays["features"], dtype=np.float32)[:, :len(ACTOR_FEATURE_NAMES)]
    _, unique_indices = np.unique(actor_raw_all, axis=0, return_index=True)
    unique_indices.sort()
    actor_raw = actor_raw_all[unique_indices]
    raw_error = np.asarray(arrays["raw_actor_state_error_m"], dtype=np.float32)[unique_indices]
    horizons = np.asarray(arrays["horizon_seconds"], dtype=np.float32)[unique_indices]
    labels_np = raw_error > float(config["evaluation"]["unreliable_actor_state_error_m"])
    mean = actor_raw.mean(axis=0)
    scale = actor_raw.std(axis=0).clip(min=1e-4)
    features = torch.from_numpy((actor_raw - mean) / scale).cuda()
    log_error = torch.from_numpy(np.log1p(raw_error)).cuda()
    groups: list[tuple[torch.Tensor, torch.Tensor]] = []
    for horizon in sorted(np.unique(horizons).tolist()):
        members = horizons == horizon
        positive = np.flatnonzero(members & labels_np)
        negative = np.flatnonzero(members & ~labels_np)
        if len(positive) and len(negative):
            groups.append((torch.from_numpy(positive).long().cuda(), torch.from_numpy(negative).long().cuda()))
    model_config = config["model"]
    model = ActorFailureMLP(model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    pair_per_group = max(1, int(model_config["pair_batch_size"]) // len(groups))
    regression_count = int(model_config["regression_batch_size"])
    final = {}
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(model_config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        positive = torch.cat([
            pos[torch.randint(len(pos), (pair_per_group,), device="cuda")] for pos, _ in groups
        ])
        negative = torch.cat([
            neg[torch.randint(len(neg), (pair_per_group,), device="cuda")] for _, neg in groups
        ])
        regression = torch.randint(len(features), (regression_count,), device="cuda")
        positive_score, _ = model(features[positive])
        negative_score, _ = model(features[negative])
        pair_loss = torch.nn.functional.softplus(
            float(model_config["pair_margin"]) - positive_score + negative_score
        ).mean()
        _, predicted_error = model(features[regression])
        regression_loss = torch.nn.functional.smooth_l1_loss(
            predicted_error, log_error[regression], beta=float(model_config["huber_beta"])
        )
        loss = pair_loss + float(model_config["regression_weight"]) * regression_loss
        loss.backward()
        optimizer.step()
        final = {"total": float(loss.detach().cpu()), "pair": float(pair_loss.detach().cpu()),
            "error_regression": float(regression_loss.detach().cpu())}
        if epoch % 250 == 0 or epoch + 1 == int(model_config["epochs"]):
            print(f"visited-actor-failure epoch={epoch + 1} " + " ".join(
                f"{key}={value:.6f}" for key, value in final.items()
            ), flush=True)
    torch.save({
        "feature_names": ACTOR_FEATURE_NAMES, "feature_mean": mean, "feature_scale": scale,
        "hidden_dimensions": model_config["hidden_dimensions"], "model_state_dict": model.state_dict(),
    }, run_dir / "VISITED_ACTOR_FAILURE.pt")
    del features, log_error
    torch.cuda.empty_cache()

    evaluation_path = args.runs_root / config["evaluation_rows"]["run"] / config["evaluation_rows"]["artifact"]
    deadline = time.monotonic() + float(config["evaluation_rows"]["readiness_timeout_seconds"])
    while not evaluation_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"P81 evaluation rows not ready: {evaluation_path}")
        print("waiting for prospectively frozen P81 visited-region rows", flush=True)
        time.sleep(10.0)
    evaluation = _load(evaluation_path)
    evaluation_actor = np.asarray(evaluation["features"], dtype=np.float32)[:, :len(ACTOR_FEATURE_NAMES)]
    with torch.no_grad():
        score, predicted_error = model(torch.from_numpy((evaluation_actor - mean) / scale).cuda())
    score_np = score.cpu().numpy()
    predicted_error_np = np.expm1(predicted_error.cpu().numpy()).clip(min=0.0)
    frozen = torch.load(
        args.runs_root / config["frozen_p75"]["run"] / config["frozen_p75"]["artifact"], map_location="cuda"
    )
    frozen_model = ReliabilityMLP(len(FEATURE_NAMES), frozen["hidden_dimensions"]).cuda()
    frozen_model.load_state_dict(frozen["query_model_state_dict"])
    frozen_score = predict_reliability(
        frozen_model.eval(), evaluation["features"], np.asarray(frozen["feature_mean"], dtype=np.float32),
        np.asarray(frozen["feature_scale"], dtype=np.float32),
    )
    scenes = np.asarray(evaluation["scene_index"])
    error = np.asarray(evaluation["raw_actor_state_error_m"], dtype=np.float64)
    visited = np.asarray(evaluation["predicted_minimum_separation_m"]) <= float(config["evaluation"]["visited_region_radius_m"])
    unreliable = error > float(config["evaluation"]["unreliable_actor_state_error_m"])
    fraction = float(config["selection"]["coverage_fraction_within_visited_region"])
    selected = _select_visited_by_scene(score_np, scenes, visited, fraction)
    frozen_selected = _select_visited_by_scene(frozen_score, scenes, visited, fraction)
    visited_indices = np.flatnonzero(visited)
    candidate_events = int(np.count_nonzero(unreliable[selected]))
    frozen_events = int(np.count_nonzero(unreliable[frozen_selected]))
    all_events = int(np.count_nonzero(unreliable[visited_indices]))
    scene_rows = []
    for scene in np.unique(scenes):
        members = np.flatnonzero((scenes == scene) & visited)
        chosen = selected[np.isin(selected, members)]
        if len(members):
            scene_rows.append({"scene_index": int(scene), "visited_row_count": int(len(members)),
                "selected_count": int(len(chosen)), "visited_event_count": int(np.count_nonzero(unreliable[members])),
                "selected_event_count": int(np.count_nonzero(unreliable[chosen])),
                "visited_event_prevalence": float(unreliable[members].mean()),
                "selected_event_prevalence": float(unreliable[chosen].mean())})
    candidate_error = float(error[selected].mean())
    frozen_error = float(error[frozen_selected].mean())
    metrics = {
        "total_row_count": int(len(error)), "visited_row_count": int(len(visited_indices)),
        "selected_visited_row_count": int(len(selected)),
        "achieved_coverage_within_visited_region": float(len(selected) / max(len(visited_indices), 1)),
        "all_visited_unreliable_events": all_events,
        "candidate_selected_unreliable_events": candidate_events,
        "frozen_p75_selected_unreliable_events": frozen_events,
        "all_visited_unreliable_prevalence": float(unreliable[visited_indices].mean()),
        "candidate_selected_unreliable_prevalence": float(unreliable[selected].mean()),
        "frozen_p75_selected_unreliable_prevalence": float(unreliable[frozen_selected].mean()),
        "candidate_event_reduction_within_visited_region": float((all_events / max(len(visited_indices), 1) - candidate_events / max(len(selected), 1)) / max(all_events / max(len(visited_indices), 1), 1e-12)),
        "candidate_event_auroc_within_visited_region": binary_auroc(unreliable[visited_indices], score_np[visited_indices]),
        "candidate_selected_mean_actor_error_m": candidate_error,
        "frozen_p75_selected_mean_actor_error_m": frozen_error,
        "candidate_error_ratio_to_frozen_p75": candidate_error / max(frozen_error, 1e-12),
        "predicted_actor_error_mean_m": float(predicted_error_np[visited_indices].mean()),
        "scene_nonincreasing_count": int(sum(row["selected_event_prevalence"] <= row["visited_event_prevalence"] for row in scene_rows)),
        "scene_count": int(len(scene_rows)), "scene_rows": scene_rows,
    }
    gates = {
        "minimum_event_reduction_within_visited_region": metrics["candidate_event_reduction_within_visited_region"] >= float(config["gates"]["minimum_event_reduction_within_visited_region"]),
        "no_more_events_than_frozen_p75": candidate_events <= frozen_events,
        "maximum_error_ratio_to_frozen_p75": metrics["candidate_error_ratio_to_frozen_p75"] <= float(config["gates"]["maximum_error_ratio_to_frozen_p75"]),
        "minimum_scene_nonincreasing_fraction": metrics["scene_nonincreasing_count"] / max(metrics["scene_count"], 1) >= float(config["gates"]["minimum_scene_nonincreasing_fraction"]),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {"schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict, "role": config["role"],
        "training": {"raw_query_row_count": int(len(arrays["features"])), "deduplicated_actor_state_count": int(len(actor_raw)),
            "unreliable_actor_state_count": int(np.count_nonzero(labels_np)), "horizon_pair_groups": int(len(groups)),
            "final_losses": final}, "fresh_test_visited_evaluation": metrics, "gate_results": gates,
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started}, "claim_boundary": config["claim_boundary"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "fresh_test_visited_evaluation": metrics, "gate_results": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
