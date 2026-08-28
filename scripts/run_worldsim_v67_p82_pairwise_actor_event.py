"""Train source-only pairwise event rankers while the fresh test cohort is prepared."""

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


class EventCostMLP(torch.nn.Module):
    def __init__(self, feature_count: int, hidden_dimensions: Sequence[int]) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = feature_count
        for hidden in hidden_dimensions:
            layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        self.encoder = torch.nn.Sequential(*layers)
        self.event_head = torch.nn.Linear(width, 1)
        self.cost_head = torch.nn.Linear(width, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        return self.event_head(encoded).squeeze(-1), torch.nn.functional.softplus(
            self.cost_head(encoded).squeeze(-1)
        )


def _load(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))


def _combine(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}


def _select_by_scene(score: np.ndarray, scenes: np.ndarray, fraction: float) -> np.ndarray:
    selected: list[int] = []
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        count = max(1, int(np.floor(len(members) * fraction)))
        selected.extend(members[np.argsort(score[members], kind="mergesort")[:count]].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


@torch.no_grad()
def _predict(model: EventCostMLP, features: np.ndarray, mean: np.ndarray, scale: np.ndarray, actor_only: bool) -> tuple[np.ndarray, np.ndarray]:
    normalized = (np.asarray(features, dtype=np.float32) - mean) / scale
    if actor_only:
        normalized = normalized[:, :len(ACTOR_FEATURE_NAMES)]
    event, cost = model(torch.from_numpy(normalized).cuda())
    return event.cpu().numpy(), np.expm1(cost.cpu().numpy()).clip(min=0.0)


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
    raw = np.asarray(arrays["features"], dtype=np.float32)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0).clip(min=1e-4)
    features = torch.from_numpy((raw - mean) / scale).cuda()
    actor_features = features[:, :len(ACTOR_FEATURE_NAMES)]
    log_cost = torch.from_numpy(np.log1p(np.asarray(arrays["target_cost"], dtype=np.float32))).cuda()
    labels_np = (
        (np.asarray(arrays["raw_actor_state_error_m"]) > float(config["evaluation"]["unreliable_actor_state_error_m"]))
        & (np.asarray(arrays["predicted_minimum_separation_m"]) <= float(config["evaluation"]["unreliable_exposure_radius_m"]))
    )
    positive = torch.from_numpy(np.flatnonzero(labels_np)).long().cuda()
    negative = torch.from_numpy(np.flatnonzero(~labels_np)).long().cuda()
    model_config = config["model"]
    query = EventCostMLP(features.shape[1], model_config["hidden_dimensions"]).cuda()
    actor = EventCostMLP(actor_features.shape[1], model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        list(query.parameters()) + list(actor.parameters()),
        lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]),
    )
    pair_count = int(model_config["pair_batch_size"])
    regression_count = int(model_config["regression_batch_size"])
    margin = float(model_config["pair_margin"])
    regression_weight = float(model_config["regression_weight"])
    final = {}
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(model_config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        pos = positive[torch.randint(len(positive), (pair_count,), device="cuda")]
        neg = negative[torch.randint(len(negative), (pair_count,), device="cuda")]
        regression = torch.randint(len(features), (regression_count,), device="cuda")
        query_pos_event, _ = query(features[pos])
        query_neg_event, _ = query(features[neg])
        actor_pos_event, _ = actor(actor_features[pos])
        actor_neg_event, _ = actor(actor_features[neg])
        query_pair = torch.nn.functional.softplus(margin - query_pos_event + query_neg_event).mean()
        actor_pair = torch.nn.functional.softplus(margin - actor_pos_event + actor_neg_event).mean()
        _, query_cost = query(features[regression])
        _, actor_cost = actor(actor_features[regression])
        query_regression = torch.nn.functional.smooth_l1_loss(
            query_cost, log_cost[regression], beta=float(model_config["huber_beta"])
        )
        actor_regression = torch.nn.functional.smooth_l1_loss(
            actor_cost, log_cost[regression], beta=float(model_config["huber_beta"])
        )
        loss = query_pair + actor_pair + regression_weight * (query_regression + actor_regression)
        loss.backward()
        optimizer.step()
        final = {"total": float(loss.detach().cpu()), "query_pair": float(query_pair.detach().cpu()),
            "actor_pair": float(actor_pair.detach().cpu()), "query_regression": float(query_regression.detach().cpu()),
            "actor_regression": float(actor_regression.detach().cpu())}
        if epoch % 250 == 0 or epoch + 1 == int(model_config["epochs"]):
            print(f"pairwise-event epoch={epoch + 1} " + " ".join(f"{key}={value:.6f}" for key, value in final.items()), flush=True)
    torch.save({
        "feature_names": FEATURE_NAMES, "feature_mean": mean, "feature_scale": scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "query_model_state_dict": query.state_dict(), "actor_only_model_state_dict": actor.state_dict(),
    }, run_dir / "PAIRWISE_ACTOR_EVENT.pt")
    del features, actor_features, log_cost, positive, negative
    torch.cuda.empty_cache()

    evaluation_path = args.runs_root / config["evaluation_rows"]["run"] / config["evaluation_rows"]["artifact"]
    deadline = time.monotonic() + float(config["evaluation_rows"]["readiness_timeout_seconds"])
    while not evaluation_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"P81 evaluation rows not ready: {evaluation_path}")
        print("waiting for prospectively frozen P81 evaluation rows", flush=True)
        time.sleep(10.0)
    evaluation = _load(evaluation_path)
    query_event, query_cost_score = _predict(query.eval(), evaluation["features"], mean, scale, False)
    actor_event, actor_cost_score = _predict(actor.eval(), evaluation["features"], mean, scale, True)
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
    fraction = float(config["selection"]["coverage_fraction"])
    query_selected = _select_by_scene(query_event, scenes, fraction)
    actor_selected = _select_by_scene(actor_event, scenes, fraction)
    frozen_selected = _select_by_scene(frozen_score, scenes, fraction)
    labels = (
        (np.asarray(evaluation["raw_actor_state_error_m"]) > float(config["evaluation"]["unreliable_actor_state_error_m"]))
        & (np.asarray(evaluation["predicted_minimum_separation_m"]) <= float(config["evaluation"]["unreliable_exposure_radius_m"]))
    )
    target = np.asarray(evaluation["target_cost"], dtype=np.float64)
    query_events = int(np.count_nonzero(labels[query_selected]))
    actor_events = int(np.count_nonzero(labels[actor_selected]))
    frozen_events = int(np.count_nonzero(labels[frozen_selected]))
    query_cost = float(target[query_selected].mean())
    frozen_cost = float(target[frozen_selected].mean())
    metrics = {
        "row_count": int(len(target)), "selected_row_count": int(len(query_selected)),
        "achieved_coverage": float(len(query_selected) / len(target)),
        "all_unreliable_events": int(np.count_nonzero(labels)),
        "query_selected_unreliable_events": query_events,
        "actor_selected_unreliable_events": actor_events,
        "frozen_p75_selected_unreliable_events": frozen_events,
        "query_selected_unreliable_prevalence": float(labels[query_selected].mean()),
        "actor_selected_unreliable_prevalence": float(labels[actor_selected].mean()),
        "frozen_p75_selected_unreliable_prevalence": float(labels[frozen_selected].mean()),
        "event_reduction_over_actor_only": float((actor_events - query_events) / max(actor_events, 1)),
        "query_event_auroc": binary_auroc(labels, query_event),
        "actor_event_auroc": binary_auroc(labels, actor_event),
        "query_selected_mean_cost": query_cost,
        "actor_selected_mean_cost": float(target[actor_selected].mean()),
        "frozen_p75_selected_mean_cost": frozen_cost,
        "query_cost_ratio_to_frozen_p75": query_cost / max(frozen_cost, 1e-12),
        "query_auxiliary_cost_score_mean": float(query_cost_score.mean()),
        "actor_auxiliary_cost_score_mean": float(actor_cost_score.mean()),
    }
    gates = {
        "minimum_event_reduction_over_actor_only": metrics["event_reduction_over_actor_only"] >= float(config["gates"]["minimum_event_reduction_over_actor_only"]),
        "no_more_events_than_frozen_p75": query_events <= frozen_events,
        "maximum_cost_ratio_to_frozen_p75": metrics["query_cost_ratio_to_frozen_p75"] <= float(config["gates"]["maximum_cost_ratio_to_frozen_p75"]),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"row_count": int(len(raw)), "unreliable_row_count": int(np.count_nonzero(labels_np)), "final_losses": final},
        "fresh_test_evaluation": metrics, "gate_results": gates,
        "resources": {"gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "fresh_test_evaluation": metrics, "gate_results": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
