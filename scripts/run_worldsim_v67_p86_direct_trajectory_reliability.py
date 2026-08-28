"""Train a direct set-summary ranker for reliability of each candidate Ego trajectory."""

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


class TrajectoryRiskMLP(torch.nn.Module):
    def __init__(self, feature_count: int, hidden_dimensions: Sequence[int]) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = feature_count
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


def _aggregate(arrays: dict[str, np.ndarray], radius: float, threshold: float) -> dict[str, np.ndarray]:
    raw = np.asarray(arrays["features"], dtype=np.float32)
    keys = np.stack((arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"]), axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    separation = np.asarray(arrays["predicted_minimum_separation_m"])
    error = np.asarray(arrays["raw_actor_state_error_m"], dtype=np.float32)
    query_features = []
    actor_features = []
    events = []
    max_errors = []
    scenes = []
    horizons = []
    identities = []
    for group in range(int(inverse.max()) + 1):
        members = np.flatnonzero(inverse == group)
        visited = members[separation[members] <= radius]
        if not len(visited):
            continue
        full = raw[visited]
        actor = full[:, :len(ACTOR_FEATURE_NAMES)]
        query_features.append(np.concatenate(([np.log1p(len(visited))], full.min(0), full.mean(0), full.max(0))))
        actor_features.append(np.concatenate(([np.log1p(len(visited))], actor.min(0), actor.mean(0), actor.max(0))))
        events.append(bool(np.any(error[visited] > threshold)))
        max_errors.append(float(np.max(error[visited])))
        identity = keys[members[0]]
        scenes.append(int(identity[0]))
        horizons.append(float(identity[1]) / 10.0)
        identities.append(identity)
    return {"query_features": np.asarray(query_features, dtype=np.float32),
        "actor_features": np.asarray(actor_features, dtype=np.float32),
        "events": np.asarray(events, dtype=bool), "max_error": np.asarray(max_errors, dtype=np.float32),
        "scene_index": np.asarray(scenes, dtype=np.int32), "horizon_seconds": np.asarray(horizons, dtype=np.float32),
        "identity": np.asarray(identities, dtype=np.int32)}


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
    source_path = args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"]
    deadline = time.monotonic() + float(config["evaluation_rows"]["readiness_timeout_seconds"])
    while not source_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"P86 source rows not ready: {source_path}")
        print("waiting for P86 source trajectory rows", flush=True)
        time.sleep(10.0)
    raw_source = dict(np.load(source_path, allow_pickle=False))
    source = _aggregate(raw_source, float(config["evaluation"]["visited_region_radius_m"]),
        float(config["evaluation"]["unreliable_actor_state_error_m"]))
    query_raw = source["query_features"]
    actor_raw = source["actor_features"]
    query_mean, query_scale = query_raw.mean(0), query_raw.std(0).clip(min=1e-4)
    actor_mean, actor_scale = actor_raw.mean(0), actor_raw.std(0).clip(min=1e-4)
    query_features = torch.from_numpy((query_raw - query_mean) / query_scale).cuda()
    actor_features = torch.from_numpy((actor_raw - actor_mean) / actor_scale).cuda()
    log_error = torch.from_numpy(np.log1p(source["max_error"])).cuda()
    labels = source["events"]
    groups = []
    for horizon in sorted(np.unique(source["horizon_seconds"]).tolist()):
        members = source["horizon_seconds"] == horizon
        pos, neg = np.flatnonzero(members & labels), np.flatnonzero(members & ~labels)
        if len(pos) and len(neg):
            groups.append((torch.from_numpy(pos).long().cuda(), torch.from_numpy(neg).long().cuda()))
    model_config = config["model"]
    query_model = TrajectoryRiskMLP(query_features.shape[1], model_config["hidden_dimensions"]).cuda()
    actor_model = TrajectoryRiskMLP(actor_features.shape[1], model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(list(query_model.parameters()) + list(actor_model.parameters()),
        lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
    pair_per_group = max(1, int(model_config["pair_batch_size"]) // len(groups))
    final = {}
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(model_config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        pos = torch.cat([x[torch.randint(len(x), (pair_per_group,), device="cuda")] for x, _ in groups])
        neg = torch.cat([x[torch.randint(len(x), (pair_per_group,), device="cuda")] for _, x in groups])
        regression = torch.randint(len(query_features), (int(model_config["regression_batch_size"]),), device="cuda")
        query_pos, _ = query_model(query_features[pos]); query_neg, _ = query_model(query_features[neg])
        actor_pos, _ = actor_model(actor_features[pos]); actor_neg, _ = actor_model(actor_features[neg])
        query_pair = torch.nn.functional.softplus(float(model_config["pair_margin"]) - query_pos + query_neg).mean()
        actor_pair = torch.nn.functional.softplus(float(model_config["pair_margin"]) - actor_pos + actor_neg).mean()
        _, query_error = query_model(query_features[regression]); _, actor_error = actor_model(actor_features[regression])
        query_reg = torch.nn.functional.smooth_l1_loss(query_error, log_error[regression], beta=float(model_config["huber_beta"]))
        actor_reg = torch.nn.functional.smooth_l1_loss(actor_error, log_error[regression], beta=float(model_config["huber_beta"]))
        loss = query_pair + actor_pair + float(model_config["regression_weight"]) * (query_reg + actor_reg)
        loss.backward(); optimizer.step()
        final = {"total": float(loss.detach().cpu()), "query_pair": float(query_pair.detach().cpu()),
            "actor_pair": float(actor_pair.detach().cpu()), "query_regression": float(query_reg.detach().cpu()),
            "actor_regression": float(actor_reg.detach().cpu())}
        if epoch % 250 == 0 or epoch + 1 == int(model_config["epochs"]):
            print(f"direct-trajectory epoch={epoch + 1} " + " ".join(f"{k}={v:.6f}" for k, v in final.items()), flush=True)
    torch.save({"query_feature_mean": query_mean, "query_feature_scale": query_scale,
        "actor_feature_mean": actor_mean, "actor_feature_scale": actor_scale,
        "hidden_dimensions": model_config["hidden_dimensions"], "query_model_state_dict": query_model.state_dict(),
        "actor_model_state_dict": actor_model.state_dict()}, run_dir / "DIRECT_TRAJECTORY_RELIABILITY.pt")
    evaluation_path = args.runs_root / config["evaluation_rows"]["run"] / config["evaluation_rows"]["artifact"]
    while not evaluation_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"P85 evaluation rows not ready: {evaluation_path}")
        print("waiting for P85 trajectory evaluation rows", flush=True); time.sleep(10.0)
    evaluation_raw = dict(np.load(evaluation_path, allow_pickle=False))
    evaluation = _aggregate(evaluation_raw, float(config["evaluation"]["visited_region_radius_m"]),
        float(config["evaluation"]["unreliable_actor_state_error_m"]))
    with torch.no_grad():
        query_score, _ = query_model.eval()(torch.from_numpy((evaluation["query_features"] - query_mean) / query_scale).cuda())
        actor_score, _ = actor_model.eval()(torch.from_numpy((evaluation["actor_features"] - actor_mean) / actor_scale).cuda())
    query_score, actor_score = query_score.cpu().numpy(), actor_score.cpu().numpy()
    frozen = torch.load(args.runs_root / config["frozen_p75"]["run"] / config["frozen_p75"]["artifact"], map_location="cuda")
    frozen_model = ReliabilityMLP(len(FEATURE_NAMES), frozen["hidden_dimensions"]).cuda(); frozen_model.load_state_dict(frozen["query_model_state_dict"])
    frozen_row_score = predict_reliability(frozen_model.eval(), evaluation_raw["features"],
        np.asarray(frozen["feature_mean"], dtype=np.float32), np.asarray(frozen["feature_scale"], dtype=np.float32))
    frozen_group_score = []
    row_keys = np.stack((evaluation_raw["scene_index"], np.rint(evaluation_raw["horizon_seconds"] * 10).astype(np.int32),
        evaluation_raw["anchor_frame"], evaluation_raw["query_id"]), axis=1)
    for identity in evaluation["identity"]:
        members = np.flatnonzero(np.all(row_keys == identity, axis=1) &
            (evaluation_raw["predicted_minimum_separation_m"] <= float(config["evaluation"]["visited_region_radius_m"])))
        frozen_group_score.append(float(np.max(frozen_row_score[members])))
    frozen_group_score = np.asarray(frozen_group_score)
    scenes, events, max_error = evaluation["scene_index"], evaluation["events"], evaluation["max_error"]
    fraction = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(query_score, scenes, fraction); actor_selected = _select_by_scene(actor_score, scenes, fraction)
    frozen_selected = _select_by_scene(frozen_group_score, scenes, fraction)
    query_events, actor_events, frozen_events = (int(np.count_nonzero(events[index])) for index in (selected, actor_selected, frozen_selected))
    all_prevalence, selected_prevalence = float(events.mean()), float(events[selected].mean())
    query_max_error, frozen_max_error = float(max_error[selected].mean()), float(max_error[frozen_selected].mean())
    metrics = {"source_trajectory_count": int(len(source["events"])), "evaluation_trajectory_count": int(len(events)),
        "selected_trajectory_count": int(len(selected)), "achieved_coverage": float(len(selected) / len(events)),
        "all_unreliable_events": int(np.count_nonzero(events)), "query_selected_unreliable_events": query_events,
        "actor_selected_unreliable_events": actor_events, "frozen_p75_selected_unreliable_events": frozen_events,
        "all_unreliable_prevalence": all_prevalence, "query_selected_unreliable_prevalence": selected_prevalence,
        "actor_selected_unreliable_prevalence": float(events[actor_selected].mean()),
        "frozen_p75_selected_unreliable_prevalence": float(events[frozen_selected].mean()),
        "query_event_reduction": float((all_prevalence - selected_prevalence) / max(all_prevalence, 1e-12)),
        "query_event_reduction_over_actor_only": float((actor_events - query_events) / max(actor_events, 1)),
        "query_event_auroc": binary_auroc(events, query_score), "actor_event_auroc": binary_auroc(events, actor_score),
        "query_selected_mean_max_error_m": query_max_error, "frozen_p75_selected_mean_max_error_m": frozen_max_error,
        "query_max_error_ratio_to_frozen_p75": query_max_error / max(frozen_max_error, 1e-12)}
    gates = {"minimum_event_reduction_over_actor_only": metrics["query_event_reduction_over_actor_only"] >= float(config["gates"]["minimum_event_reduction_over_actor_only"]),
        "minimum_absolute_trajectory_event_reduction": metrics["query_event_reduction"] >= float(config["gates"]["minimum_absolute_trajectory_event_reduction"]),
        "no_more_events_than_frozen_p75": query_events <= frozen_events,
        "maximum_max_error_ratio_to_frozen_p75": metrics["query_max_error_ratio_to_frozen_p75"] <= float(config["gates"]["maximum_max_error_ratio_to_frozen_p75"])}
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {"schema_version": config["output_schema_version"], "task_id": config["task_id"], "hypothesis_id": config["hypothesis_id"],
        "status": "done", "verdict": verdict, "role": config["role"],
        "training": {"source_trajectory_count": int(len(labels)), "source_unreliable_count": int(np.count_nonzero(labels)),
            "horizon_pair_groups": int(len(groups)), "final_losses": final}, "fresh_test_evaluation": metrics, "gate_results": gates,
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "fresh_test_evaluation": metrics, "gate_results": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
