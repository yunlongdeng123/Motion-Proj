"""Train a permutation-invariant Deep Sets risk model over visited Actor states."""

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
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import (
    _group_max_visited_score, _select_by_scene,
)


class DeepSetRisk(torch.nn.Module):
    def __init__(
        self, feature_count: int, element_dimensions: Sequence[int], decoder_dimensions: Sequence[int],
    ) -> None:
        super().__init__()
        element_layers: list[torch.nn.Module] = []
        width = feature_count
        for hidden in element_dimensions:
            element_layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        self.element_encoder = torch.nn.Sequential(*element_layers)
        decoder_layers: list[torch.nn.Module] = []
        decoder_width = width * 2
        for hidden in decoder_dimensions:
            decoder_layers.extend((torch.nn.Linear(decoder_width, int(hidden)), torch.nn.SiLU()))
            decoder_width = int(hidden)
        self.decoder = torch.nn.Sequential(*decoder_layers)
        self.event_head = torch.nn.Linear(decoder_width, 1)
        self.error_head = torch.nn.Linear(decoder_width, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.element_encoder(features)
        expanded_mask = mask.unsqueeze(-1)
        mean = (encoded * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1)
        maximum = encoded.masked_fill(~expanded_mask, -torch.inf).max(dim=1).values
        decoded = self.decoder(torch.cat((mean, maximum), dim=1))
        return self.event_head(decoded).squeeze(-1), torch.nn.functional.softplus(
            self.error_head(decoded).squeeze(-1)
        )


class SetAttentionRisk(torch.nn.Module):
    def __init__(
        self, feature_count: int, model_dimension: int, attention_heads: int,
        attention_layers: int, decoder_dimensions: Sequence[int],
    ) -> None:
        super().__init__()
        self.input_projection = torch.nn.Sequential(
            torch.nn.Linear(feature_count, model_dimension), torch.nn.SiLU(),
        )
        layer = torch.nn.TransformerEncoderLayer(
            d_model=model_dimension, nhead=attention_heads,
            dim_feedforward=model_dimension * 2, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=attention_layers)
        self.pool_seed = torch.nn.Parameter(torch.zeros(1, 1, model_dimension))
        self.pool = torch.nn.MultiheadAttention(
            model_dimension, attention_heads, dropout=0.0, batch_first=True,
        )
        decoder_layers: list[torch.nn.Module] = []
        width = model_dimension
        for hidden in decoder_dimensions:
            decoder_layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        self.decoder = torch.nn.Sequential(*decoder_layers)
        self.event_head = torch.nn.Linear(width, 1)
        self.error_head = torch.nn.Linear(width, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(
            self.input_projection(features), src_key_padding_mask=~mask,
        )
        seed = self.pool_seed.expand(len(features), -1, -1)
        pooled, _ = self.pool(seed, encoded, encoded, key_padding_mask=~mask, need_weights=False)
        decoded = self.decoder(pooled[:, 0])
        return self.event_head(decoded).squeeze(-1), torch.nn.functional.softplus(
            self.error_head(decoded).squeeze(-1)
        )


def _make_model(feature_count: int, model_config: dict[str, object]) -> torch.nn.Module:
    if str(model_config.get("architecture", "deepset")) == "set_attention":
        return SetAttentionRisk(
            feature_count, int(model_config["model_dimension"]),
            int(model_config["attention_heads"]), int(model_config["attention_layers"]),
            model_config["decoder_dimensions"],
        )
    return DeepSetRisk(
        feature_count, model_config["element_dimensions"], model_config["decoder_dimensions"]
    )


def _build_sets(
    arrays: dict[str, np.ndarray], radius: float, threshold: float, maximum_actors: int,
) -> dict[str, np.ndarray]:
    raw = np.asarray(arrays["features"], dtype=np.float32)
    keys = np.stack((arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"]), axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]
    ends = np.r_[starts[1:], len(order)]
    separation = np.asarray(arrays["predicted_minimum_separation_m"])
    error = np.asarray(arrays["raw_actor_state_error_m"], dtype=np.float32)
    query_sets = []
    actor_sets = []
    masks = []
    events = []
    max_errors = []
    scenes = []
    horizons = []
    identities = []
    all_visited_counts = []
    for start, end in zip(starts, ends):
        members = order[start:end]
        visited = members[separation[members] <= radius]
        if not len(visited):
            continue
        chosen = visited[np.argsort(separation[visited], kind="stable")[:maximum_actors]]
        query = np.zeros((maximum_actors, len(FEATURE_NAMES)), dtype=np.float32)
        actor = np.zeros((maximum_actors, len(ACTOR_FEATURE_NAMES)), dtype=np.float32)
        mask = np.zeros(maximum_actors, dtype=bool)
        query[:len(chosen)] = raw[chosen]
        actor[:len(chosen)] = raw[chosen, :len(ACTOR_FEATURE_NAMES)]
        mask[:len(chosen)] = True
        query_sets.append(query); actor_sets.append(actor); masks.append(mask)
        events.append(bool(np.any(error[visited] > threshold)))
        max_errors.append(float(np.max(error[visited])))
        identity = keys[members[0]]
        scenes.append(int(identity[0])); horizons.append(float(identity[1]) / 10.0); identities.append(identity)
        all_visited_counts.append(int(len(visited)))
    return {"query_sets": np.asarray(query_sets), "actor_sets": np.asarray(actor_sets),
        "mask": np.asarray(masks), "events": np.asarray(events, dtype=bool),
        "max_error": np.asarray(max_errors, dtype=np.float32), "scene_index": np.asarray(scenes, dtype=np.int32),
        "horizon_seconds": np.asarray(horizons, dtype=np.float32), "identity": np.asarray(identities, dtype=np.int32),
        "all_visited_actor_count": np.asarray(all_visited_counts, dtype=np.int16)}


@torch.no_grad()
def _predict_in_batches(
    model: torch.nn.Module, features: torch.Tensor, mask: torch.Tensor, batch_size: int = 4096,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), batch_size):
        score, _ = model(features[start:start + batch_size], mask[start:start + batch_size])
        outputs.append(score.cpu().numpy())
    return np.concatenate(outputs)


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
    raw_source = dict(np.load(source_path, allow_pickle=False))
    model_config = config["model"]
    source = _build_sets(raw_source, float(config["evaluation"]["visited_region_radius_m"]),
        float(config["evaluation"]["unreliable_actor_state_error_m"]), int(model_config["maximum_visited_actors"]))
    query_rows = np.asarray(raw_source["features"], dtype=np.float32)
    actor_rows = query_rows[:, :len(ACTOR_FEATURE_NAMES)]
    query_mean, query_scale = query_rows.mean(0), query_rows.std(0).clip(min=1e-4)
    actor_mean, actor_scale = actor_rows.mean(0), actor_rows.std(0).clip(min=1e-4)
    query_sets_np = (source["query_sets"] - query_mean) / query_scale
    actor_sets_np = (source["actor_sets"] - actor_mean) / actor_scale
    query_sets_np[~source["mask"]] = 0.0
    actor_sets_np[~source["mask"]] = 0.0
    query_sets = torch.from_numpy(query_sets_np).cuda()
    actor_sets = torch.from_numpy(actor_sets_np).cuda()
    mask = torch.from_numpy(source["mask"]).cuda()
    log_error = torch.from_numpy(np.log1p(source["max_error"])).cuda()
    labels = source["events"]
    groups = []
    for horizon in sorted(np.unique(source["horizon_seconds"]).tolist()):
        members = source["horizon_seconds"] == horizon
        pos, neg = np.flatnonzero(members & labels), np.flatnonzero(members & ~labels)
        if len(pos) and len(neg):
            groups.append((torch.from_numpy(pos).long().cuda(), torch.from_numpy(neg).long().cuda()))
    query_model = _make_model(len(FEATURE_NAMES), model_config).cuda()
    actor_model = _make_model(len(ACTOR_FEATURE_NAMES), model_config).cuda()
    optimizer = torch.optim.AdamW(list(query_model.parameters()) + list(actor_model.parameters()),
        lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
    pair_per_group = max(1, int(model_config["pair_batch_size"]) // len(groups))
    final = {}
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(model_config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        pos = torch.cat([x[torch.randint(len(x), (pair_per_group,), device="cuda")] for x, _ in groups])
        neg = torch.cat([x[torch.randint(len(x), (pair_per_group,), device="cuda")] for _, x in groups])
        regression = torch.randint(len(query_sets), (int(model_config["regression_batch_size"]),), device="cuda")
        query_pos, _ = query_model(query_sets[pos], mask[pos]); query_neg, _ = query_model(query_sets[neg], mask[neg])
        actor_pos, _ = actor_model(actor_sets[pos], mask[pos]); actor_neg, _ = actor_model(actor_sets[neg], mask[neg])
        query_pair = torch.nn.functional.softplus(float(model_config["pair_margin"]) - query_pos + query_neg).mean()
        actor_pair = torch.nn.functional.softplus(float(model_config["pair_margin"]) - actor_pos + actor_neg).mean()
        _, query_error = query_model(query_sets[regression], mask[regression])
        _, actor_error = actor_model(actor_sets[regression], mask[regression])
        query_reg = torch.nn.functional.smooth_l1_loss(query_error, log_error[regression], beta=float(model_config["huber_beta"]))
        actor_reg = torch.nn.functional.smooth_l1_loss(actor_error, log_error[regression], beta=float(model_config["huber_beta"]))
        loss = query_pair + actor_pair + float(model_config["regression_weight"]) * (query_reg + actor_reg)
        loss.backward(); optimizer.step()
        final = {"total": float(loss.detach().cpu()), "query_pair": float(query_pair.detach().cpu()),
            "actor_pair": float(actor_pair.detach().cpu()), "query_regression": float(query_reg.detach().cpu()),
            "actor_regression": float(actor_reg.detach().cpu())}
        if epoch % 250 == 0 or epoch + 1 == int(model_config["epochs"]):
            print(f"deepset-trajectory epoch={epoch + 1} " + " ".join(f"{k}={v:.6f}" for k, v in final.items()), flush=True)
    torch.save({"query_feature_mean": query_mean, "query_feature_scale": query_scale,
        "actor_feature_mean": actor_mean, "actor_feature_scale": actor_scale,
        "maximum_visited_actors": model_config["maximum_visited_actors"],
        "architecture": model_config.get("architecture", "deepset"),
        "element_dimensions": model_config.get("element_dimensions"),
        "model_dimension": model_config.get("model_dimension"),
        "attention_heads": model_config.get("attention_heads"),
        "attention_layers": model_config.get("attention_layers"),
        "decoder_dimensions": model_config["decoder_dimensions"],
        "query_model_state_dict": query_model.state_dict(), "actor_model_state_dict": actor_model.state_dict()},
        run_dir / "DEEPSET_TRAJECTORY_RELIABILITY.pt")
    evaluation_path = args.runs_root / config["evaluation_rows"]["run"] / config["evaluation_rows"]["artifact"]
    deadline = time.monotonic() + float(config["evaluation_rows"]["readiness_timeout_seconds"])
    while not evaluation_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"P85 evaluation rows not ready: {evaluation_path}")
        print("waiting for P85 trajectory rows", flush=True); time.sleep(10.0)
    evaluation_raw = dict(np.load(evaluation_path, allow_pickle=False))
    evaluation = _build_sets(evaluation_raw, float(config["evaluation"]["visited_region_radius_m"]),
        float(config["evaluation"]["unreliable_actor_state_error_m"]), int(model_config["maximum_visited_actors"]))
    evaluation_query_np = (evaluation["query_sets"] - query_mean) / query_scale
    evaluation_actor_np = (evaluation["actor_sets"] - actor_mean) / actor_scale
    evaluation_query_np[~evaluation["mask"]] = 0.0; evaluation_actor_np[~evaluation["mask"]] = 0.0
    evaluation_query = torch.from_numpy(evaluation_query_np).cuda(); evaluation_actor = torch.from_numpy(evaluation_actor_np).cuda()
    evaluation_mask = torch.from_numpy(evaluation["mask"]).cuda()
    query_score = _predict_in_batches(query_model.eval(), evaluation_query, evaluation_mask)
    actor_score = _predict_in_batches(actor_model.eval(), evaluation_actor, evaluation_mask)
    frozen = torch.load(args.runs_root / config["frozen_p75"]["run"] / config["frozen_p75"]["artifact"], map_location="cuda")
    frozen_model = ReliabilityMLP(len(FEATURE_NAMES), frozen["hidden_dimensions"]).cuda(); frozen_model.load_state_dict(frozen["query_model_state_dict"])
    frozen_row_score = predict_reliability(frozen_model.eval(), evaluation_raw["features"],
        np.asarray(frozen["feature_mean"], dtype=np.float32), np.asarray(frozen["feature_scale"], dtype=np.float32))
    row_keys = np.stack((evaluation_raw["scene_index"], np.rint(evaluation_raw["horizon_seconds"] * 10).astype(np.int32),
        evaluation_raw["anchor_frame"], evaluation_raw["query_id"]), axis=1)
    frozen_score = _group_max_visited_score(row_keys, frozen_row_score,
        np.asarray(evaluation_raw["predicted_minimum_separation_m"]) <= float(config["evaluation"]["visited_region_radius_m"]))
    scenes, events, max_error = evaluation["scene_index"], evaluation["events"], evaluation["max_error"]
    fraction = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(query_score, scenes, fraction); actor_selected = _select_by_scene(actor_score, scenes, fraction)
    frozen_selected = _select_by_scene(frozen_score, scenes, fraction)
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
        "query_max_error_ratio_to_frozen_p75": query_max_error / max(frozen_max_error, 1e-12),
        "maximum_observed_visited_actor_count": int(evaluation["all_visited_actor_count"].max())}
    gates = {"minimum_event_reduction_over_actor_only": metrics["query_event_reduction_over_actor_only"] >= float(config["gates"]["minimum_event_reduction_over_actor_only"]),
        "minimum_absolute_trajectory_event_reduction": metrics["query_event_reduction"] >= float(config["gates"]["minimum_absolute_trajectory_event_reduction"]),
        "no_more_events_than_frozen_p75": query_events <= frozen_events,
        "maximum_max_error_ratio_to_frozen_p75": metrics["query_max_error_ratio_to_frozen_p75"] <= float(config["gates"]["maximum_max_error_ratio_to_frozen_p75"])}
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {"schema_version": config["output_schema_version"], "task_id": config["task_id"], "hypothesis_id": config["hypothesis_id"],
        "status": "done", "verdict": verdict, "role": config["role"],
        "training": {"source_trajectory_count": int(len(labels)), "source_unreliable_count": int(np.count_nonzero(labels)),
            "horizon_pair_groups": int(len(groups)), "maximum_source_visited_actor_count": int(source["all_visited_actor_count"].max()),
            "final_losses": final}, "fresh_test_evaluation": metrics, "gate_results": gates,
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "fresh_test_evaluation": metrics, "gate_results": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
