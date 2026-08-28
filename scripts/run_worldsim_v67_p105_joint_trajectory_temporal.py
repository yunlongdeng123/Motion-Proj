"""Joint trajectory-level and time-local occupancy-flip supervision."""

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
    ACTOR_FEATURE_NAMES, FEATURE_NAMES, ReliabilityMLP, binary_auroc, predict_reliability,
)
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import (
    _group_max_visited_score, _select_by_scene,
)
from scripts.run_worldsim_v67_p87_deepset_trajectory_reliability import DeepSetRisk


class JointTemporalActorRisk(torch.nn.Module):
    def __init__(self, decoder_dimensions: list[int]) -> None:
        super().__init__()
        self.actor_encoder = torch.nn.Sequential(
            torch.nn.Linear(24, 256), torch.nn.SiLU(),
            torch.nn.Linear(256, 128), torch.nn.SiLU(),
        )
        self.temporal_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 64), torch.nn.SiLU(),
        )
        self.token_fuse = torch.nn.Sequential(torch.nn.Linear(192, 128), torch.nn.SiLU())
        self.token_head = torch.nn.Linear(128, 1)
        self.actor_fuse = torch.nn.Sequential(torch.nn.Linear(256, 128), torch.nn.SiLU())
        layers: list[torch.nn.Module] = []
        width = 256
        for hidden in decoder_dimensions:
            layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        self.decoder = torch.nn.Sequential(*layers)
        self.trajectory_head = torch.nn.Linear(width, 1)
        self.register_buffer("fractions", torch.linspace(0.0, 1.0, 9).view(1, 1, 9, 1))

    def forward(self, features: torch.Tensor, actor_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actor = self.actor_encoder(features[..., :24]).unsqueeze(2).expand(-1, -1, 9, -1)
        signed = features[..., 24:33].unsqueeze(-1)
        boundary = features[..., 33:42].unsqueeze(-1)
        fractions = self.fractions.expand(features.shape[0], features.shape[1], -1, -1)
        temporal = self.temporal_encoder(torch.cat((signed, boundary, fractions), dim=-1))
        token = self.token_fuse(torch.cat((actor, temporal), dim=-1))
        token_logits = self.token_head(token).squeeze(-1)
        actor_encoded = self.actor_fuse(torch.cat((token.mean(dim=2), token.max(dim=2).values), dim=-1))
        expanded_mask = actor_mask.unsqueeze(-1)
        mean = (actor_encoded * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1)
        maximum = actor_encoded.masked_fill(~expanded_mask, -torch.inf).max(dim=1).values
        trajectory = self.trajectory_head(self.decoder(torch.cat((mean, maximum), dim=-1))).squeeze(-1)
        return trajectory, token_logits


def _augment(arrays: dict[str, np.ndarray]) -> np.ndarray:
    features = np.asarray(arrays["features"], dtype=np.float32)
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    return np.concatenate((features, signed, np.abs(signed)), axis=1).astype(np.float32)


def _build_sets(arrays: dict[str, np.ndarray], maximum_actors: int) -> dict[str, np.ndarray]:
    raw = _augment(arrays)
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]
    ends = np.r_[starts[1:], len(order)]
    separation = np.asarray(arrays["predicted_minimum_separation_m"], dtype=np.float32)
    row_events = np.asarray(arrays["occupancy_decision_flip"], dtype=bool)
    token_events = np.asarray(arrays["occupancy_decision_flip_profile"], dtype=bool)
    query_sets = []; actor_sets = []; masks = []; token_sets = []; events = []
    scenes = []; horizons = []; identities = []
    for start, end in zip(starts, ends):
        members = order[start:end]
        chosen = members[np.argsort(separation[members], kind="stable")[:maximum_actors]]
        query = np.zeros((maximum_actors, raw.shape[1]), dtype=np.float32)
        actor = np.zeros((maximum_actors, len(ACTOR_FEATURE_NAMES)), dtype=np.float32)
        mask = np.zeros(maximum_actors, dtype=bool)
        temporal = np.zeros((maximum_actors, token_events.shape[1]), dtype=bool)
        query[:len(chosen)] = raw[chosen]
        actor[:len(chosen)] = raw[chosen, :len(ACTOR_FEATURE_NAMES)]
        mask[:len(chosen)] = True
        temporal[:len(chosen)] = token_events[chosen]
        identity = keys[members[0]]
        query_sets.append(query); actor_sets.append(actor); masks.append(mask); token_sets.append(temporal)
        events.append(bool(np.any(row_events[members])))
        scenes.append(int(identity[0])); horizons.append(float(identity[1]) / 10.0); identities.append(identity)
    return {
        "query_sets": np.asarray(query_sets), "actor_sets": np.asarray(actor_sets),
        "mask": np.asarray(masks), "token_events": np.asarray(token_sets),
        "events": np.asarray(events, dtype=bool), "scene_index": np.asarray(scenes, dtype=np.int32),
        "horizon_seconds": np.asarray(horizons, dtype=np.float32),
        "identity": np.asarray(identities, dtype=np.int32),
    }


@torch.no_grad()
def _predict_query(
    model: JointTemporalActorRisk, features: torch.Tensor, mask: torch.Tensor, batch_size: int = 2048,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), batch_size):
        logits, _ = model(features[start:start + batch_size], mask[start:start + batch_size])
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


@torch.no_grad()
def _predict_actor(
    model: DeepSetRisk, features: torch.Tensor, mask: torch.Tensor, batch_size: int = 4096,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), batch_size):
        logits, _ = model(features[start:start + batch_size], mask[start:start + batch_size])
        outputs.append(torch.sigmoid(logits).cpu().numpy())
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
    source_raw = dict(np.load(args.runs_root / config["source_rows"]["run"] /
                              config["source_rows"]["artifact"], allow_pickle=False))
    model_config = config["model"]
    source = _build_sets(source_raw, int(model_config["maximum_visited_actors"]))
    augmented_rows = _augment(source_raw)
    query_mean, query_scale = augmented_rows.mean(0), augmented_rows.std(0).clip(min=1e-4)
    actor_rows = np.asarray(source_raw["features"], dtype=np.float32)[:, :len(ACTOR_FEATURE_NAMES)]
    actor_mean, actor_scale = actor_rows.mean(0), actor_rows.std(0).clip(min=1e-4)
    query_np = (source["query_sets"] - query_mean) / query_scale
    actor_np = (source["actor_sets"] - actor_mean) / actor_scale
    query_np[~source["mask"]] = 0.0; actor_np[~source["mask"]] = 0.0
    query = torch.from_numpy(query_np).cuda(); actor = torch.from_numpy(actor_np).cuda()
    mask = torch.from_numpy(source["mask"]).cuda()
    trajectory_labels = torch.from_numpy(source["events"].astype(np.float32)).cuda()
    token_labels = torch.from_numpy(source["token_events"].astype(np.float32)).cuda()
    horizon_groups = [
        torch.from_numpy(np.flatnonzero(source["horizon_seconds"] == horizon)).long().cuda()
        for horizon in sorted(np.unique(source["horizon_seconds"]).tolist())
    ]
    query_model = JointTemporalActorRisk(model_config["decoder_dimensions"]).cuda()
    actor_model = DeepSetRisk(
        len(ACTOR_FEATURE_NAMES), model_config["element_dimensions"], model_config["decoder_dimensions"],
    ).cuda()
    optimizer = torch.optim.AdamW(
        list(query_model.parameters()) + list(actor_model.parameters()),
        lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]),
    )
    final = {}
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(model_config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        batch = torch.cat([
            group[torch.randint(len(group), (int(model_config["horizon_batch_size"]),), device="cuda")]
            for group in horizon_groups
        ])
        query_logits, token_logits = query_model(query[batch], mask[batch])
        actor_logits, _ = actor_model(actor[batch], mask[batch])
        target = trajectory_labels[batch]
        query_loss = torch.nn.functional.binary_cross_entropy_with_logits(query_logits, target)
        actor_loss = torch.nn.functional.binary_cross_entropy_with_logits(actor_logits, target)
        valid = mask[batch].unsqueeze(-1).expand_as(token_logits)
        local_target = token_labels[batch]
        positive = torch.nonzero((valid & (local_target > 0.5)).reshape(-1), as_tuple=False).flatten()
        negative = torch.nonzero((valid & (local_target < 0.5)).reshape(-1), as_tuple=False).flatten()
        if len(positive):
            sampled_negative = negative[torch.randint(len(negative), (len(positive),), device="cuda")]
            local_indices = torch.cat((positive, sampled_negative))
            auxiliary = torch.nn.functional.binary_cross_entropy_with_logits(
                token_logits.reshape(-1)[local_indices], local_target.reshape(-1)[local_indices],
            )
        else:
            auxiliary = token_logits.sum() * 0.0
        loss = query_loss + actor_loss + float(model_config["temporal_auxiliary_weight"]) * auxiliary
        loss.backward(); optimizer.step()
        final = {"query_trajectory": float(query_loss.detach().cpu()),
                 "actor_trajectory": float(actor_loss.detach().cpu()),
                 "query_temporal_auxiliary": float(auxiliary.detach().cpu())}
        if epoch % 250 == 0 or epoch + 1 == int(model_config["epochs"]):
            print(f"joint-temporal epoch={epoch + 1} " + " ".join(f"{k}={v:.6f}" for k, v in final.items()), flush=True)
    torch.save({
        "query_feature_mean": query_mean, "query_feature_scale": query_scale,
        "actor_feature_mean": actor_mean, "actor_feature_scale": actor_scale,
        "element_dimensions": model_config["element_dimensions"],
        "decoder_dimensions": model_config["decoder_dimensions"],
        "query_model_state_dict": query_model.state_dict(), "actor_model_state_dict": actor_model.state_dict(),
    }, run_dir / config["model_artifact"])

    evaluation_raw = dict(np.load(args.runs_root / config["evaluation_rows"]["run"] /
                                  config["evaluation_rows"]["artifact"], allow_pickle=False))
    evaluation = _build_sets(evaluation_raw, int(model_config["maximum_visited_actors"]))
    evaluation_query_np = (evaluation["query_sets"] - query_mean) / query_scale
    evaluation_actor_np = (evaluation["actor_sets"] - actor_mean) / actor_scale
    evaluation_query_np[~evaluation["mask"]] = 0.0; evaluation_actor_np[~evaluation["mask"]] = 0.0
    evaluation_query = torch.from_numpy(evaluation_query_np).cuda()
    evaluation_actor = torch.from_numpy(evaluation_actor_np).cuda()
    evaluation_mask = torch.from_numpy(evaluation["mask"]).cuda()
    query_score = _predict_query(query_model.eval(), evaluation_query, evaluation_mask)
    actor_score = _predict_actor(actor_model.eval(), evaluation_actor, evaluation_mask)
    frozen = torch.load(args.runs_root / config["frozen_p75"]["run"] /
                        config["frozen_p75"]["artifact"], map_location="cuda")
    frozen_model = ReliabilityMLP(len(FEATURE_NAMES), frozen["hidden_dimensions"]).cuda()
    frozen_model.load_state_dict(frozen["query_model_state_dict"])
    frozen_row_score = predict_reliability(
        frozen_model.eval(), evaluation_raw["features"][:, :len(FEATURE_NAMES)],
        np.asarray(frozen["feature_mean"], dtype=np.float32),
        np.asarray(frozen["feature_scale"], dtype=np.float32),
    )
    row_keys = np.stack((
        evaluation_raw["scene_index"], np.rint(evaluation_raw["horizon_seconds"] * 10).astype(np.int32),
        evaluation_raw["anchor_frame"], evaluation_raw["query_id"],
    ), axis=1)
    frozen_score = _group_max_visited_score(
        row_keys, frozen_row_score,
        np.asarray(evaluation_raw["predicted_minimum_separation_m"])
        <= float(config["evaluation"]["visited_region_radius_m"]),
    )
    scenes, events = evaluation["scene_index"], evaluation["events"]
    fraction = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(query_score, scenes, fraction)
    actor_selected = _select_by_scene(actor_score, scenes, fraction)
    frozen_selected = _select_by_scene(frozen_score, scenes, fraction)
    query_events, actor_events, frozen_events = (
        int(np.count_nonzero(events[index])) for index in (selected, actor_selected, frozen_selected)
    )
    all_prevalence, selected_prevalence = float(events.mean()), float(events[selected].mean())
    metrics = {
        "source_trajectory_count": int(len(source["events"])),
        "evaluation_trajectory_count": int(len(events)), "all_occupancy_flip_events": int(np.count_nonzero(events)),
        "selected_trajectory_count": int(len(selected)), "achieved_coverage": float(len(selected) / len(events)),
        "query_selected_occupancy_flip_events": query_events,
        "actor_selected_occupancy_flip_events": actor_events,
        "frozen_p75_selected_occupancy_flip_events": frozen_events,
        "all_occupancy_flip_prevalence": all_prevalence,
        "query_selected_occupancy_flip_prevalence": selected_prevalence,
        "actor_selected_occupancy_flip_prevalence": float(events[actor_selected].mean()),
        "frozen_p75_selected_occupancy_flip_prevalence": float(events[frozen_selected].mean()),
        "query_event_reduction": float((all_prevalence - selected_prevalence) / max(all_prevalence, 1e-12)),
        "query_event_reduction_over_actor_only": float((actor_events - query_events) / max(actor_events, 1)),
        "query_event_auroc": binary_auroc(events, query_score), "actor_event_auroc": binary_auroc(events, actor_score),
    }
    gates = {
        "minimum_event_reduction_over_actor_only": metrics["query_event_reduction_over_actor_only"]
        >= float(config["gates"]["minimum_event_reduction_over_actor_only"]),
        "minimum_absolute_trajectory_event_reduction": metrics["query_event_reduction"]
        >= float(config["gates"]["minimum_absolute_trajectory_event_reduction"]),
        "no_more_events_than_frozen_p75": query_events <= frozen_events,
        "maximum_max_error_ratio_to_frozen_p75": selected_prevalence
        <= float(events[frozen_selected].mean()) * float(config["gates"]["maximum_max_error_ratio_to_frozen_p75"]),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "training": {"final_losses": final},
        "development_evaluation": metrics, "gate_results": gates,
        "resources": {"gpu": torch.cuda.get_device_name(0),
                      "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict,
                      "development_evaluation": metrics, "gate_results": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
