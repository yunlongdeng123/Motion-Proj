"""Train shared false-safe/false-alarm heads for development occupancy reliability."""

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
from scripts.run_worldsim_v67_p87_deepset_trajectory_reliability import _build_sets


class MultiTaskDeepSet(torch.nn.Module):
    def __init__(self, feature_count: int, element_dimensions: Sequence[int], decoder_dimensions: Sequence[int]) -> None:
        super().__init__()
        layers = []
        width = feature_count
        for hidden in element_dimensions:
            layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        self.element_encoder = torch.nn.Sequential(*layers)
        layers = []
        decoder_width = width * 2
        for hidden in decoder_dimensions:
            layers.extend((torch.nn.Linear(decoder_width, int(hidden)), torch.nn.SiLU()))
            decoder_width = int(hidden)
        self.decoder = torch.nn.Sequential(*layers)
        self.false_safe_head = torch.nn.Linear(decoder_width, 1)
        self.false_alarm_head = torch.nn.Linear(decoder_width, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.element_encoder(features)
        expanded = mask.unsqueeze(-1)
        mean = (encoded * expanded).sum(1) / expanded.sum(1).clamp(min=1)
        maximum = encoded.masked_fill(~expanded, -torch.inf).max(1).values
        decoded = self.decoder(torch.cat((mean, maximum), dim=1))
        return self.false_safe_head(decoded).squeeze(-1), self.false_alarm_head(decoded).squeeze(-1)


def _group_any(arrays: dict[str, np.ndarray], field: str, identities: np.ndarray) -> np.ndarray:
    keys = np.stack((arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
                     arrays["anchor_frame"], arrays["query_id"]), axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    values = np.zeros(len(unique), dtype=bool)
    np.logical_or.at(values, inverse, np.asarray(arrays[field], dtype=bool))
    table = {tuple(key.tolist()): value for key, value in zip(unique, values)}
    return np.asarray([table[tuple(key.tolist())] for key in identities], dtype=bool)


@torch.no_grad()
def _predict(model: MultiTaskDeepSet, features: torch.Tensor, mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    safe, alarm = [], []
    for start in range(0, len(features), 4096):
        safe_logit, alarm_logit = model(features[start:start + 4096], mask[start:start + 4096])
        safe.append(torch.sigmoid(safe_logit).cpu().numpy())
        alarm.append(torch.sigmoid(alarm_logit).cpu().numpy())
    return np.concatenate(safe), np.concatenate(alarm)


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
    row_root = args.runs_root / config["row_run"]["run"]
    raw_source = dict(np.load(row_root / config["row_run"]["source_artifact"], allow_pickle=False))
    model_config = config["model"]
    source = _build_sets(raw_source, float(config["evaluation"]["visited_region_radius_m"]),
                         float(config["evaluation"]["unreliable_actor_state_error_m"]),
                         int(model_config["maximum_visited_actors"]))
    source_safe = _group_any(raw_source, "occupancy_false_safe", source["identity"])
    source_alarm = _group_any(raw_source, "occupancy_false_alarm", source["identity"])
    rows = np.asarray(raw_source["features"], dtype=np.float32)
    actor_rows = rows[:, :len(ACTOR_FEATURE_NAMES)]
    query_mean, query_scale = rows.mean(0), rows.std(0).clip(min=1e-4)
    actor_mean, actor_scale = actor_rows.mean(0), actor_rows.std(0).clip(min=1e-4)
    query_np = (source["query_sets"] - query_mean) / query_scale
    actor_np = (source["actor_sets"] - actor_mean) / actor_scale
    query_np[~source["mask"]] = 0.0
    actor_np[~source["mask"]] = 0.0
    query_sets = torch.from_numpy(query_np).cuda()
    actor_sets = torch.from_numpy(actor_np).cuda()
    mask = torch.from_numpy(source["mask"]).cuda()
    safe_target = torch.from_numpy(source_safe.astype(np.float32)).cuda()
    alarm_target = torch.from_numpy(source_alarm.astype(np.float32)).cuda()
    horizon_groups = [torch.from_numpy(np.flatnonzero(source["horizon_seconds"] == horizon)).long().cuda()
                      for horizon in sorted(np.unique(source["horizon_seconds"]).tolist())]
    query_model = MultiTaskDeepSet(len(FEATURE_NAMES), model_config["element_dimensions"], model_config["decoder_dimensions"]).cuda()
    actor_model = MultiTaskDeepSet(len(ACTOR_FEATURE_NAMES), model_config["element_dimensions"], model_config["decoder_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(list(query_model.parameters()) + list(actor_model.parameters()),
                                  lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
    final = {}
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(model_config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        batch = torch.cat([group[torch.randint(len(group), (int(model_config["horizon_batch_size"]),), device="cuda")]
                           for group in horizon_groups])
        query_safe, query_alarm = query_model(query_sets[batch], mask[batch])
        actor_safe, actor_alarm = actor_model(actor_sets[batch], mask[batch])
        qs = torch.nn.functional.binary_cross_entropy_with_logits(query_safe, safe_target[batch])
        qa = torch.nn.functional.binary_cross_entropy_with_logits(query_alarm, alarm_target[batch])
        ass = torch.nn.functional.binary_cross_entropy_with_logits(actor_safe, safe_target[batch])
        aa = torch.nn.functional.binary_cross_entropy_with_logits(actor_alarm, alarm_target[batch])
        loss = float(model_config["false_safe_loss_weight"]) * (qs + ass) + float(model_config["false_alarm_loss_weight"]) * (qa + aa)
        loss.backward(); optimizer.step()
        final = {"total": float(loss.detach().cpu()), "query_false_safe": float(qs.detach().cpu()),
                 "query_false_alarm": float(qa.detach().cpu()), "actor_false_safe": float(ass.detach().cpu()),
                 "actor_false_alarm": float(aa.detach().cpu())}
        if epoch % 250 == 0 or epoch + 1 == int(model_config["epochs"]):
            print(f"multitask-occupancy epoch={epoch + 1} " + " ".join(f"{k}={v:.6f}" for k, v in final.items()), flush=True)
    torch.save({"query_feature_mean": query_mean, "query_feature_scale": query_scale,
                "actor_feature_mean": actor_mean, "actor_feature_scale": actor_scale,
                "element_dimensions": model_config["element_dimensions"], "decoder_dimensions": model_config["decoder_dimensions"],
                "query_model_state_dict": query_model.state_dict(), "actor_model_state_dict": actor_model.state_dict()},
               run_dir / "MULTITASK_OCCUPANCY_FLIP.pt")

    raw = dict(np.load(row_root / config["row_run"]["development_artifact"], allow_pickle=False))
    evaluation = _build_sets(raw, float(config["evaluation"]["visited_region_radius_m"]),
                             float(config["evaluation"]["unreliable_actor_state_error_m"]),
                             int(model_config["maximum_visited_actors"]))
    false_safe = _group_any(raw, "occupancy_false_safe", evaluation["identity"])
    false_alarm = _group_any(raw, "occupancy_false_alarm", evaluation["identity"])
    query_np = (evaluation["query_sets"] - query_mean) / query_scale
    actor_np = (evaluation["actor_sets"] - actor_mean) / actor_scale
    query_np[~evaluation["mask"]] = 0.0; actor_np[~evaluation["mask"]] = 0.0
    eval_query = torch.from_numpy(query_np).cuda(); eval_actor = torch.from_numpy(actor_np).cuda()
    eval_mask = torch.from_numpy(evaluation["mask"]).cuda()
    query_safe, query_alarm = _predict(query_model.eval(), eval_query, eval_mask)
    actor_safe, actor_alarm = _predict(actor_model.eval(), eval_actor, eval_mask)
    query_score = 1.0 - (1.0 - query_safe) * (1.0 - query_alarm)
    actor_score = 1.0 - (1.0 - actor_safe) * (1.0 - actor_alarm)
    frozen = torch.load(args.runs_root / config["frozen_p75"]["run"] / config["frozen_p75"]["artifact"], map_location="cuda")
    frozen_model = ReliabilityMLP(len(FEATURE_NAMES), frozen["hidden_dimensions"]).cuda()
    frozen_model.load_state_dict(frozen["query_model_state_dict"])
    frozen_row_score = predict_reliability(frozen_model.eval(), raw["features"],
        np.asarray(frozen["feature_mean"], dtype=np.float32), np.asarray(frozen["feature_scale"], dtype=np.float32))
    keys = np.stack((raw["scene_index"], np.rint(raw["horizon_seconds"] * 10).astype(np.int32),
                     raw["anchor_frame"], raw["query_id"]), axis=1)
    frozen_score = _group_max_visited_score(keys, frozen_row_score,
        np.asarray(raw["predicted_minimum_separation_m"]) <= float(config["evaluation"]["visited_region_radius_m"]))
    scenes, events = evaluation["scene_index"], evaluation["events"]
    fraction = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(query_score, scenes, fraction)
    actor_selected = _select_by_scene(actor_score, scenes, fraction)
    frozen_selected = _select_by_scene(frozen_score, scenes, fraction)
    query_events, actor_events, frozen_events = (int(np.count_nonzero(events[x])) for x in (selected, actor_selected, frozen_selected))
    all_prevalence, selected_prevalence = float(events.mean()), float(events[selected].mean())
    metrics = {"evaluation_trajectory_count": int(len(events)), "all_occupancy_flip_events": int(events.sum()),
        "query_selected_occupancy_flip_events": query_events, "actor_selected_occupancy_flip_events": actor_events,
        "frozen_p75_selected_occupancy_flip_events": frozen_events,
        "query_event_reduction": float((all_prevalence - selected_prevalence) / max(all_prevalence, 1e-12)),
        "query_event_reduction_over_actor_only": float((actor_events - query_events) / max(actor_events, 1)),
        "query_event_auroc": binary_auroc(events, query_score), "actor_event_auroc": binary_auroc(events, actor_score),
        "query_selected_false_safe_events": int(false_safe[selected].sum()),
        "actor_selected_false_safe_events": int(false_safe[actor_selected].sum()),
        "query_selected_false_alarm_events": int(false_alarm[selected].sum()),
        "actor_selected_false_alarm_events": int(false_alarm[actor_selected].sum())}
    gates = {"minimum_event_reduction_over_actor_only": metrics["query_event_reduction_over_actor_only"] >= float(config["gates"]["minimum_event_reduction_over_actor_only"]),
        "minimum_absolute_trajectory_event_reduction": metrics["query_event_reduction"] >= float(config["gates"]["minimum_absolute_trajectory_event_reduction"]),
        "no_more_events_than_frozen_p75": query_events <= frozen_events}
    verdict = "supported_development_multitask_occupancy_flip" if all(gates.values()) else "rejected_development_multitask_occupancy_flip"
    summary = {"schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict, "role": config["role"],
        "training": {"source_trajectory_count": int(len(source["events"])), "final_losses": final},
        "development_evaluation": metrics, "gate_results": gates,
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "development_evaluation": metrics, "gate_results": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
