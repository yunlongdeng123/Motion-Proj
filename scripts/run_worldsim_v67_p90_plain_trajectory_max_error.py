"""Train plain continuous max-error regressors over visited Actor sets."""

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
from scripts.run_worldsim_v67_p87_deepset_trajectory_reliability import (
    DeepSetRisk, _build_sets,
)


@torch.no_grad()
def _predict_error(
    model: DeepSetRisk, features: torch.Tensor, mask: torch.Tensor, batch_size: int = 4096,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), batch_size):
        _, prediction = model(features[start:start + batch_size], mask[start:start + batch_size])
        outputs.append(prediction.cpu().numpy())
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
    raw_source = dict(np.load(args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], allow_pickle=False))
    model_config = config["model"]
    source = _build_sets(raw_source, float(config["evaluation"]["visited_region_radius_m"]),
        float(config["evaluation"]["unreliable_actor_state_error_m"]), int(model_config["maximum_visited_actors"]))
    query_rows = np.asarray(raw_source["features"], dtype=np.float32)
    actor_rows = query_rows[:, :len(ACTOR_FEATURE_NAMES)]
    query_mean, query_scale = query_rows.mean(0), query_rows.std(0).clip(min=1e-4)
    actor_mean, actor_scale = actor_rows.mean(0), actor_rows.std(0).clip(min=1e-4)
    query_np = (source["query_sets"] - query_mean) / query_scale
    actor_np = (source["actor_sets"] - actor_mean) / actor_scale
    query_np[~source["mask"]] = 0.0; actor_np[~source["mask"]] = 0.0
    query_sets = torch.from_numpy(query_np).cuda(); actor_sets = torch.from_numpy(actor_np).cuda()
    mask = torch.from_numpy(source["mask"]).cuda()
    target = torch.from_numpy(np.log1p(source["max_error"])).cuda()
    horizon_groups = [torch.from_numpy(np.flatnonzero(source["horizon_seconds"] == horizon)).long().cuda()
        for horizon in sorted(np.unique(source["horizon_seconds"]).tolist())]
    query_model = DeepSetRisk(len(FEATURE_NAMES), model_config["element_dimensions"], model_config["decoder_dimensions"]).cuda()
    actor_model = DeepSetRisk(len(ACTOR_FEATURE_NAMES), model_config["element_dimensions"], model_config["decoder_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(list(query_model.parameters()) + list(actor_model.parameters()),
        lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
    final_query = final_actor = 0.0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(model_config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        batch = torch.cat([group[torch.randint(len(group), (int(model_config["horizon_batch_size"]),), device="cuda")]
            for group in horizon_groups])
        _, query_prediction = query_model(query_sets[batch], mask[batch])
        _, actor_prediction = actor_model(actor_sets[batch], mask[batch])
        query_loss = torch.nn.functional.smooth_l1_loss(query_prediction, target[batch], beta=float(model_config["huber_beta"]))
        actor_loss = torch.nn.functional.smooth_l1_loss(actor_prediction, target[batch], beta=float(model_config["huber_beta"]))
        (query_loss + actor_loss).backward(); optimizer.step()
        final_query, final_actor = float(query_loss.detach().cpu()), float(actor_loss.detach().cpu())
        if epoch % 250 == 0 or epoch + 1 == int(model_config["epochs"]):
            print(f"plain-trajectory epoch={epoch + 1} query={final_query:.6f} actor={final_actor:.6f}", flush=True)
    torch.save({"query_feature_mean": query_mean, "query_feature_scale": query_scale,
        "actor_feature_mean": actor_mean, "actor_feature_scale": actor_scale,
        "maximum_visited_actors": model_config["maximum_visited_actors"],
        "element_dimensions": model_config["element_dimensions"], "decoder_dimensions": model_config["decoder_dimensions"],
        "query_model_state_dict": query_model.state_dict(), "actor_model_state_dict": actor_model.state_dict()},
        run_dir / "PLAIN_TRAJECTORY_MAX_ERROR.pt")
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
    query_score = _predict_error(query_model.eval(), evaluation_query, evaluation_mask)
    actor_score = _predict_error(actor_model.eval(), evaluation_actor, evaluation_mask)
    frozen = torch.load(args.runs_root / config["frozen_p75"]["run"] / config["frozen_p75"]["artifact"], map_location="cuda")
    frozen_model = ReliabilityMLP(len(FEATURE_NAMES), frozen["hidden_dimensions"]).cuda(); frozen_model.load_state_dict(frozen["query_model_state_dict"])
    frozen_row_score = predict_reliability(frozen_model.eval(), evaluation_raw["features"],
        np.asarray(frozen["feature_mean"], dtype=np.float32), np.asarray(frozen["feature_scale"], dtype=np.float32))
    row_keys = np.stack((evaluation_raw["scene_index"], np.rint(evaluation_raw["horizon_seconds"] * 10).astype(np.int32),
        evaluation_raw["anchor_frame"], evaluation_raw["query_id"]), axis=1)
    frozen_score = _group_max_visited_score(row_keys, frozen_row_score,
        np.asarray(evaluation_raw["predicted_minimum_separation_m"]) <= float(config["evaluation"]["visited_region_radius_m"]))
    scenes = evaluation["scene_index"]
    events = evaluation["max_error"] > float(config["evaluation"]["unreliable_actor_state_error_m"])
    max_error = evaluation["max_error"]
    fraction = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(query_score, scenes, fraction); actor_selected = _select_by_scene(actor_score, scenes, fraction)
    frozen_selected = _select_by_scene(frozen_score, scenes, fraction)
    query_events, actor_events, frozen_events = (int(np.count_nonzero(events[index])) for index in (selected, actor_selected, frozen_selected))
    all_prevalence, selected_prevalence = float(events.mean()), float(events[selected].mean())
    query_max_error, frozen_max_error = float(max_error[selected].mean()), float(max_error[frozen_selected].mean())
    metrics = {"source_trajectory_count": int(len(source["max_error"])), "evaluation_trajectory_count": int(len(events)),
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
        "training": {"source_trajectory_count": int(len(source["max_error"])), "horizon_group_count": int(len(horizon_groups)),
            "final_query_loss": final_query, "final_actor_loss": final_actor},
        "fresh_test_evaluation": metrics, "gate_results": gates,
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "fresh_test_evaluation": metrics, "gate_results": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
