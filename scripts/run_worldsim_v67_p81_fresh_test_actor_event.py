"""Confirm frozen trajectory-conditioned Actor event triage on unread test-role scenes."""

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
    ACTOR_FEATURE_NAMES, FEATURE_NAMES, ReliabilityMLP, binary_auroc,
    materialize_actor_query_rows, predict_reliability,
)


def _select_by_scene(score: np.ndarray, scenes: np.ndarray, fraction: float) -> np.ndarray:
    selected: list[int] = []
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        count = max(1, int(np.floor(len(members) * fraction)))
        selected.extend(members[np.argsort(score[members], kind="mergesort")[:count]].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _load_model(path: Path) -> tuple[ReliabilityMLP, ReliabilityMLP, np.ndarray, np.ndarray]:
    artifact = torch.load(path, map_location="cuda")
    query = ReliabilityMLP(len(FEATURE_NAMES), artifact["hidden_dimensions"]).cuda()
    actor = ReliabilityMLP(len(ACTOR_FEATURE_NAMES), artifact["hidden_dimensions"]).cuda()
    query.load_state_dict(artifact["query_model_state_dict"])
    actor.load_state_dict(artifact["actor_only_model_state_dict"])
    return (
        query.eval(), actor.eval(),
        np.asarray(artifact["feature_mean"], dtype=np.float32),
        np.asarray(artifact["feature_scale"], dtype=np.float32),
    )


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

    data = config["evaluation_data"]
    metadata = Path(data["metadata_root"]) / "v1.0-trainval"
    rows = json.loads((metadata / "scene.json").read_text(encoding="utf-8"))
    index_by_name = {str(row["name"]): index for index, row in enumerate(rows)}
    names = [str(name) for name in data["scene_names"]]
    identities = [(name, int(index_by_name[name])) for name in names]
    root = Path(data["processed_root"])
    deadline = time.monotonic() + float(data["readiness_timeout_seconds"])
    while True:
        ready = [
            (root / f"{index:03d}" / "instances" / "instances_info.json").is_file()
            and (root / f"{index:03d}" / "lidar_pose").is_dir()
            for _, index in identities
        ]
        if all(ready):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"fresh test Actor scenes not ready: {sum(ready)}/{len(ready)}")
        print(f"waiting for fresh test Actor scenes ready={sum(ready)}/{len(ready)}", flush=True)
        time.sleep(10.0)
    arrays = materialize_actor_query_rows(
        [root / f"{index:03d}" for _, index in identities],
        [float(data["horizon_seconds"])], data,
    )
    np.savez_compressed(run_dir / "FRESH_TEST_H3P5_ACTOR_QUERY_ROWS.npz", **arrays)

    query_model, actor_model, mean, scale = _load_model(
        args.runs_root / config["source"]["run"] / config["source"]["artifact"]
    )
    p73_query, _, p73_mean, p73_scale = _load_model(
        args.runs_root / config["p73_baseline"]["run"] / config["p73_baseline"]["artifact"]
    )
    query_score = predict_reliability(query_model, arrays["features"], mean, scale)
    actor_score = predict_reliability(actor_model, arrays["features"], mean, scale, actor_only=True)
    p73_score = predict_reliability(p73_query, arrays["features"], p73_mean, p73_scale)
    scenes = np.asarray(arrays["scene_index"])
    fraction = float(config["selection"]["coverage_fraction"])
    query_selected = _select_by_scene(query_score, scenes, fraction)
    actor_selected = _select_by_scene(actor_score, scenes, fraction)
    p73_selected = _select_by_scene(p73_score, scenes, fraction)
    target = np.asarray(arrays["target_cost"], dtype=np.float64)
    unreliable = (
        (np.asarray(arrays["raw_actor_state_error_m"]) > float(config["evaluation"]["unreliable_actor_state_error_m"]))
        & (np.asarray(arrays["predicted_minimum_separation_m"]) <= float(config["evaluation"]["unreliable_exposure_radius_m"]))
    )
    query_events = int(np.count_nonzero(unreliable[query_selected]))
    actor_events = int(np.count_nonzero(unreliable[actor_selected]))
    p73_events = int(np.count_nonzero(unreliable[p73_selected]))
    scene_rows = []
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        chosen = query_selected[np.isin(query_selected, members)]
        scene_rows.append({
            "scene_index": int(scene), "row_count": int(len(members)),
            "selected_count": int(len(chosen)),
            "all_unreliable_events": int(np.count_nonzero(unreliable[members])),
            "selected_unreliable_events": int(np.count_nonzero(unreliable[chosen])),
            "all_unreliable_prevalence": float(unreliable[members].mean()),
            "selected_unreliable_prevalence": float(unreliable[chosen].mean()),
        })
    reduction = (actor_events - query_events) / max(actor_events, 1)
    selection = {
        "row_count": int(len(target)), "selected_row_count": int(len(query_selected)),
        "achieved_coverage": float(len(query_selected) / len(target)),
        "all_unreliable_events": int(np.count_nonzero(unreliable)),
        "query_selected_unreliable_events": query_events,
        "actor_selected_unreliable_events": actor_events,
        "p73_selected_unreliable_events": p73_events,
        "all_unreliable_prevalence": float(unreliable.mean()),
        "query_selected_unreliable_prevalence": float(unreliable[query_selected].mean()),
        "actor_selected_unreliable_prevalence": float(unreliable[actor_selected].mean()),
        "p73_selected_unreliable_prevalence": float(unreliable[p73_selected].mean()),
        "event_reduction_over_actor_only": float(reduction),
        "all_mean_cost": float(target.mean()),
        "query_selected_mean_cost": float(target[query_selected].mean()),
        "actor_selected_mean_cost": float(target[actor_selected].mean()),
        "p73_selected_mean_cost": float(target[p73_selected].mean()),
        "query_event_auroc": binary_auroc(unreliable, query_score),
        "actor_event_auroc": binary_auroc(unreliable, actor_score),
        "p73_event_auroc": binary_auroc(unreliable, p73_score),
        "scene_nonincreasing_count": int(sum(
            row["selected_unreliable_prevalence"] <= row["all_unreliable_prevalence"]
            for row in scene_rows
        )),
        "scene_count": int(len(scene_rows)), "scene_rows": scene_rows,
    }
    gates = {
        "minimum_event_reduction_over_actor_only": reduction >= float(config["gates"]["minimum_event_reduction_over_actor_only"]),
        "fewer_events_than_p73": query_events < p73_events,
        "minimum_scene_nonincreasing_fraction": (
            selection["scene_nonincreasing_count"] / max(selection["scene_count"], 1)
            >= float(config["gates"]["minimum_scene_nonincreasing_fraction"])
        ),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "fresh_scene_names": names,
        "fresh_scene_indices": [index for _, index in identities],
        "selection": selection, "gate_results": gates,
        "resources": {"gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "selection": selection, "gate_results": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
