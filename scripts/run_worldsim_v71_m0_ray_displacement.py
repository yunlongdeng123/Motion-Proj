"""训练并在source Selection评估V7.1 M0位移模型。"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.actor_corpus import load_actor_cache
from motion_proj.worldsim_v71.dataset_nuscenes import build_v71_index, compile_source_scene
from motion_proj.worldsim_v71.evaluate_surface import (
    differentiable_symmetric_chamfer,
    evaluate_actor_surface,
    summarize_surface_rows,
)
from motion_proj.worldsim_v71.first_return_renderer import differentiable_first_return_depth
from motion_proj.worldsim_v71.ray_displacement import (
    M0_EXTRA_FEATURE_NAMES,
    RaySurfaceDisplacementMLP,
    apply_predicted_displacement,
    build_m0_features,
    hard_collision_surface,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _paths(cache_root: Path, maximum: int) -> list[Path]:
    complete = [
        path
        for path in (cache_root / "train").glob("*/*.npz")
        if not path.name.endswith(".tmp.npz")
    ]
    return sorted(complete)[: int(maximum)]


def _wait_for_corpus(cache_root: Path, minimum: int, timeout_seconds: int) -> list[Path]:
    started = time.monotonic()
    while True:
        paths = _paths(cache_root, 10**9)
        if len(paths) >= minimum:
            return paths
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"S2 corpus did not reach {minimum} actors")
        time.sleep(10)


def _raw_features(payload: Mapping[str, Any], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return build_m0_features(
        payload["base_features"],
        payload["candidates"],
        payload["size_lwh_m"],
        payload["evidence_masses"],
        payload["query_sensor_origin"],
        device=device,
    )


def _prepare_actor(path: Path, standardizer: FeatureStandardizer, device: torch.device) -> dict[str, Any] | None:
    payload = load_actor_cache(path)
    if len(payload["candidates"]) == 0:
        return None
    features, rays, normals = _raw_features(payload, device)
    return {
        **payload,
        "path": str(path),
        "features": torch.as_tensor(standardizer.transform(features), dtype=torch.float32, device=device),
        "candidates_t": torch.as_tensor(payload["candidates"], dtype=torch.float32, device=device),
        "anchors_t": torch.as_tensor(payload["anchors"], dtype=torch.float32, device=device),
        "ray_directions_t": torch.as_tensor(rays, dtype=torch.float32, device=device),
        "normals_t": torch.as_tensor(normals, dtype=torch.float32, device=device),
        "size_t": torch.as_tensor(payload["size_lwh_m"], dtype=torch.float32, device=device),
        "evidence_unknown_t": torch.as_tensor(payload["evidence_masses"][:, 2], dtype=torch.float32, device=device),
    }


def _predict(model: RaySurfaceDisplacementMLP, actor: Mapping[str, Any], config: Mapping[str, Any]):
    prediction = model(actor["features"])
    moved, unknown = apply_predicted_displacement(
        actor["candidates_t"],
        actor["ray_directions_t"],
        actor["normals_t"],
        prediction,
        maximum_ray_displacement_m=float(config["maximum_ray_displacement_m"]),
        maximum_normal_displacement_m=float(config["maximum_normal_displacement_m"]),
        actor_half_size_m=actor["size_t"] * 0.5,
        cuboid_padding_m=float(config["cuboid_padding_m"]),
    )
    return prediction, moved, unknown


def _distill(
    model: RaySurfaceDisplacementMLP,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, float | int]]:
    oracle = [actor for actor in actors if len(actor["oracle_displacement"]) == len(actor["candidates_t"])]
    if not oracle:
        raise RuntimeError("M0 distillation has no S1 oracle targets")
    history = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["distillation_epochs"])):
        permutation = torch.randperm(len(oracle)).tolist()
        total = 0.0
        for start in range(0, len(permutation), batch_size):
            losses = []
            for index in permutation[start : start + batch_size]:
                actor = oracle[index]
                _, moved, unknown = _predict(model, actor, config)
                target = torch.as_tensor(actor["oracle_displacement"], dtype=torch.float32, device=moved.device)
                displacement_loss = F.smooth_l1_loss(moved - actor["candidates_t"], target)
                evidence_loss = F.binary_cross_entropy(unknown, actor["evidence_unknown_t"])
                losses.append(displacement_loss + float(config["evidence_loss_weight"]) * evidence_loss)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(losses)
        if epoch in {0, int(config["distillation_epochs"]) - 1}:
            history.append({"epoch": epoch + 1, "mean_loss": total / len(oracle)})
        print(json.dumps({"stage": "m0_distill", "epoch": epoch + 1, "actors": len(oracle), "loss": total / len(oracle)}), flush=True)
    return history


def _limit_tensor(values: torch.Tensor, maximum: int) -> torch.Tensor:
    if len(values) <= maximum:
        return values
    indices = torch.linspace(0, len(values) - 1, steps=maximum, device=values.device).long()
    return values.index_select(0, indices)


def _physical_actor_loss(
    model: RaySurfaceDisplacementMLP,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> torch.Tensor:
    _, moved, unknown = _predict(model, actor, config)
    anchors = _limit_tensor(actor["anchors_t"], int(config["maximum_training_anchors"]))
    surface = torch.cat([anchors, moved], dim=0)
    targets = torch.as_tensor(actor["target"], dtype=torch.float32, device=moved.device)
    origins = torch.as_tensor(actor["target_sensor_origins"], dtype=torch.float32, device=moved.device)
    if len(targets) > int(config["maximum_training_rays"]):
        indices = torch.linspace(0, len(targets) - 1, steps=int(config["maximum_training_rays"]), device=moved.device).long()
        targets = targets.index_select(0, indices)
        origins = origins.index_select(0, indices)
    target_depth = torch.linalg.vector_norm(targets - origins, dim=1)
    with torch.no_grad():
        baseline = torch.cat([anchors, actor["candidates_t"]], dim=0)
        baseline_depth = differentiable_first_return_depth(baseline, origins, targets, **config["renderer"])
        reference_first = F.smooth_l1_loss(baseline_depth, target_depth).clamp_min(1.0e-3)
        reference_surface = differentiable_symmetric_chamfer(baseline, targets).clamp_min(1.0e-3)
    predicted_depth = differentiable_first_return_depth(surface, origins, targets, **config["renderer"])
    first_loss = F.smooth_l1_loss(predicted_depth, target_depth) / reference_first
    surface_loss = differentiable_symmetric_chamfer(surface, targets) / reference_surface
    displacement = moved - actor["candidates_t"]
    anchor_loss = displacement.square().mean() / max(float(config["maximum_ray_displacement_m"]) ** 2, 1.0e-6)
    if len(moved) > 1:
        nearest = torch.cdist(actor["candidates_t"], actor["candidates_t"]).topk(min(5, len(moved)), largest=False).indices[:, 1:]
        center = displacement[:, None, :].expand(-1, nearest.shape[1], -1)
        smooth_loss = (center - displacement[nearest]).square().mean() / max(float(config["maximum_ray_displacement_m"]) ** 2, 1.0e-6)
    else:
        smooth_loss = torch.zeros((), dtype=moved.dtype, device=moved.device)
    evidence_loss = F.binary_cross_entropy(unknown, actor["evidence_unknown_t"])
    return (
        first_loss
        + surface_loss
        + float(config["anchor_loss_weight"]) * anchor_loss
        + float(config["smooth_loss_weight"]) * smooth_loss
        + float(config["evidence_loss_weight"]) * evidence_loss
    )


def _physical_train(
    model: RaySurfaceDisplacementMLP,
    standardizer: FeatureStandardizer,
    cache_root: Path,
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[list[dict[str, float | int]], dict[str, dict[str, Any]]]:
    actor_cache: dict[str, dict[str, Any]] = {}
    history = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(config["physical_epochs"])):
        paths = _paths(cache_root, int(config["maximum_training_actors"]))
        for path in paths:
            key = str(path)
            if key not in actor_cache:
                actor = _prepare_actor(path, standardizer, device)
                if actor is not None:
                    actor_cache[key] = actor
        actors = list(actor_cache.values())
        permutation = torch.randperm(len(actors)).tolist()
        total = 0.0
        for start in range(0, len(permutation), batch_size):
            losses = [
                _physical_actor_loss(model, actors[index], config)
                for index in permutation[start : start + batch_size]
            ]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(losses)
        mean_loss = total / len(actors)
        if epoch in {0, int(config["physical_epochs"]) - 1}:
            history.append({"epoch": epoch + 1, "mean_loss": mean_loss, "actor_count": len(actors)})
        print(json.dumps({"stage": "m0_physical", "epoch": epoch + 1, "actors": len(actors), "loss": mean_loss}), flush=True)
    return history, actor_cache


def _evaluate_bundle(
    bundle: Mapping[str, Any],
    model: RaySurfaceDisplacementMLP,
    standardizer: FeatureStandardizer,
    model_config: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any] | None:
    diagnostics = bundle["diagnostics"]
    candidates = np.asarray(diagnostics["completion_candidates"], dtype=np.float32).reshape(-1, 3)
    if len(candidates) == 0:
        return None
    payload = {
        "base_features": np.asarray(diagnostics["completion_features"], dtype=np.float32),
        "candidates": candidates,
        "size_lwh_m": np.asarray(diagnostics["track"].size_lwh_m, dtype=np.float32),
        "evidence_masses": np.column_stack(
            [
                np.zeros(len(candidates), dtype=np.float32),
                np.ones(len(candidates), dtype=np.float32),
                np.zeros(len(candidates), dtype=np.float32),
            ]
        ),
        "query_sensor_origin": np.asarray(diagnostics["query_sensor_origin"], dtype=np.float32),
        "anchors": np.concatenate(
            [
                np.asarray(diagnostics["kept"], dtype=np.float32).reshape(-1, 3),
                np.asarray(diagnostics["projected"], dtype=np.float32).reshape(-1, 3),
            ],
            axis=0,
        ),
    }
    build_points = np.concatenate([np.asarray(points, dtype=np.float32) for points in diagnostics["build_frame_points"]], axis=0)
    build_origins = np.concatenate(
        [
            np.repeat(np.asarray(origin, dtype=np.float32).reshape(1, 3), len(points), axis=0)
            for points, origin in zip(diagnostics["build_frame_points"], diagnostics["build_sensor_origins"])
        ],
        axis=0,
    )
    if len(build_points) > 16384:
        selected = np.linspace(0, len(build_points) - 1, num=16384, dtype=np.int64)
        build_points = build_points[selected]
        build_origins = build_origins[selected]
    from motion_proj.worldsim_v71.evidence_volume import build_evidential_queries
    payload["evidence_masses"] = build_evidential_queries(
        candidates,
        build_origins,
        build_points,
        beam_radius_m=0.20,
        endpoint_radius_m=0.12,
        device=device,
        query_chunk_size=256,
    ).masses
    features, rays, normals = _raw_features(payload, device)
    actor = {
        "features": torch.as_tensor(standardizer.transform(features), dtype=torch.float32, device=device),
        "candidates_t": torch.as_tensor(candidates, dtype=torch.float32, device=device),
        "anchors_t": torch.as_tensor(payload["anchors"], dtype=torch.float32, device=device),
        "ray_directions_t": torch.as_tensor(rays, dtype=torch.float32, device=device),
        "normals_t": torch.as_tensor(normals, dtype=torch.float32, device=device),
        "size_t": torch.as_tensor(payload["size_lwh_m"], dtype=torch.float32, device=device),
    }
    with torch.inference_mode():
        _, moved, unknown = _predict(model, actor, model_config)
        output = hard_collision_surface(
            actor["anchors_t"], moved, unknown, unknown_threshold=float(model_config["unknown_threshold"])
        ).cpu().numpy()
    output = _voxel_unique(output, float(evaluation["output_voxel_size_m"]))
    row = evaluate_actor_surface(
        np.asarray(diagnostics["compiled"], dtype=np.float32),
        output,
        np.asarray(diagnostics["target"], dtype=np.float32),
        np.asarray(diagnostics["target_sensor_origins"], dtype=np.float32),
        hazardous=bool(bundle["row"]["hazardous"]),
        device=device,
        lateral_tolerance_m=float(evaluation["literal_lateral_tolerance_m"]),
        depth_tolerance_m=float(evaluation["literal_depth_tolerance_m"]),
        distance_chunk_size=int(evaluation["distance_chunk_size"]),
    )
    row.update(
        {
            "scene_name": bundle["scene_name"],
            "track_id": str(bundle["row"]["track_id"]),
            "category": str(bundle["row"]["category"]),
            "candidate_count": len(candidates),
            "unknown_count": int(torch.count_nonzero(unknown >= float(model_config["unknown_threshold"]))),
            "mean_displacement_m": float(torch.linalg.vector_norm(moved - actor["candidates_t"], dim=1).mean()),
        }
    )
    return row


def _decisions(summary: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, bool]:
    hazard_reduction = summary["hazard"]["relative_early_reduction"]
    return {
        "hazard_literal_first_return_relative_reduction": hazard_reduction is not None and float(hazard_reduction) >= float(config["minimum_hazard_literal_relative_reduction"]),
        "chamfer_non_degradation": float(summary["chamfer_delta_m"]) <= float(config["maximum_chamfer_delta_m"]),
        "target_hit_recall": float(summary["hit_recall_delta"]) >= float(config["minimum_hit_recall_delta"]),
        "actor_state_retention": float(summary["minimum_actor_state_retention"]) == float(config["required_actor_state_retention"]),
        "hazard_state_retention": float(summary["minimum_hazard_state_retention"]) == float(config["required_hazard_state_retention"]),
    }


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = json.loads((repo_root / config["source_split"]).read_text(encoding="utf-8"))
    compiler = yaml.safe_load((repo_root / config["p2_config"]).read_text(encoding="utf-8"))
    _deep_update(compiler, config["compiler_overrides"])
    cache_root = Path(config["cache_root"])
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "waiting_initial_corpus"})
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("M0 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        initial_paths = _wait_for_corpus(
            cache_root,
            int(config["source"]["minimum_start_actors"]),
            int(config["source"]["corpus_wait_seconds"]),
        )
        initial_payloads = [load_actor_cache(path) for path in initial_paths]
        feature_arrays = [
            _raw_features(payload, device)[0]
            for payload in initial_payloads
            if len(payload["candidates"])
        ]
        standardizer = FeatureStandardizer.fit(np.concatenate(feature_arrays, axis=0))
        initial_actors = [
            actor
            for path in initial_paths
            if (actor := _prepare_actor(path, standardizer, device)) is not None
        ]
        input_dim = initial_actors[0]["features"].shape[1]
        torch.manual_seed(int(config["model"]["seed"]))
        model = RaySurfaceDisplacementMLP(input_dim, int(config["model"]["hidden_dim"])).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        _write_json(run_dir / "status.json", {"status": "running", "phase": "oracle_distillation"})
        distill_history = _distill(model, initial_actors, config["model"], optimizer)
        _write_json(run_dir / "status.json", {"status": "running", "phase": "physical_training"})
        physical_history, actor_cache = _physical_train(
            model, standardizer, cache_root, config["model"], optimizer, device
        )
        while not (cache_root / "TRAIN_COMPLETE").is_file():
            if time.monotonic() - started > int(config["source"]["corpus_wait_seconds"]):
                raise TimeoutError("S2 corpus did not complete before Selection")
            time.sleep(15)
        corpus_manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
        if corpus_manifest.get("verdict") != "train_corpus_target_met":
            raise RuntimeError("S2 exhausted available train scenes below the frozen tracklet target")
        model.eval()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "standardizer": standardizer.payload(),
                "input_dim": int(input_dim),
                "base_feature_count": int(input_dim - len(M0_EXTRA_FEATURE_NAMES)),
                "extra_feature_names": M0_EXTRA_FEATURE_NAMES,
                "seed": int(config["model"]["seed"]),
            },
            run_dir / "MODEL.pt",
        )
        _write_json(run_dir / "status.json", {"status": "running", "phase": "source_selection"})
        index = build_v71_index(Path(config["source"]["dataset_root"]), split)
        selection_rows = []
        for position, scene_name in enumerate(split["roles"]["selection"]):
            bundles = compile_source_scene(scene_name, index, config["actors"], compiler, device)
            for bundle in bundles:
                row = _evaluate_bundle(bundle, model, standardizer, config["model"], config["evaluation"], device)
                if row is not None:
                    selection_rows.append(row)
            print(json.dumps({"stage": "source_selection", "progress": f"{position + 1}/{len(split['roles']['selection'])}", "scene": scene_name, "actors": len(selection_rows)}), flush=True)
        metrics = summarize_surface_rows(selection_rows)
        decisions = _decisions(metrics, config["decision"])
        passed = all(decisions.values())
        _write_jsonl(run_dir / "SOURCE_SELECTION_ACTORS.jsonl", selection_rows)
        summary = {
            "schema_version": "worldsim_v71.m0_ray_surface_displacement.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m0_source_selection_passed_frozen" if passed else "m0_source_selection_rejected",
            "initial_training_actor_count": len(initial_actors),
            "final_training_actor_count": len(actor_cache),
            "corpus_actor_count": int(corpus_manifest["actor_count"]),
            "selection_actor_count": len(selection_rows),
            "distillation_history": distill_history,
            "physical_history": physical_history,
            "source_selection": metrics,
            "decisions": decisions,
            "selection_read": True,
            "source_final_read": False,
            "external_read": False,
            "failure_ledger_refs": config["failure_ledger_refs"],
            "failure_ledger_delta": "none" if passed else "failure-id-required-at-closeout",
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "status.json", {"status": "done", "phase": "source_selection", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        return {"run_dir": str(run_dir), "verdict": summary["verdict"], "decisions": decisions}
    except Exception as error:
        _write_json(run_dir / "status.json", {"status": "failed", "phase": "m0", "error": f"{type(error).__name__}: {error}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.repo_root.resolve(), args.run_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
