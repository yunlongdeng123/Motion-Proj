"""把冻结的干净 nuScenes 日志物化为 V7.2 ActorBundleV2 与独立 target 分片。"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v71.dataset_nuscenes import build_v71_index, compile_source_scene
from motion_proj.worldsim_v72.data.actor_dataset import save_actor_bundle_v2
from motion_proj.worldsim_v72.data.schema import (
    ActorBundleV2,
    QueryRayBatch,
    RayTargets,
    SurfaceTargets,
)
from motion_proj.worldsim_v72.data.splits import load_data_roles, require_role_access
from motion_proj.worldsim_v72.evaluation.surface_metrics import deterministic_farthest_point_sample


TARGET_SCHEMA_VERSION = "worldsim_v72.actor_targets.v1"


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _limit_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, num=maximum, dtype=np.int64)


def _fixed_count(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if not len(points):
        raise ValueError("不能从空点集生成固定点数输入")
    if len(points) >= count:
        return deterministic_farthest_point_sample(points, count)
    repeats, remainder = divmod(count, len(points))
    parts = [np.tile(points, (repeats, 1))]
    if remainder:
        parts.append(deterministic_farthest_point_sample(points, remainder))
    return np.concatenate(parts).astype(np.float32, copy=False)


def _fixed_pair(
    points: np.ndarray, origins: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    origins = np.asarray(origins, dtype=np.float32).reshape(-1, 3)
    if len(points) != len(origins) or not len(points):
        raise ValueError("target points 与 origins 必须非空且逐点对应")
    if len(points) >= count:
        indices = np.linspace(0, len(points) - 1, num=count, dtype=np.int64)
    else:
        indices = np.resize(np.arange(len(points), dtype=np.int64), count)
    return points[indices], origins[indices]


def _scene_names_for_logs(metadata_root: Path, log_ids: list[str]) -> list[str]:
    wanted = set(log_ids)
    scenes = json.loads((metadata_root / "scene.json").read_text(encoding="utf-8"))
    return [str(row["name"]) for row in scenes if str(row["log_token"]) in wanted]


def _concatenate_build_records(
    records: list[Mapping[str, Any]], maximum_points: int
) -> dict[str, np.ndarray]:
    points = np.concatenate([np.asarray(row["points"], dtype=np.float32) for row in records])
    origins = np.concatenate(
        [
            np.repeat(np.asarray(row["sensor_origin"], dtype=np.float32)[None, :], len(row["points"]), axis=0)
            for row in records
        ]
    )
    frame_ids = np.concatenate(
        [np.repeat(str(row["frame_id"]), len(row["points"])) for row in records]
    )
    sensor_ids = np.concatenate(
        [np.repeat(str(row["sensor_id"]), len(row["points"])) for row in records]
    )
    intensity = np.concatenate([np.asarray(row["intensity"], dtype=np.float32) for row in records])
    rings = np.concatenate([np.asarray(row["beam_or_ring_id"], dtype=np.int32) for row in records])
    selected = _limit_indices(len(points), maximum_points)
    points = points[selected]
    origins = origins[selected]
    displacement = points - origins
    ranges = np.linalg.norm(displacement, axis=1).astype(np.float32)
    directions = displacement / np.maximum(ranges[:, None], 1.0e-8)
    return {
        "points": points,
        "origins": origins,
        "directions": directions.astype(np.float32),
        "ranges": ranges,
        "frame_ids": frame_ids[selected],
        "sensor_ids": sensor_ids[selected],
        "intensity": intensity[selected],
        "rings": rings[selected],
    }


def _target_query(diagnostics: Mapping[str, Any], log_id: str) -> tuple[QueryRayBatch, RayTargets]:
    points = np.asarray(diagnostics["target"], dtype=np.float32)
    origins = np.asarray(diagnostics["target_sensor_origins"], dtype=np.float32)
    displacement = points - origins
    ranges = np.linalg.norm(displacement, axis=1).astype(np.float32)
    directions = displacement / np.maximum(ranges[:, None], 1.0e-8)
    frame_ids = np.asarray(diagnostics["target_frame_ids"], dtype=str)
    timestamps = np.asarray(diagnostics["target_timestamps_ns"], dtype=np.int64)
    rings = np.asarray(diagnostics["target_beam_or_ring_id"], dtype=np.int32)
    poses_by_frame = {
        str(row["frame_id"]): np.asarray(row["world_from_sensor"], dtype=np.float64)
        for row in diagnostics["target_records"]
    }
    poses = np.stack([poses_by_frame[frame_id] for frame_id in frame_ids])
    query = QueryRayBatch(
        ray_id=np.asarray([f"{frame_id}:{index}" for index, frame_id in enumerate(frame_ids)]),
        log_id=log_id,
        frame_id=frame_ids,
        time_ns=timestamps,
        sensor_id=np.repeat("LIDAR_TOP", len(points)),
        origin_m=origins,
        unit_direction=directions,
        range_min_m=np.zeros(len(points), dtype=np.float32),
        range_max_m=np.maximum(ranges + 2.0, 80.0).astype(np.float32),
        query_pose=poses,
        beam_parameters=rings[:, None].astype(np.float32),
        valid_emission_mask=np.ones(len(points), dtype=bool),
    )
    targets = RayTargets(
        return_valid=np.ones(len(points), dtype=bool),
        return_index=np.zeros(len(points), dtype=np.int32),
        range_m=ranges,
        intensity=np.asarray(diagnostics["target_intensity"], dtype=np.float32),
        intensity_valid=np.isfinite(diagnostics["target_intensity"]),
        censoring_or_missing_mask=np.zeros(len(points), dtype=bool),
        object_id_for_evaluation_only=np.repeat(str(diagnostics["track"].track_id), len(points)),
    )
    return query, targets


def _save_targets(
    path: Path,
    query: QueryRayBatch,
    targets: RayTargets,
    surfaces: SurfaceTargets,
    metadata: Mapping[str, Any],
) -> None:
    temporary = path.with_suffix(".tmp.npz")
    arrays: dict[str, Any] = {"schema_version": np.asarray(TARGET_SCHEMA_VERSION)}
    for prefix, value in (("query", query), ("ray", targets), ("surface", surfaces)):
        for name, field in value.__dataclass_fields__.items():
            if name == "log_id":
                arrays[f"{prefix}_{name}"] = np.asarray(getattr(value, name))
            else:
                arrays[f"{prefix}_{name}"] = np.asarray(getattr(value, name))
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    _write_json(path.with_suffix(".json"), {"schema_version": TARGET_SCHEMA_VERSION, **metadata})


def _materialize_actor(
    bundle: Mapping[str, Any], log_id: str, role_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    diagnostics = bundle["diagnostics"]
    track = diagnostics["track"]
    scene_name = str(bundle["scene_name"])
    track_id = str(bundle["row"]["track_id"])
    records = list(diagnostics["build_records"])
    build = _concatenate_build_records(
        records, int(config["cache"]["maximum_build_evidence_points"])
    )
    evidence = np.zeros((len(build["points"]), 3), dtype=np.float32)
    evidence[:, 1] = 1.0
    actor_bundle = ActorBundleV2(
        dataset="nuScenes",
        log_id=log_id,
        scene_id=scene_name,
        actor_id=track_id,
        category=str(bundle["row"]["category"]),
        size_lwh_m=np.asarray(track.size_lwh_m, dtype=np.float32),
        build_frame_ids=np.asarray([str(row["frame_id"]) for row in records]),
        build_time_ns=np.asarray([int(row["timestamp_ns"]) for row in records], dtype=np.int64),
        world_from_sensor=np.stack([row["world_from_sensor"] for row in records]),
        world_from_actor=np.stack([row["world_from_actor"] for row in records]),
        build_points_actor_m=build["points"],
        point_frame_id=build["frame_ids"],
        point_sensor_id=build["sensor_ids"],
        build_ray_origin_actor_m=build["origins"],
        build_ray_direction_actor=build["directions"],
        build_range_m=build["ranges"],
        build_intensity=build["intensity"],
        build_intensity_valid=np.isfinite(build["intensity"]),
        sensor_model_ref=str(config["sensor_model_ref"]),
        beam_or_ring_id=build["rings"].astype(str),
        per_point_time_offset_ns=np.zeros(len(build["points"]), dtype=np.int64),
        evidence_fou=evidence,
        conflict_count=np.zeros(len(build["points"]), dtype=np.int32),
        opportunity_count=np.ones(len(build["points"]), dtype=np.int32),
        build_only_surface_m=np.asarray(diagnostics["canonical"], dtype=np.float32),
        provenance={
            "role": str(bundle["row"]["role"]),
            "split_rule": "frame_rank_mod_3_eq_2_held_out",
            "target_separate": True,
            "source_test_read": False,
            "external_test_read": False,
        },
    )
    actor_dir = role_root / scene_name
    actor_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = actor_dir / f"{track_id}.bundle.npz"
    save_actor_bundle_v2(actor_bundle, bundle_path)

    query, targets = _target_query(diagnostics, log_id)
    free_points = query.origin_m + float(config["cache"]["known_free_fraction"]) * targets.range_m[:, None] * query.unit_direction
    surface_targets = SurfaceTargets(
        surface_points_m=query.origin_m + targets.range_m[:, None] * query.unit_direction,
        frame_id=query.frame_id,
        normals=np.zeros((len(query), 3), dtype=np.float32),
        normal_valid=np.zeros(len(query), dtype=bool),
        free_space_queries_m=free_points.astype(np.float32),
        occupied_queries_m=(query.origin_m + targets.range_m[:, None] * query.unit_direction).astype(np.float32),
        known_label_mask=np.ones(2 * len(query), dtype=bool),
        label_provenance=np.concatenate(
            [np.repeat("measured_prehit_free_sample", len(query)), np.repeat("measured_lidar_endpoint", len(query))]
        ),
        confidence_or_sensor_tolerance_m=np.full(
            2 * len(query), float(config["cache"]["sensor_tolerance_m"]), dtype=np.float32
        ),
    )
    target_path = actor_dir / f"{track_id}.targets.npz"
    _save_targets(
        target_path,
        query,
        targets,
        surface_targets,
        {
            "dataset": "nuScenes",
            "log_id": log_id,
            "scene_id": scene_name,
            "actor_id": track_id,
            "query_ray_count": len(query),
            "target_used_for_input_sampling": False,
        },
    )

    canonical = np.asarray(diagnostics["canonical"], dtype=np.float32)
    target_points = np.asarray(surface_targets.surface_points_m, dtype=np.float32)
    scale_m = float(np.max(track.size_lwh_m) * 0.5)
    target_fixed, target_origins_fixed = _fixed_pair(
        target_points, query.origin_m, int(config["cache"]["target_point_count"])
    )
    adapter_path = actor_dir / f"{track_id}.adapter.npz"
    temporary_adapter = adapter_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_adapter,
        schema_version=np.asarray("worldsim_v72.clean_adapointr_example.v1"),
        partial_normalized=_fixed_count(canonical, int(config["cache"]["input_point_count"])) / scale_m,
        target_normalized=target_fixed / scale_m,
        target_origins_normalized=target_origins_fixed / scale_m,
        canonical=canonical,
        target=target_points,
        target_sensor_origins=query.origin_m,
        size_lwh_m=np.asarray(track.size_lwh_m, dtype=np.float32),
        trajectory_xyz_m=np.asarray(track.city_centers_m, dtype=np.float32),
        scale_m=np.asarray(scale_m, dtype=np.float32),
        hazardous=np.asarray(bool(bundle["row"]["hazardous"])),
        category=np.asarray(str(bundle["row"]["category"])),
    )
    temporary_adapter.replace(adapter_path)
    return {
        "log_id": log_id,
        "scene_name": scene_name,
        "track_id": track_id,
        "category": str(bundle["row"]["category"]),
        "hazardous": bool(bundle["row"]["hazardous"]),
        "relative_bundle": str(bundle_path.relative_to(role_root)),
        "relative_targets": str(target_path.relative_to(role_root)),
        "relative_adapter": str(adapter_path.relative_to(role_root)),
        "build_frame_count": len(records),
        "build_point_count": actor_bundle.point_count,
        "surface_point_count": len(canonical),
        "query_ray_count": len(query),
        "scale_m": scale_m,
    }


def run(config_path: Path, role: str, run_id: str) -> dict[str, Any]:
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    if role not in {"dev", "route_select"}:
        raise PermissionError("clean actor builder 只允许 dev 或 route_select")
    if config["source_test_read"] or config["external_test_read"]:
        raise PermissionError("clean actor builder 不允许读取最终角色")
    roles = load_data_roles(REPO_ROOT / config["roles"])
    log_ids = require_role_access(roles, "nuscenes", role)
    metadata_root = Path(config["dataset_root"]) / "v1.0-trainval"
    scene_names = _scene_names_for_logs(metadata_root, log_ids)
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    role_root = Path(config["output_root"]) / role
    if role_root.exists():
        raise FileExistsError(f"角色缓存已存在: {role_root}")
    role_root.mkdir(parents=True)
    started = time.monotonic()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    compiler = yaml.safe_load((REPO_ROOT / config["compiler_config"]).read_text(encoding="utf-8"))
    _deep_update(compiler, config["compiler_overrides"])
    manifest = {
        "schema_version": "worldsim_v72.clean_actor_data_run.v1",
        "task_id": config["task_id"],
        "run_id": run_id,
        "run_uri": f"run://worldsim_v72/{config['task_id']}/{run_id}",
        "status": "running",
        "role": role,
        "log_ids": log_ids,
        "scene_count": len(scene_names),
        "git_commit": git_commit,
        "source_test_read": False,
        "external_test_read": False,
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": config["failure_ledger_delta"],
    }
    _write_json(run_dir / "manifest.json", manifest)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump({**config, "role": role, "run_id": run_id}, sort_keys=False), encoding="utf-8"
    )
    _write_json(run_dir / "status.json", {"status": "running", "phase": "index"})
    try:
        index = build_v71_index(Path(config["dataset_root"]), {"roles": {role: scene_names}})
        device = torch.device(config["device"])
        rows: list[dict[str, Any]] = []
        scene_to_log = {
            str(row["name"]): str(row["log_token"])
            for row in json.loads((metadata_root / "scene.json").read_text(encoding="utf-8"))
        }
        for scene_index, scene_name in enumerate(scene_names):
            bundles = compile_source_scene(
                scene_name, index, config["actors"], compiler, device
            )
            for bundle in bundles:
                rows.append(_materialize_actor(bundle, scene_to_log[scene_name], role_root, config))
            _write_json(
                run_dir / "status.json",
                {"status": "running", "phase": "materialize", "scenes": scene_index + 1, "actors": len(rows)},
            )
            print(
                json.dumps({"stage": "clean_actor_data", "role": role, "scene": scene_name, "progress": f"{scene_index + 1}/{len(scene_names)}", "actors": len(rows)}),
                flush=True,
            )
        if not rows:
            raise RuntimeError(f"{role} 没有可评价 Actor")
        _write_jsonl(role_root / "ACTORS.jsonl", rows)
        fingerprint = hashlib.sha256(
            json.dumps([[row["log_id"], row["scene_name"], row["track_id"]] for row in rows]).encode()
        ).hexdigest()
        summary = {
            **manifest,
            "status": "done",
            "actor_count": len(rows),
            "log_count": len({row["log_id"] for row in rows}),
            "materialized_scene_count": len({row["scene_name"] for row in rows}),
            "hazard_actor_count": sum(int(row["hazardous"]) for row in rows),
            "build_point_count": sum(row["build_point_count"] for row in rows),
            "query_ray_count": sum(row["query_ray_count"] for row in rows),
            "fingerprint": fingerprint,
            "resources": {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "peak_gpu_memory_gib": torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0,
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(role_root / "manifest.json", summary)
        _write_json(run_dir / "manifest.json", {**manifest, "status": "done", "summary": "summary.json"})
        _write_json(run_dir / "status.json", {"status": "done", "phase": "complete"})
        return summary
    except Exception as error:
        _write_json(run_dir / "status.json", {"status": "failed", "phase": "runtime", "error": f"{type(error).__name__}: {error}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.role, args.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
