"""Materialize producer-side anchor provenance and ray evidence for V7.1."""

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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v71_m0_ray_displacement as m0_runner
from motion_proj.worldsim_v71.actor_corpus import (
    _deterministic_limit,
    load_actor_cache,
)
from motion_proj.worldsim_v71.dataset_drivestudio import (
    compile_processed_scene,
    discover_processed_train_scenes,
)
from motion_proj.worldsim_v71.dataset_nuscenes import build_v71_index, compile_source_scene
from motion_proj.worldsim_v71.evidence_volume import build_evidential_queries


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _scalar_text(value: Any) -> str:
    array = np.asarray(value)
    return str(array.item() if array.ndim == 0 else value)


def _mass_stats(masses: np.ndarray, opportunities: np.ndarray) -> dict[str, Any]:
    if not len(masses):
        return {
            "mean_free": 0.0,
            "mean_occupied": 0.0,
            "mean_unknown": 0.0,
            "unsupported_count": 0,
        }
    return {
        "mean_free": float(np.mean(masses[:, 0])),
        "mean_occupied": float(np.mean(masses[:, 1])),
        "mean_unknown": float(np.mean(masses[:, 2])),
        "unsupported_count": int(np.count_nonzero(opportunities == 0)),
    }


def _materialize(
    bundle: Mapping[str, Any],
    cache_actor: Mapping[str, Any],
    output_path: Path,
    config: Mapping[str, Any],
    source_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    diagnostics = bundle["diagnostics"]
    kept = np.asarray(diagnostics["kept"], dtype=np.float32).reshape(-1, 3)
    projected = np.asarray(diagnostics["projected"], dtype=np.float32).reshape(-1, 3)
    anchors = np.concatenate([kept, projected], axis=0)
    cached_anchors = np.asarray(cache_actor["anchors"], dtype=np.float32).reshape(-1, 3)
    if anchors.shape != cached_anchors.shape or not np.allclose(anchors, cached_anchors):
        raise RuntimeError(
            f"anchor ordering mismatch for {bundle['scene_name']}/{bundle['row']['track_id']}"
        )

    kept_query_indices = np.asarray(
        diagnostics["kept_query_indices"], dtype=np.int64
    )
    projected_query_indices = np.asarray(
        diagnostics["projected_query_indices"], dtype=np.int64
    )
    source_query_indices = np.concatenate(
        [kept_query_indices, projected_query_indices]
    )
    canonical_indices = np.concatenate(
        [
            np.asarray(diagnostics["kept_surface_indices"], dtype=np.int64),
            np.asarray(diagnostics["projected_surface_indices"], dtype=np.int64),
        ]
    )
    provenance = np.concatenate(
        [
            np.zeros(len(kept), dtype=np.int8),
            np.ones(len(projected), dtype=np.int8),
        ]
    )

    build_points_parts = [
        np.asarray(points, dtype=np.float32).reshape(-1, 3)
        for points in diagnostics["build_frame_points"]
    ]
    build_origins_parts = [
        np.repeat(np.asarray(origin, dtype=np.float32).reshape(1, 3), len(points), axis=0)
        for points, origin in zip(build_points_parts, diagnostics["build_sensor_origins"])
    ]
    build_points = np.concatenate(build_points_parts, axis=0)
    build_origins = np.concatenate(build_origins_parts, axis=0)
    build_points, selected = _deterministic_limit(
        build_points, int(config["evidence"]["maximum_build_evidence_points"])
    )
    build_origins = build_origins[selected]
    build = build_evidential_queries(
        anchors,
        build_origins,
        build_points,
        beam_radius_m=float(config["evidence"]["build_beam_radius_m"]),
        endpoint_radius_m=float(config["evidence"]["build_endpoint_radius_m"]),
        device=device,
        query_chunk_size=int(config["evidence"]["query_chunk_size"]),
    )

    target = np.asarray(cache_actor["target"], dtype=np.float32).reshape(-1, 3)
    target_origins = np.asarray(
        cache_actor["target_sensor_origins"], dtype=np.float32
    ).reshape(-1, 3)
    supervision = build_evidential_queries(
        anchors,
        target_origins,
        target,
        beam_radius_m=float(config["evidence"]["supervision_beam_radius_m"]),
        endpoint_radius_m=float(config["evidence"]["supervision_endpoint_radius_m"]),
        device=device,
        query_chunk_size=int(config["evidence"]["query_chunk_size"]),
    )

    query = np.asarray(diagnostics["query"], dtype=np.float32).reshape(-1, 3)
    source_points = query[source_query_indices]
    source_origin = np.asarray(
        diagnostics["query_sensor_origin"], dtype=np.float32
    ).reshape(3)
    source_vectors = source_points - source_origin[None, :]
    source_ranges = np.linalg.norm(source_vectors, axis=1).astype(np.float32)
    source_directions = source_vectors / np.maximum(source_ranges[:, None], 1.0e-6)

    hit_count = np.asarray(diagnostics["canonical_hit_count"], dtype=np.int32)
    temporal_support = np.asarray(
        diagnostics["canonical_temporal_support"], dtype=np.int16
    )
    view_support = np.asarray(
        diagnostics["canonical_view_support"], dtype=np.int8
    )
    canonical_origins = np.asarray(
        diagnostics["canonical_sensor_origins"], dtype=np.float32
    ).reshape(-1, 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray("worldsim_v71.anchor_evidence_sidecar.v1"),
        scene_name=np.asarray(str(bundle["scene_name"])),
        track_id=np.asarray(str(bundle["row"]["track_id"])),
        source_mode=np.asarray(source_mode),
        provenance_names=np.asarray(["KEEP", "PROJECT"]),
        anchors=anchors,
        input_provenance=provenance,
        input_source_query_indices=source_query_indices.astype(np.int32),
        input_source_query_points=source_points.astype(np.float32),
        input_source_query_frame_rank=np.asarray(
            int(diagnostics["query_frame_rank"]), dtype=np.int32
        ),
        input_source_query_timestamp_ns=np.asarray(
            int(diagnostics["query_timestamp_ns"]), dtype=np.int64
        ),
        input_source_sensor_origin=source_origin,
        input_source_ray_directions=source_directions.astype(np.float32),
        input_source_ranges_m=source_ranges,
        input_projection_displacement_xyz_m=(anchors - source_points).astype(np.float32),
        input_canonical_surface_indices=canonical_indices.astype(np.int32),
        input_canonical_hit_count=hit_count[canonical_indices],
        input_canonical_temporal_support=temporal_support[canonical_indices],
        input_canonical_view_support=view_support[canonical_indices],
        input_canonical_sensor_origins=canonical_origins[canonical_indices],
        input_build_frame_count=np.asarray(len(build_points_parts), dtype=np.int16),
        input_build_evidence_masses=build.masses,
        input_build_evidence_opportunities=build.opportunity_count,
        input_build_free_count=build.free_count,
        input_build_occupied_count=build.occupied_count,
        supervision_evidence_masses=supervision.masses,
        supervision_evidence_opportunities=supervision.opportunity_count,
        supervision_free_count=supervision.free_count,
        supervision_occupied_count=supervision.occupied_count,
    )
    temporary.replace(output_path)

    return {
        "scene_name": str(bundle["scene_name"]),
        "track_id": str(bundle["row"]["track_id"]),
        "source_mode": source_mode,
        "anchor_count": int(len(anchors)),
        "kept_count": int(len(kept)),
        "projected_count": int(len(projected)),
        "build": _mass_stats(build.masses, build.opportunity_count),
        "supervision": _mass_stats(
            supervision.masses, supervision.opportunity_count
        ),
        "supervision_conflicted_count": int(
            np.count_nonzero(
                (supervision.free_count > 0) & (supervision.occupied_count > 0)
            )
        ),
        "bytes": int(output_path.stat().st_size),
    }


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    anchor_count = sum(int(row["anchor_count"]) for row in rows)
    kept_count = sum(int(row["kept_count"]) for row in rows)
    projected_count = sum(int(row["projected_count"]) for row in rows)
    return {
        "actor_count": len(rows),
        "anchor_count": anchor_count,
        "kept_count": kept_count,
        "projected_count": projected_count,
        "projected_fraction": projected_count / max(anchor_count, 1),
        "raw_actor_count": sum(row["source_mode"] == "raw_s2" for row in rows),
        "processed_actor_count": sum(
            row["source_mode"] == "processed_recovery" for row in rows
        ),
        "supervision_unsupported_count": sum(
            int(row["supervision"]["unsupported_count"]) for row in rows
        ),
        "supervision_conflicted_count": sum(
            int(row["supervision_conflicted_count"]) for row in rows
        ),
        "cache_bytes": sum(int(row["bytes"]) for row in rows),
    }


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "indexing"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M33 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        desired: dict[tuple[str, str], dict[str, Any]] = {}
        for path in m0_runner._paths(
            Path(config["cache_root"]), int(config["maximum_training_actors"])
        ):
            actor = load_actor_cache(path)
            key = (_scalar_text(actor["scene_name"]), _scalar_text(actor["track_id"]))
            desired[key] = actor
        scenes = sorted({scene for scene, _ in desired})
        manifest = json.loads(
            (Path(config["cache_root"]) / "manifest.json").read_text(encoding="utf-8")
        )
        processed_scenes = set(str(value) for value in manifest["processed_completed_scenes"])
        recovery = yaml.safe_load(
            (repo_root / config["processed_recovery_config"]).read_text(encoding="utf-8")
        )
        processed_lookup = {
            str(row["scene_name"]): Path(row["scene_root"])
            for row in discover_processed_train_scenes(
                [Path(value) for value in recovery["source"]["processed_roots"]],
                Path(recovery["source"]["scene_metadata_path"]),
                [],
            )
        }
        raw_scenes = [scene for scene in scenes if scene not in processed_scenes]
        raw_index = build_v71_index(
            Path(config["source"]["dataset_root"]), {"roles": {"train": raw_scenes}}
        )
        compiler = yaml.safe_load(
            (repo_root / config["p2_config"]).read_text(encoding="utf-8")
        )
        _deep_update(compiler, config["compiler_overrides"])
        sidecar_root = Path(config["sidecar_root"])
        sidecar_root.mkdir(parents=True, exist_ok=False)
        rows: list[dict[str, Any]] = []
        matched: set[tuple[str, str]] = set()
        _write_json(
            run_dir / "status.json",
            {
                "status": "running",
                "phase": "materializing",
                "desired_actors": len(desired),
                "source_scenes": len(scenes),
            },
        )
        for scene_index, scene_name in enumerate(scenes):
            if scene_name in processed_scenes:
                bundles = compile_processed_scene(
                    scene_name,
                    processed_lookup[scene_name],
                    config["actors"],
                    compiler,
                    device,
                    keyframe_stride=int(recovery["source"]["keyframe_stride"]),
                    lidar_record_width=int(recovery["source"]["lidar_record_width"]),
                )
                source_mode = "processed_recovery"
            else:
                bundles = compile_source_scene(
                    scene_name, raw_index, config["actors"], compiler, device
                )
                source_mode = "raw_s2"
            for bundle in bundles:
                key = (scene_name, str(bundle["row"]["track_id"]))
                if key not in desired:
                    continue
                rows.append(
                    _materialize(
                        bundle,
                        desired[key],
                        sidecar_root / "train" / scene_name / f"{key[1]}.npz",
                        config,
                        source_mode,
                        device,
                    )
                )
                matched.add(key)
            progress = {
                "schema_version": "worldsim_v71.anchor_evidence_manifest.v1",
                "status": "running",
                "scene_progress": f"{scene_index + 1}/{len(scenes)}",
                "actor_count": len(rows),
                "desired_actor_count": len(desired),
            }
            _write_json(sidecar_root / "manifest.json", progress)
            print(
                json.dumps(
                    {
                        "stage": "m33_anchor_evidence",
                        "scene_progress": progress["scene_progress"],
                        "actors": len(rows),
                    }
                ),
                flush=True,
            )

        missing = [
            {"scene_name": scene, "track_id": track}
            for scene, track in sorted(set(desired) - matched)
        ]
        if missing:
            raise RuntimeError(f"M33 source replay missed {len(missing)} cached actors")
        aggregate = _aggregate(rows)
        summary = {
            "schema_version": "worldsim_v71.m33_anchor_evidence_sidecar.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "exact_anchor_evidence_sidecar_materialized",
            **aggregate,
            "source_scene_count": len(scenes),
            "raw_source_scene_count": len(raw_scenes),
            "processed_recovery_scene_count": len(scenes) - len(raw_scenes),
            "input_contract": "query/build-only provenance and ray evidence",
            "supervision_contract": "held-out native LiDAR FREE/OCCUPIED/UNKNOWN",
            "training": False,
            "checkpoint_written": False,
            "selection_read": False,
            "source_final_read": False,
            "external_read": False,
            "m21_partial_quality_read": False,
            "surface_filtered_for_metric": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_jsonl(run_dir / "ANCHOR_EVIDENCE_INDEX.jsonl", rows)
        _write_json(run_dir / "summary.json", summary)
        _write_json(sidecar_root / "manifest.json", summary)
        (sidecar_root / "COMPLETE").write_text("done\n", encoding="utf-8")
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "materializing",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": "m33",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config.resolve(), args.repo_root.resolve(), args.run_id),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
