"""Attribute anchor first-return contradictions to supervision provenance."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
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
from motion_proj.worldsim_v71.actor_corpus import load_actor_cache
from motion_proj.worldsim_v71.dataset_nuscenes import build_v71_index, compile_source_scene
from motion_proj.worldsim_v71.first_return_renderer import literal_first_return_partition


SURFACES = ("raw_query", "unknown_query", "kept", "projected", "anchors")


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


def _motion_displacement(track: Any) -> float:
    trajectory = np.asarray(track.city_centers_m, dtype=np.float64).reshape(-1, 3)
    if len(trajectory) == 0:
        return 0.0
    return float(np.linalg.norm(trajectory - trajectory[0], axis=1).max(initial=0.0))


def _target_frame_ordinals(origins: np.ndarray) -> np.ndarray:
    mapping: dict[tuple[float, float, float], int] = {}
    ordinals = []
    for origin in np.asarray(origins, dtype=np.float32).reshape(-1, 3):
        key = tuple(float(value) for value in origin)
        if key not in mapping:
            mapping[key] = len(mapping)
        ordinals.append(mapping[key])
    return np.asarray(ordinals, dtype=np.int64)


def _partition(
    surface: np.ndarray,
    target: np.ndarray,
    origins: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    return literal_first_return_partition(
        surface,
        target,
        origins,
        lateral_tolerance_m=float(config["lateral_tolerance_m"]),
        depth_tolerance_m=float(config["depth_tolerance_m"]),
        device=device,
        ray_chunk_size=int(config["ray_chunk_size"]),
        point_chunk_size=int(config["point_chunk_size"]),
    )


def _counts(partition: Mapping[str, np.ndarray], point_count: int) -> dict[str, int]:
    return {
        "point_count": int(point_count),
        "observable_count": int(np.count_nonzero(partition["observable"])),
        "early_count": int(np.count_nonzero(partition["early"])),
        "hit_count": int(np.count_nonzero(partition["hit"])),
    }


def _actor_row(
    scene_name: str,
    bundle: Mapping[str, Any],
    cache_actor: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    diagnostics = bundle["diagnostics"]
    target = np.asarray(diagnostics["target"], dtype=np.float32).reshape(-1, 3)
    origins = np.asarray(diagnostics["target_sensor_origins"], dtype=np.float32).reshape(-1, 3)
    sources = {
        "raw_query": np.asarray(diagnostics["query"], dtype=np.float32).reshape(-1, 3),
        "unknown_query": np.asarray(diagnostics["unknown_query"], dtype=np.float32).reshape(-1, 3),
        "kept": np.asarray(diagnostics["kept"], dtype=np.float32).reshape(-1, 3),
        "projected": np.asarray(diagnostics["projected"], dtype=np.float32).reshape(-1, 3),
    }
    sources["anchors"] = np.concatenate([sources["kept"], sources["projected"]], axis=0)
    partitions = {
        name: _partition(surface, target, origins, config, device)
        for name, surface in sources.items()
    }
    anchor_partition = partitions["anchors"]
    anchor_early = np.asarray(anchor_partition["early"], dtype=bool)
    first_indices = np.asarray(anchor_partition["first_indices"], dtype=np.int64)
    first_is_kept = first_indices < len(sources["kept"])
    kept_first_early = anchor_early & first_is_kept
    projected_first_early = anchor_early & ~first_is_kept
    frame_ordinals = _target_frame_ordinals(origins)
    frame_rows = []
    for ordinal in sorted(set(frame_ordinals.tolist())):
        selected = frame_ordinals == ordinal
        frame_rows.append(
            {
                "target_frame_ordinal": int(ordinal),
                "ray_count": int(np.count_nonzero(selected)),
                "anchor_early_count": int(np.count_nonzero(anchor_early & selected)),
                "anchor_hit_count": int(np.count_nonzero(anchor_partition["hit"] & selected)),
                "kept_first_early_count": int(np.count_nonzero(kept_first_early & selected)),
                "projected_first_early_count": int(np.count_nonzero(projected_first_early & selected)),
            }
        )
    displacement = _motion_displacement(diagnostics["track"])
    row = bundle["row"]
    return {
        "scene_name": scene_name,
        "track_id": str(row["track_id"]),
        "category": str(row["category"]),
        "hazardous": bool(row["hazardous"]),
        "moving": displacement > float(config["moving_max_displacement_m"]),
        "trajectory_max_displacement_m": displacement,
        "ray_count": int(len(target)),
        "cache_target_count": int(len(cache_actor["target"])),
        "cache_anchor_count": int(len(cache_actor["anchors"])),
        "surface_counts": {
            name: _counts(partitions[name], len(sources[name])) for name in SURFACES
        },
        "anchor_first_early_by_provenance": {
            "kept": int(np.count_nonzero(kept_first_early)),
            "projected": int(np.count_nonzero(projected_first_early)),
        },
        "target_frame_rows": frame_rows,
    }


def _stratum(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    rays = sum(int(row["ray_count"]) for row in rows)
    result: dict[str, Any] = {"actor_count": len(rows), "ray_count": rays}
    for name in SURFACES:
        totals = {
            field: sum(int(row["surface_counts"][name][field]) for row in rows)
            for field in ("point_count", "observable_count", "early_count", "hit_count")
        }
        result[name] = {
            **totals,
            "observable_rate": totals["observable_count"] / rays if rays else None,
            "early_rate": totals["early_count"] / rays if rays else None,
            "hit_rate": totals["hit_count"] / rays if rays else None,
        }
    kept = sum(int(row["anchor_first_early_by_provenance"]["kept"]) for row in rows)
    projected = sum(int(row["anchor_first_early_by_provenance"]["projected"]) for row in rows)
    attributed = kept + projected
    result["anchor_first_early_by_provenance"] = {
        "kept_count": kept,
        "projected_count": projected,
        "kept_fraction": kept / attributed if attributed else None,
        "projected_fraction": projected / attributed if attributed else None,
    }
    frame_totals: dict[int, dict[str, int]] = defaultdict(
        lambda: {
            "ray_count": 0,
            "anchor_early_count": 0,
            "anchor_hit_count": 0,
            "kept_first_early_count": 0,
            "projected_first_early_count": 0,
        }
    )
    for row in rows:
        for frame in row["target_frame_rows"]:
            totals = frame_totals[int(frame["target_frame_ordinal"])]
            for field in totals:
                totals[field] += int(frame[field])
    result["target_frame_ordinal"] = {
        str(ordinal): {
            **totals,
            "anchor_early_rate": totals["anchor_early_count"] / totals["ray_count"],
            "anchor_hit_rate": totals["anchor_hit_count"] / totals["ray_count"],
        }
        for ordinal, totals in sorted(frame_totals.items())
    }
    return result


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    categories = sorted({str(row["category"]) for row in rows})
    return {
        "all": _stratum(rows),
        "hazard": _stratum([row for row in rows if bool(row["hazardous"])]),
        "clear": _stratum([row for row in rows if not bool(row["hazardous"])]),
        "moving": _stratum([row for row in rows if bool(row["moving"])]),
        "quasi_static": _stratum([row for row in rows if not bool(row["moving"])]),
        "category": {
            category: _stratum([row for row in rows if str(row["category"]) == category])
            for category in categories
        },
    }


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "selecting"})
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M31 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        eligible = []
        for path in m0_runner._paths(Path(config["cache_root"]), int(config["maximum_training_actors"])):
            actor = load_actor_cache(path)
            if len(actor["candidates"]) > 0:
                eligible.append(actor)
        selected = [
            actor for index, actor in enumerate(eligible)
            if index % int(config["holdout_stride"]) == 0
        ]
        desired = {
            (_scalar_text(actor["scene_name"]), _scalar_text(actor["track_id"])): actor
            for actor in selected
        }
        compiler = yaml.safe_load((repo_root / config["p2_config"]).read_text(encoding="utf-8"))
        _deep_update(compiler, config["compiler_overrides"])
        rows: list[dict[str, Any]] = []
        matched: set[tuple[str, str]] = set()
        scenes = sorted({scene for scene, _ in desired})
        index = build_v71_index(
            Path(config["source"]["dataset_root"]),
            {"roles": {"train": scenes}},
        )
        _write_json(
            run_dir / "status.json",
            {"status": "running", "phase": "attribution", "selected_actors": len(desired), "source_scenes": len(scenes)},
        )
        for scene_index, scene_name in enumerate(scenes):
            bundles = compile_source_scene(scene_name, index, config["actors"], compiler, device)
            for bundle in bundles:
                key = (scene_name, str(bundle["row"]["track_id"]))
                if key not in desired:
                    continue
                rows.append(
                    _actor_row(scene_name, bundle, desired[key], config["evaluation"], device)
                )
                matched.add(key)
            print(
                json.dumps(
                    {
                        "stage": "m31_anchor_attribution",
                        "scene_progress": f"{scene_index + 1}/{len(scenes)}",
                        "matched_actors": len(rows),
                    }
                ),
                flush=True,
            )
        metrics = _summarize(rows) if rows else {}
        missing = [
            {"scene_name": scene, "track_id": track}
            for scene, track in sorted(set(desired) - matched)
        ]
        summary = {
            "schema_version": "worldsim_v71.m31_anchor_contradiction_attribution.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "anchor_contradiction_attributed" if rows else "anchor_attribution_no_source_match",
            "selected_actor_count": len(desired),
            "matched_actor_count": len(rows),
            "missing_actor_count": len(missing),
            "missing_actors": missing,
            "source_scene_count": len(scenes),
            "metrics": metrics,
            "training": False,
            "checkpoint_written": False,
            "surface_filtered_for_metric": False,
            "pretrained_holdout_exposure": True,
            "external_read": False,
            "m21_partial_quality_read": False,
            "causal_attribution_claimed": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_jsonl(run_dir / "ANCHOR_ATTRIBUTION_ROWS.jsonl", rows)
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {"status": "done", "phase": "attribution", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m31", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(run(args.config.resolve(), args.repo_root.resolve(), args.run_id), ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
