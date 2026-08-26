#!/usr/bin/env python3
"""Materialize the V6.2 sparse evidence-query dataset."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motion_proj.worldsim_v62.evidence import (
    build_evidence_grid,
    grid_spec_from_config,
    save_evidence_grid,
)
from motion_proj.worldsim_v62.query_dataset import build_query_arrays


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _unit(task: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    config = task["config"]
    scene = task["scene"]
    target_frame = int(task["target_frame"])
    scene_root = Path(task["scene_root"])
    run_dir = Path(task["run_dir"])
    spec = grid_spec_from_config(config, target_frame)

    candidate_offsets = [int(value) for value in config["sweep_roles"]["method_candidate_offsets"]]
    ordinal = int(task["target_ordinal"])
    dropout_offset = candidate_offsets[ordinal % len(candidate_offsets)]
    method_offsets = [value for value in candidate_offsets if value != dropout_offset]
    target_offsets = [int(value) for value in config["sweep_roles"]["target_evidence_offsets"]]
    method_frames = [target_frame + value for value in method_offsets]
    dropout_frames = [target_frame + dropout_offset]
    target_frames = [target_frame + value for value in target_offsets]

    evidence_kwargs = {
        "record_width": int(config["raw_lidar"]["point_record_float32_width"]),
        "dynamic_box_margin_m": float(config["raw_lidar"]["dynamic_box_margin_m"]),
        "maximum_free_rays_per_sweep": int(config["ray_carving"]["maximum_rays_per_sweep"]),
        "maximum_range_m": float(config["ray_carving"]["maximum_range_m"]),
        "behind_hit_steps": int(config["ray_carving"]["behind_hit_steps"]),
    }
    method = build_evidence_grid(
        scene_root, target_frame, method_frames, spec, **evidence_kwargs
    )
    dropout = build_evidence_grid(
        scene_root, target_frame, dropout_frames, spec, **evidence_kwargs
    )
    target = build_evidence_grid(
        scene_root, target_frame, target_frames, spec, **evidence_kwargs
    )

    unit_root = run_dir / "units" / scene / f"f{target_frame:03d}"
    method_path = unit_root / "METHOD_EVIDENCE.npz"
    dropout_path = unit_root / "DROPOUT_TARGET.npz"
    target_path = unit_root / "TARGET_EVIDENCE.npz"
    query_path = unit_root / "QUERIES.npz"
    save_evidence_grid(method_path, method)
    save_evidence_grid(dropout_path, dropout)
    save_evidence_grid(target_path, target)

    if bool(config.get("queries", {}).get("enabled", True)):
        query_arrays, query_summary = build_query_arrays(
            method,
            dropout,
            target,
            config["queries"],
            int(config["queries"]["seed"])
            + int(task["scene_ordinal"]) * 1000
            + target_frame,
        )
        np.savez_compressed(query_path, **query_arrays)
        query_summary["path"] = str(query_path.relative_to(run_dir))
    else:
        query_summary = {
            "query_count": 0,
            "candidate_pool_counts": {},
            "path": None,
        }

    source_roles = {
        "method": [str(scene_root / f"lidar/{frame:03d}.bin") for frame in method_frames],
        "dropout": [str(scene_root / f"lidar/{frame:03d}.bin") for frame in dropout_frames],
        "target": [str(scene_root / f"lidar/{frame:03d}.bin") for frame in target_frames],
    }
    return {
        "scene": scene,
        "target_frame": target_frame,
        "source_roles": source_roles,
        "method": {
            **method.summary,
            "path": str(method_path.relative_to(run_dir)),
        },
        "dropout": {
            **dropout.summary,
            "path": str(dropout_path.relative_to(run_dir)),
        },
        "target": {
            **target.summary,
            "path": str(target_path.relative_to(run_dir)),
        },
        "queries": {
            **query_summary,
        },
        "wall_seconds": time.monotonic() - started,
    }


def run(
    config_path: Path,
    processed_root: Path,
    run_dir: Path,
    maximum_workers: int,
    limit_units: int | None,
) -> dict[str, Any]:
    started = time.monotonic()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task_id = str(config.get("task_id", "WS-V62-P2-EVIDENCE-QUERY-DATASET-01"))
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)

    tasks: list[dict[str, Any]] = []
    for scene_ordinal, scene in enumerate(config["scenes"]):
        scene_root = processed_root / f"{int(scene['processed_index']):03d}"
        for target_ordinal, target_frame in enumerate(config["targets"]["frame_indices"]):
            tasks.append(
                {
                    "config": config,
                    "scene": scene["name"],
                    "scene_ordinal": scene_ordinal,
                    "target_frame": int(target_frame),
                    "target_ordinal": target_ordinal,
                    "scene_root": str(scene_root),
                    "run_dir": str(run_dir),
                }
            )
    if limit_units is not None:
        tasks = tasks[: int(limit_units)]

    workers = max(1, min(int(maximum_workers), len(tasks)))
    if workers == 1:
        rows = [_unit(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_unit, tasks))
    rows.sort(key=lambda row: (row["scene"], row["target_frame"]))

    role_sets = {name: set() for name in ("method", "dropout", "target")}
    for row in rows:
        for name in role_sets:
            role_sets[name].update(row["source_roles"][name])
    overlap = {
        "method_dropout": sorted(role_sets["method"] & role_sets["dropout"]),
        "method_target": sorted(role_sets["method"] & role_sets["target"]),
        "dropout_target": sorted(role_sets["dropout"] & role_sets["target"]),
    }
    overlap_count = sum(len(value) for value in overlap.values())
    if overlap_count:
        raise RuntimeError(f"source role overlap: {overlap}")

    method_rows = [
        {
            "scene": row["scene"],
            "target_frame": row["target_frame"],
            **row["method"],
        }
        for row in rows
    ]
    dropout_rows = [
        {
            "scene": row["scene"],
            "target_frame": row["target_frame"],
            **row["dropout"],
        }
        for row in rows
    ]
    target_rows = [
        {
            "scene": row["scene"],
            "target_frame": row["target_frame"],
            **row["target"],
        }
        for row in rows
    ]
    query_rows = [
        {
            "scene": row["scene"],
            "target_frame": row["target_frame"],
            **row["queries"],
        }
        for row in rows
    ]
    _write_jsonl(run_dir / "METHOD_EVIDENCE.jsonl", method_rows)
    _write_jsonl(run_dir / "DROPOUT_TARGETS.jsonl", dropout_rows)
    _write_jsonl(run_dir / "TARGET_EVIDENCE.jsonl", target_rows)
    _write_jsonl(run_dir / "QUERY_MANIFEST.jsonl", query_rows)

    split_manifest = {
        "schema_version": "worldsim_v62.p2_split_manifest.v1",
        "task_id": task_id,
        "mode": "probe" if limit_units is not None else "formal",
        "scene_names": sorted({row["scene"] for row in rows}),
        "unit_count": len(rows),
        "target_frames": sorted({row["target_frame"] for row in rows}),
        "source_role_counts": {name: len(values) for name, values in role_sets.items()},
        "source_role_overlap_count": overlap_count,
        "confirmation_read": False,
        "exact_once_test_read": False,
    }
    _write_json(run_dir / "SPLIT_MANIFEST.json", split_manifest)
    disk_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    summary = {
        "schema_version": "worldsim_v62.p2_summary.v1",
        "task_id": task_id,
        "mode": split_manifest["mode"],
        "unit_count": len(rows),
        "scene_count": len(split_manifest["scene_names"]),
        "query_count": sum(row["queries"]["query_count"] for row in rows),
        "source_role_overlap_count": overlap_count,
        "minimum_candidate_pool_counts": (
            {
                name: min(
                    row["queries"]["candidate_pool_counts"][name] for row in rows
                )
                for name in config["queries"]["quotas"]
            }
            if bool(config.get("queries", {}).get("enabled", True))
            else {}
        ),
        "disk_bytes": disk_bytes,
        "maximum_unit_wall_seconds": max((row["wall_seconds"] for row in rows), default=0.0),
        "wall_seconds": time.monotonic() - started,
        "passed": overlap_count == 0,
    }
    _write_json(run_dir / "P2_SUMMARY.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--maximum-workers", type=int, default=2)
    parser.add_argument("--limit-units", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                args.processed_root,
                args.run_dir,
                args.maximum_workers,
                args.limit_units,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
