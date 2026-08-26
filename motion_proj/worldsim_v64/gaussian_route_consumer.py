"""Consume frozen Gaussian BEV states on logged future-route corridors."""

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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _future_route_in_target_lidar(
    processed_scene: Path, target_frame: int, future_frame_count: int
) -> np.ndarray:
    pose_root = processed_scene / "lidar_pose"
    target_pose = np.loadtxt(pose_root / f"{target_frame:03d}.txt")
    target_from_world = np.linalg.inv(target_pose)
    route = []
    for frame in range(target_frame + 1, target_frame + future_frame_count + 1):
        path = pose_root / f"{frame:03d}.txt"
        if not path.exists():
            break
        future_pose = np.loadtxt(path)
        world_origin = np.append(future_pose[:3, 3], 1.0)
        route.append((target_from_world @ world_origin)[:2])
    if not route:
        raise RuntimeError(f"future lidar route is empty at frame {target_frame}")
    return np.asarray(route, dtype=np.float32)


def _route_length(route_xy: np.ndarray) -> float:
    points = np.vstack((np.zeros((1, 2), dtype=np.float32), route_xy))
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def run(config_path: Path, runs_root: Path, processed_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v64" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()

    source_root = runs_root / config["inputs"]["gaussian_run"]
    source_paths = sorted((source_root / "units").glob("*/*/GAUSSIAN_STATE.npz"))
    expected_count = int(config["gates"]["expected_case_count"])
    if len(source_paths) != expected_count:
        raise RuntimeError(f"expected {expected_count} Gaussian cases, found {len(source_paths)}")
    processed_indices = {
        str(row["name"]): int(row["processed_index"]) for row in config["scenes"]
    }
    route_config = config["route"]
    future_frame_count = int(route_config["future_frame_count"])
    corridor_radius_m = float(route_config["corridor_radius_m"])
    support_threshold = float(route_config["bev_support_threshold"])
    origin = np.asarray(route_config["grid_origin_xy_m"], dtype=np.float32)
    voxel_size = float(route_config["voxel_size_m"])
    shape = tuple(int(value) for value in route_config["grid_shape_xy"])
    x = origin[0] + (np.arange(shape[0], dtype=np.float32) + 0.5) * voxel_size
    y = origin[1] + (np.arange(shape[1], dtype=np.float32) + 0.5) * voxel_size
    xx, yy = np.meshgrid(x, y, indexing="ij")
    grid_xy = torch.from_numpy(np.stack((xx, yy), axis=-1)).to("cuda")

    rows = []
    rows_path = run_dir / "ROUTE_ROWS.jsonl"
    with torch.inference_mode():
        for source_path in source_paths:
            relative = source_path.relative_to(source_root / "units")
            scene, unit = relative.parts[:2]
            target_frame = int(unit.removeprefix("f"))
            route_xy = _future_route_in_target_lidar(
                processed_root / f"{processed_indices[scene]:03d}",
                target_frame,
                future_frame_count,
            )
            route_tensor = torch.from_numpy(route_xy).to("cuda")
            distance_squared = (
                grid_xy[None] - route_tensor[:, None, None]
            ).square().sum(dim=-1)
            corridor = distance_squared.amin(dim=0) <= corridor_radius_m**2
            corridor_cpu = corridor.cpu().numpy()
            with np.load(source_path, allow_pickle=False) as source:
                c0_density = np.asarray(source["c0_bev_density"], dtype=np.float32)
                m0_density = np.asarray(source["m0_bev_density"], dtype=np.float32)
            c0_support = int(np.count_nonzero(corridor_cpu & (c0_density >= support_threshold)))
            m0_support = int(np.count_nonzero(corridor_cpu & (m0_density >= support_threshold)))
            row = {
                "scene": scene,
                "unit": unit,
                "target_frame": target_frame,
                "future_route_sample_count": int(route_xy.shape[0]),
                "future_route_length_m": _route_length(route_xy),
                "corridor_cell_count": int(np.count_nonzero(corridor_cpu)),
                "c0_route_support_cells": c0_support,
                "m0_route_support_cells": m0_support,
                "additional_route_support_cells": m0_support - c0_support,
                "c0_route_exposure_mass": float(c0_density[corridor_cpu].sum()),
                "m0_route_exposure_mass": float(m0_density[corridor_cpu].sum()),
                "c0_route_intercept": c0_support > 0,
                "m0_route_intercept": m0_support > 0,
            }
            rows.append(row)
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    c0_support = sum(int(row["c0_route_support_cells"]) for row in rows)
    m0_support = sum(int(row["m0_route_support_cells"]) for row in rows)
    c0_intercepts = sum(bool(row["c0_route_intercept"]) for row in rows)
    m0_intercepts = sum(bool(row["m0_route_intercept"]) for row in rows)
    additional_intercept_cases = sum(
        bool(row["m0_route_intercept"]) and not bool(row["c0_route_intercept"])
        for row in rows
    )
    gates = {
        "all_cases_consumed": len(rows) == expected_count,
        "positive_conditional_route_support_gain": m0_support > c0_support,
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "case_count": len(rows),
        "total_logged_future_route_length_m": float(
            sum(float(row["future_route_length_m"]) for row in rows)
        ),
        "c0_route_support_cells": c0_support,
        "m0_route_support_cells": m0_support,
        "additional_route_support_cells": m0_support - c0_support,
        "c0_route_intercept_case_count": c0_intercepts,
        "m0_route_intercept_case_count": m0_intercepts,
        "additional_route_intercept_case_count": additional_intercept_cases,
        "route": {
            "future_seconds": float(route_config["future_seconds"]),
            "future_frame_count": future_frame_count,
            "corridor_radius_m": corridor_radius_m,
            "bev_support_threshold": support_threshold,
        },
        "target_evidence_read": False,
        "model_access": False,
        "collision_ground_truth_read": False,
        "gate_results": gates,
        "resources": {
            "gpu_used": True,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        },
        "failure_ledger_refs": config["failure_ledger_refs"],
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "resource.json", summary["resources"])
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {
        "run_dir": str(run_dir),
        "verdict": verdict,
        "additional_route_support_cells": m0_support - c0_support,
        "additional_route_intercept_case_count": additional_intercept_cases,
        "gate_results": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config.resolve(),
                args.runs_root.resolve(),
                args.processed_root.resolve(),
                args.run_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
