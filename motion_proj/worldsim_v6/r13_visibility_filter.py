"""WorldSim V6 R13 目标视角实测 LiDAR 可见性筛选实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

from motion_proj.worldsim_v6.r13_worldspace_route import (
    _metric_row,
    _project_points,
    _render_index,
    _resize_depth,
    _resize_rgb,
)
from motion_proj.worldsim_v6.r3_support import _project_observations


TASK_ID = "WS-V6-R13-WORLDSIM-01"


class R13VisibilityError(RuntimeError):
    """R13 visibility 正式合同失败。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R13VisibilityError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def _common_lateral(rows: list[dict[str, Any]], frames: list[int]) -> float:
    passed: dict[int, set[float]] = {frame: set() for frame in frames}
    for row in rows:
        if row["route_support_pass"] and row["deviation_id"].startswith("lateral_"):
            frame = int(row["case_id"].split("__f")[1].split("__")[0])
            passed[frame].add(float(row["deviation_id"].split("_")[1].removesuffix("m")))
    common = set.intersection(*(passed[frame] for frame in frames))
    return max(common) if common else 0.0


def _observation_depth_raster(
    support: Mapping[str, np.ndarray],
    lateral_m: float,
    forward_m: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, int]:
    """把三相机冻结 LiDAR observation 投到目标视角并生成最近表面 z-buffer。"""
    observations = _project_observations(support, lateral_m, forward_m)
    source_width = int(observations["width"])
    source_height = int(observations["height"])
    x = np.rint(observations["x"].astype(np.float64) * width / source_width).astype(np.int64)
    y = np.rint(observations["y"].astype(np.float64) * height / source_height).astype(np.int64)
    z = observations["z"].astype(np.float32)
    keep = (x >= 0) & (x < width) & (y >= 0) & (y < height) & np.isfinite(z) & (z > 0)
    x, y, z = x[keep], y[keep], z[keep]
    linear = y * width + x
    order = np.lexsort((z, linear))
    ordered_linear = linear[order]
    first = np.r_[True, ordered_linear[1:] != ordered_linear[:-1]] if order.size else np.zeros(0, bool)
    selected = order[first]
    raster = np.full((height, width), np.nan, dtype=np.float32)
    raster[y[selected], x[selected]] = z[selected]
    return raster, int(selected.size)


def _visibility_keep(
    x: np.ndarray,
    y: np.ndarray,
    projected_z: np.ndarray,
    observation_depth: np.ndarray,
    maximum_distance_px: float,
    maximum_relative_disagreement: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.isfinite(observation_depth) & (observation_depth > 0)
    if not np.any(observed) or x.size == 0:
        return np.zeros(x.size, dtype=bool), np.full(x.size, np.inf), np.full(x.size, np.inf)
    distance, nearest = distance_transform_edt(
        ~observed, return_distances=True, return_indices=True
    )
    nearest_z = observation_depth[nearest[0, y, x], nearest[1, y, x]]
    disagreement = np.abs(projected_z - nearest_z) / np.maximum(np.abs(nearest_z), 1.0e-6)
    keep = (
        np.isfinite(nearest_z)
        & (distance[y, x] <= maximum_distance_px)
        & (disagreement <= maximum_relative_disagreement)
    )
    return keep, distance[y, x].astype(np.float32), disagreement.astype(np.float32)


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R13VisibilityError("正式 R13 visibility run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R13VisibilityError("R13 visibility task_id 漂移")
    sources = config["sources"]
    base = _resolve_runs_uri(sources["base_run"])
    r3 = _resolve_runs_uri(sources["r3_render_run"])
    street_root = r3 / "renders/scene-0242/streetgs"
    frozen = {
        base / "MANIFEST.json": sources["base_manifest_sha256"],
        base / "R13_GATE.json": sources["base_gate_sha256"],
        base / "ROUTE_METRICS.jsonl": sources["base_route_metrics_sha256"],
        street_root / "RENDER_MAP.jsonl": sources["streetgs_render_map_sha256"],
    }
    for frame, digest in sources["support_frame_sha256"].items():
        frozen[street_root / f"support_frame_{int(frame):03d}.npz"] = digest
    for path, expected in frozen.items():
        if _sha256(path) != expected:
            raise R13VisibilityError(f"冻结输入漂移：{path}")
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R13VisibilityError("R13 visibility 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__worldspace-visibility-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        world_rows = _read_jsonl(base / "WORLD_CHUNKS.jsonl")
        chunks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for row in world_rows:
            path = base / "world_chunks" / f"{row['case_id']}.npz"
            if _sha256(path) != row["world_chunk_sha256"]:
                raise R13VisibilityError(f"base world chunk 漂移：{row['case_id']}")
            frozen[path] = row["world_chunk_sha256"]
            with np.load(path, allow_pickle=False) as archive:
                chunks[row["case_id"]] = (
                    np.asarray(archive["points_world_m"], dtype=np.float32),
                    np.asarray(archive["rgb_float32"], dtype=np.float32),
                )
        immutable_before = {str(path): _sha256(path) for path in frozen}
        street_index = _render_index(street_root)
        width, height = [int(value) for value in config["metrics"]["resolution_px"]]
        frames = [int(value) for value in config["cohort"]["frame_indices"]]
        visibility = config["visibility"]

        def evaluate() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            metric_rows: list[dict[str, Any]] = []
            ablation_rows: list[dict[str, Any]] = []
            for frame in frames:
                case_id = config["cohort"]["case_ids"][frame]
                points, colors = chunks[case_id]
                with np.load(street_root / f"support_frame_{frame:03d}.npz", allow_pickle=False) as support:
                    source_height, source_width = np.asarray(support["cam0_rgb"]).shape[:2]
                    intrinsics = np.asarray(support["cam0_intrinsics"], dtype=np.float64).copy()
                    intrinsics[0] *= width / source_width
                    intrinsics[1] *= height / source_height
                    camera_to_world = np.asarray(support["cam0_camera_to_world"], dtype=np.float64)
                    for deviation in config["cohort"]["route_deviations"]:
                        lateral_m = float(deviation["lateral_m"])
                        forward_m = float(deviation["forward_m"])
                        target_c2w = camera_to_world.copy()
                        target_c2w[:3, 3] += target_c2w[:3, 0] * lateral_m
                        target_c2w[:3, 3] += target_c2w[:3, 2] * forward_m
                        x, y, z, source_index = _project_points(
                            points, target_c2w, intrinsics, width, height
                        )
                        observation_depth, observation_count = _observation_depth_raster(
                            support, lateral_m, forward_m, width, height
                        )
                        keep, nearest_distance, disagreement = _visibility_keep(
                            x,
                            y,
                            z,
                            observation_depth,
                            float(visibility["maximum_nearest_observation_distance_px"]),
                            float(visibility["maximum_relative_depth_disagreement"]),
                        )
                        key = (frame, deviation["variant"], lateral_m)
                        with np.load(street_index[key], allow_pickle=False) as archive:
                            target_rgb = _resize_rgb(np.asarray(archive["rgb"]), width, height)
                            target_depth = _resize_depth(np.asarray(archive["depth"]), width, height)
                        row = _metric_row(
                            "v6_target_view_lidar_visibility",
                            case_id,
                            deviation["id"],
                            x[keep],
                            y[keep],
                            z[keep],
                            colors[source_index[keep]],
                            target_rgb,
                            target_depth,
                            config["metrics"],
                        )
                        metric_rows.append(row)
                        ablation_rows.append(
                            {
                                "schema_version": "worldsim_v6.r13_visibility_ablation.v1",
                                "case_id": case_id,
                                "deviation_id": deviation["id"],
                                "before_projected_pixels": int(x.size),
                                "after_projected_pixels": int(np.count_nonzero(keep)),
                                "keep_fraction": float(np.mean(keep)) if keep.size else 0.0,
                                "target_view_observation_pixels": observation_count,
                                "kept_nearest_distance_px_mean": float(np.mean(nearest_distance[keep]))
                                if np.any(keep)
                                else None,
                                "kept_relative_depth_disagreement_mean": float(np.mean(disagreement[keep]))
                                if np.any(keep)
                                else None,
                            }
                        )
            return metric_rows, ablation_rows

        rows, ablation_rows = evaluate()
        repeated_rows, repeated_ablation = evaluate()
        repeat_exact = _canonical([rows, ablation_rows]) == _canonical(
            [repeated_rows, repeated_ablation]
        )
        _write_jsonl(run_dir / "VISIBILITY_METRICS.jsonl", rows)
        _write_jsonl(run_dir / "VISIBILITY_ABLATION.jsonl", ablation_rows)
        base_rows = [
            row
            for row in _read_jsonl(base / "ROUTE_METRICS.jsonl")
            if row["method"] == "v6_generate_verify_bake"
        ]
        base_index = {(row["case_id"], row["deviation_id"]): row for row in base_rows}
        comparisons = []
        for row in rows:
            before = base_index[(row["case_id"], row["deviation_id"])]
            comparisons.append(
                {
                    "case_id": row["case_id"],
                    "deviation_id": row["deviation_id"],
                    "base_pass": before["route_support_pass"],
                    "filtered_pass": row["route_support_pass"],
                    "base_projected_pixels": before["projected_pixel_count"],
                    "filtered_projected_pixels": row["projected_pixel_count"],
                    "base_photo_mae": before["photo_mae"],
                    "filtered_photo_mae": row["photo_mae"],
                    "base_geometry_mre": before["geometry_mean_relative_error"],
                    "filtered_geometry_mre": row["geometry_mean_relative_error"],
                }
            )
        _write_jsonl(run_dir / "BASE_COMPARISON.jsonl", comparisons)
        base_gate = json.loads((base / "R13_GATE.json").read_text(encoding="utf-8"))
        base_summary = json.loads((base / "SUMMARY.json").read_text(encoding="utf-8"))
        base_route = float(base_gate["usable_lateral_route_m"])
        filtered_route = _common_lateral(rows, frames)
        fifth = [row for row in rows if row["deviation_id"] == "lateral_5m"]
        fifth_pass_both = len(fifth) == len(frames) and all(row["route_support_pass"] for row in fifth)
        v6_false_safe_rate = float(base_summary["baseline_safety"]["v6"]["joint_false_safe_rate"])
        wall_seconds = time.monotonic() - started
        unsupported = config["unsupported_metrics"]
        checks = {
            "target_view_observation_filter": all(
                row["target_view_observation_pixels"] > 0 for row in ablation_rows
            ),
            "route_non_regression": filtered_route
            >= float(config["metrics"]["minimum_non_regression_lateral_route_m"]),
            "fifth_meter_pass_both_frames": fifth_pass_both
            if config["metrics"]["require_fifth_meter_pass_both_frames"]
            else True,
            "repeat_exact": repeat_exact,
            "source_immutable": immutable_before == {str(path): _sha256(path) for path in frozen},
            "v6_false_safe_rate_zero": v6_false_safe_rate == 0.0,
            "unsupported_metrics_abstain": all(str(value).startswith("ABSTAIN") for value in unsupported.values()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.r13_visibility_gate.v1",
            "checks": checks,
            "base_usable_lateral_route_m": base_route,
            "filtered_usable_lateral_route_m": filtered_route,
            "fifth_meter_pass_both_frames": fifth_pass_both,
            "v6_joint_false_safe_rate": v6_false_safe_rate,
            "unsupported_metrics": unsupported,
            "decision": "accept_target_view_lidar_visibility"
            if checks["passed"]
            else "reject_target_view_lidar_visibility",
        }
        _write_json(run_dir / "R13_VISIBILITY_GATE.json", gate)
        summary = {
            "schema_version": "worldsim_v6.r13_visibility_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_target_view_lidar_visibility"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "base_usable_lateral_route_m": base_route,
            "filtered_usable_lateral_route_m": filtered_route,
            "fifth_meter_pass_both_frames": fifth_pass_both,
            "v6_joint_false_safe_rate": v6_false_safe_rate,
            "unsupported_metrics": unsupported,
            "wall_seconds": wall_seconds,
            "claim_boundary": config["claim_boundary"],
            "training_started": False,
            "confirmation_content_read": False,
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "BASE_COMPARISON.jsonl",
            "VISIBILITY_ABLATION.jsonl",
            "VISIBILITY_METRICS.jsonl",
            "R13_VISIBILITY_GATE.json",
            "SUMMARY.json",
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r13_visibility_manifest.v1",
                "source_commit": source_commit,
                "config": str(config_path),
                "files": {
                    relative: {
                        "bytes": (run_dir / relative).stat().st_size,
                        "sha256": _sha256(run_dir / relative),
                    }
                    for relative in tracked
                },
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "task_id": TASK_ID,
                "hypothesis_id": config["hypothesis_id"],
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            },
        )
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "blocked",
                "task_id": TASK_ID,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r13_visibility_filter_v0.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_dir = run_experiment(args.repo_root, args.config, args.run_root)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
