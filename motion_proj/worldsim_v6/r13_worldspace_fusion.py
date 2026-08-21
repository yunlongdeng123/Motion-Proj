"""WorldSim V6 R13 两关键帧世界点融合实验。"""

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

from motion_proj.worldsim_v6.r13_worldspace_route import (
    _metric_row,
    _project_points,
    _render_index,
    _resize_depth,
    _resize_rgb,
)


TASK_ID = "WS-V6-R13-WORLDSIM-01"


class R13FusionError(RuntimeError):
    """R13 fusion 正式合同失败。"""


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
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix):]).parts:
        raise R13FusionError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix):]).resolve()


def _voxel_union(
    point_sets: list[np.ndarray], color_sets: list[np.ndarray], voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate(point_sets, axis=0).astype(np.float32)
    colors = np.concatenate(color_sets, axis=0).astype(np.float32)
    keys = np.floor(points.astype(np.float64) / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    selected = np.sort(first)
    return points[selected], colors[selected]


def _common_lateral(rows: list[dict[str, Any]], frames: list[int]) -> float:
    passed: dict[int, set[float]] = {frame: set() for frame in frames}
    for row in rows:
        if row["route_support_pass"] and row["deviation_id"].startswith("lateral_"):
            frame = int(row["case_id"].split("__f")[1].split("__")[0])
            passed[frame].add(float(row["deviation_id"].split("_")[1].removesuffix("m")))
    common = set.intersection(*(passed[frame] for frame in frames))
    return max(common) if common else 0.0


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R13FusionError("正式 R13 fusion run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R13FusionError("R13 fusion task_id 漂移")
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
            raise R13FusionError(f"冻结输入漂移：{path}")
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R13FusionError("R13 fusion 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__worldspace-fusion-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        immutable_before = {str(path): _sha256(path) for path in frozen}
        world_rows = _read_jsonl(base / "WORLD_CHUNKS.jsonl")
        point_sets: list[np.ndarray] = []
        color_sets: list[np.ndarray] = []
        for row in world_rows:
            path = base / "world_chunks" / f"{row['case_id']}.npz"
            if _sha256(path) != row["world_chunk_sha256"]:
                raise R13FusionError(f"base world chunk 漂移：{row['case_id']}")
            with np.load(path, allow_pickle=False) as archive:
                point_sets.append(np.asarray(archive["points_world_m"], dtype=np.float32))
                color_sets.append(np.asarray(archive["rgb_float32"], dtype=np.float32))
        fused_points, fused_colors = _voxel_union(
            point_sets, color_sets, float(config["fusion"]["voxel_size_m"])
        )
        fused_path = run_dir / "FUSED_WORLD_CHUNK.npz"
        np.savez_compressed(
            fused_path,
            points_world_m=fused_points,
            rgb_float32=fused_colors,
        )
        street_index = _render_index(street_root)
        width, height = [int(value) for value in config["metrics"]["resolution_px"]]
        frames = [int(value) for value in config["cohort"]["frame_indices"]]

        def evaluate() -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for frame in frames:
                case_id = config["cohort"]["case_ids"][frame]
                with np.load(street_root / f"support_frame_{frame:03d}.npz", allow_pickle=False) as support:
                    source_height, source_width = np.asarray(support["cam0_rgb"]).shape[:2]
                    intrinsics = np.asarray(support["cam0_intrinsics"], dtype=np.float64).copy()
                    intrinsics[0] *= width / source_width
                    intrinsics[1] *= height / source_height
                    camera_to_world = np.asarray(support["cam0_camera_to_world"], dtype=np.float64)
                for deviation in config["cohort"]["route_deviations"]:
                    target_c2w = camera_to_world.copy()
                    target_c2w[:3, 3] += target_c2w[:3, 0] * float(deviation["lateral_m"])
                    target_c2w[:3, 3] += target_c2w[:3, 2] * float(deviation["forward_m"])
                    x, y, z, source_index = _project_points(
                        fused_points, target_c2w, intrinsics, width, height
                    )
                    key = (frame, deviation["variant"], float(deviation["lateral_m"]))
                    with np.load(street_index[key], allow_pickle=False) as archive:
                        target_rgb = _resize_rgb(np.asarray(archive["rgb"]), width, height)
                        target_depth = _resize_depth(np.asarray(archive["depth"]), width, height)
                    row = _metric_row(
                        "v6_temporal_voxel_union",
                        case_id,
                        deviation["id"],
                        x,
                        y,
                        z,
                        fused_colors[source_index],
                        target_rgb,
                        target_depth,
                        config["metrics"],
                    )
                    rows.append(row)
            return rows

        rows = evaluate()
        repeated = evaluate()
        repeat_exact = _canonical(rows) == _canonical(repeated)
        _write_jsonl(run_dir / "FUSION_ROUTE_METRICS.jsonl", rows)
        base_rows = [
            row
            for row in _read_jsonl(base / "ROUTE_METRICS.jsonl")
            if row["method"] == "v6_generate_verify_bake"
        ]
        base_index = {(row["case_id"], row["deviation_id"]): row for row in base_rows}
        ablation_rows = []
        for row in rows:
            before = base_index[(row["case_id"], row["deviation_id"])]
            ablation_rows.append(
                {
                    "case_id": row["case_id"],
                    "deviation_id": row["deviation_id"],
                    "base_projected_pixels": before["projected_pixel_count"],
                    "fused_projected_pixels": row["projected_pixel_count"],
                    "base_photo_mae": before["photo_mae"],
                    "fused_photo_mae": row["photo_mae"],
                    "base_geometry_mre": before["geometry_mean_relative_error"],
                    "fused_geometry_mre": row["geometry_mean_relative_error"],
                    "base_pass": before["route_support_pass"],
                    "fused_pass": row["route_support_pass"],
                }
            )
        _write_jsonl(run_dir / "FUSION_ABLATION.jsonl", ablation_rows)
        base_route = float(json.loads((base / "R13_GATE.json").read_text(encoding="utf-8"))["usable_lateral_route_m"])
        fused_route = _common_lateral(rows, frames)
        fifth = [row for row in ablation_rows if row["deviation_id"] == "lateral_5m"]
        base_fifth_geometry = float(np.mean([row["base_geometry_mre"] for row in fifth]))
        fused_fifth_geometry = float(np.mean([row["fused_geometry_mre"] for row in fifth]))
        reduction = (base_fifth_geometry - fused_fifth_geometry) / base_fifth_geometry
        fifth_pass_both = len(fifth) == 2 and all(row["fused_pass"] for row in fifth)
        wall_seconds = time.monotonic() - started
        unsupported = config["unsupported_metrics"]
        checks = {
            "fused_point_densification": int(fused_points.shape[0]) > max(value.shape[0] for value in point_sets),
            "route_non_regression": fused_route
            >= float(config["metrics"]["minimum_non_regression_lateral_route_m"]),
            "fifth_meter_geometry_reduction": reduction
            >= float(config["metrics"]["minimum_fifth_meter_geometry_error_reduction"]),
            "fifth_meter_pass_both_frames": fifth_pass_both
            if config["metrics"]["require_fifth_meter_pass_both_frames"]
            else True,
            "repeat_exact": repeat_exact,
            "source_immutable": immutable_before == {str(path): _sha256(path) for path in frozen},
            "unsupported_metrics_abstain": all(str(value).startswith("ABSTAIN") for value in unsupported.values()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.r13_fusion_gate.v1",
            "checks": checks,
            "base_usable_lateral_route_m": base_route,
            "fused_usable_lateral_route_m": fused_route,
            "fifth_meter_geometry_error_reduction": reduction,
            "unsupported_metrics": unsupported,
            "decision": "proceed_to_target_view_visibility_verification"
            if checks["passed"]
            else "reject_temporal_voxel_union",
        }
        _write_json(run_dir / "R13_FUSION_GATE.json", gate)
        summary = {
            "schema_version": "worldsim_v6.r13_fusion_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_temporal_union"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "input_point_count": sum(value.shape[0] for value in point_sets),
            "fused_point_count": int(fused_points.shape[0]),
            "base_usable_lateral_route_m": base_route,
            "fused_usable_lateral_route_m": fused_route,
            "base_fifth_meter_geometry_mre": base_fifth_geometry,
            "fused_fifth_meter_geometry_mre": fused_fifth_geometry,
            "fifth_meter_geometry_error_reduction": reduction,
            "unsupported_metrics": unsupported,
            "wall_seconds": wall_seconds,
            "claim_boundary": config["claim_boundary"],
            "training_started": False,
            "confirmation_content_read": False,
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "FUSED_WORLD_CHUNK.npz",
            "FUSION_ABLATION.jsonl",
            "FUSION_ROUTE_METRICS.jsonl",
            "R13_FUSION_GATE.json",
            "SUMMARY.json",
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r13_fusion_manifest.v1",
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
        default=Path("configs/worldsim_v6/r13_worldspace_fusion_v0.yaml"),
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
