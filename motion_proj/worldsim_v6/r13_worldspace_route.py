"""WorldSim V6 R13 静态 chunk 世界坐标提升与路线偏离实验。"""

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
from PIL import Image


TASK_ID = "WS-V6-R13-WORLDSIM-01"


class R13ExperimentError(RuntimeError):
    """R13 正式合同失败。"""


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
        raise R13ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix):]).resolve()


def _resize_rgb(value: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = np.clip(value.astype(np.float32), 0.0, 1.0)
    image = Image.fromarray(np.rint(clipped * 255.0).astype(np.uint8), mode="RGB")
    return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def _resize_depth(value: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(value.astype(np.float32), mode="F")
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST), dtype=np.float32)


def _render_index(root: Path) -> dict[tuple[int, str, float], Path]:
    result: dict[tuple[int, str, float], Path] = {}
    for row in _read_jsonl(root / "RENDER_MAP.jsonl"):
        path = root / row["path"]
        if _sha256(path) != row["sha256"]:
            raise R13ExperimentError(f"R3 render 漂移：{path}")
        result[(int(row["frame_index"]), row["variant"], float(row["lateral_offset_m"]))] = path
    return result


def _lift_points(
    coordinates: np.ndarray,
    depth: np.ndarray,
    valid_depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = coordinates[:, 0].astype(np.int64)
    x = coordinates[:, 1].astype(np.int64)
    keep = valid_depth[y, x] & np.isfinite(depth[y, x]) & (depth[y, x] > 1.0e-6)
    selected = coordinates[keep]
    z = depth[selected[:, 0], selected[:, 1]].astype(np.float64)
    pixels = np.stack(
        [selected[:, 1], selected[:, 0], np.ones(selected.shape[0], dtype=np.int64)], axis=0
    ).astype(np.float64)
    camera = (np.linalg.inv(intrinsics) @ pixels) * z[None]
    world = camera_to_world[:3, :3] @ camera + camera_to_world[:3, 3:4]
    return world.T.astype(np.float32), keep


def _project_points(
    points_world: np.ndarray,
    target_camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(target_camera_to_world)
    camera = world_to_camera[:3, :3] @ points_world.astype(np.float64).T + world_to_camera[:3, 3:4]
    z = camera[2]
    uvw = intrinsics @ camera
    x = np.rint(uvw[0] / np.maximum(uvw[2], 1.0e-8)).astype(np.int64)
    y = np.rint(uvw[1] / np.maximum(uvw[2], 1.0e-8)).astype(np.int64)
    keep = (z > 1.0e-6) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    source_index = np.nonzero(keep)[0]
    x = x[keep]
    y = y[keep]
    z = z[keep]
    if source_index.size == 0:
        return x, y, z.astype(np.float32), source_index
    linear = y * width + x
    order = np.lexsort((z, linear))
    ordered_linear = linear[order]
    first = np.r_[True, ordered_linear[1:] != ordered_linear[:-1]]
    chosen = order[first]
    return x[chosen], y[chosen], z[chosen].astype(np.float32), source_index[chosen]


def _metric_row(
    method: str,
    case_id: str,
    deviation_id: str,
    x: np.ndarray,
    y: np.ndarray,
    projected_z: np.ndarray,
    colors: np.ndarray,
    target_rgb: np.ndarray,
    target_depth: np.ndarray,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    if x.size == 0:
        photo_mae = None
        geometry_mre = None
        geometry_pixels = 0
    else:
        photo_mae = float(np.abs(colors.astype(np.float32) - target_rgb[y, x]).mean())
        target_z = target_depth[y, x]
        valid = np.isfinite(target_z) & (target_z > 1.0e-6) & np.isfinite(projected_z)
        geometry_pixels = int(np.count_nonzero(valid))
        geometry_mre = (
            float(
                np.mean(
                    np.abs(projected_z[valid] - target_z[valid])
                    / np.maximum(np.abs(target_z[valid]), 1.0e-6)
                )
            )
            if geometry_pixels
            else None
        )
    passed = (
        x.size >= int(thresholds["minimum_projected_pixels"])
        and photo_mae is not None
        and photo_mae <= float(thresholds["maximum_photo_mae"])
        and geometry_mre is not None
        and geometry_mre <= float(thresholds["maximum_geometry_mean_relative_error"])
    )
    return {
        "schema_version": "worldsim_v6.r13_route_metric.v1",
        "method": method,
        "case_id": case_id,
        "deviation_id": deviation_id,
        "projected_pixel_count": int(x.size),
        "geometry_pixel_count": geometry_pixels,
        "photo_mae": photo_mae,
        "geometry_mean_relative_error": geometry_mre,
        "route_support_pass": bool(passed),
    }


def _joint_false_safe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    false_count = sum(not (row["P1"]["truth_safe"] and row["P2"]["truth_safe"]) for row in rows)
    return {
        "accepted_count": len(rows),
        "false_safe_count": false_count,
        "false_safe_rate": false_count / len(rows),
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R13ExperimentError("正式 R13 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R13ExperimentError("R13 task_id 漂移")
    sources = config["sources"]
    r11 = _resolve_runs_uri(sources["r11_run"])
    cross = _resolve_runs_uri(sources["r9_cross_run"])
    big = _resolve_runs_uri(sources["r9_big_lama_run"])
    r10 = _resolve_runs_uri(sources["r10_run"])
    r3 = _resolve_runs_uri(sources["r3_render_run"])
    ad_root = r3 / "renders/scene-0242/ad_gs"
    street_root = r3 / "renders/scene-0242/streetgs"
    frozen = {
        r11 / "MANIFEST.json": sources["r11_manifest_sha256"],
        r11 / "package/PACKAGE_MANIFEST.json": sources["r11_package_manifest_sha256"],
        cross / "MANIFEST.json": sources["r9_cross_manifest_sha256"],
        big / "MANIFEST.json": sources["r9_big_lama_manifest_sha256"],
        big / "verifier_worker/PER_CASE_ARMS.jsonl": sources["r9_big_lama_arms_sha256"],
        r10 / "MANIFEST.json": sources["r10_manifest_sha256"],
        ad_root / "RENDER_MAP.jsonl": sources["ad_gs_render_map_sha256"],
        street_root / "RENDER_MAP.jsonl": sources["streetgs_render_map_sha256"],
    }
    for frame, digest in sources["support_frame_sha256"].items():
        frozen[street_root / f"support_frame_{int(frame):03d}.npz"] = digest
    for path, expected in frozen.items():
        if _sha256(path) != expected:
            raise R13ExperimentError(f"冻结输入漂移：{path}")
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R13ExperimentError("R13 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__worldspace-route-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        world_dir = run_dir / "world_chunks"
        world_dir.mkdir()
        immutable_before = {str(path): _sha256(path) for path in frozen}
        ad_index = _render_index(ad_root)
        street_index = _render_index(street_root)
        assets = {row["case_id"]: row for row in _read_jsonl(r11 / "package/ASSET_REGISTRY.jsonl")}
        provenance = {row["asset_id"]: row for row in _read_jsonl(r11 / "package/PROVENANCE.jsonl")}
        big_result = json.loads((big / "big_lama_proposals/WORKER_RESULT.json").read_text(encoding="utf-8"))
        big_records = {
            row["case_id"]: row["repeats"][0] for row in big_result["cases"]
        }
        width, height = [int(value) for value in config["world_lift"]["resolution_px"]]
        case_data: dict[str, dict[str, Any]] = {}
        world_rows: list[dict[str, Any]] = []
        for case_id in config["cohort"]["case_ids"]:
            asset = assets[case_id]
            frame = int(case_id.split("__f")[1].split("__")[0])
            payload = r11 / "package" / asset["payload"]
            if _sha256(payload) != asset["payload_sha256"]:
                raise R13ExperimentError(f"R11 payload 漂移：{case_id}")
            with np.load(payload, allow_pickle=False) as archive:
                coordinates = np.asarray(archive["coordinates_yx"], dtype=np.int64)
                v6_rgb = np.asarray(archive["rgb_uint8"], dtype=np.uint8).astype(np.float32) / 255.0
            verifier_input = cross / "verifier_inputs" / f"{case_id}.npz"
            with np.load(verifier_input, allow_pickle=False) as archive:
                depth = np.asarray(archive["target_depth"], dtype=np.float32)
                depth_valid = np.asarray(archive["target_depth_valid"], dtype=bool)
            with np.load(street_root / f"support_frame_{frame:03d}.npz", allow_pickle=False) as support:
                base_height, base_width = np.asarray(support["cam0_rgb"]).shape[:2]
                intrinsics = np.asarray(support["cam0_intrinsics"], dtype=np.float64).copy()
                intrinsics[0] *= width / base_width
                intrinsics[1] *= height / base_height
                camera_to_world = np.asarray(support["cam0_camera_to_world"], dtype=np.float64)
            points_world, keep = _lift_points(
                coordinates, depth, depth_valid, intrinsics, camera_to_world
            )
            v6_rgb = v6_rgb[keep]
            big_path = big / "big_lama_proposals" / big_records[case_id]["output"]
            if _sha256(big_path) != big_records[case_id]["output_sha256"]:
                raise R13ExperimentError(f"Big-LaMa proposal 漂移：{case_id}")
            big_image = np.load(big_path, allow_pickle=False).astype(np.float32) / 255.0
            big_rgb = big_image[coordinates[keep, 0], coordinates[keep, 1]]
            source_proposal = cross / "cross_frontend_reconstruction_proposals" / f"{case_id}__repeat1.npy"
            expected_source_sha = provenance[asset["asset_id"]]["source_proposal_sha256"]
            if _sha256(source_proposal) != expected_source_sha:
                raise R13ExperimentError(f"cross proposal 漂移：{case_id}")
            naive_image = np.load(source_proposal, allow_pickle=False).astype(np.float32) / 255.0
            naive_rgb = naive_image[coordinates[keep, 0], coordinates[keep, 1]]
            if not np.array_equal(np.rint(naive_rgb * 255).astype(np.uint8), np.rint(v6_rgb * 255).astype(np.uint8)):
                raise R13ExperimentError(f"R11 与 cross proposal 不一致：{case_id}")
            z = depth[coordinates[keep, 0], coordinates[keep, 1]].astype(np.float64)
            area_m2 = float(np.sum((z * z) / (intrinsics[0, 0] * intrinsics[1, 1])))
            world_path = world_dir / f"{case_id}.npz"
            np.savez_compressed(
                world_path,
                points_world_m=points_world,
                rgb_float32=v6_rgb.astype(np.float32),
            )
            world_rows.append(
                {
                    "case_id": case_id,
                    "frame_index": frame,
                    "point_count": int(points_world.shape[0]),
                    "estimated_surface_area_m2": area_m2,
                    "world_chunk_sha256": _sha256(world_path),
                }
            )
            case_data[case_id] = {
                "frame": frame,
                "points": points_world,
                "v6_rgb": v6_rgb,
                "naive_rgb": naive_rgb,
                "big_rgb": big_rgb,
                "intrinsics": intrinsics,
                "camera_to_world": camera_to_world,
            }

        def evaluate() -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for case_id, data in case_data.items():
                frame = data["frame"]
                for deviation in config["cohort"]["route_deviations"]:
                    target_c2w = data["camera_to_world"].copy()
                    target_c2w[:3, 3] += target_c2w[:3, 0] * float(deviation["lateral_m"])
                    target_c2w[:3, 3] += target_c2w[:3, 2] * float(deviation["forward_m"])
                    x, y, projected_z, source_index = _project_points(
                        data["points"], target_c2w, data["intrinsics"], width, height
                    )
                    key = (frame, deviation["variant"], float(deviation["lateral_m"]))
                    with np.load(street_index[key], allow_pickle=False) as archive:
                        target_rgb = _resize_rgb(np.asarray(archive["rgb"]), width, height)
                        target_depth = _resize_depth(np.asarray(archive["depth"]), width, height)
                    with np.load(ad_index[key], allow_pickle=False) as archive:
                        native_rgb = _resize_rgb(np.asarray(archive["rgb"]), width, height)
                        native_depth = _resize_depth(np.asarray(archive["depth"]), width, height)
                    rows.append(
                        _metric_row(
                            "native_reconstruction_only",
                            case_id,
                            deviation["id"],
                            x,
                            y,
                            native_depth[y, x],
                            native_rgb[y, x],
                            target_rgb,
                            target_depth,
                            config["metrics"],
                        )
                    )
                    for method, colors in [
                        ("generator_only_big_lama", data["big_rgb"]),
                        ("reconstruction_plus_naive_generation", data["naive_rgb"]),
                        ("v6_generate_verify_bake", data["v6_rgb"]),
                    ]:
                        rows.append(
                            _metric_row(
                                method,
                                case_id,
                                deviation["id"],
                                x,
                                y,
                                projected_z,
                                colors[source_index],
                                target_rgb,
                                target_depth,
                                config["metrics"],
                            )
                        )
            return rows

        metric_rows = evaluate()
        repeat_rows = evaluate()
        repeat_exact = _canonical(metric_rows) == _canonical(repeat_rows)
        _write_jsonl(run_dir / "ROUTE_METRICS.jsonl", metric_rows)
        _write_jsonl(run_dir / "WORLD_CHUNKS.jsonl", world_rows)
        big_safety = _joint_false_safe(
            _read_jsonl(big / "verifier_worker/PER_CASE_ARMS.jsonl")
        )
        naive_safety = _joint_false_safe(
            _read_jsonl(cross / "verifier_worker/PER_CASE_ARMS.jsonl")
        )
        decisions = _read_jsonl(r10 / "FACTORIZED_DECISIONS.jsonl")
        v6_accepted = [row for row in decisions if row["overall_decision"] == "ACCEPT"]
        v6_false = sum(bool(row["false_safe"]) for row in v6_accepted)
        safety = {
            "native_reconstruction_only": {"status": "ABSTAIN_NOT_A_PROPOSAL_ACCEPTANCE_POLICY"},
            "generator_only_big_lama": big_safety,
            "reconstruction_plus_naive_generation": naive_safety,
            "v6_generate_verify_bake": {
                "accepted_count": len(v6_accepted),
                "false_safe_count": v6_false,
                "false_safe_rate": v6_false / len(v6_accepted),
            },
        }
        _write_json(run_dir / "BASELINE_SAFETY.json", safety)
        v6_rows = [row for row in metric_rows if row["method"] == "v6_generate_verify_bake"]
        passing = [row for row in v6_rows if row["route_support_pass"]]
        lateral_pass_by_frame: dict[int, set[float]] = {}
        for row in passing:
            if row["deviation_id"].startswith("lateral_"):
                frame = int(row["case_id"].split("__f")[1].split("__")[0])
                value = float(row["deviation_id"].split("_")[1].removesuffix("m"))
                lateral_pass_by_frame.setdefault(frame, set()).add(value)
        common_lateral = set.intersection(*lateral_pass_by_frame.values()) if len(lateral_pass_by_frame) == 2 else set()
        usable_lateral_m = max(common_lateral) if common_lateral else 0.0
        unsupported = config["unsupported_metrics"]
        wall_seconds = time.monotonic() - started
        checks = {
            "all_four_methods": {row["method"] for row in metric_rows} == set(config["methods"]),
            "expected_case_deviation_count": len(v6_rows)
            == int(config["cohort"]["expected_case_deviation_count"]),
            "world_space_chunks": len(world_rows) == 2
            and all(row["point_count"] >= int(config["metrics"]["minimum_projected_pixels"]) for row in world_rows),
            "minimum_passing_case_deviations": len(passing)
            >= int(config["metrics"]["minimum_v6_passing_case_deviations"]),
            "minimum_usable_lateral_route": usable_lateral_m
            >= float(config["metrics"]["minimum_usable_lateral_route_m"]),
            "v6_false_safe_rate": safety["v6_generate_verify_bake"]["false_safe_rate"]
            <= float(config["metrics"]["maximum_v6_false_safe_rate"]),
            "false_safe_reduction_vs_naive": (
                safety["reconstruction_plus_naive_generation"]["false_safe_rate"]
                - safety["v6_generate_verify_bake"]["false_safe_rate"]
            )
            >= float(config["metrics"]["minimum_false_safe_reduction_vs_naive"]),
            "repeat_exact": repeat_exact,
            "source_immutable": immutable_before == {str(path): _sha256(path) for path in frozen},
            "unsupported_metrics_abstain": all(str(value).startswith("ABSTAIN") for value in unsupported.values()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(
            run_dir / "R13_GATE.json",
            {
                "schema_version": "worldsim_v6.r13_gate.v1",
                "checks": checks,
                "usable_lateral_route_m": usable_lateral_m,
                "v6_passing_case_deviation_count": len(passing),
                "unsupported_metrics": unsupported,
                "decision": "proceed_to_worldspace_chunk_densification"
                if checks["passed"]
                else "reject_point_lifted_worldsim_route",
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r13_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_point_lift_scope"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "world_chunk_count": len(world_rows),
            "world_point_count": sum(row["point_count"] for row in world_rows),
            "estimated_world_area_m2": sum(row["estimated_surface_area_m2"] for row in world_rows),
            "usable_lateral_route_m": usable_lateral_m,
            "v6_passing_case_deviation_count": len(passing),
            "baseline_safety": safety,
            "unsupported_metrics": unsupported,
            "wall_seconds": wall_seconds,
            "claim_boundary": config["claim_boundary"],
            "training_started": False,
            "confirmation_content_read": False,
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "BASELINE_SAFETY.json",
            "R13_GATE.json",
            "ROUTE_METRICS.jsonl",
            "SUMMARY.json",
            "WORLD_CHUNKS.jsonl",
        ] + [str(path.relative_to(run_dir)) for path in sorted(world_dir.glob("*.npz"))]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r13_manifest.v1",
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
        default=Path("configs/worldsim_v6/r13_worldspace_route_v0.yaml"),
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
