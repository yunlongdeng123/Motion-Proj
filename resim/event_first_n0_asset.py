#!/usr/bin/env python
"""固化 event-first 路线的 nuScenes map-expansion 资产与坐标合同。"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import yaml
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from motion_proj.runtime.fingerprint import file_fingerprint, git_state
from motion_proj.runtime.v71_contract import generate_run_id, utc_now


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是 YAML object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected_sha256: str, expected_size: int | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"文件大小不符: {path}: {size} != {expected_size}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA256 不符: {path}: {actual} != {expected_sha256}")
    return {"path": str(path), "size_bytes": size, "sha256": actual}


def _transform(translation: list[float], rotation: list[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def _rotation_error(left: np.ndarray, right: np.ndarray) -> float:
    delta = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


def _scene_samples(nusc: NuScenes, scene: dict) -> list[dict]:
    samples = []
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        samples.append(sample)
        token = sample["next"]
    return samples


def _raw_lidar_pose(nusc: NuScenes, sample: dict) -> np.ndarray:
    sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    calibrated = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sample_data["ego_pose_token"])
    return _transform(ego["translation"], ego["rotation"]) @ _transform(
        calibrated["translation"], calibrated["rotation"]
    )


def _actor_positions(processed_root: Path, scene_id: str, actor_ids: list[int]) -> list[dict]:
    path = processed_root / scene_id / "instances" / "instances_info.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for actor_id in actor_ids:
        actor = raw[str(actor_id)]
        values = actor["frame_annotations"]
        for frame, transform in zip(values["frame_idx"], values["obj_to_world"]):
            matrix = np.asarray(transform, dtype=float)
            output.append(
                {
                    "actor_id": int(actor_id),
                    "frame_index": int(frame),
                    "xy": [float(matrix[0, 3]), float(matrix[1, 3])],
                }
            )
    return output


def _lane_audit(nmap: NuScenesMap, xy_rows: list[dict], radius_m: float) -> dict:
    matched = 0
    layer_hits = 0
    examples = []
    for row in xy_rows:
        x, y = row["xy"]
        lane_token = nmap.get_closest_lane(x, y, radius=radius_m)
        layers = nmap.get_records_in_radius(
            x, y, radius=0.1, layer_names=["drivable_area", "lane", "lane_connector"]
        )
        matched += int(bool(lane_token))
        layer_hits += int(any(layers.values()))
        if len(examples) < 5:
            examples.append({**row, "closest_lane_token": lane_token, "layers": layers})
    count = len(xy_rows)
    return {
        "query_count": count,
        "closest_lane_match_count": matched,
        "closest_lane_match_fraction": matched / count if count else None,
        "drivable_or_lane_point_hit_count": layer_hits,
        "drivable_or_lane_point_hit_fraction": layer_hits / count if count else None,
        "examples": examples,
    }


def run(config_path: Path, output_root: Path | None) -> Path:
    config = _load_yaml(config_path)
    config_sha = file_fingerprint(str(config_path))
    dataset_root = Path(config["dataset_root"]).resolve()
    processed_root = Path(config["processed_root"]).resolve()
    archive_cfg = config["archive"]
    archive = _verify_file(
        Path(archive_cfg["path"]).resolve(),
        archive_cfg["sha256"],
        int(archive_cfg["size_bytes"]),
    )

    files = {"archive": archive, "maps": {}, "prediction": None, "license": None}
    for map_name, record in config["maps"]["files"].items():
        files["maps"][map_name] = _verify_file(
            dataset_root / record["path"], record["sha256"]
        )
    files["prediction"] = _verify_file(
        dataset_root / config["prediction"]["path"], config["prediction"]["sha256"]
    )
    files["license"] = _verify_file(
        dataset_root / config["license"]["path"], config["license"]["sha256"]
    )

    maps = {}
    required_layers = tuple(config["maps"]["required_layers"])
    for map_name in sorted(config["maps"]["files"]):
        nmap = NuScenesMap(dataroot=str(dataset_root), map_name=map_name)
        missing_layers = [name for name in required_layers if not hasattr(nmap, name)]
        if missing_layers:
            raise RuntimeError(f"{map_name} 缺少层: {missing_layers}")
        if str(nmap.version) != str(config["gates"]["map_version_exact"]):
            raise RuntimeError(f"{map_name} version={nmap.version}")
        maps[map_name] = {
            "version": str(nmap.version),
            "layer_counts": {
                name: len(getattr(nmap, name))
                for name in required_layers
                if isinstance(getattr(nmap, name), (list, dict))
            },
            "api": nmap,
        }

    nusc = NuScenes(version="v1.0-mini", dataroot=str(dataset_root), verbose=False)
    scene_registry = []
    max_translation = 0.0
    max_rotation = 0.0
    radius = float(config["gates"]["closest_lane_radius_m"])
    for expected in config["scenes"]:
        index = int(expected["nuscenes_scene_index"])
        scene = nusc.scene[index]
        exact = {
            "scene_name": scene["name"] == expected["scene_name"],
            "scene_token": scene["token"] == expected["scene_token"],
            "sample_count": int(scene["nbr_samples"]) == int(expected["sample_count"]),
        }
        log = nusc.get("log", scene["log_token"])
        exact["map_name"] = log["location"] == expected["map_name"]
        if not all(exact.values()):
            raise RuntimeError(f"scene 映射不符: {expected['processed_scene_id']}: {exact}")

        ego_rows = []
        pose_rows = []
        for sample_index, sample in enumerate(_scene_samples(nusc, scene)):
            raw_pose = _raw_lidar_pose(nusc, sample)
            processed_frame = sample_index * 5
            processed_pose = np.loadtxt(
                processed_root
                / expected["processed_scene_id"]
                / "lidar_pose"
                / f"{processed_frame:03d}.txt"
            ).reshape(4, 4)
            translation_error = float(
                np.linalg.norm(raw_pose[:3, 3] - processed_pose[:3, 3])
            )
            rotation_error = _rotation_error(raw_pose, processed_pose)
            max_translation = max(max_translation, translation_error)
            max_rotation = max(max_rotation, rotation_error)
            ego_rows.append(
                {
                    "frame_index": processed_frame,
                    "xy": [float(processed_pose[0, 3]), float(processed_pose[1, 3])],
                }
            )
            pose_rows.append(
                {
                    "frame_index": processed_frame,
                    "translation_error_m": translation_error,
                    "rotation_error_rad": rotation_error,
                }
            )
        nmap = maps[expected["map_name"]]["api"]
        actor_rows = _actor_positions(
            processed_root,
            expected["processed_scene_id"],
            [int(value) for value in expected["selected_actor_ids"]],
        )
        scene_registry.append(
            {
                **expected,
                "log_token": scene["log_token"],
                "logfile": log["logfile"],
                "mapping_checks": exact,
                "pose_roundtrip": {
                    "rows": pose_rows,
                    "max_translation_error_m": max(
                        row["translation_error_m"] for row in pose_rows
                    ),
                    "max_rotation_error_rad": max(
                        row["rotation_error_rad"] for row in pose_rows
                    ),
                },
                "ego_lane_audit": _lane_audit(nmap, ego_rows, radius),
                "selected_actor_lane_audit": _lane_audit(nmap, actor_rows, radius),
            }
        )

    translation_tolerance = float(config["gates"]["pose_translation_tolerance_m"])
    rotation_tolerance = float(config["gates"]["pose_rotation_tolerance_rad"])
    if max_translation > translation_tolerance or max_rotation > rotation_tolerance:
        raise RuntimeError(
            "raw/processed pose round-trip 失败: "
            f"{max_translation} m, {max_rotation} rad"
        )

    code = git_state(str(Path(config["repo_root"])))
    run_id = generate_run_id(config["task_id"], "map-v1.3", int(config["seed"]), config_sha)
    run_root = output_root or Path(config["run_root"])
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    maps_serializable = {
        name: {key: value for key, value in record.items() if key != "api"}
        for name, record in maps.items()
    }
    asset_manifest = {
        "schema_version": config["schema_version"],
        "task_id": config["task_id"],
        "source": archive_cfg["source"],
        "config_sha256": config_sha,
        "files": files,
        "maps": maps_serializable,
    }
    registry = {
        "schema_version": "scene-map-registry-v1",
        "coordinate_contract": "nuScenes global world XY equals map global XY; no transform applied",
        "scenes": scene_registry,
    }
    asset_manifest["asset_manifest_sha256"] = canonical_sha256(asset_manifest)
    registry["scene_map_registry_sha256"] = canonical_sha256(registry)
    summary = {
        "task_id": config["task_id"],
        "run_id": run_id,
        "split": config["split"],
        "seed": int(config["seed"]),
        "terminal_status": "COMPLETE",
        "research_verdict": "n0_asset_pass",
        "map_count": len(maps_serializable),
        "scene_count": len(scene_registry),
        "max_pose_translation_error_m": max_translation,
        "max_pose_rotation_error_rad": max_rotation,
        "asset_manifest_sha256": asset_manifest["asset_manifest_sha256"],
        "scene_map_registry_sha256": registry["scene_map_registry_sha256"],
    }
    manifest = {
        "schema_version": 1,
        "task_id": config["task_id"],
        "run_id": run_id,
        "command": list(sys.argv),
        "code_commit": code["commit"],
        "code_dirty": code["dirty"],
        "dirty_diff_hash": code["dirty_diff_hash"],
        "config_fingerprint": config_sha,
        "data_fingerprint": canonical_sha256(files),
        "split": config["split"],
        "seed": int(config["seed"]),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "started_at": started_at,
        "ended_at": utc_now(),
        "terminal_status": "COMPLETE",
        "exit_reason": "n0_asset_gates_passed",
    }
    atomic_write_text(str(run_dir / "resolved.yaml"), config_path.read_text(encoding="utf-8"))
    atomic_write_json(str(run_dir / "manifest.json"), manifest)
    atomic_write_json(str(run_dir / "asset_manifest.json"), asset_manifest)
    atomic_write_json(str(run_dir / "scene_map_registry.json"), registry)
    atomic_write_text(
        str(run_dir / "metrics.jsonl"),
        json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n",
    )
    atomic_write_json(str(run_dir / "summary.json"), summary)
    atomic_write_text(str(run_dir / "COMPLETE"), "n0_asset_gates_passed\n")
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resim/event_first_n0_asset_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    run(args.config, args.output_root)


if __name__ == "__main__":
    main()
