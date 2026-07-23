#!/usr/bin/env python
"""从冻结 registry/evidence/support 与 S0 pose 生成正式 WorldStateSequence。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.resim.coordinates import drivestudio_intrinsics
from motion_proj.resim.schema import world_state_hash, world_state_payload
from motion_proj.runtime.atomic import atomic_write_json
from motion_proj.runtime.fingerprint import file_fingerprint


def _camera_models(processed: Path, frame: int, cameras: list[dict]) -> list[dict]:
    output = []
    for camera in cameras:
        index = int(camera["dataset_index"])
        T_world_camera = np.loadtxt(
            processed / "extrinsics" / f"{frame:03d}_{index}.txt"
        ).reshape(4, 4)
        raw_k = np.loadtxt(
            processed / "intrinsics" / f"{index}.txt"
        )
        output.append(
            {
                "camera_id": camera["camera_id"],
                "dataset_index": index,
                "T_world_camera": T_world_camera.tolist(),
                "intrinsics": drivestudio_intrinsics(raw_k).tolist(),
                "intrinsics_raw": np.asarray(raw_k).reshape(-1).tolist(),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/resim/v71/render_and_label_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    world_config = yaml.safe_load(
        Path(config["world_config"]).read_text(encoding="utf-8")
    )
    world_scene = {value["scene_id"]: value for value in world_config["scenes"]}
    output_root = args.output_root or Path(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"world-state output 已存在，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True)
    evidence = json.loads(
        (Path(config["evidence_root"]) / "summary.json").read_text(encoding="utf-8")
    )
    support = json.loads(
        (Path(config["support_root"]) / "summary.json").read_text(encoding="utf-8")
    )
    summary = {"schema_version": 1, "task_id": "V7-H1-11C", "sequences": {}}
    for scene in config["scenes"]:
        scene_id = scene["scene_id"]
        edit = json.loads(Path(scene["edit_json"]).read_text(encoding="utf-8"))
        registry = json.loads(
            (
                Path(config["registry_root"]) / scene_id / "actor_registry.json"
            ).read_text(encoding="utf-8")
        )
        if int(edit["actor_id"]) != int(scene["actor_id"]):
            raise RuntimeError("edit actor 与冻结 config 不一致")
        processed = Path(world_config["processed_root"]) / scene_id
        for variant_name in config["variants"]:
            variant = edit["variants"][variant_name]
            if not variant.get("accepted", True):
                raise RuntimeError(f"{scene_id}/{variant_name} 未被 source editor 接受")
            frames = []
            for value in variant["frames"]:
                frame = int(value["frame"])
                T_world_lidar = np.loadtxt(
                    processed / "lidar_pose" / f"{frame:03d}.txt"
                ).reshape(4, 4)
                frames.append(
                    {
                        "frame_index": frame,
                        "timestamp": frame / 10.0,
                        "timestamp_provenance": "observed" if frame % 5 == 0 else "interpolated",
                        "T_world_ego": T_world_lidar.tolist(),
                        "ego_pose_source": "T_world_lidar_sensor_proxy",
                        "camera_models": _camera_models(processed, frame, config["cameras"]),
                        "actor_nodes": [
                            {
                                "true_instance_id": int(scene["actor_id"]),
                                "occupancy_instance_id": int(scene["actor_id"]) + 1,
                                "T_world_actor": value["obj_to_world"],
                                "dimensions_lwh": value["box_size"],
                                "state_source": f"S0_{variant_name}",
                            }
                        ],
                    }
                )
            payload = {
                "sequence_id": f"{scene_id}:{scene['actor_id']}:{variant_name}",
                "scene_id": scene_id,
                "split": config["split"],
                "data_fingerprint": canonical_sha256(
                    {
                        "edit": file_fingerprint(scene["edit_json"]),
                        "registry": registry["actor_registry_sha256"],
                        "evidence": evidence["scenes"][scene_id]["scene_evidence_sha256"],
                        "support": support["scenes"][scene_id]["scene_support_sha256"],
                    }
                ),
                "coordinate_convention": "T_dst_src-v1",
                "timebase_hz": 10,
                "reconstruction": {
                    "checkpoint": world_scene[scene_id]["checkpoint"],
                    "checkpoint_sha256": registry["checkpoint_sha256"],
                },
                "actor_registry_sha256": registry["actor_registry_sha256"],
                "safety_geometry_sha256": canonical_sha256(
                    {"source": "raw_annotations", "scene_id": scene_id}
                ),
                "observation_evidence_sha256": evidence["scenes"][scene_id]["scene_evidence_sha256"],
                "render_support_sha256": support["scenes"][scene_id]["scene_support_sha256"],
                "edit_spec": {
                    "variant": variant_name,
                    "actor_id": int(scene["actor_id"]),
                    "source_edit_sha256": file_fingerprint(scene["edit_json"]),
                    "peak_abs_dy": variant.get("peak_abs_dy"),
                },
                "frames": frames,
            }
            normalized = world_state_payload(payload)
            digest = world_state_hash(payload)
            target = output_root / scene_id / variant_name
            target.mkdir(parents=True)
            atomic_write_json(
                str(target / "world_state.json"),
                {**normalized, "world_state_hash": digest},
            )
            summary["sequences"][f"{scene_id}:{variant_name}"] = {
                "world_state_hash": digest,
                "frame_count": len(frames),
                "actor_id": int(scene["actor_id"]),
            }
    summary["world_state_set_sha256"] = canonical_sha256(summary["sequences"])
    atomic_write_json(str(output_root / "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
