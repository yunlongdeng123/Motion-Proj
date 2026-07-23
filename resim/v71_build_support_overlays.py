#!/usr/bin/env python
"""构建 threshold-free render support，并分别输出三类机器诊断 overlay。"""
from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.resim.render_support import RenderSupportFrame
from motion_proj.runtime.atomic import atomic_write_json
from motion_proj.runtime.fingerprint import file_fingerprint


COLORS = np.asarray(
    [
        [220, 30, 30], [30, 180, 60], [30, 90, 220], [230, 160, 20],
        [170, 50, 210], [20, 190, 190], [240, 90, 140], [120, 120, 30],
    ],
    dtype=np.uint8,
)


def _save_rgb(path: Path, value: np.ndarray) -> None:
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB").save(path)


def _decode_layers(frame: np.lib.npyio.NpzFile) -> dict[int, np.ndarray]:
    shape = frame["base_state"].shape
    ids = frame["dynamic_actor_ids"]
    offsets = frame["dynamic_layer_offsets"]
    flat = frame["dynamic_layer_flat_indices"]
    layers = {}
    for index, actor_id in enumerate(ids):
        mask = np.zeros(np.prod(shape), dtype=bool)
        mask[flat[offsets[index] : offsets[index + 1]]] = True
        layers[int(actor_id)] = mask.reshape(shape)
    return layers


def _safety_overlay(layers: dict[int, np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for index, (actor_id, mask) in enumerate(sorted(layers.items())):
        footprint = np.any(mask, axis=2)
        image[footprint] = COLORS[index % len(COLORS)]
    return np.flipud(np.swapaxes(image, 0, 1))


def _evidence_overlay(base: np.ndarray, layers: dict[int, np.ndarray]) -> np.ndarray:
    # max-z projection：unknown 黑、ray-free 蓝、static 灰、dynamic 黄。
    top = np.max(base, axis=2)
    image = np.zeros((*top.shape, 3), dtype=np.uint8)
    image[top == 1] = [40, 90, 200]
    image[top == 2] = [170, 170, 170]
    dynamic = np.zeros(top.shape, dtype=bool)
    for mask in layers.values():
        dynamic |= np.any(mask, axis=2)
    image[dynamic] = [245, 200, 30]
    return np.flipud(np.swapaxes(image, 0, 1))


def _render_support_overlay(active: np.ndarray) -> np.ndarray:
    # time × actor 的 raw instances_fv，不施加 alpha/visibility 阈值。
    value = np.asarray(active > 0, dtype=np.uint8) * 255
    image = np.stack([value // 3, value, value // 2], axis=-1)
    return np.repeat(np.repeat(image, 8, axis=0), 12, axis=1)


def _tensor_hash(value: np.ndarray) -> str:
    digest = sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world-config", type=Path,
        default=Path("configs/resim/v71/world_state_v1.yaml"),
    )
    parser.add_argument(
        "--render-config", type=Path,
        default=Path("configs/resim/v71/render_support_v1.yaml"),
    )
    parser.add_argument(
        "--evidence-root", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11B/evidence_v2"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11B/support_overlays"),
    )
    parser.add_argument("--overlay-frame", type=int, default=40)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"support output 已存在，拒绝覆盖: {args.output_root}")
    args.output_root.mkdir(parents=True)
    world = yaml.safe_load(args.world_config.read_text(encoding="utf-8"))
    render_config_hash = file_fingerprint(str(args.render_config))
    scenes = {}
    for scene in world["scenes"]:
        scene_id = scene["scene_id"]
        scene_out = args.output_root / scene_id
        scene_out.mkdir()
        checkpoint = torch.load(scene["checkpoint"], map_location="cpu")
        models = checkpoint["models"]
        background = models["Background"]["_means"]
        rigid = models["RigidNodes"]["_means"]
        active = models["RigidNodes"]["instances_fv"].cpu().numpy()
        point_ids = models["RigidNodes"]["points_ids"].cpu().numpy().reshape(-1)
        actor_counts = {
            str(index): int(np.count_nonzero(point_ids == index))
            for index in range(active.shape[1])
        }
        np.savez_compressed(
            scene_out / "render_support_raw.npz",
            instances_fv=active,
            rigid_point_model_index=point_ids,
        )
        source_observations = tuple(
            f"{camera}:{frame:03d}"
            for frame in range(active.shape[0])
            for camera in ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
        )
        support = RenderSupportFrame(
            checkpoint_sha256=file_fingerprint(scene["checkpoint"]),
            gaussian_primitive_count=int(len(background) + len(rigid)),
            source_observations=source_observations,
            supporting_camera_times=source_observations,
            reprojection_residual_px=None,
            uncertainty=None,
        )
        frame_path = (
            args.evidence_root / scene_id / f"frame_{args.overlay_frame:03d}.npz"
        )
        frame = np.load(frame_path, allow_pickle=False)
        layers = _decode_layers(frame)
        safety_path = scene_out / "safety_geometry_overlay.png"
        evidence_path = scene_out / "observation_evidence_overlay.png"
        render_path = scene_out / "render_support_overlay.png"
        _save_rgb(safety_path, _safety_overlay(layers, frame["base_state"].shape))
        _save_rgb(
            evidence_path, _evidence_overlay(frame["base_state"], layers)
        )
        _save_rgb(render_path, _render_support_overlay(active))
        overlays = {
            "safety_geometry": {
                "path": safety_path.name,
                "sha256": file_fingerprint(str(safety_path)),
                "source": "continuous_annotation_OBB_only",
            },
            "observation_evidence": {
                "path": evidence_path.name,
                "sha256": file_fingerprint(str(evidence_path)),
                "source": "base_free_static_unknown_plus_dynamic_layers",
            },
            "render_support": {
                "path": render_path.name,
                "sha256": file_fingerprint(str(render_path)),
                "source": "checkpoint_instances_fv_raw",
            },
        }
        if len({value["path"] for value in overlays.values()}) != 3:
            raise RuntimeError("三类 overlay 路径发生混用")
        scene_summary = {
            "scene_id": scene_id,
            "render_support_hash": support.content_hash(),
            "render_support_raw_sha256": file_fingerprint(
                str(scene_out / "render_support_raw.npz")
            ),
            "background_gaussian_count": int(len(background)),
            "rigid_gaussian_count": int(len(rigid)),
            "total_gaussian_count": support.gaussian_primitive_count,
            "actor_gaussian_count_by_model_index": actor_counts,
            "instances_fv_sha256": _tensor_hash(active),
            "source_observation_count": len(source_observations),
            "thresholded_visibility": "not_computed_in_world_state",
            "overlays": overlays,
            "human_verdict": "not_collected",
        }
        scene_summary["scene_support_sha256"] = canonical_sha256(scene_summary)
        atomic_write_json(str(scene_out / "summary.json"), scene_summary)
        scenes[scene_id] = scene_summary
    summary = {
        "schema_version": "render-support-v1-build",
        "task_id": "V7-H1-11B",
        "render_config_sha256": render_config_hash,
        "overlay_frame": args.overlay_frame,
        "scenes": scenes,
        "overlays_separate": all(
            len({value["path"] for value in scene["overlays"].values()}) == 3
            for scene in scenes.values()
        ),
        "agent_filled_human_verdict": False,
    }
    summary["support_set_sha256"] = canonical_sha256(scenes)
    atomic_write_json(str(args.output_root / "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
