#!/usr/bin/env python3
"""Execute a two-actor transform/lifecycle-owned scene patch in StreetGS."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy(value):
    return value.detach().cpu().numpy()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _quat_error(left: np.ndarray, right: np.ndarray) -> float:
    direct = np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)), axis=-1)
    antipodal = np.max(np.abs(left.astype(np.float64) + right.astype(np.float64)), axis=-1)
    return float(np.max(np.minimum(direct, antipodal)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--scene-package", required=True, type=Path)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frames = [int(value) for value in args.frames.split(",")]
    output = args.output.resolve()
    sensor_dir = output / "sensors"
    sensor_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve()
    run_root = checkpoint.parent
    backup = run_root / "backup"
    scene_package = args.scene_package.resolve()
    sys.path.insert(0, str(args.repo_root.resolve()))
    sys.path.insert(0, str(backup))
    sys.path.append(str(args.upstream_root.resolve()))

    import torch
    from datasets.driving_dataset import DrivingDataset
    from models.gaussians.basics import dataclass_gs, spherical_harmonics
    from omegaconf import OmegaConf
    from utils.misc import import_str

    checkpoint_before = _sha256(checkpoint)
    scene_manifest_before = _sha256(scene_package / "SCENE_PACKAGE_MANIFEST.json")
    composition = json.loads(
        (scene_package / "SCENE_COMPOSITION.json").read_text(encoding="utf-8")
    )
    runtime_contract = json.loads(
        (scene_package / "RUNTIME_CONTRACT.json").read_text(encoding="utf-8")
    )
    actors = []
    for actor_row in composition["actors"]:
        actor_package = scene_package / actor_row["actor_package_path"]
        geometry = json.loads(
            (actor_package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8")
        )
        arrays = geometry["base_arrays"]
        transform_record = geometry["proposal_transform_world"]
        transforms = np.load(actor_package / transform_record["path"], allow_pickle=False)
        if (
            transforms.dtype != np.float64
            or transforms.ndim != 3
            or transforms.shape[1:] != (4, 4)
            or not np.isfinite(transforms).all()
        ):
            raise ValueError("nested actor transform contract drift")
        lifecycle_record = geometry.get("actor_frame_validity", arrays.get("actor_frame_validity"))
        if lifecycle_record is None:
            raise ValueError("nested actor lifecycle missing")
        lifecycle = np.load(actor_package / lifecycle_record["path"], allow_pickle=False)
        if lifecycle.dtype != np.bool_ or lifecycle.shape != (len(geometry["trajectory"]),):
            raise ValueError("nested actor lifecycle contract drift")
        timestamp_to_index = {
            int(row["timestamp_us"]): index for index, row in enumerate(geometry["trajectory"])
        }
        if transforms.shape[0] != len(timestamp_to_index):
            raise ValueError("nested actor transform trajectory denominator drift")
        actors.append(
            {
                "actor_id": str(actor_row["actor_id"]),
                "actor_model_index": int(actor_row["actor_model_index"]),
                "proposal_id": str(actor_row["proposal_id"]),
                "translation_delta_m": list(actor_row["translation_delta_m"]),
                "package": actor_package,
                "package_manifest_sha256_before": _sha256(
                    actor_package / "PACKAGE_MANIFEST.json"
                ),
                "geometry": geometry,
                "arrays": arrays,
                "transforms": transforms,
                "lifecycle": lifecycle,
                "timestamp_to_index": timestamp_to_index,
                "means_world": np.load(
                    actor_package / arrays["means_world_m"]["path"], allow_pickle=False
                ),
                "quaternions_world": np.load(
                    actor_package / arrays["quaternions_world_wxyz"]["path"], allow_pickle=False
                ),
                "scales": np.load(actor_package / arrays["scales_m"]["path"], allow_pickle=False),
                "opacities": np.load(
                    actor_package / arrays["opacities"]["path"], allow_pickle=False
                ),
                "features_dc": np.load(
                    actor_package / arrays["features_dc"]["path"], allow_pickle=False
                ),
                "features_rest": np.load(
                    actor_package / arrays["features_rest"]["path"], allow_pickle=False
                ),
                "source_indices": np.load(
                    actor_package / arrays["source_indices"]["path"], allow_pickle=False
                ),
            }
        )
    if [actor["actor_id"] for actor in actors] != runtime_contract["actor_order"]:
        raise ValueError("scene runtime actor order drift")

    cfg = OmegaConf.load(run_root / "config.yaml")
    cfg.data.preload_device = "cpu"
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=torch.device("cuda"),
    )
    trainer.resume_from_checkpoint(ckpt_path=str(checkpoint), load_only_model=True)
    trainer.set_eval()
    if set(trainer.gaussian_classes) != {"Background", "RigidNodes"}:
        raise RuntimeError(f"unsupported Gaussian classes: {set(trainer.gaussian_classes)}")
    rigid = trainer.models["RigidNodes"]
    point_ids = rigid.point_ids.detach().reshape(-1).to(torch.int64)
    combined_mask = torch.zeros_like(point_ids, dtype=torch.bool)
    for actor in actors:
        actor_mask = point_ids == actor["actor_model_index"]
        if int(actor_mask.sum().item()) != int(actor["geometry"]["primitive_count"]):
            raise RuntimeError(f"native primitive denominator drift: {actor['actor_id']}")
        native_indices = torch.nonzero(actor_mask, as_tuple=False).reshape(-1).cpu().numpy()
        if not np.array_equal(
            native_indices.astype(actor["source_indices"].dtype), actor["source_indices"]
        ):
            raise RuntimeError(f"native source index drift: {actor['actor_id']}")
        if bool(torch.any(combined_mask & actor_mask)):
            raise RuntimeError("actor masks overlap")
        combined_mask |= actor_mask
        actor["mask"] = actor_mask
        actor["device_arrays"] = {
            "scales": torch.from_numpy(actor["scales"]).cuda(),
            "opacities": torch.from_numpy(actor["opacities"]).cuda(),
            "features_dc": torch.from_numpy(actor["features_dc"]).cuda(),
            "features_rest": torch.from_numpy(actor["features_rest"]).cuda(),
        }

    def compiled_actor(actor, frame_index: int, cam):
        trajectory_index = actor["timestamp_to_index"][frame_index * 100000]
        translation = actor["transforms"][trajectory_index, :3, 3].astype(np.float32)
        means = torch.from_numpy(actor["means_world"][trajectory_index]).cuda()
        means = means + torch.from_numpy(translation).cuda()[None, :]
        quats = torch.from_numpy(actor["quaternions_world"][trajectory_index]).cuda()
        device_arrays = actor["device_arrays"]
        colors = torch.cat(
            (device_arrays["features_dc"][:, None, :], device_arrays["features_rest"]), dim=1
        )
        viewdirs = means.detach() - cam.camtoworlds.data[..., :3, 3]
        viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        degree = min(rigid.step // rigid.ctrl_cfg.sh_degree_interval, rigid.sh_degree)
        rgbs = torch.clamp(spherical_harmonics(degree, viewdirs, colors) + 0.5, 0.0, 1.0)
        opacities = device_arrays["opacities"]
        if not bool(actor["lifecycle"][trajectory_index]):
            opacities = torch.zeros_like(opacities)
        return {
            "_means": means,
            "_quats": quats,
            "_scales": device_arrays["scales"],
            "_opacities": opacities,
            "_rgbs": rgbs,
        }, translation, bool(actor["lifecycle"][trajectory_index])

    def render_compiled_scene(frame_index: int, image_infos, cam):
        fields = {name: [] for name in ("_means", "_scales", "_quats", "_rgbs", "_opacities")}
        compiled_fields = {}
        actor_runtime = {}
        for actor in actors:
            values, translation, lifecycle = compiled_actor(actor, frame_index, cam)
            compiled_fields[actor["actor_id"]] = values
            actor_runtime[actor["actor_id"]] = {
                "translation_delta_m": translation.astype(float).tolist(),
                "package_actor_frame_valid": lifecycle,
            }
        for class_name in trainer.gaussian_classes:
            values = trainer.models[class_name].get_gaussians(cam)
            if class_name == "RigidNodes":
                values = {name: value.clone() for name, value in values.items()}
                for actor in actors:
                    for name, replacement in compiled_fields[actor["actor_id"]].items():
                        values[name][actor["mask"]] = replacement
            for name in fields:
                fields[name].append(values[name])
        merged = {name: torch.cat(values, dim=0) for name, values in fields.items()}
        gs = dataclass_gs(
            _means=merged["_means"],
            _scales=merged["_scales"],
            _quats=merged["_quats"],
            _rgbs=merged["_rgbs"],
            _opacities=merged["_opacities"],
            detach_keys=[],
            extras=None,
        )
        rendered, _ = trainer.render_gaussians(
            gs=gs,
            cam=cam,
            near_plane=trainer.render_cfg.near_plane,
            far_plane=trainer.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=trainer.render_cfg.get("radius_clip", 0.0),
        )
        sky = trainer.models["Sky"](image_infos)
        rendered["rgb"] = trainer.affine_transformation(
            rendered["rgb_gaussians"] + sky * (1.0 - rendered["opacity"]), image_infos
        )
        return rendered, compiled_fields, actor_runtime

    def actor_effect_pixels(native_rigid, mask, cam) -> int:
        actor_only = dataclass_gs(
            _means=native_rigid["_means"][mask],
            _scales=native_rigid["_scales"][mask],
            _quats=native_rigid["_quats"][mask],
            _rgbs=native_rigid["_rgbs"][mask],
            _opacities=native_rigid["_opacities"][mask],
            detach_keys=[],
            extras=None,
        )
        rendered, _ = trainer.render_gaussians(
            gs=actor_only,
            cam=cam,
            near_plane=trainer.render_cfg.near_plane,
            far_plane=trainer.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=trainer.render_cfg.get("radius_clip", 0.0),
        )
        return int((_numpy(rendered["opacity"]).squeeze(-1) > 0.01).sum())

    rows = []
    camera_downscale = trainer._get_downscale_factor()
    with torch.inference_mode():
        for frame_index in frames:
            image_infos, camera_infos = dataset.full_image_set.get_image(
                frame_index * 3, camera_downscale
            )
            for values in (image_infos, camera_infos):
                for key, value in values.items():
                    if isinstance(value, torch.Tensor):
                        values[key] = value.cuda(non_blocking=True)
            state_before = rigid.instances_trans.detach().clone()
            for actor in actors:
                trajectory_index = actor["timestamp_to_index"][frame_index * 100000]
                translation = actor["transforms"][trajectory_index, :3, 3].astype(np.float32)
                rigid.instances_trans[frame_index, actor["actor_model_index"]].add_(
                    torch.from_numpy(translation).cuda()
                )
            native = trainer(image_infos, camera_infos)
            cam = trainer.process_camera(
                camera_infos=camera_infos,
                image_ids=image_infos["img_idx"].flatten()[0],
                novel_view=False,
            )
            native_rigid = rigid.get_gaussians(cam)
            rigid.instances_trans.copy_(state_before)
            state_restored = torch.equal(rigid.instances_trans, state_before)
            compiled_1, compiled_fields, actor_runtime = render_compiled_scene(
                frame_index, image_infos, cam
            )
            compiled_2, _, _ = render_compiled_scene(frame_index, image_infos, cam)
            native_rgb = _numpy(native["rgb"]).astype(np.float32)
            compiled_rgb_1 = _numpy(compiled_1["rgb"]).astype(np.float32)
            compiled_rgb_2 = _numpy(compiled_2["rgb"]).astype(np.float32)
            native_depth = _numpy(native["depth"]).astype(np.float32)
            compiled_depth_1 = _numpy(compiled_1["depth"]).astype(np.float32)
            compiled_depth_2 = _numpy(compiled_2["depth"]).astype(np.float32)
            native_opacity = _numpy(native["opacity"]).astype(np.float32)
            compiled_opacity_1 = _numpy(compiled_1["opacity"]).astype(np.float32)
            compiled_opacity_2 = _numpy(compiled_2["opacity"]).astype(np.float32)
            rgb_error = np.abs(compiled_rgb_1.astype(np.float64) - native_rgb.astype(np.float64))
            depth_error = np.abs(compiled_depth_1.astype(np.float64) - native_depth.astype(np.float64))
            opacity_error = np.abs(
                compiled_opacity_1.astype(np.float64) - native_opacity.astype(np.float64)
            )
            per_actor = {}
            for actor in actors:
                actor_id = actor["actor_id"]
                native_fields = {
                    name: _numpy(value[actor["mask"]]) for name, value in native_rigid.items()
                }
                compiled = {name: _numpy(value) for name, value in compiled_fields[actor_id].items()}
                per_actor[actor_id] = {
                    **actor_runtime[actor_id],
                    "actor_model_index": actor["actor_model_index"],
                    "primitive_count": int(actor["mask"].sum().item()),
                    "nonzero_opacity_primitives": int(
                        (native_rigid["_opacities"][actor["mask"]].squeeze(-1) > 0).sum().item()
                    ),
                    "actor_effect_pixels": actor_effect_pixels(native_rigid, actor["mask"], cam),
                    "native_actor_field_max_error": {
                        "means_m": float(np.max(np.abs(native_fields["_means"] - compiled["_means"]))),
                        "quaternions_wxyz": _quat_error(
                            native_fields["_quats"], compiled["_quats"]
                        ),
                        "scales_m": float(
                            np.max(np.abs(native_fields["_scales"] - compiled["_scales"]))
                        ),
                        "opacities": float(
                            np.max(
                                np.abs(native_fields["_opacities"] - compiled["_opacities"])
                            )
                        ),
                        "view_dependent_rgb": float(
                            np.max(np.abs(native_fields["_rgbs"] - compiled["_rgbs"]))
                        ),
                    },
                }
            sensor_name = f"sensors/frame{frame_index:03d}.npz"
            np.savez_compressed(
                output / sensor_name,
                native_rgb=native_rgb.astype(np.float16),
                compiled_rgb=compiled_rgb_1.astype(np.float16),
                native_depth=native_depth,
                compiled_depth=compiled_depth_1,
                native_opacity=native_opacity.astype(np.float16),
                compiled_opacity=compiled_opacity_1.astype(np.float16),
            )
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_us": frame_index * 100000,
                    "actors": per_actor,
                    "joint_actor_effect_pixels": actor_effect_pixels(native_rigid, combined_mask, cam),
                    "native_translation_state_restored_exact": state_restored,
                    "full_sensor_rgb_mae": float(rgb_error.mean()),
                    "full_sensor_rgb_p99_absolute_error": float(np.quantile(rgb_error, 0.99)),
                    "full_sensor_rgb_max_absolute_error": float(rgb_error.max()),
                    "full_sensor_depth_mae_m": float(depth_error.mean()),
                    "full_sensor_opacity_mae": float(opacity_error.mean()),
                    "compiled_repeat_exact": all(
                        np.array_equal(left, right)
                        for left, right in (
                            (compiled_rgb_1, compiled_rgb_2),
                            (compiled_depth_1, compiled_depth_2),
                            (compiled_opacity_1, compiled_opacity_2),
                        )
                    ),
                    "sensor_path": sensor_name,
                    "sensor_sha256": _sha256(output / sensor_name),
                }
            )
    (output / "FRAME_METRICS.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_json(
        output / "WORKER_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r71_worker_audit.v1",
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": _sha256(checkpoint),
            "scene_package_manifest_sha256_before": scene_manifest_before,
            "scene_package_manifest_sha256_after": _sha256(
                scene_package / "SCENE_PACKAGE_MANIFEST.json"
            ),
            "nested_actor_package_manifest_sha256_before": {
                actor["actor_id"]: actor["package_manifest_sha256_before"] for actor in actors
            },
            "nested_actor_package_manifest_sha256_after": {
                actor["actor_id"]: _sha256(actor["package"] / "PACKAGE_MANIFEST.json")
                for actor in actors
            },
            "runtime_mode": runtime_contract["runtime_mode"],
            "actor_order": [actor["actor_id"] for actor in actors],
            "actor_model_indices": [actor["actor_model_index"] for actor in actors],
            "translation_source": "nested_package_transform_trajectory",
            "lifecycle_source": "nested_package_actor_frame_validity",
            "upstream_commit": subprocess.check_output(
                ["git", "-C", str(args.upstream_root.resolve()), "rev-parse", "HEAD"], text=True
            ).strip(),
            "frame_count": len(rows),
            "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "wall_seconds": time.monotonic() - started,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
