#!/usr/bin/env python3
"""在冻结 StreetGS renderer 中替换 actor_0000 为 R35 compiled payload。"""

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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
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
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--actor-model-index", required=True, type=int)
    parser.add_argument(
        "--translation-delta-m",
        default="0,0,0",
        help="可选的 world-frame actor 轨迹平移，仅供后续反事实合同使用",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frames = [int(value) for value in args.frames.split(",")]
    cli_translation_delta = np.asarray(
        [float(value) for value in args.translation_delta_m.split(",")], dtype=np.float32
    )
    if cli_translation_delta.shape != (3,) or not np.isfinite(cli_translation_delta).all():
        raise ValueError("translation delta 必须是三个有限数")
    output = args.output.resolve()
    sensor_dir = output / "sensors"
    sensor_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve()
    run_root = checkpoint.parent
    backup = run_root / "backup"
    package = args.package.resolve()
    sys.path.insert(0, str(args.repo_root.resolve()))
    sys.path.insert(0, str(backup))
    sys.path.append(str(args.upstream_root.resolve()))

    import torch
    from datasets.driving_dataset import DrivingDataset
    from models.gaussians.basics import dataclass_gs, spherical_harmonics
    from omegaconf import OmegaConf
    from utils.misc import import_str

    checkpoint_before = _sha256(checkpoint)
    package_manifest_before = _sha256(package / "PACKAGE_MANIFEST.json")
    geometry = json.loads((package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8"))
    if "arrays" in geometry:
        arrays = geometry["arrays"]
        proposal_transforms = None
        runtime_mode = "legacy_materialized_geometry_with_cli_translation"
    elif "base_arrays" in geometry and "proposal_transform_world" in geometry:
        if np.any(cli_translation_delta != 0):
            raise ValueError("transform-owned package 禁止叠加 CLI translation")
        arrays = geometry["base_arrays"]
        transform_record = geometry["proposal_transform_world"]
        proposal_transforms = np.load(package / transform_record["path"], allow_pickle=False)
        if (
            proposal_transforms.dtype != np.float64
            or proposal_transforms.ndim != 3
            or proposal_transforms.shape[1:] != (4, 4)
            or not np.isfinite(proposal_transforms).all()
        ):
            raise ValueError("transform-owned package 的 float64 齐次变换合同漂移")
        runtime_mode = "transform_owned_package_direct"
    else:
        raise ValueError("未支持的 actor package geometry schema")
    means_world = np.load(package / arrays["means_world_m"]["path"], allow_pickle=False)
    quaternions_world = np.load(
        package / arrays["quaternions_world_wxyz"]["path"], allow_pickle=False
    )
    scales = np.load(package / arrays["scales_m"]["path"], allow_pickle=False)
    opacities = np.load(package / arrays["opacities"]["path"], allow_pickle=False)
    features_dc = np.load(package / arrays["features_dc"]["path"], allow_pickle=False)
    features_rest = np.load(package / arrays["features_rest"]["path"], allow_pickle=False)
    source_indices = np.load(package / arrays["source_indices"]["path"], allow_pickle=False)
    timestamp_to_index = {
        int(row["timestamp_us"]): index for index, row in enumerate(geometry["trajectory"])
    }
    if proposal_transforms is not None and proposal_transforms.shape[0] != len(timestamp_to_index):
        raise ValueError("transform trajectory denominator 漂移")

    def translation_for_index(trajectory_index: int) -> np.ndarray:
        if proposal_transforms is None:
            return cli_translation_delta
        return proposal_transforms[trajectory_index, :3, 3].astype(np.float32)

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
        raise RuntimeError(f"未支持的 Gaussian class 集合：{set(trainer.gaussian_classes)}")
    rigid = trainer.models["RigidNodes"]
    point_ids = rigid.point_ids.detach().reshape(-1).to(torch.int64)
    actor_mask = point_ids == int(args.actor_model_index)
    if int(actor_mask.sum().item()) != int(geometry["primitive_count"]):
        raise RuntimeError("native actor primitive denominator 漂移")
    native_source_indices = torch.nonzero(actor_mask, as_tuple=False).reshape(-1).cpu().numpy()
    if not np.array_equal(native_source_indices.astype(source_indices.dtype), source_indices):
        raise RuntimeError("actor source_indices 与 native mask 顺序漂移")

    device_arrays = {
        "scales": torch.from_numpy(scales).cuda(),
        "opacities": torch.from_numpy(opacities).cuda(),
        "features_dc": torch.from_numpy(features_dc).cuda(),
        "features_rest": torch.from_numpy(features_rest).cuda(),
    }

    def compiled_actor(frame_index: int, cam):
        timestamp_us = frame_index * 100000
        trajectory_index = timestamp_to_index[timestamp_us]
        translation_delta = translation_for_index(trajectory_index)
        means = torch.from_numpy(means_world[trajectory_index]).cuda()
        means = means + torch.from_numpy(translation_delta).cuda()[None, :]
        quats = torch.from_numpy(quaternions_world[trajectory_index]).cuda()
        colors = torch.cat(
            (device_arrays["features_dc"][:, None, :], device_arrays["features_rest"]),
            dim=1,
        )
        viewdirs = means.detach() - cam.camtoworlds.data[..., :3, 3]
        viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        degree = min(rigid.step // rigid.ctrl_cfg.sh_degree_interval, rigid.sh_degree)
        rgbs = torch.clamp(spherical_harmonics(degree, viewdirs, colors) + 0.5, 0.0, 1.0)
        return {
            "_means": means,
            "_quats": quats,
            "_scales": device_arrays["scales"],
            "_opacities": device_arrays["opacities"],
            "_rgbs": rgbs,
        }

    def render_with_compiled_actor(frame_index: int, image_infos, cam):
        fields = {name: [] for name in ("_means", "_scales", "_quats", "_rgbs", "_opacities")}
        compiled = compiled_actor(frame_index, cam)
        for class_name in trainer.gaussian_classes:
            values = trainer.models[class_name].get_gaussians(cam)
            if class_name == "RigidNodes":
                values = {name: value.clone() for name, value in values.items()}
                for name, replacement in compiled.items():
                    values[name][actor_mask] = replacement
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
        return rendered, compiled

    rows = []
    camera_downscale = trainer._get_downscale_factor()
    with torch.inference_mode():
        for frame_index in frames:
            trajectory_index = timestamp_to_index[frame_index * 100000]
            translation_delta = translation_for_index(trajectory_index)
            image_infos, camera_infos = dataset.full_image_set.get_image(
                frame_index * 3, camera_downscale
            )
            for values in (image_infos, camera_infos):
                for key, value in values.items():
                    if isinstance(value, torch.Tensor):
                        values[key] = value.cuda(non_blocking=True)
            translation_state_before = rigid.instances_trans.detach().clone()
            if np.any(translation_delta != 0):
                rigid.instances_trans[frame_index, int(args.actor_model_index)].add_(
                    torch.from_numpy(translation_delta).cuda()
                )
            native = trainer(image_infos, camera_infos)
            cam = trainer.process_camera(
                camera_infos=camera_infos,
                image_ids=image_infos["img_idx"].flatten()[0],
                novel_view=False,
            )
            native_actor = rigid.get_gaussians(cam)
            rigid.instances_trans.copy_(translation_state_before)
            translation_state_restored = torch.equal(
                rigid.instances_trans, translation_state_before
            )
            compiled_1, compiled_actor_fields = render_with_compiled_actor(
                frame_index, image_infos, cam
            )
            compiled_2, _ = render_with_compiled_actor(frame_index, image_infos, cam)
            actor_native = {name: _numpy(value[actor_mask]) for name, value in native_actor.items()}
            actor_compiled = {name: _numpy(value) for name, value in compiled_actor_fields.items()}
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
            depth_error = np.abs(
                compiled_depth_1.astype(np.float64) - native_depth.astype(np.float64)
            )
            opacity_error = np.abs(
                compiled_opacity_1.astype(np.float64) - native_opacity.astype(np.float64)
            )
            actor_support = native_actor["_opacities"][actor_mask].squeeze(-1) > 0
            actor_only = dataclass_gs(
                _means=native_actor["_means"][actor_mask],
                _scales=native_actor["_scales"][actor_mask],
                _quats=native_actor["_quats"][actor_mask],
                _rgbs=native_actor["_rgbs"][actor_mask],
                _opacities=native_actor["_opacities"][actor_mask],
                detach_keys=[],
                extras=None,
            )
            actor_render, _ = trainer.render_gaussians(
                gs=actor_only,
                cam=cam,
                near_plane=trainer.render_cfg.near_plane,
                far_plane=trainer.render_cfg.far_plane,
                render_mode="RGB+ED",
                radius_clip=trainer.render_cfg.get("radius_clip", 0.0),
            )
            effect_pixels = int((_numpy(actor_render["opacity"]).squeeze(-1) > 0.01).sum())
            repeat_exact = all(
                np.array_equal(left, right)
                for left, right in (
                    (compiled_rgb_1, compiled_rgb_2),
                    (compiled_depth_1, compiled_depth_2),
                    (compiled_opacity_1, compiled_opacity_2),
                )
            )
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
                    "translation_delta_m": translation_delta.astype(float).tolist(),
                    "native_translation_state_restored_exact": translation_state_restored,
                    "actor_effect_pixels": effect_pixels,
                    "actor_nonzero_opacity_primitives": int(actor_support.sum().item()),
                    "native_actor_field_max_error": {
                        "means_m": float(
                            np.max(np.abs(actor_native["_means"] - actor_compiled["_means"]))
                        ),
                        "quaternions_wxyz": _quat_error(
                            actor_native["_quats"], actor_compiled["_quats"]
                        ),
                        "scales_m": float(
                            np.max(np.abs(actor_native["_scales"] - actor_compiled["_scales"]))
                        ),
                        "opacities": float(
                            np.max(
                                np.abs(actor_native["_opacities"] - actor_compiled["_opacities"])
                            )
                        ),
                        "view_dependent_rgb": float(
                            np.max(np.abs(actor_native["_rgbs"] - actor_compiled["_rgbs"]))
                        ),
                    },
                    "full_sensor_rgb_mae": float(rgb_error.mean()),
                    "full_sensor_rgb_p99_absolute_error": float(np.quantile(rgb_error, 0.99)),
                    "full_sensor_rgb_max_absolute_error": float(rgb_error.max()),
                    "full_sensor_depth_mae_m": float(depth_error.mean()),
                    "full_sensor_opacity_mae": float(opacity_error.mean()),
                    "compiled_repeat_exact": repeat_exact,
                    "sensor_path": sensor_name,
                    "sensor_sha256": _sha256(output / sensor_name),
                }
            )
    (output / "FRAME_METRICS.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    checkpoint_after = _sha256(checkpoint)
    package_manifest_after = _sha256(package / "PACKAGE_MANIFEST.json")
    _write_json(
        output / "WORKER_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r36_worker_audit.v1",
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "package_manifest_sha256_before": package_manifest_before,
            "package_manifest_sha256_after": package_manifest_after,
            "package_geometry_schema_version": geometry.get("schema_version"),
            "runtime_mode": runtime_mode,
            "translation_source": "package_transform_trajectory"
            if proposal_transforms is not None
            else "cli_argument",
            "upstream_commit": subprocess.check_output(
                ["git", "-C", str(args.upstream_root.resolve()), "rev-parse", "HEAD"],
                text=True,
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
