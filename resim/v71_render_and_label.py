#!/usr/bin/env python
"""从 WorldState 驱动 StreetGS，并同步写出 V7.1 typed labels。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, "/root/autodl-tmp/third_party/drivestudio")

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.resim.coordinates import box_corners_actor
from motion_proj.resim.drivestudio_adapter import (
    global_actor_gaussian_mask,
    gsplat_first_hit_from_info,
)
from motion_proj.resim.label_regeneration import (
    audit_label_bundle,
    bytes_sha256,
    cumulative_first_hit,
    lidar_measured_depth,
    limited_semantic_mask,
    raw_projected_box,
    render_expected_depth,
    vehicle_instance_mask,
    visible_box_from_mask,
)
from motion_proj.resim.schema import render_request_hash, render_request_payload
from motion_proj.runtime.atomic import atomic_write_json
from motion_proj.runtime.fingerprint import file_fingerprint


def _to_device(value, device):
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value.to(device) if torch.is_tensor(value) else value


def _quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    # 稳定 matrix→quaternion，输出 DriveStudio wxyz。
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _write_array(path: Path, value: np.ndarray, metadata: dict) -> dict:
    np.save(path, value)
    actual = path.with_suffix(path.suffix + ".npy") if path.suffix != ".npy" else path
    sidecar = {
        **metadata,
        "artifact_path": actual.name,
        "artifact_hash": bytes_sha256(actual),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }
    atomic_write_json(str(actual.with_suffix(actual.suffix + ".sidecar.json")), sidecar)
    return sidecar


def _write_png(path: Path, value: np.ndarray, metadata: dict) -> dict:
    imageio.imwrite(path, value)
    sidecar = {
        **metadata,
        "artifact_path": path.name,
        "artifact_hash": bytes_sha256(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }
    atomic_write_json(str(path.with_suffix(path.suffix + ".sidecar.json")), sidecar)
    return sidecar


def _write_json_artifact(path: Path, value: dict, metadata: dict) -> dict:
    atomic_write_json(str(path), value)
    sidecar = {
        **metadata,
        "artifact_path": path.name,
        "artifact_hash": bytes_sha256(path),
        "media_type": "application/json",
    }
    atomic_write_json(str(path.with_suffix(path.suffix + ".sidecar.json")), sidecar)
    return sidecar


def _frame_record(world: dict, frame_index: int) -> dict:
    matches = [value for value in world["frames"] if int(value["frame_index"]) == frame_index]
    if len(matches) != 1:
        raise RuntimeError(f"WorldState 中 frame {frame_index} 必须恰好一次")
    return matches[0]


def _render_request(world_hash: str, frame: int, camera: dict, config: dict, renderer_hash: str) -> dict:
    render = config["render"]
    value = {
        "world_state_hash": world_hash,
        "frame_index": frame,
        "camera_id": camera["camera_id"],
        "output_resolution": [800, 450],
        "rasterizer_mode": render["rasterizer_mode"],
        "precision": render["precision"],
        "deterministic_flags": render["deterministic_flags"],
        "renderer_config_sha256": renderer_hash,
        "depth_definitions": {
            "depth_render_expected": "gsplat_RGB+ED",
            "depth_surface_first_hit": "cumulative_alpha_first_crossing",
            "depth_lidar_measured": "DriveStudio calibrated sparse LiDAR",
        },
        "alpha_first_hit_threshold": float(render["alpha_first_hit_threshold"]),
        "compositing_policy": "near-layer alpha over far-layer",
        "instance_policy": "nearest per-actor expected depth above alpha threshold",
        "limited_semantic_policy": "unknown/static_background/vehicle/ignore",
        "color_space": "sRGB_float_then_png8",
        "encoding": {"rgb": "png8", "numeric": "npy"},
    }
    return {**render_request_payload(value), "render_request_hash": render_request_hash(value)}


def _load_trainer(checkpoint: Path, device: torch.device):
    cfg = OmegaConf.load(checkpoint.parent / "config.yaml")
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(ckpt_path=str(checkpoint), load_only_model=True)
    trainer.set_eval()
    for model in trainer.models.values():
        if hasattr(model, "in_test_set"):
            model.in_test_set = False
    return cfg, dataset, trainer


def _apply_world_pose(trainer, world: dict, frame_record: dict, model_index: int, T_model_world: np.ndarray) -> None:
    actor = frame_record["actor_nodes"][0]
    transform = T_model_world @ np.asarray(actor["T_world_actor"], dtype=float)
    local = int(frame_record["frame_index"]) - int(world["frames"][0]["frame_index"])
    # DriveStudio checkpoint 时间从 config start=0；显式使用绝对帧更安全。
    local = int(frame_record["frame_index"])
    rigid = trainer.models["RigidNodes"]
    rigid.instances_trans.data[local, model_index] = torch.as_tensor(
        transform[:3, 3], device=rigid.instances_trans.device,
        dtype=rigid.instances_trans.dtype,
    )
    rigid.instances_quats.data[local, model_index] = torch.as_tensor(
        _quat_wxyz(transform[:3, :3]), device=rigid.instances_quats.device,
        dtype=rigid.instances_quats.dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/resim/v71/render_and_label_v1.yaml"),
    )
    parser.add_argument(
        "--world-root", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11C/world_states"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11C/render_labels"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"render-label output 已存在，拒绝覆盖: {args.output_root}")
    args.output_root.mkdir(parents=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    world_cfg = yaml.safe_load(Path(config["world_config"]).read_text(encoding="utf-8"))
    world_cfg_by_scene = {value["scene_id"]: value for value in world_cfg["scenes"]}
    renderer_hash = file_fingerprint(str(args.config))
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    summary = {"schema_version": 1, "task_id": "V7-H1-11C", "samples": {}}
    peak = 0
    for scene in config["scenes"]:
        scene_id, actor_id, frame_index = scene["scene_id"], int(scene["actor_id"]), int(scene["render_frame"])
        checkpoint = Path(world_cfg_by_scene[scene_id]["checkpoint"])
        cfg, dataset, trainer = _load_trainer(checkpoint, device)
        registry = json.loads(
            (Path(config["registry_root"]) / scene_id / "actor_registry.json").read_text()
        )
        actor_by_model = {int(value["rigid_model_index"]): value for value in registry["actors"]}
        target_rows = [value for value in registry["actors"] if int(value["true_instance_id"]) == actor_id]
        if len(target_rows) != 1:
            raise RuntimeError("target actor registry 映射不唯一")
        model_index = int(target_rows[0]["rigid_model_index"])
        processed = Path(world_cfg["processed_root"]) / scene_id
        start_camera = np.loadtxt(processed / "extrinsics" / "000_0.txt").reshape(4, 4)
        T_model_world = np.linalg.inv(start_camera)
        original_trans = trainer.models["RigidNodes"].instances_trans.detach().clone()
        original_quats = trainer.models["RigidNodes"].instances_quats.detach().clone()
        for variant_name in config["variants"]:
            world = json.loads(
                (args.world_root / scene_id / variant_name / "world_state.json").read_text()
            )
            frame_record = _frame_record(world, frame_index)
            trainer.models["RigidNodes"].instances_trans.data.copy_(original_trans)
            trainer.models["RigidNodes"].instances_quats.data.copy_(original_quats)
            _apply_world_pose(trainer, world, frame_record, model_index, T_model_world)
            for camera in config["cameras"]:
                camera_index = int(camera["dataset_index"])
                image_index = frame_index * dataset.pixel_source.num_cams + camera_index
                image_infos, camera_infos = dataset.full_image_set.get_image(
                    image_index, camera_downscale=float(config["render"]["camera_downscale"])
                )
                image_infos = _to_device(image_infos, device)
                camera_infos = _to_device(camera_infos, device)
                captured = {}
                original_render = trainer.render_gaussians

                def capture_render(*call_args, **call_kwargs):
                    outputs, render_fn = original_render(*call_args, **call_kwargs)
                    captured["render_fn"] = render_fn
                    captured["gs"] = call_args[0] if call_args else call_kwargs["gs"]
                    captured["cam"] = call_args[1] if len(call_args) > 1 else call_kwargs["cam"]
                    return outputs, render_fn

                trainer.render_gaussians = capture_render
                with torch.no_grad():
                    output = trainer(image_infos, camera_infos)
                trainer.render_gaussians = original_render
                info = {key: value for key, value in trainer.info.items()}
                render_fn = captured["render_fn"]
                rigid_label = int(trainer.gaussian_classes["RigidNodes"])
                rigid_global = trainer.pts_labels == rigid_label
                background_global = ~rigid_global
                rigid_model = trainer.models["RigidNodes"]
                with torch.no_grad():
                    background_rgb_raw, _, background_opacity_raw = render_fn(
                        background_global
                    )
                    rigid_rgb_raw, _, rigid_opacity_raw = render_fn(rigid_global)
                actor_opacities, actor_depths = {}, {}
                for index, actor in actor_by_model.items():
                    mask = global_actor_gaussian_mask(
                        trainer.pts_labels, rigid_label, rigid_model.point_ids, index
                    )
                    with torch.no_grad():
                        _, depth, opacity = render_fn(mask)
                    actor_opacities[int(actor["limited_label_id"])] = opacity.cpu().numpy().squeeze()
                    actor_depths[int(actor["limited_label_id"])] = depth.cpu().numpy().squeeze()
                instance = vehicle_instance_mask(
                    actor_opacities, actor_depths,
                    alpha_threshold=float(config["render"]["instance_alpha_threshold"]),
                )
                semantic = limited_semantic_mask(
                    output["Background_opacity"].cpu().numpy().squeeze(),
                    instance,
                    alpha_threshold=float(config["render"]["instance_alpha_threshold"]),
                    ignore_mask=image_infos["human_masks"].cpu().numpy() > 0,
                )
                first_depth, first_valid = gsplat_first_hit_from_info(
                    info, alpha_threshold=float(config["render"]["alpha_first_hit_threshold"])
                )
                bg_first, bg_valid = gsplat_first_hit_from_info(
                    info,
                    alpha_threshold=float(config["render"]["alpha_first_hit_threshold"]),
                    gaussian_mask=background_global,
                )
                rigid_first, rigid_valid = gsplat_first_hit_from_info(
                    info,
                    alpha_threshold=float(config["render"]["alpha_first_hit_threshold"]),
                    gaussian_mask=rigid_global,
                )
                expected = render_expected_depth(
                    output["depth"].cpu().numpy().squeeze(),
                    output["opacity"].cpu().numpy().squeeze(),
                )
                measured = lidar_measured_depth(
                    image_infos["lidar_depth_map"].cpu().numpy()
                )
                first = cumulative_first_hit(
                    first_depth[..., None], first_valid[..., None].astype(np.float32),
                    threshold=0.5,
                )
                # first 已由 gsplat contribution 求得；TypedDepth 这里只绑定类型。
                first = type(first)(
                    first_depth.astype(np.float32), first_valid,
                    "depth_surface_first_hit", "T1",
                    f"cumulative_alpha_first_crossing_{config['render']['alpha_first_hit_threshold']}",
                )
                actor_label_id = int(target_rows[0]["limited_label_id"])
                visible = {str(actor_label_id): visible_box_from_mask(instance == actor_label_id)}
                actor_transform_model = T_model_world @ np.asarray(
                    frame_record["actor_nodes"][0]["T_world_actor"], dtype=float
                )
                corners_model = (
                    actor_transform_model[:3, :3]
                    @ box_corners_actor(frame_record["actor_nodes"][0]["dimensions_lwh"]).T
                ).T + actor_transform_model[:3, 3]
                processed_cam = captured["cam"]
                T_camera_model = torch.linalg.inv(processed_cam.camtoworlds).cpu().numpy()
                corners_camera = (
                    T_camera_model[:3, :3] @ corners_model.T
                ).T + T_camera_model[:3, 3]
                raw_box = raw_projected_box(
                    corners_camera,
                    processed_cam.Ks.cpu().numpy(),
                    (int(processed_cam.W), int(processed_cam.H)),
                )
                request = _render_request(
                    world["world_state_hash"], frame_index, camera, config, renderer_hash
                )
                metadata = {
                    "world_state_hash": world["world_state_hash"],
                    "render_request_hash": request["render_request_hash"],
                    "scene_id": scene_id,
                    "variant": variant_name,
                    "frame_index": frame_index,
                    "camera_id": camera["camera_id"],
                }
                sample_dir = args.output_root / scene_id / variant_name / camera["camera_id"]
                sample_dir.mkdir(parents=True)
                artifacts = []
                rgb = (np.clip(output["rgb"].cpu().numpy(), 0, 1) * 255).round().astype(np.uint8)
                artifacts.append(_write_png(sample_dir / "rgb.png", rgb, metadata))
                arrays = {
                    "alpha.npy": output["opacity"].cpu().numpy().squeeze().astype(np.float32),
                    "depth_render_expected.npy": expected.value.astype(np.float32),
                    "depth_surface_first_hit.npy": first.value,
                    "depth_surface_first_hit_valid.npy": first.valid.astype(np.uint8),
                    "depth_lidar_measured.npy": measured.value,
                    "depth_lidar_measured_valid.npy": measured.valid.astype(np.uint8),
                    "rgb_background.npy": output["Background_rgb"].cpu().numpy().astype(np.float32),
                    "alpha_background.npy": output["Background_opacity"].cpu().numpy().squeeze().astype(np.float32),
                    "depth_surface_first_hit_background.npy": bg_first.astype(np.float32),
                    "depth_surface_first_hit_background_valid.npy": bg_valid.astype(np.uint8),
                    "rgb_actor_only.npy": output["RigidNodes_rgb"].cpu().numpy().astype(np.float32),
                    "alpha_actor_only.npy": output["RigidNodes_opacity"].cpu().numpy().squeeze().astype(np.float32),
                    "depth_surface_first_hit_actor_only.npy": rigid_first.astype(np.float32),
                    "depth_surface_first_hit_actor_only_valid.npy": rigid_valid.astype(np.uint8),
                    "vehicle_instance_mask.npy": instance,
                    "limited_semantic_mask.npy": semantic,
                }
                for name, value in arrays.items():
                    artifacts.append(_write_array(sample_dir / name, value, metadata))
                artifacts.append(
                    _write_json_artifact(
                        sample_dir / "boxes.json",
                        {
                            "actor_id": actor_id,
                            "limited_label_id": actor_label_id,
                            "box3d": frame_record["actor_nodes"][0],
                            "box2d_projected_raw": raw_box,
                            "box2d_visible": visible[str(actor_label_id)],
                        },
                        metadata,
                    )
                )
                artifacts.append(
                    _write_json_artifact(
                        sample_dir / "evidence_refs.json",
                        {
                            "safety_geometry_sha256": world["safety_geometry_sha256"],
                            "observation_evidence_sha256": world["observation_evidence_sha256"],
                            "render_support_sha256": world["render_support_sha256"],
                            "observation_evidence_frame": str(
                                Path(config["evidence_root"]) / scene_id / f"frame_{frame_index:03d}.npz"
                            ),
                        },
                        metadata,
                    )
                )
                bundle = {
                    **metadata,
                    "typed_depths": [expected, first, measured],
                    "instance_mask": instance,
                    "limited_semantic_mask": semantic,
                    "visible_boxes": visible,
                }
                audit = audit_label_bundle(bundle)
                # 两个互斥 Gaussian layer 依 first-hit z-order 做 alpha-over 回合成。
                bg_rgb_np = background_rgb_raw.cpu().numpy()
                rigid_rgb_np = rigid_rgb_raw.cpu().numpy()
                bg_alpha_np = background_opacity_raw.cpu().numpy()
                rigid_alpha_np = rigid_opacity_raw.cpu().numpy()
                rigid_front = (
                    rigid_valid
                    & (~bg_valid | (rigid_first < bg_first))
                )[..., None]
                front_rigid = rigid_rgb_np + (1.0 - rigid_alpha_np) * bg_rgb_np
                front_background = bg_rgb_np + (1.0 - bg_alpha_np) * rigid_rgb_np
                recomposed = np.where(rigid_front, front_rigid, front_background)
                full_gaussian = output["rgb_gaussians"].cpu().numpy()
                composition_mean = float(np.mean(np.abs(recomposed - full_gaussian)))
                audit["layer_composition_mean_abs"] = composition_mean
                audit["layer_composition_pass"] = composition_mean <= float(
                    config["render"]["layer_composition_mean_abs_tolerance"]
                )
                if variant_name == "V0" and camera["camera_id"] == "CAM_FRONT":
                    legacy_scene = {"003": "s0", "005": "s1", "004": "s2"}[scene_id]
                    legacy_path = Path(
                        f"/root/autodl-tmp/runs/occgs_resim/c0_cf/{legacy_scene}/V0/"
                        f"rgb_t{frame_index:03d}.png"
                    )
                    legacy = imageio.imread(legacy_path).astype(np.float32) / 255.0
                    audit["legacy_v0_mean_abs"] = float(
                        np.mean(np.abs(legacy - rgb.astype(np.float32) / 255.0))
                    )
                    audit["legacy_v0_regression_pass"] = (
                        audit["legacy_v0_mean_abs"]
                        <= float(config["render"]["legacy_png_mean_abs_tolerance"])
                    )
                else:
                    audit["legacy_v0_regression_pass"] = True
                # 同一 state/request 立即重复 render，测确定性。
                with torch.no_grad():
                    repeated = trainer(image_infos, camera_infos)
                repeat_max = float(
                    torch.max(torch.abs(output["rgb"] - repeated["rgb"])).item()
                )
                audit["repeated_render_max_abs"] = repeat_max
                audit["repeated_render_pass"] = repeat_max <= float(
                    config["render"]["repeated_render_max_abs_tolerance"]
                )
                audit["artifact_sidecars_complete"] = all(
                    value["world_state_hash"] == world["world_state_hash"]
                    and value["render_request_hash"] == request["render_request_hash"]
                    for value in artifacts
                )
                audit["pass"] = (
                    audit["pass"]
                    and audit["repeated_render_pass"]
                    and audit["artifact_sidecars_complete"]
                    and audit["layer_composition_pass"]
                    and audit["legacy_v0_regression_pass"]
                )
                _write_json_artifact(sample_dir / "render_request.json", request, metadata)
                _write_json_artifact(sample_dir / "label_sync_audit.json", audit, metadata)
                key = f"{scene_id}:{variant_name}:{camera['camera_id']}:{frame_index}"
                summary["samples"][key] = {
                    "world_state_hash": world["world_state_hash"],
                    "render_request_hash": request["render_request_hash"],
                    "label_sync_pass": audit["pass"],
                    "repeat_max_abs": repeat_max,
                    "visible_status": visible[str(actor_label_id)]["status"],
                    "artifact_count": len(artifacts) + 2,
                }
                peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
        del trainer, dataset
        torch.cuda.empty_cache()
    summary["sample_count"] = len(summary["samples"])
    summary["label_sync_all_pass"] = all(
        value["label_sync_pass"] for value in summary["samples"].values()
    )
    summary["peak_cuda_bytes"] = peak
    summary["render_label_set_sha256"] = canonical_sha256(summary["samples"])
    atomic_write_json(str(args.output_root / "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
