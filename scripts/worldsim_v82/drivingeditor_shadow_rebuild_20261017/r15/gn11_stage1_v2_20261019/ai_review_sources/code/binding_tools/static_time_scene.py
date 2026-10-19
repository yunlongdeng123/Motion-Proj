"""同一冻结检查的scene-aware副本；无scene17锚点fallback。"""
import json
from pathlib import Path
import numpy as np


def run_static_time_check(scene, render_func, render_args, env_map, iteration):
    import torch
    spec = getattr(scene, "followup_scene_spec", None)
    if not isinstance(spec, dict):
        raise ValueError("真实Scene尚未绑定followup sceneSpec")
    from utils.hybrid30k_runtime import camera_id
    cameras = sorted(scene.getTrainCameras(), key=lambda camera: float(camera.timestamp))
    ids = [camera_id(camera) for camera in cameras]
    if ids != spec["frame_ids_with_camera"]:
        raise ValueError("静态时间实际camera身份与sceneSpec不匹配")
    if any(abs(float(camera.timestamp) - t) > 1e-7 for camera, t in zip(cameras, spec["model_timestamps"])):
        raise ValueError("静态时间actual timestamps不匹配")
    anchor_id = spec["camera_directory"] + "/" + spec["anchor_frame"]
    matches = [camera for camera in cameras if camera_id(camera) == anchor_id]
    if len(matches) != 1:
        raise ValueError("sceneSpec实际anchor缺失或重复")
    anchor = matches[0]
    original_time = anchor.timestamp
    destination = Path(scene.model_path) / ("static_time_%d" % iteration)
    destination.mkdir(exist_ok=False)
    rows = []
    try:
        with torch.no_grad():
            base = render_func(anchor, scene.gaussians, scene.dynamic_gaussians, *render_args, env_map=env_map)
            reference = base["render_s"].detach().cpu().numpy().clip(0, 1)
            reference_alpha = base["static_alpha"].detach().cpu().numpy()
            for index, camera in enumerate(cameras):
                # camera10就是anchor：enumeration期间取不可变spec timestamp，不读被修改的anchor.timestamp。
                timestamp = spec["model_timestamps"][index]
                anchor.timestamp = timestamp
                package = render_func(anchor, scene.gaussians, scene.dynamic_gaussians, *render_args, env_map=env_map)
                image = package["render_s"].detach().cpu().numpy().astype(np.float32)
                alpha = package["static_alpha"].detach().cpu().numpy().astype(np.float32)
                if image.shape != (3, 320, 480) or alpha.shape != (1, 320, 480):
                    raise ValueError("static-time actual FP32 dimensions与固定协议不符")
                if not np.isfinite(image).all() or not np.isfinite(alpha).all():
                    raise FloatingPointError("静态时间控制出现非finite输出")
                np.savez_compressed(destination / ("%03d.npz" % index), static_rgb=image, static_alpha=alpha)
                difference = np.abs(image.clip(0, 1) - reference)
                rows.append({"source_timestamp_frame": camera.image_name, "timestamp_model": timestamp,
                             "static_rgb_MAE_from_anchor": float(difference.mean()),
                             "static_rgb_max_abs_from_anchor": float(difference.max()),
                             "static_alpha_MAE_from_anchor": float(np.abs(alpha - reference_alpha).mean())})
    finally:
        anchor.timestamp = original_time
    velocity = scene.gaussians._velocity.detach()
    widths = scene.gaussians.get_scaling_t.detach()
    norm = torch.linalg.vector_norm(velocity, dim=1)
    row = {"iteration": iteration, "anchor_pose_frame": anchor.image_name,
           "anchor_camera_directory": spec["camera_directory"],
           "anchor_timestamp_model": float(original_time), "rows": rows,
           "raw_static_velocity_norm_model_mean": float(norm.mean().item()),
           "raw_static_velocity_norm_model_max": float(norm.max().item()),
           "temporal_width_model_mean": float(widths.mean().item()),
           "static_SHM_peak_to_peak_displacement_bound_model_max": float((norm * scene.gaussians.T / np.pi).max().item()),
           "scope": "fixed pose/rays with observed model timestamps; counterfactual combinations, no GT quality score",
           "unit": "model coordinates and model time; not physical velocity or seconds"}
    with (destination / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(row, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return row
