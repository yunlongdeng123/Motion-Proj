"""固定真实位姿枚举49个模型timestamp；不把反事实组合作为GT质量评价。"""
import json
from pathlib import Path
import numpy as np


def run_static_time_check(scene, render_func, render_args, env_map, iteration):
    import torch
    cameras = sorted(scene.getTrainCameras(), key=lambda camera: float(camera.timestamp))
    timestamps = [(camera.image_name, float(camera.timestamp)) for camera in cameras]
    anchor = next(camera for camera in cameras if camera.image_name == '0017010')
    original_time = anchor.timestamp
    destination = Path(scene.model_path) / ('static_time_%d' % iteration)
    destination.mkdir(exist_ok=False)
    rows = []
    try:
        with torch.no_grad():
            base = render_func(anchor, scene.gaussians, scene.dynamic_gaussians, *render_args, env_map=env_map)
            reference = base['render_s'].detach().cpu().numpy().clip(0, 1)
            reference_alpha = base['static_alpha'].detach().cpu().numpy()
            for index, (frame_name, timestamp) in enumerate(timestamps):
                anchor.timestamp = timestamp
                package = render_func(anchor, scene.gaussians, scene.dynamic_gaussians, *render_args, env_map=env_map)
                image = package['render_s'].detach().cpu().numpy().astype(np.float32)
                alpha = package['static_alpha'].detach().cpu().numpy().astype(np.float32)
                if not np.isfinite(image).all() or not np.isfinite(alpha).all():
                    raise FloatingPointError('静态时间控制出现非finite输出')
                np.savez_compressed(destination / ('%03d.npz' % index), static_rgb=image, static_alpha=alpha)
                difference = np.abs(image.clip(0, 1) - reference)
                rows.append({'source_timestamp_frame': frame_name, 'timestamp_model': timestamp,
                             'static_rgb_MAE_from_anchor': float(difference.mean()),
                             'static_rgb_max_abs_from_anchor': float(difference.max()),
                             'static_alpha_MAE_from_anchor': float(np.abs(alpha - reference_alpha).mean())})
    finally:
        anchor.timestamp = original_time
    raw_velocity = scene.gaussians._velocity.detach()
    widths = scene.gaussians.get_scaling_t.detach()
    norm = torch.linalg.vector_norm(raw_velocity, dim=1)
    row = {'iteration': iteration, 'anchor_pose_frame': anchor.image_name,
           'anchor_timestamp_model': float(original_time), 'rows': rows,
           'raw_static_velocity_norm_model_mean': float(norm.mean().item()),
           'raw_static_velocity_norm_model_max': float(norm.max().item()),
           'temporal_width_model_mean': float(widths.mean().item()),
           'static_SHM_peak_to_peak_displacement_bound_model_max': float((norm * scene.gaussians.T / np.pi).max().item()),
           'scope': 'fixed pose/rays with observed model timestamps; counterfactual combinations, no GT quality score',
           'unit': 'model coordinates and model time; not physical velocity or seconds'}
    (destination / 'summary.json').write_text(json.dumps(row, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    return row
