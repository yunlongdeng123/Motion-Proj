"""导出官方 SplatAD 留出传感器输出；两条原生点云路径并列保留。"""
import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from splatad_compat import apply
apply()
from nerfstudio.utils.eval_utils import eval_setup

N = Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
ap = argparse.ArgumentParser()
ap.add_argument('--fit', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--checkpoint-step', type=int, required=True)
ap.add_argument('--all-lidars', action='store_true')
ap.add_argument('--all-cameras', action='store_true')
args = ap.parse_args()
if args.out.exists():
    raise RuntimeError(f'Preserve existing validation: {args.out}')
args.out.mkdir(parents=True)
snapshot = args.out / 'checkpoint'
snapshot.mkdir()
name = f'step-{args.checkpoint_step:09d}.ckpt'
shutil.copy2(args.fit / 'nerfstudio_models' / name, snapshot / name)
seed = 20260915
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_num_threads(4)

registration = {
    'task_id': 'WS-SIM-NATIVE-SENSOR-VALIDATION-01', 'seed': seed,
    'fit': str(args.fit), 'checkpoint_step': args.checkpoint_step,
    'selection': 'All held-out sensors when requested; otherwise four evenly spaced LiDAR scans and earliest held-out image per camera, chosen before rendering.',
    'role': 'Native held-out sensor fidelity and adapter validation; not downstream impact or a geometry badcase.',
    'readouts': ['official point_cloud from raw accumulated depth', 'official median_point_cloud'],
    'information': 'Full-scene RGB+LiDAR fit, metric calibration and annotated dynamic actor trajectories. Official validation uses measured/imputed ray directions and validity masks.',
    'completed': False, 'start_unix': time.time(), 'human_verdict': None,
}
(args.out / 'registration.json').write_text(json.dumps(registration, indent=2))

def update_config(config):
    config.load_dir = snapshot
    config.load_step = args.checkpoint_step
    config.pipeline.datamanager.cache_images = 'cpu'
    config.pipeline.datamanager.cache_lidars = 'cpu'
    config.pipeline.datamanager.max_thread_workers = 4
    config.pipeline.calc_fid_steps = ()
    return config

config, pipeline, ckpt, step = eval_setup(
    args.fit / 'config.yml', test_mode='val', update_config_callback=update_config)
dm, model = pipeline.datamanager, pipeline.model
model.eval()

def arr(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def scalar(x):
    return float(arr(x).reshape(-1)[0])

def err_stats(pred, gt):
    d = arr(pred).reshape(-1) - arr(gt).reshape(-1)
    good = np.isfinite(d)
    if not good.any():
        return {'n': 0, 'nonfinite': int((~good).sum())}
    d = d[good]
    return {'n': int(d.size), 'nonfinite': int((~good).sum()),
            'bias_m': float(d.mean()), 'mae_m': float(np.abs(d).mean()),
            'median_abs_m': float(np.median(np.abs(d))),
            'p90_abs_m': float(np.quantile(np.abs(d), .9)),
            'rmse_m': float(np.sqrt(np.mean(d ** 2)))}

lidars = dm.eval_lidar_dataset.lidars
cameras = dm.eval_dataset.cameras
lidar_indices = list(range(len(lidars))) if args.all_lidars else sorted(set(
    np.linspace(0, len(lidars) - 1, min(4, len(lidars))).astype(int).tolist()))
sensor_ids = arr(cameras.metadata['sensor_idxs']).reshape(-1)
camera_indices = list(range(len(cameras))) if args.all_cameras else [
    int(np.flatnonzero(sensor_ids == i)[0]) for i in np.unique(sensor_ids)]
registration.update(lidar_indices=lidar_indices, camera_indices=camera_indices,
                    heldout_lidars=len(lidars), heldout_cameras=len(cameras),
                    dataparser_transform=arr(dm.train_dataparser_outputs.dataparser_transform).tolist(),
                    dataparser_scale=float(dm.train_dataparser_outputs.dataparser_scale),
                    time_offset=float(dm.train_dataparser_outputs.time_offset))
(args.out / 'registration.json').write_text(json.dumps(registration, indent=2))
rows = {'lidars': [], 'cameras': []}

def save_summary():
    (args.out / 'summary.json').write_text(json.dumps(rows, indent=2))

with torch.inference_mode():
    for i in lidar_indices:
        data = copy.deepcopy(dm.cached_lidar_eval[i])
        lidar = copy.deepcopy(lidars[i:i+1]).to(model.device)
        lidar.metadata['lidar_idx'] = i
        dm._add_metadata(lidar, data, len(dm.eval_dataset))
        # 官方渲染器会原位居中扫描时间；独立副本避免跨条件累积修改。
        raster_before = data['raster_pts'].clone()
        begin = time.time()
        outputs = model.get_lidar_outputs(lidar)
        pred, gt = model.filter_lidar_pred_and_gt(outputs, data, output_point_cloud=True)
        valid = gt['valid']
        drop = pred['ray_drop'].sigmoid() > .5
        gt_drop = gt['ray_drop']
        row = {'index': i, 'time_s': scalar(lidar.times),
               'raw': err_stats(pred['depth'], gt['depth']),
               'median': err_stats(pred['median_depth'], gt['depth']),
               'valid_rays': int(valid.sum()),
               'gt_returns': int((valid & ~gt_drop).sum()),
               'pred_returns': int((valid & ~drop).sum()),
               'false_drop': int((valid & ~gt_drop & drop).sum()),
               'false_return': int((valid & gt_drop & ~drop).sum()),
               'gt_point_count': len(gt['point_cloud']),
               'raw_point_count': len(pred['point_cloud']),
               'median_point_count': len(pred['median_point_cloud']),
               'render_seconds': time.time()-begin}
        prefix = args.out / f'lidar_{i:04d}'
        np.savez_compressed(str(prefix)+'.npz',
            raw_points=arr(pred['point_cloud']), median_points=arr(pred['median_point_cloud']),
            gt_points=arr(gt['point_cloud']),
            lidar_to_world=arr(lidar.lidar_to_worlds),
            linear_velocities_local=arr(data['linear_velocities_local']),
            raster_before=arr(raster_before), raster_after=arr(data['raster_pts']),
            raw_depth=arr(outputs['depth']), median_depth=arr(outputs['median_depth']),
            intensity=arr(outputs['intensity']), ray_drop_prob=arr(outputs['ray_drop_prob']),
            valid=arr(valid), gt_drop=arr(gt_drop), accumulation=arr(outputs['accumulation']))
        rows['lidars'].append(row)
        save_summary()
        print('VALIDATE_LIDAR', json.dumps(row), flush=True)
        del outputs, pred, gt, data, lidar

    for i in camera_indices:
        camera = copy.deepcopy(cameras[i:i+1]).to(model.device)
        camera.metadata['cam_idx'] = i
        data = dm.cached_eval[i]
        outputs = model.get_camera_outputs(camera)
        gt = model.get_gt_img(data['image'])
        rgb = outputs['rgb']
        gt = gt[:rgb.shape[0], :rgb.shape[1], :3]
        mse = float(((rgb-gt)**2).mean())
        row = {'index': i, 'sensor_idx': int(sensor_ids[i]),
               'time_s': scalar(camera.times), 'psnr_db': float(-10*np.log10(max(mse, 1e-12))),
               'source_filename': str(dm.eval_dataset.image_filenames[i]),
               'shape': list(rgb.shape), 'gt_original_shape': list(data['image'].shape)}
        for label, tensor in [('real', gt), ('render', rgb)]:
            Image.fromarray((np.clip(arr(tensor), 0, 1)*255).astype(np.uint8)).save(
                args.out / f'camera_{i:04d}_{label}.jpg', quality=95)
        np.savez_compressed(args.out / f'camera_{i:04d}.npz',
            camera_to_world=arr(camera.camera_to_worlds), K=arr(camera.get_intrinsics_matrices()),
            depth=arr(outputs['depth']), accumulation=arr(outputs['accumulation']))
        rows['cameras'].append(row)
        save_summary()
        print('VALIDATE_CAMERA', json.dumps(row), flush=True)
        del outputs, gt, rgb

registration.update(completed=True, end_unix=time.time())
(args.out / 'registration.json').write_text(json.dumps(registration, indent=2))
print('VALIDATION_COMPLETED', args.out, flush=True)
