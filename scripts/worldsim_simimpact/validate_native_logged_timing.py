"""只验证新增的日志逐传感器时间/位姿接口；不重跑发现批次。"""
import json
from pathlib import Path
import numpy as np
import torch
from native_splatad_bridge import NativeSplatADBridge, CAMERAS
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
I=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1')
token=json.loads((I/'inputs/scene-0004.json').read_text())['views'][0]['sample_token']
torch.manual_seed(20260915);np.random.seed(20260915);torch.set_num_threads(4)
b=NativeSplatADBridge(N/'fits/scene-0004/splatad/native-r2',N/'validation/scene0004-step008000-integration/checkpoint',8000,token)
sample=b.sample;obs=b.render_log_sample(sample)
rgb,gt,ego,front=b.real_observation(sample)
checks={}
for name in CAMERAS:
    sd=b.nusc.get('sample_data',sample['data'][name])
    expected=(b.A@b.ego_pose(sd)@b.extrinsics[name])[:3]
    delta=float(abs(obs['camera_poses'][name]-expected).max())
    assert delta<1e-4
    assert obs['sensor_times_absolute_s'][name]==sd['timestamp']/1e6
    checks[name]={'matrix_max_abs_difference':delta,'actual_time_offset_from_front_s':(sd['timestamp']-front['timestamp'])/1e6,
        'rgb_shape':list(obs['rgb'][name].shape)}
sd=b.nusc.get('sample_data',sample['data']['LIDAR_TOP'])
relative=np.linalg.inv(ego)@b.ego_pose(sd)@b.extrinsics['LIDAR_TOP']
err=float(abs(obs['scan']['policy_lidar_origin_ego']-relative[:3,3]).max());assert err<1e-8
out={'role':'Single-frame API validation for final native-timestamp replay; no new policy or research condition.',
    'camera_checks':checks,'lidar_origin_max_abs_difference_m':err,
    'lidar_time_offset_from_front_s':(sd['timestamp']-front['timestamp'])/1e6,
    'raw_points':len(obs['scan']['raw']),'median_points':len(obs['scan']['median']),
    'finite_pointclouds':all(bool(np.isfinite(obs['scan'][k]).all()) for k in ['raw','median'])}
assert out['finite_pointclouds'] and out['raw_points']>100
(N/'native_logged_timing_validation.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2),flush=True)
