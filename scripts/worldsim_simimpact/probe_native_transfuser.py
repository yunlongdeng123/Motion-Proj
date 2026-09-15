"""确认原生传感器环境可复用冻结策略，并与既有同输入基线比较。"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
B=Path('/root/autodl-tmp/external/worldsim_simimpact')
I=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1')
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
sys.path.insert(0,str(B/'NAVSIM'))
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import Lidar
from hugsim.dataparser import parse_raw
torch.set_num_threads(4);torch.manual_seed(20260915)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
c=TransfuserConfig();c.latent=False
agent=TransfuserAgent(c,1e-4,str(I/'lidar_policy/assets/transfuser_seed_0.ckpt'))
agent.initialize();agent.cuda().eval()
ref=json.loads((I/'lidar_policy/scene-0004_log_reference.json').read_text())
views=json.loads((I/'inputs/scene-0004.json').read_text())['views']
rgb={v['camera']:np.array(Image.open(v['image']).convert('RGB')) for v in views[:6] if v['camera'] in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']}
data=parse_raw(({'rgb':rgb},{'ego_pos':[0,0,0],'ego_steer':0,'ego_velo':ref['initial_velocity_xy_mps'][0],'accelerate':0}))['input']
data.ego_statuses[-1].ego_velocity=np.array(ref['initial_velocity_xy_mps'],np.float32)
data.ego_statuses[-1].ego_acceleration=np.array(ref['initial_acceleration_xy_mps2'],np.float32)
xyz=np.load(I/'lidar_policy/scans/scene-0004/real.npz')['points']
pc=np.zeros((6,len(xyz)),np.float32);pc[:3]=xyz.T;data.lidars[-1]=Lidar(pc)
f={}
for builder in agent.get_feature_builders():f.update(builder.compute_features(data))
with torch.no_grad():traj=agent({k:v.unsqueeze(0).cuda() for k,v in f.items()})['trajectory'][0].cpu().numpy()
prior=json.loads((I/'lidar_policy/policy_outputs/scene-0004/real.json').read_text())
delta=float(abs(traj-np.array(prior['trajectory'])).max())
r={'role':'Policy runtime compatibility, same input and official checkpoint. No new experiment condition.',
   'max_abs_vs_prior_trajectory_m':delta,'trajectory':traj.tolist(),'torch':torch.__version__}
(N/'native_policy_runtime.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r),flush=True)
assert delta<1e-3, 'Policy runtime changed the reference'
