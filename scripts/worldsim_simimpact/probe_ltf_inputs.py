"""Verify the actual frozen LTF input dependency on real scene RGB.

This is a policy dependency probe, not a simulated driving outcome.
"""
import json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
B=Path('/root/autodl-tmp/external/worldsim_simimpact')
sys.path[:0]=[str(B/'NAVSIM'),str(B/'nuplan-devkit')]
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from hugsim.dataparser import parse_raw
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
torch.set_num_threads(4);torch.manual_seed(20260915)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
c=TransfuserConfig();c.latent=True
agent=TransfuserAgent(c,1e-4,str(R/'assets/ltf_seed_0.ckpt'))
agent.initialize();agent.cuda().eval()
source=json.loads((R/'raw_source_availability.json').read_text())
rows=[]
for name,s in source.items():
    rgb={}
    for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']:
        record=next(x for x in s['records'] if x['sample_token']==s['samples'][0] and f'/{cam}/' in x['filename'])
        rgb[cam]=np.asarray(Image.open(R/'raw_initial'/record['filename']).convert('RGB'))
    info={'ego_pos':[0.,0.,0.],'ego_steer':0.,'ego_velo':1.,'accelerate':0.}
    obs={'rgb':rgb}
    data=parse_raw((obs,info))['input']
    features={}
    for builder in agent.get_feature_builders():features.update(builder.compute_features(data))
    features={k:v.unsqueeze(0).cuda() for k,v in features.items()}
    with torch.no_grad():
        prediction=agent(features)['trajectory'].cpu().numpy()
        altered=dict(features)
        altered['lidar_feature']=torch.ones_like(features['lidar_feature'])
        changed=agent(altered)['trajectory'].cpu().numpy()
    row={'scene':name,'strict_checkpoint_loading':True,'source':'real RGB, official start speed/status, policy-only input dependency probe',
         'intervention':'Replace the unused incoming LiDAR histogram with ones; this is not a natural geometric badcase',
         'max_abs_trajectory_delta_m':float(np.max(np.abs(prediction-changed))),
         'trajectory':prediction.tolist(),'feature_shapes':{k:list(v.shape) for k,v in features.items()}}
    rows.append(row);print(name,'delta_m',row['max_abs_trajectory_delta_m'],flush=True)
(R/'ltf_input_dependency.json').write_text(json.dumps(rows,indent=2))
