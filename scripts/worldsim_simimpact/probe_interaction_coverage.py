"""只对0061候选做有限恢复：共同覆盖、视野内外缺失与路锥局部因果。"""
import sys,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
B=Path('/root/autodl-tmp/external/worldsim_simimpact');sys.path[:0]=[str(B/'NAVSIM'),str(B/'HUGSIM'),str(B/'HUGSIM/sim')]
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import Lidar
from hugsim.dataparser import parse_raw
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1');L=R/'lidar_policy';C=L/'coverage_controls';C.mkdir(exist_ok=True);name='scene-0061'
protocol={'task':'WS-SIM-INTERACTION-COVERAGE-01','parent':'WS-SIM-INTERACTION-01','scene':name,
 'selection':'Only new nominal actor-overlap scene in frozen six-log cohort; selected after discovery, not independent validation',
 'purpose':'Separate common camera field-of-view limits, missing-return effect and cone-local errors before a physics-harm claim',
 'conditions':['real','GT_FOV_only','GT_without_common_missing','missing_inside_fov_only','missing_outside_fov_only','full_restore_all_missing','full_restore_outside_fov','full_restore_inside_fov_missing','cone_only_error','full_restore_cones'],
 'information':'All restore/hybrid conditions use oracle real LiDAR; FOV uses known cameras. These are diagnostics, not same-budget method improvements.',
 'fov_boundary':'Geometric image field of view, not occlusion-tested visibility',
 'human_verdict':None,'stop_rule':'One coverage/local restoration batch on the discovered scene; no more extreme synthetic perturbations'}
(C/'registration.json').write_text(json.dumps(protocol,indent=2))
ref=json.loads((L/f'{name}_log_reference.json').read_text());(C/f'{name}_log_reference.json').write_text(json.dumps(ref,indent=2))
views=json.loads((R/'inputs'/f'{name}.json').read_text())['views'];real=np.load(L/'scans'/name/'real.npz');gt=real['points'];ranges=real['ranges'];dirs=real['directions'];origin=real['origin'];ego=np.array(views[0]['world_from_ego_camera'])
covered=np.zeros(len(gt),bool)
for v in views[:6]:
 T=np.linalg.inv(ego)@np.array(v['world_from_camera']);cp=(gt-T[:3,3])@T[:3,:3];uv=cp@np.array(v['intrinsics_original']).T;uv=uv[:,:2]/uv[:,2:];w,h=v['original_wh'];covered|=(cp[:,2]>.2)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
cone=np.zeros(len(gt),bool)
for a in ref['records'][0]['boxes']:
 b=np.array(a['box'])
 if a['category']!='movable_object.trafficcone' or not (0<b[0]<25 and abs(b[1])<5):continue
 y=b[6];rot=np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]]);p=(gt-b[:3])@rot;cone|=(abs(p)<np.array([b[4],b[3],b[5]])/2+.05).all(1)
scans={};common_missing=np.ones(len(gt),bool)
for method in ['dvgt1','vggt','omega512','pi3x']:
 for variant in ['six','twelve']:
  f=np.load(L/'scans'/name/f'{method}_{variant}_build_scale.npz')['first_range'];scans[(method,variant)]=f;common_missing&=~np.isfinite(f)
torch.set_num_threads(4);torch.manual_seed(20260915);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
c=TransfuserConfig();c.latent=False;agent=TransfuserAgent(c,1e-4,str(L/'assets/transfuser_seed_0.ckpt'));agent.initialize();agent.cuda().eval()
rgb={v['camera']:np.array(Image.open(v['image']).convert('RGB')) for v in views[:6] if v['camera'] in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']}
data=parse_raw(({'rgb':rgb},{'ego_pos':[0,0,0],'ego_steer':0,'ego_velo':float(ref['initial_velocity_xy_mps'][0]),'accelerate':0}))['input']
data.ego_statuses[-1].ego_velocity=np.array(ref['initial_velocity_xy_mps'],np.float32);data.ego_statuses[-1].ego_acceleration=np.array(ref['initial_acceleration_xy_mps2'],np.float32)
out=[]
def predict(points,condition,method=None,variant=None):
 pc=np.zeros((6,len(points)),np.float32);pc[:3]=points.T;data.lidars[-1]=Lidar(pc);features={}
 for b in agent.get_feature_builders():features.update(b.compute_features(data))
 features={k:v.unsqueeze(0).cuda() for k,v in features.items()}
 with torch.no_grad():p=agent(features)['trajectory'][0].cpu().numpy()
 row={'scene':name,'condition':condition,'points':len(points),'trajectory':p.tolist(),'input_scope':'Oracle causal coverage diagnostic','protocol':'build_scale'}
 if method:row.update(method=method,variant=variant)
 out.append(row);print(condition,method,variant,len(points),flush=True)
predict(gt,'real');predict(gt[covered],'GT_FOV_only');predict(gt[~common_missing],'GT_without_common_missing')
for (method,variant),f in scans.items():
 finite=np.isfinite(f);pp=origin+dirs*f[:,None]
 for condition in ['missing_inside_fov_only','missing_outside_fov_only','full_restore_all_missing','full_restore_outside_fov','full_restore_inside_fov_missing','cone_only_error','full_restore_cones']:
  if condition=='missing_inside_fov_only':p=gt[finite|~covered]
  elif condition=='missing_outside_fov_only':p=gt[finite|covered]
  elif condition=='full_restore_all_missing':p=gt.copy();p[finite]=pp[finite]
  elif condition=='full_restore_outside_fov':p=np.concatenate([pp[finite],gt[~finite&~covered]])
  elif condition=='full_restore_inside_fov_missing':p=np.concatenate([pp[finite],gt[~finite&covered]])
  elif condition=='cone_only_error':p=np.concatenate([gt[~cone],pp[cone&finite]])
  else:p=np.concatenate([pp[~cone&finite],gt[cone]])
  predict(p,condition,method,variant)
(C/'policy_probe_summary.json').write_text(json.dumps(out,indent=2))
(C/'coverage_counts.json').write_text(json.dumps({'gt':len(gt),'inside_fov':int(covered.sum()),'outside_fov':int((~covered).sum()),'near_cone_returns':int(cone.sum()),'common_missing':int(common_missing.sum()),'common_missing_inside_fov':int((common_missing&covered).sum()),'policy_predictions':len(out)},indent=2))
print('DONE',len(out),flush=True)
