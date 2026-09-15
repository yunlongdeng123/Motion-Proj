"""两个已发现残余的有限空间定位；这是传感器oracle，尚未修复几何资产。"""
import os,sys,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
B=Path('/root/autodl-tmp/external/worldsim_simimpact')
sys.path[:0]=[str(B/'NAVSIM'),str(B/'HUGSIM'),str(B/'HUGSIM/sim')]
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import Lidar
from hugsim.dataparser import parse_raw
I=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1')
L=I/'lidar_policy';C=L/'pi3x_localization';C.mkdir(exist_ok=True)
name='scene-0061';ref=json.loads((L/f'{name}_log_reference.json').read_text())
regions=['forward_corridor','forward_left','forward_right','near_rear','outside_local_regions']
registration={'task_id':'WS-SIM-PI3X-LOCAL-01','run_id':'20260915-r1','scene':name,'variants':['six','twelve'],
 'trigger':'Only two residual contact conditions after all missing rays restored in completed COVERAGE-01; selected on downstream event, not depth error',
 'role':'Finite sensor-domain localization before geometry-asset oracle repair; a policy recovery here is not yet a reconstruction-causality result',
 'regions':{'forward_corridor':'0<x<=25 m, abs(y)<=3 m','forward_left':'0<x<=25 m, 3<y<=12 m','forward_right':'0<x<=25 m, -12<=y<-3 m','near_rear':'-10<x<=0 m, abs(y)<=12 m','outside_local_regions':'Complement; deliberately nonlocal control, not a hero patch'},
 'membership':'Union of GT endpoint and reconstructed endpoint membership; all z. Unknown/no-return rays restored before every residual/intervention.',
 'conditions':'Real + each residual + each of five local repairs and corresponding only-local-error hybrids = 23 forwards. No adaptive partition search within this batch.',
 'downstream':'Frozen official TransFuser RGB+LiDAR, then official PDM execution. Keep GPU-reference numerical comparison and signed clearance; contact remains annotation-rectangle diagnostic.',
 'next_step':'Only a bounded localized recovery warrants a geometry-asset substitution; no recovery closes these two candidates. Novel-pose loop still required for final claim.',
 'device':'CPU to leave single GPU available for native SplatAD fit','seed':20260915,'human_verdict':None}
p=C/'registration.json'
if p.exists():raise RuntimeError('Existing localization; inspect before rerun')
p.write_text(json.dumps(registration,indent=2));(C/f'{name}_log_reference.json').write_text(json.dumps(ref,indent=2))
views=json.loads((I/'inputs'/f'{name}.json').read_text())['views']
scan=np.load(L/'scans'/name/'real.npz');gt=scan['points'];origin=scan['origin'];dirs=scan['directions']
def membership(points):
 x,y=points[:,:2].T
 m={'forward_corridor':(x>0)&(x<=25)&(abs(y)<=3),
    'forward_left':(x>0)&(x<=25)&(y>3)&(y<=12),
    'forward_right':(x>0)&(x<=25)&(y>=-12)&(y<-3),
    'near_rear':(x>-10)&(x<=0)&(abs(y)<=12)}
 m['outside_local_regions']=~np.logical_or.reduce(list(m.values()))
 return m
gm=membership(gt)
torch.set_num_threads(4);torch.manual_seed(20260915)
c=TransfuserConfig();c.latent=False
agent=TransfuserAgent(c,1e-4,str(L/'assets/transfuser_seed_0.ckpt'));agent.initialize();agent.cpu().eval()
rgb={v['camera']:np.array(Image.open(v['image']).convert('RGB')) for v in views[:6] if v['camera'] in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']}
data=parse_raw(({'rgb':rgb},{'ego_pos':[0,0,0],'ego_steer':0,'ego_velo':float(ref['initial_velocity_xy_mps'][0]),'accelerate':0}))['input']
data.ego_statuses[-1].ego_velocity=np.array(ref['initial_velocity_xy_mps'],np.float32)
data.ego_statuses[-1].ego_acceleration=np.array(ref['initial_acceleration_xy_mps2'],np.float32)
rows=[];masks={}
old=json.loads((L/'coverage_controls/policy_probe_summary.json').read_text())
def predict(points,condition,variant=None,region=None,changed=0):
 pc=np.zeros((6,len(points)),np.float32);pc[:3]=points.T;data.lidars[-1]=Lidar(pc);features={}
 for b in agent.get_feature_builders():features.update(b.compute_features(data))
 features={k:v.unsqueeze(0) for k,v in features.items()}
 start=time.time()
 with torch.no_grad():p=agent(features)['trajectory'][0].cpu().numpy()
 row={'scene':name,'condition':condition,'trajectory':p.tolist(),'protocol':'build_scale','points':len(points),'changed_returns':changed,'seconds':time.time()-start,'input_scope':'Oracle sensor-localization hybrid; geometry asset unchanged'}
 if variant:row.update(method='pi3x',variant=variant)
 if region:row['region']=region
 if condition in ['real','full_restore_all_missing']:
  refp=next(r for r in old if r['condition']==condition and (condition=='real' or (r.get('method')=='pi3x' and r.get('variant')==variant)))
  row['max_abs_vs_prior_GPU_trajectory']=float(abs(p-np.array(refp['trajectory'])).max())
 rows.append(row);(C/'policy_probe_summary.json').write_text(json.dumps(rows,indent=2))
 print('LOCAL',condition,variant,region,changed,row.get('max_abs_vs_prior_GPU_trajectory'),flush=True)
predict(gt,'real')
for variant in ['six','twelve']:
 f=np.load(L/'scans'/name/f'pi3x_{variant}_build_scale.npz')['first_range'];finite=np.isfinite(f)
 pp=gt.copy();pp[finite]=origin+dirs[finite]*f[finite,None]
 pm=membership(pp);predict(pp,'full_restore_all_missing',variant)
 for region in regions:
  mask=gm[region]|pm[region];masks[f'{variant}_{region}']=mask
  repaired=pp.copy();repaired[mask]=gt[mask]
  only=gt.copy();only[mask]=pp[mask]
  changed=int((mask&finite&(np.linalg.norm(pp-gt,axis=1)>1e-6)).sum())
  predict(repaired,'repair_'+region,variant,region,changed)
  predict(only,'only_error_'+region,variant,region,changed)
np.savez_compressed(C/'region_masks.npz',**masks)
print('LOCALIZATION_DONE',len(rows),flush=True)
