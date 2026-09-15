"""先按真实交通状态冻结小批交互窗口，不读取任何模型结果。"""
import json, collections
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1')
M=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval/v1.0-trainval')
R.mkdir(parents=True,exist_ok=True)
protocol={'task_id':'WS-SIM-INTERACTION-01','run_id':'20260915-r1','seed':20260915,
 'stage':'Metadata-selected natural interaction discovery; no independent test claim',
 'selection':{'scene_name_range':'scene-0001 through scene-0100','excluded_scenes':['scene-0013','scene-0038','scene-0041'],
 'excluded_logs':'All logs containing the three initial simulation scenes','max_windows':6,'distinct_logs':True,
 'window':'First qualifying keyframe from index 2 with >= 4 seconds of future reference',
 'order':'scene name then chronological index; no severity sorting',
 'lead_vehicle_center_forward_m':[4,25],'lead_vehicle_center_lateral_abs_max_m':2.,
 'lead_vehicle_min_lidar_points':5,'ego_speed_mps':[2,15],
 'four_second_heading_change_abs_max_rad':.2},
 'models':['vggt','omega512','dvgt1','pi3x'],'variants':['six','twelve'],
 'geometry_inputs':'First frame or first two frames RGB; common first six output maps',
 'controls':['Known calibration / poses','native scale and one BUILD LiDAR scalar',
 'real LiDAR baseline','full reconstructed scan','oracle early/late/missing hybrids'],
 'downstream':'Official TransFuser RGB+LiDAR and official NAVSIM PDMSimulator; nuScenes adapter, not full native benchmark',
 'outcomes':['tracked position and acceleration','recorded-actor overlap with baseline checks','reference trajectory error'],
 'stop_rule':'One six-log metadata-selected batch. If no residual harm after ordinary scale controls, lower first-return-to-serious-harm claim; do not make synthetic geometry more extreme.',
 'old_AV2_reserve_access':False,'human_verdict':None,'failure_ledger_refs':['V74-H2-F13','V74-H2-F14']}
p=R/'registration.json'
if p.exists():raise RuntimeError('Registration already exists; preserve frozen selection')
p.write_text(json.dumps(protocol,indent=2))
scenes=sorted(json.loads((M/'scene.json').read_text()),key=lambda x:x['name'])
exclude={s['log_token'] for s in scenes if s['name'] in protocol['selection']['excluded_scenes']}
scenes=[s for s in scenes if s['name']<='scene-0100' and s['log_token'] not in exclude]
samples={s['token']:s for s in json.loads((M/'sample.json').read_text())}
scene_samples={};token_set=set()
for s in scenes:
 seq=[];t=s['first_sample_token']
 while t:seq.append(samples[t]);token_set.add(t);t=samples[t]['next']
 scene_samples[s['name']]=seq
sd=collections.defaultdict(list)
for d in json.loads((M/'sample_data.json').read_text()):
 if d['is_key_frame'] and d['sample_token'] in token_set:sd[d['sample_token']].append(d)
ego={x['token']:x for x in json.loads((M/'ego_pose.json').read_text())}
instances={x['token']:x for x in json.loads((M/'instance.json').read_text())}
categories={x['token']:x['name'] for x in json.loads((M/'category.json').read_text())}
ann=collections.defaultdict(list)
for a in json.loads((M/'sample_annotation.json').read_text()):
 if a['sample_token'] in token_set and categories[instances[a['instance_token']]['category_token']].startswith('vehicle.'):
  ann[a['sample_token']].append(a)
def transform(e):
 t=np.eye(4);t[:3,:3]=Quaternion(e['rotation']).rotation_matrix;t[:3,3]=e['translation'];return t
selected={};used=set();candidates=[]
for s in scenes:
 if s['log_token'] in used:continue
 seq=scene_samples[s['name']]
 fronts=[next(d for d in sd[x['token']] if '/CAM_FRONT/' in d['filename']) for x in seq]
 poses=[transform(ego[d['ego_pose_token']]) for d in fronts]
 for i in range(2,len(seq)-8):
  inv=np.linalg.inv(poses[i]);previous=inv@poses[i-1];future=inv@poses[i+8]
  dt=(fronts[i]['timestamp']-fronts[i-1]['timestamp'])/1e6
  speed=-previous[0,3]/dt;yaw=abs(np.arctan2(future[1,0],future[0,0]))
  if not (2<=speed<=15 and yaw<=.2):continue
  leads=[]
  for a in ann[seq[i]['token']]:
   pos=inv@transform(a);x,y=pos[:2,3]
   if 4<=x<=25 and abs(y)<=2 and a['num_lidar_pts']>=5:
    leads.append({'instance':a['instance_token'],'category':categories[instances[a['instance_token']]['category_token']],
      'center_forward_m':float(x),'center_lateral_m':float(y),'lidar_points':a['num_lidar_pts']})
  if not leads:continue
  chosen=seq[i:i+9];tokens=[x['token'] for x in chosen]
  records=[{k:d[k] for k in ['token','sample_token','timestamp','filename','calibrated_sensor_token','ego_pose_token']}
    for token in tokens for d in sd[token]]
  selected[s['name']]={'log_token':s['log_token'],'original_start_index':i,'samples':tokens,'records':records,
   'metadata_selection':{'speed_mps':float(speed),'four_second_heading_change_rad':float(yaw),'leads':leads}}
  used.add(s['log_token']);print(s['name'],s['log_token'],i,round(speed,2),leads,flush=True);break
 if len(selected)==6:break
(R/'raw_source_availability.json').write_text(json.dumps(selected,indent=2))
(R/'selection_summary.json').write_text(json.dumps({k:{x:v[x] for x in ['log_token','original_start_index','metadata_selection']} for k,v in selected.items()},indent=2))
print('FROZEN',len(selected),'distinct logs',flush=True)
