import os,json,pathlib,datetime,subprocess,torch,numpy as np
from PIL import Image
from pyquaternion import Quaternion
torch.set_num_threads(1)
P=pathlib.Path('/root/autodl-tmp');R=P/'runs/worldsim_v74_h2/WS-V74-MAINFIG-01/20260915-first-return-r1'
D=P/'runs/worldsim_v73/WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2'
RAW=P/'runs/worldsim_v73/WS-V73-M1-NATIVE-GEOMETRY-ADAPT-01/20260907T161500Z__native-dpt-surround25-dev6-s7301-r3/build_observations.pt'
META=P/'data/worldsim_v4/drivestudio_raw_trainval/v1.0-trainval'
R.mkdir(parents=True,exist_ok=True)
assert not (R/'registration.json').exists()
entries=[x for x in json.loads((D/'index.json').read_text())['cases'] if x['role']=='development']
order=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']
# DVGT camera order is explicit in input directories; all six views retained.
reg={'task_id':'WS-V74-MAINFIG-01','run_id':R.name,'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=P/'motion_proj',text=True).strip(),'seed':7403,'role':'REEXPOSED_DISCOVERY','methods':['official_vggt','vggt_omega_416_reproduce','dvgt1','pi3x'],'cohort':{'scenes':sorted(set(x['scene'] for x in entries)),'logs':sorted(set(x['log_id'] for x in entries)),'actors':len(entries)},'variants':{'six':{'sample_indices':[3],'views':6},'twelve':{'sample_indices':[3,4],'views':12}},'query_sample_indices':[2,5],'registration_rule':'All existing DEV windows, metadata only. No heldout sensor arrays loaded by inference. No reserve consumed. No training.','evaluation':{'epsilon_m':0.2,'sensitivity_epsilon_m':[0.1,0.3,0.5],'controls':['native metric / camera-baseline scale for relative VGGT','one scalar fitted on common BUILD sample3 background; additional LiDAR information','known camera pose/intrinsics and actor motion diagnostic','confidence retention 1,0.9,0.75,0.5','pointmap native coordinates versus calibrated depth grid'],'scale':'median BUILD background z/pred z, shared sample3 views within each run; not tuned on actor or QUERY','surface':'fixed image-grid triangles; each edge <= max(0.15 m, 0.05*minimum camera depth); actor box expanded by 1m; never repair/fit using QUERY','recall':'fraction owned heldout endpoints within 0.2m of retained vertices; report separately from ray-hit','static_stratum':'metadata translation_speed_mps <= 0.5; dynamic separate','downstream':'opaque first-hit LiDAR readout plus single-beam occupancy; beyond early hit UNKNOWN, not FREE; no closed-loop claim'},'failure_ledger_refs':['V73-F03','V73-F09','V74-F06','V74-H2-F12'],'failure_ledger_delta':'pending','human_verdict':None,'resources':{'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip(),'cpu_max':pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'memory_max':pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip()},'power':'No shutdown or scheduler authorized'}
(R/'registration.json').write_text(json.dumps(reg,indent=2));(R/'cohort.json').write_text(json.dumps(entries,indent=2))
sd={x['filename']:x for x in json.loads((META/'sample_data.json').read_text()) if x['is_key_frame']}
ego={x['token']:x for x in json.loads((META/'ego_pose.json').read_text())}
cal={x['token']:x for x in json.loads((META/'calibrated_sensor.json').read_text())}
scenes=torch.load(RAW,weights_only=False,map_location='cpu',mmap=True)
def mat(q,t):
 T=np.eye(4);T[:3,:3]=Quaternion(q).rotation_matrix;T[:3,3]=t;return T
for name in reg['cohort']['scenes']:
 s=next(x for x in scenes if x['scene_id']==name);views=[]
 samples=list(dict.fromkeys(v['sample_id'] for v in s['views']))
 assert len(samples)==4
 for idx in [1,2]:
  for cam in order:
   j,v=next((j,v) for j,v in enumerate(s['views']) if v['sample_id']==samples[idx] and v['camera_id']==cam)
   filename=str(pathlib.Path(v['image_path']).relative_to(META.parent));d=sd[filename];e=ego[d['ego_pose_token']];c=cal[d['calibrated_sensor_token']]
   views.append({'source_view_index':j,'sample_token':v['sample_id'],'sample_index':3 if idx==1 else 4,'camera':cam,'image':v['image_path'],'timestamp_us':v['camera_time_us'],'world_from_camera':np.asarray(v['world_from_camera']).tolist(),'world_from_ego_camera':mat(e['rotation'],e['translation']).tolist(),'intrinsics_original':c['camera_intrinsic'],'original_wh':list(Image.open(v['image_path']).size)})
  if idx==1:first=views[0]['timestamp_us']
 assert abs(views[6]['timestamp_us']-first-500000)<120000
 (R/'inputs').mkdir(exist_ok=True)
 (R/'inputs'/f'{name}.json').write_text(json.dumps({'scene':name,'log':s['log_id'],'role':'BUILD_RGB_ONLY','views':views},indent=2))
 print(name,len(views),flush=True)
print('REGISTERED',R,flush=True)
