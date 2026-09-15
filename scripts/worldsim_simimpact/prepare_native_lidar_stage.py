"""登记原生新位姿传感器闭环，并提取两个明确场景的完整日志资产。"""
import json,subprocess
from pathlib import Path
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1');M=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval/v1.0-trainval')
R.mkdir(parents=True,exist_ok=True);(R/'scripts').mkdir(exist_ok=True)
reg={'task_id':'WS-SIM-NATIVE-LIDAR-01','run_id':'20260915-r1','parent_goal':'Actual reconstruction impact in end-to-end driving simulation',
 'purpose':'Establish native SplatAD camera+LiDAR novel-pose sensor loop with official driving components and metric log reference; not another extreme first-return perturbation search',
 'scenes':['scene-0004','scene-0061'],'selection':'0004 is first lexicographic scene in the completed ordinary interaction cohort; 0061 is its sole nominal contact candidate. Both exposed discovery sources, not independent confirmation.',
 'official_renderer':'georghess/neurad-studio SplatAD with carlinds/splatad CUDA kernels',
 'asset_fitting':'Official scene-specific RGB+LiDAR reconstruction, with known calibration and annotated dynamics. This supplies a strong metric simulator reference; not a feed-forward model training or same-information ranking.',
 'initial_fit_budget':'Official 30001-step schedule, checkpointed; first establish one scene, assess actual validation and resource cost before fitting second',
 'model_and_policy_scope':'Existing official geometry checkpoints, official RGB+LiDAR TransFuser, official vehicle controller/dynamics; all adapters and extra information reported',
 'required_evidence':['Native RGB and LiDAR change with actually executed ego pose','Same-run observer calibration and independent log sensor agreement','Frozen sensor protocol and physics query semantics','Closed-loop native baseline before geometric intervention','Residual harm separated from FOV, scale, RGB artifacts and near-boundary vehicle-box effects'],
 'counterexamples_retained':['V74-H2-F13','V74-H2-F14','V74-H2-F15','V74-H2-F16'],
 'resource':'Single RTX3090 24GiB, 14CPU/90GiB cgroup; no old H2 restart','seed':20260915,'human_verdict':None,'failure_ledger_delta':'pending'}
p=R/'registration.json'
if not p.exists():p.write_text(json.dumps(reg,indent=2))
scenes={s['name']:s for s in json.loads((M/'scene.json').read_text()) if s['name'] in reg['scenes']};samples={s['token']:s for s in json.loads((M/'sample.json').read_text())};scene_tokens={}
for name,s in scenes.items():
 seq=[];t=s['first_sample_token']
 while t:seq.append(t);t=samples[t]['next']
 scene_tokens[name]=set(seq)
tokens=set.union(*scene_tokens.values());sd=[d for d in json.loads((M/'sample_data.json').read_text()) if d['sample_token'] in tokens and ('/CAM_' in d['filename'] or '/LIDAR_TOP/' in d['filename'])]
members=sorted({d['filename'] for d in sd});data=R/'data';data.mkdir(exist_ok=True)
link=data/'v1.0-trainval'
if not link.exists():link.symlink_to(M,target_is_directory=True)
# nuScenes metadata references global map images; resolve them to existing data when available.
for map_root in [M.parent/'maps',Path('/root/autodl-tmp/data/worldsim_v4/nuscenes_meta/maps')]:
 if map_root.exists() and not (data/'maps').exists():(data/'maps').symlink_to(map_root,target_is_directory=True)
(R/'raw_members.txt').write_text('\n'.join(members)+'\n')
(R/'source_manifest.json').write_text(json.dumps({'scenes':reg['scenes'],'keyframes':{n:len(s) for n,s in scene_tokens.items()},'records_per_scene':{n:sum(d['sample_token'] in s for d in sd) for n,s in scene_tokens.items()},'members':members},indent=2))
print('EXTRACT',len(members),'members',flush=True)
archive='/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval01_blobs.tgz'
proc=subprocess.run(['tar','-xzf',archive,'-C',str(data),'--no-recursion','-T',str(R/'raw_members.txt')])
missing=[f for f in members if not (data/f).exists()];result={'archive':archive,'returncode':proc.returncode,'requested':len(members),'missing':missing}
(R/'extraction_result.json').write_text(json.dumps(result,indent=2));print('EXTRACTION',len(members)-len(missing),'present',len(missing),'missing',flush=True)
