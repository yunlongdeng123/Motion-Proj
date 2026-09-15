"""以自然交互和日志独立性冻结原生仿真候选；不读取重建输出。"""
import json,time,collections,shutil
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import Polygon
P=Path('/root/autodl-tmp');M=P/'data/worldsim_v4/drivestudio_raw_trainval/v1.0-trainval'
O=P/'runs/worldsim_simimpact/WS-SIM-FRESH-NATIVE-01/20260915-r1'
if O.exists():raise RuntimeError(f'Preserve {O}')
O.mkdir(parents=True);shutil.copy2(__file__,O/'selection_source_snapshot.py')
reg={'task_id':'WS-SIM-FRESH-NATIVE-01','run_id':O.name,'seed':20260915,
    'purpose':'New-log native RGB+3D-LiDAR simulation, baseline-first before any new scene fitting. No synthetic perturbation or geometric-error selection.',
    'selection':{'candidate_scene_range':'scene-0001..scene-0100 (the locally mounted first public shard discovery window)',
        'max_distinct_logs':6,'order':'scene name then first qualifying chronological keyframe, starting at index2',
        'future_keyframes_available':16,'initial_ego_speed_mps':[2,15],
        'initial_actor_center_forward_m':[4,30],'initial_actor_lateral_abs_max_m':6,'initial_actor_min_lidar_points':8,
        'actor_categories':'vehicle or human pedestrian','recorded_ego_actor_min_4s_box_gap_m':[.8,5],
        'interaction':'Within4s, target lateral position in initial ego frame changes >=1.5m OR recorded ego forward speed falls by >=2m/s.',
        'ego_box':'Same Pacifica adapter envelope as prior PDM: length5.176,width2.297,rear-axle-to-center1.461m. Metadata screening only.'},
    'exclusion':'Every log containing an already evaluated scene in initial simulation cohort, six-log interaction cohort, or V74 official-mainfigure cohort. Excludes shared logs, not merely scene names. Foundation-model or detector training overlap is not ruled out.',
    'stage1_real_baseline':'Eight logged timestamps per selected window, one unchanged TransFuser seed0 and official PDM vehicle tracking; no reconstruction outputs. All timestamps and failures retained.',
    'fit_admission':'First scene in frozen order with no real-baseline executed annotated-actor overlap across eight starts, mean4s ADE<=1m and mean4s FDE<=2m. Assess data availability before any fit. At most one full30001-step SplatAD fit in this batch; no automatic second fit.',
    'future_stage':'Only a qualified source justifies complete scene sensor fitting/validation and actual feedback loops. Geometry diagnostics alone do not admit a Hero candidate.',
    'stop_rule':'Finite6-log window; if no real baseline qualifies, do not reinterpret domain-transfer failures as reconstruction harm. Do not adjust thresholds or synthesize more extreme actors.',
    'information':'Future ego and annotations are metadata selection/evaluation only; policy receives current images/LiDAR and backward-derived state. Scene fitting uses extra full RGB+LiDAR/actor observations and is separately labelled.',
    'failure_ledger_refs':['V74-H2-F18','V74-H2-F19','V74-H2-F20','V74-H2-F21'],
    'failure_ledger_delta':'pending','human_verdict':None,'start_unix':time.time(),'selection_completed':False}
(O/'registration.json').write_text(json.dumps(reg,indent=2))
scenes=sorted(json.loads((M/'scene.json').read_text()),key=lambda x:x['name']);prior=set()
for root in [P/'runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1',P/'runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1',P/'runs/worldsim_v74_h2/WS-V74-MAINFIG-01/20260915-first-return-r1']:
    prior.update(p.stem for p in (root/'inputs').glob('*.json') if p.stem.startswith('scene-'))
    for file in ['selection_summary.json','raw_source_availability.json']:
        if (root/file).exists():prior.update(k for k in json.loads((root/file).read_text()) if k.startswith('scene-'))
assert {'scene-0004','scene-0061','scene-0013'}<=prior
excluded={s['log_token'] for s in scenes if s['name'] in prior}
reg.update(excluded_scenes=sorted(prior),excluded_log_tokens=sorted(excluded))
(O/'registration.json').write_text(json.dumps(reg,indent=2))
scenes=[s for s in scenes if s['name']<='scene-0100' and s['log_token'] not in excluded]
samples={s['token']:s for s in json.loads((M/'sample.json').read_text())};seqs={};tokens=set()
for scene in scenes:
    seq=[];t=scene['first_sample_token']
    while t:seq.append(samples[t]);tokens.add(t);t=samples[t]['next']
    seqs[scene['name']]=seq
sd=collections.defaultdict(list)
for d in json.loads((M/'sample_data.json').read_text()):
    if d['is_key_frame'] and d['sample_token'] in tokens:sd[d['sample_token']].append(d)
ego={e['token']:e for e in json.loads((M/'ego_pose.json').read_text())}
instances={x['token']:x for x in json.loads((M/'instance.json').read_text())};categories={x['token']:x['name'] for x in json.loads((M/'category.json').read_text())}
ann=collections.defaultdict(dict)
for a in json.loads((M/'sample_annotation.json').read_text()):
    if a['sample_token'] in tokens and categories[instances[a['instance_token']]['category_token']].startswith(('vehicle.','human.pedestrian.')):ann[a['sample_token']][a['instance_token']]=a
def T(x):
    m=np.eye(4);m[:3,:3]=Quaternion(x['rotation']).rotation_matrix;m[:3,3]=x['translation'];return m
def poly(transform,length,width,offset=0.):
    xy=np.array([[length/2+offset,width/2,0],[length/2+offset,-width/2,0],[-length/2+offset,-width/2,0],[-length/2+offset,width/2,0]])@transform[:3,:3].T+transform[:3,3];return Polygon(xy[:,:2])
selected={};used=set()
for scene in scenes:
    if scene['log_token'] in used:continue
    seq=seqs[scene['name']];fronts=[next(d for d in sd[s['token']] if '/CAM_FRONT/' in d['filename']) for s in seq];poses=[T(ego[d['ego_pose_token']]) for d in fronts];times=np.array([d['timestamp'] for d in fronts])/1e6
    for i in range(2,len(seq)-16):
        inv=np.linalg.inv(poses[i]);past=inv@poses[i-1];speed=-past[0,3]/(times[i]-times[i-1])
        if not 2<=speed<=15:continue
        future_speeds=[((inv@poses[j])[:2,3]-(inv@poses[j-1])[:2,3])[0]/(times[j]-times[j-1]) for j in range(i+1,i+9)]
        braking=speed-min(future_speeds);qualified=[]
        for instance,a in sorted(ann[seq[i]['token']].items()):
            local=inv@T(a);x,y=local[:2,3]
            if not (4<=x<=30 and abs(y)<=6 and a['num_lidar_pts']>=8):continue
            future=[ann[s['token']].get(instance) for s in seq[i:i+9]]
            if any(a is None for a in future):continue
            gaps=[];lateral=[]
            for j,a in enumerate(future):
                w,l,h=a['size'];actor=T(a);gaps.append(poly(poses[i+j],5.176,2.297,1.461).distance(poly(actor,l,w)));lateral.append((inv@actor)[1,3])
            gap=min(gaps);change=max(lateral)-min(lateral)
            if not (.8<=gap<=5 and (change>=1.5 or braking>=2)):continue
            qualified.append({'instance':instance,'category':categories[instances[instance]['category_token']],'center_forward_m':float(x),'center_lateral_m':float(y),'lidar_points':future[0]['num_lidar_pts'],'min_recorded_box_gap_m':float(gap),'lateral_span_m':float(change),'ego_speed_drop_mps':float(braking)})
        if not qualified:continue
        chosen=seq[i:i+17];needed={s['token'] for s in seq[i-2:i+17]};records=[{k:d[k] for k in ['token','sample_token','timestamp','filename','calibrated_sensor_token','ego_pose_token']} for t in needed for d in sd[t]]
        selected[scene['name']]={'log_token':scene['log_token'],'original_start_index':i,'samples':[s['token'] for s in chosen],'past_samples':[s['token'] for s in seq[i-2:i]],'records':records,'metadata_selection':{'speed_mps':float(speed),'targets':qualified}}
        used.add(scene['log_token']);print('SELECTED',scene['name'],i,round(speed,2),qualified,flush=True);break
    if len(selected)==6:break
assert not ({x['log_token'] for x in selected.values()}&excluded)
(O/'raw_source_availability.json').write_text(json.dumps(selected,indent=2));(O/'selection_summary.json').write_text(json.dumps({k:{x:v[x] for x in ['log_token','original_start_index','metadata_selection']} for k,v in selected.items()},indent=2))
reg.update(scenes=list(selected),selection_completed=True,selection_end_unix=time.time(),selected_count=len(selected))
(O/'registration.json').write_text(json.dumps(reg,indent=2));print('FROZEN',len(selected),'new-to-this-evaluation logs',flush=True)
