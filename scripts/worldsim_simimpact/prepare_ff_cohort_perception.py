"""补齐已冻结六日志原生感知矩阵，复用所有已完成推理，保留额外信息边界。"""
import argparse,json,time,shutil,subprocess
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion

ROOT=Path('/root/autodl-tmp/runs/worldsim_simimpact')
I=ROOT/'WS-SIM-INTERACTION-01/20260915-r1'
N=ROOT/'WS-SIM-NATIVE-LIDAR-01/20260915-r1'
F=ROOT/'WS-SIM-FF-PERCEPTION-01/20260915-r1'
O=ROOT/'WS-SIM-FF-COHORT-PERCEPTION-01/20260915-r1'
M=N/'data/v1.0-trainval'
METHODS=['vggt','omega512','dvgt1','pi3x']
ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','extract','prepare']);a=ap.parse_args()
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def table(name):return {x['token']:x for x in json.loads((M/f'{name}.json').read_text())}
def current_raw(name):
    for base in [N/'data',I/'raw_initial',O/'raw_extra']:
        if (base/name).is_file():return base/name
    return O/'raw_extra'/name

if a.mode=='register':
    if O.exists():raise RuntimeError(f'Preserve {O}')
    O.mkdir(parents=True);shutil.copy2(__file__,O/'source_snapshot.py')
    selection=json.loads((I/'selection_summary.json').read_text())
    reg={'task_id':'WS-SIM-FF-COHORT-PERCEPTION-01','run_id':O.name,'seed':20260915,
      'scenes':list(selection),'selection':selection,
      'scope':'Complete the six-log already exposed interaction cohort for frozen CenterPoint. Four logs have not yet been evaluated by this detector. No independent test claim.',
      'inputs':'Four official six/twelve-view predictions already computed; common six output depth maps; fixed known calibration and one BUILD LiDAR global scale. Only current mesh scan changes; identical nine causal real sweeps and current-ray real intensity are extra information. fill_missing restores originally missing current ranges with real data.',
      'conditions':['real']+[f'{m}_{v}_{c}' for m in METHODS for v in ['six','twelve'] for c in ['full','fill_missing']],
      'evaluation':'Unchanged native CenterPoint score0.3/0.5, class-correct center2m and BEV IoU0.5. Front0..32m/lateral16m, >=5 annotated current LiDAR points. Keep every eligible object and positive/negative result; counts are observations, not AP/NDS.',
      'candidate_rule':'Start with targets matched on real inputs at score0.5 in both center and IoU. Report losses at both score0.3 and0.5, and continuous same-class box errors. Only new-log candidates surviving fill_missing in both input budgets merit causal audit. Prioritize metadata-selected lead actor, then robust class-correct center loss, then IoU loss; ties scene/instance order. At most two new targets, no geometric severity selection.',
      'planner_boundary':'CenterPoint output is not fed into the previously used TransFuser. Detection mismatch or TTC computed from boxes is not an executed planner or closed-loop consequence. No new safety claim.',
      'stop_rule':'One complete existing six-log matrix. Do not retune thresholds, ROIs, or reconstruction to force a problem. No new model fitting or geometry forwards. Existing Omega/Pi3X scene0004 local audits remain closed.',
      'failure_ledger_refs':['V74-H2-F16','V74-H2-F19','V74-H2-F20'],
      'failure_ledger_delta':'pending','human_verdict':None,'start_unix':time.time(),'prepared':False}
    dump(O/'registration.json',reg)
    sdall=table('sample_data');samples=table('sample');sd={};views={};seqs={}
    for scene in selection:
        views[scene]=json.loads((I/f'inputs/{scene}.json').read_text())['views']
        token=views[scene][0]['sample_token']
        current=next(x for x in sdall.values() if x['sample_token']==token and x['is_key_frame'] and '/LIDAR_TOP/' in x['filename'])
        seq=[]
        for j in range(10):
            sd[current['token']]=current;seq.append(current['token'])
            if j<9:current=sdall[current['prev']]
        seqs[scene]=seq
    ego_all=table('ego_pose');cal_all=table('calibrated_sensor')
    et={x['ego_pose_token'] for x in sd.values()};ct={x['calibrated_sensor_token'] for x in sd.values()}
    dump(O/'metadata.json',{'sample_data':sd,'ego_pose':{k:ego_all[k] for k in et},'calibrated_sensor':{k:cal_all[k] for k in ct},'sequences':seqs,'views':views})
    missing=sorted({x['filename'] for x in sd.values() if not current_raw(x['filename']).is_file()})
    (O/'raw_members.txt').write_text('\n'.join(missing)+'\n')
    reg.update(missing_raw_files=len(missing),expected_new_detector_forwards=82,reused_detector_forwards=20,total_matrix_forwards=102)
    dump(O/'registration.json',reg);print('REGISTERED',reg['scenes'],'missing raw',len(missing),flush=True)
elif a.mode=='extract':
    targets=(O/'raw_members.txt').read_text().splitlines();out=O/'raw_extra';out.mkdir(exist_ok=True)
    archive='/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval01_blobs.tgz'
    if not targets:code=0
    else:code=subprocess.run(['tar','-xzf',archive,'-C',str(out),'--no-recursion','-T',str(O/'raw_members.txt')]).returncode
    result={'archive':archive,'tar_returncode':code,'requested':len(targets),'missing':[x for x in targets if not current_raw(x).is_file()]}
    dump(O/'extraction_result.json',result);print(result,flush=True)
    assert not result['missing']
else:
    if (O/'input_manifest.json').exists():raise RuntimeError('Preserve prepared inputs')
    reg=json.loads((O/'registration.json').read_text());meta=json.loads((O/'metadata.json').read_text())
    def pose(tab,t):
        x=meta[tab][t];p=np.eye(4);p[:3,:3]=Quaternion(x['rotation']).rotation_matrix;p[:3,3]=x['translation'];return p
    def world(sd):return pose('ego_pose',sd['ego_pose_token'])@pose('calibrated_sensor',sd['calibrated_sensor_token'])
    frames=[];checks=[]
    for scene in reg['scenes']:
        seq=[meta['sample_data'][k] for k in meta['sequences'][scene]];sd=seq[0];inv=np.linalg.inv(world(sd))
        raw=np.fromfile(current_raw(sd['filename']),np.float32).reshape(-1,5);r=np.linalg.norm(raw[:,:3],axis=1)
        valid=np.isfinite(raw[:,:3]).all(1)&(r>1)&(r<80);current=raw[valid,:4].copy()
        z=np.load(I/f'lidar_policy/scans/{scene}/real.npz');T=inv@np.asarray(meta['views'][scene][0]['world_from_ego_camera'])
        xyz=z['points']@T[:3,:3].T+T[:3,3]
        assert len(xyz)==len(current)
        err=float(np.max(abs(xyz-current[:,:3])));assert err<.001
        past=[]
        for p in seq[1:]:
            rawp=np.fromfile(current_raw(p['filename']),np.float32).reshape(-1,5)[:,:4].copy()
            rawp=rawp[~((abs(rawp[:,0])<1)&(abs(rawp[:,1])<1))];rel=inv@world(p)
            rawp[:,:3]=rawp[:,:3]@rel[:3,:3].T+rel[:3,3]
            past.append(np.c_[rawp,np.full(len(rawp),(sd['timestamp']-p['timestamp'])/1e6)])
        history=np.concatenate(past).astype(np.float32);dest=O/scene;dest.mkdir();arrays={};check={'scene':scene,'current_points':len(current),'alignment_max_abs_m':err,'history_points':len(history),'conditions':{}}
        def save(name,points):
            points=np.concatenate([np.c_[points,np.zeros(len(points))],history]).astype(np.float32)
            path=dest/f'{name}.npz';np.savez_compressed(path,points=points);arrays[name]=str(path)
        if scene not in ['scene-0004','scene-0061']:save('real',current)
        for method in METHODS:
            for variant in ['six','twelve']:
                if scene in ['scene-0004','scene-0061'] and variant=='six':continue
                f=np.load(I/f'lidar_policy/scans/{scene}/{method}_{variant}_build_scale.npz')['first_range'];finite=np.isfinite(f)
                pt=z['origin']+z['directions']*f[:,None];pt=pt@T[:3,:3].T+T[:3,3]
                save(f'{method}_{variant}_full',np.c_[pt[finite],current[finite,3]])
                if not (scene=='scene-0004' and variant=='twelve' and method in ['omega512','pi3x']):
                    p=current.copy();p[finite,:3]=pt[finite];save(f'{method}_{variant}_fill_missing',p)
                check['conditions'][f'{method}_{variant}']={'returned':int(finite.sum()),'missing':int((~finite).sum())}
        checks.append(check);frames.append({'scene':scene,'sample_token':sd['sample_token'],'points':arrays,'history_tokens':[x['token'] for x in seq[1:]]})
        print('PREPARED',scene,len(arrays),'conditions',flush=True)
    assert sum(len(f['points']) for f in frames)==82
    reg.update(prepared=True,preparation_end_unix=time.time(),alignment_checks=checks)
    dump(O/'registration.json',reg);dump(O/'input_manifest.json',{'registration':reg,'frames':frames})
