"""从完整 SplatAD 资产生成检测器所需的真实时序十帧扫描，保留两种原生距离读出。"""
import argparse,json,time,sys,copy
from pathlib import Path
import numpy as np
import torch
from native_splatad_bridge import NativeSplatADBridge
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
D=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-DETECTOR-01/20260915-r1')
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--checkpoint-dir',type=Path,required=True);ap.add_argument('--pattern',choices=['fixed','logged'],default='fixed');a=ap.parse_args()
assert json.loads((N/'fits/scene-0004/splatad/native-r2/fit_manifest.json').read_text())['completed']
if a.out.exists():raise RuntimeError(f'Preserve existing export: {a.out}')
a.out.mkdir(parents=True)
real=json.loads((D/'real_scene0004_r1/summary.json').read_text());torch.manual_seed(20260915);torch.set_num_threads(4)
b=NativeSplatADBridge(N/'fits/scene-0004/splatad/native-r2',a.checkpoint_dir,30000,real[0]['sample_token'])
reg={'task_id':'WS-SIM-NATIVE-DETECTOR-01','run_id':a.out.name,'checkpoint_step':30000,
     'role':'Full-budget native SplatAD scans at logged LiDAR poses/times for the existing eight-frame CenterPoint reference.',
     'scope':'Not a feed-forward reconstruction condition; full-scene RGB+LiDAR asset, known poses/dynamics and learned intensity/drop are extra information. Natural sensor differences precede geometry attribution.',
     'timing':'Current scan plus the same nine historical timestamps as the real baseline; every scan rendered with its causal velocity.',
     'pattern':b.lidar_check,'readouts':['raw','median'],'intensity':'Native sigmoid output multiplied by255, matching nuScenes reflectance units.',
     'failure_ledger_refs':['V74-H2-F18'],'failure_ledger_delta':'pending','human_verdict':None,'completed':False,'start_unix':time.time()}
reg['scan_pattern_control']=a.pattern
if a.pattern=='logged':reg['pattern_control_information']='Official per-scan measured/imputed directions and validity, an extra-information logged-sensor diagnostic. Valid depth placeholders set to1; model eval does not consume actual GT ranges. Not a counterfactual closed-loop firing pattern.'
(a.out/'registration.json').write_text(json.dumps(reg,indent=2))
def set_logged_pattern(sd):
    target=sd['timestamp']/1e6-b.time_offset;options=[]
    for split,dataset in [('train',b.dm.train_lidar_dataset),('eval',b.dm.eval_lidar_dataset)]:
        times=dataset.lidars.times.detach().cpu().numpy().reshape(-1);idx=int(abs(times-target).argmin())
        options.append((abs(times[idx]-target),split,dataset,idx))
    diff,split,dataset,idx=min(options,key=lambda x:x[0]);assert diff<1e-4
    b.lidar=copy.deepcopy(dataset.lidars[idx:idx+1]).to(b.device)
    data=copy.deepcopy((b.dm.cached_lidar_train if split=='train' else b.dm.cached_lidar_eval)[idx]);data['is_eval']=True
    b.lidar.metadata['lidar_idx']=idx;b.dm._add_metadata(b.lidar,data,len(b.dm.train_dataset if split=='train' else b.dm.eval_dataset))
    b.raster=data['raster_pts'].clone();b.pattern_valid=b.raster[...,2]>0
    b.raster[...,2]=b.pattern_valid.float();b.raster[...,4]=0
    angles=torch.deg2rad(b.raster[...,:2]);b.directions=torch.stack([angles[...,1].cos()*angles[...,0].cos(),angles[...,1].cos()*angles[...,0].sin(),angles[...,1].sin()],-1)
cache={};summary=[]
for row in real:
    current=b.nusc.get('sample_data',row['lidar_token']);current_world=b.ego_pose(current)@b.extrinsics['LIDAR_TOP'];inv=np.linalg.inv(current_world)
    parts={k:[] for k in ['raw','median']}
    for j,source in enumerate(row['sweeps']):
        token=source['token'];sd=b.nusc.get('sample_data',token)
        if token not in cache:
            if a.pattern=='logged':set_logged_pattern(sd)
            ego=b.ego_pose(sd);v,_,w=b.status_at(sd);scan=b.lidar_scan(ego,sd['timestamp']/1e6,v,w)
            cache[token]={'ego':ego,**{k:scan[k] for k in ['raw','median','intensity']}}
        scan=cache[token];local_inv=np.linalg.inv(b.extrinsics['LIDAR_TOP']);relative=inv@scan['ego']
        for k in parts:
            p=scan[k];intensity=scan['intensity']*255
            local=p@local_inv[:3,:3].T+local_inv[:3,3]
            keep=np.isfinite(p).all(1)
            if j:keep &= ~((abs(local[:,0])<1)&(abs(local[:,1])<1))
            p=p[keep]@relative[:3,:3].T+relative[:3,3]
            parts[k].append(np.c_[p,intensity[keep],np.full(len(p),source['age_s'])])
    counts={}
    for k in parts:
        p=np.concatenate(parts[k]).astype(np.float32);counts[k]=len(p)
        np.savez_compressed(a.out/f'{k}_{row["index"]:02d}.npz',points=p)
    summary.append({'index':row['index'],'sample_token':row['sample_token'],'lidar_token':row['lidar_token'],'point_counts':counts})
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2));print('NATIVE_DETECTOR_SCANS',row['index'],counts,flush=True)
reg.update(completed=True,end_unix=time.time(),native_lidar_renders=len(cache));(a.out/'registration.json').write_text(json.dumps(reg,indent=2))
print('NATIVE_DETECTOR_SCANS_COMPLETED',a.out,flush=True)
