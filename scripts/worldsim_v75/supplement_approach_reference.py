"""有限参考补证：当前部分扫描＋之前两次扫描，点时间均不得越过生成起点。"""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from prepare_argoverse import poses,POS,QUAT,ROOT
from readout_natural import read_center

OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-APPROACH-CLOSEDLOOP-01/20260920-association-r2')


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=OUT)
    parser.add_argument('--task-id',default='WS-V75-APPROACH-REFERENCE-01')
    args=parser.parse_args();OUT=args.run_dir
    read=OUT/'reconstruction'; old=json.loads((read/'readout_ray_control_result.json').read_text()); p=json.loads((read/'protocol.json').read_text())
    dest=read/'reference_supplement_result.json'; assert not dest.exists()
    base=json.loads((Path(p['base_run'])/'input_manifest.json').read_text()); raw=ROOT/base['log_id']; cutoff=p['cutoff_ns']; origin=np.array(base['city_origin'])
    files=sorted(x for x in (raw/'sensors/lidar').glob('*.feather') if int(x.stem)<=cutoff)[-3:]
    assert len(files)==3 and str(files[-1])==p['lidar_anchor_file']
    protocol={'task_id':args.task_id,'run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'source_readout':str(read/'readout_ray_control_result.json'),'files':list(map(str,files)),'cutoff_ns':cutoff,'target':p['target'],
        'role':f'post hoc missing-reference supplementation; original{old["readouts"]["reference_lidar"]["n"]}-point rejection preserved; no new DVGT input or model selection',
        'information':'two additional past LiDAR sweeps, known ego poses and GT actor masks; no GT translation motion compensation',
        'rules':{'max_target_center_displacement_m':.2,'minimum_core_points':6,'reference_center_error_max_m':.5,'same_face_retention_min':.8},
        'stop':'exactly3 sweeps including original partial scan; no further scan/threshold expansion',
        'human_verdict':None,'failure_ledger_delta':'none'}
    (read/'reference_supplement_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    ego_df=pd.read_feather(raw/'city_SE3_egovehicle.feather'); annotations=pd.read_feather(raw/'annotations.feather')
    camera=np.array(p['views'][0]['camera_world']); K=np.array(p['views'][0]['K_network']); dims=np.array(p['target_dimensions_oracle']); rotation=camera[:3,:3].T@np.array(p['target_rotation_world_oracle'])
    observed=np.array(p['bbox_network']); center=(observed[:2]+observed[2:])/2; half=(observed[2:]-observed[:2])*.3
    scans=[]; all_points=[]; centers=[]; all_uv=[]
    for file in files:
        stamp=int(file.stem); sweep=pd.read_feather(file); allowed=stamp+sweep.offset_ns.to_numpy(np.int64)<=cutoff
        E=poses(ego_df,[stamp])[0]; E[:3,3]-=origin
        target=annotations[(annotations.timestamp_ns==stamp)&(annotations.track_uuid==p['target'])]
        assert len(target)==1; target=target.iloc[0]
        rot=E[:3,:3]@Rotation.from_quat(target[QUAT].to_numpy(float)).as_matrix(); c=E[:3,:3]@target[POS].to_numpy(float)+E[:3,3]; centers.append(c)
        points=sweep[['x','y','z']].to_numpy(float)[allowed]@E[:3,:3].T+E[:3,3]
        inside=np.all(abs((points-c)@rot)<=dims/2+.15,axis=1); pc=(points-camera[:3,3])@camera[:3,:3]
        pixels=pc@K.T; uv=pixels[:,:2]/np.where(abs(pixels[:,2:])>1e-8,pixels[:,2:],np.nan)
        core=inside&(pc[:,2]>.2)&np.all(uv>=center-half,1)&np.all(uv<=center+half,1)
        all_points.extend(pc[core]); all_uv.extend(uv[core]); scans.append({'file':str(file),'allowed_points':int(allowed.sum()),'excluded_future_points':int((~allowed).sum()),'target_box_points':int(inside.sum()),'core_points':int(core.sum()),'center_world_reference':c.tolist()})
    points=np.array(all_points); uv=np.array(all_uv); displacement=float(np.max(np.linalg.norm(np.array(centers)-centers[-1],axis=1)))
    ordinary=np.array(old['readouts']['ordinary_bbox']['center_camera'])
    reference=read_center(np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T,points[:,2],ordinary,rotation,dims) if len(points) else None
    if reference:
        c=np.array(reference['center_camera']); world=c@camera[:3,:3].T+camera[:3,3]; gt=np.array(old['reference_only_gt_center_world'])
        reference.update(center_world=world.tolist(),offset_world_m=(world-gt).tolist(),center_error_m=float(np.linalg.norm(world-gt)))
    model=old['readouts']['dvgt_metric']; model_ok=model is not None and model['n']>=20 and model['face_retention']>=.8 and model['radial_scale']>0
    reference_ok=reference is not None and reference['n']>=6 and reference['face_retention']>=.8 and reference['center_error_m']<=.5 and displacement<=.2
    result={'status':'complete','target':p['target'],'source_readout':protocol['source_readout'],'scans':scans,
            'reference_readout':reference,'target_center_displacement_m':displacement,'reference_ok':bool(reference_ok),
            'generation_admitted':bool(reference_ok and model_ok),'raw_model_readout_unchanged':True,'scale_readout_unchanged':True,
            'role':protocol['role'],'human_verdict':None,'failure_ledger_delta':'none'}
    dest.write_text(json.dumps(result,indent=2)+'\n'); np.savez(read/'reference_supplement_points.npz',camera_points=points,uv=uv)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
