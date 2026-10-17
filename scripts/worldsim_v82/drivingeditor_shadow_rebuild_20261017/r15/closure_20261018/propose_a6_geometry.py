"""R15-A06：绕开失效的渲染候选入口，但不跳过真实3D正证据和保体否决。CPU。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--audit',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();base=Path(a.base);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    saved=torch.load(base/'paused_checkpoint_004000_20261017/checkpoint.pt',map_location='cpu',weights_only=False)
    state=saved['state'];vals=lambda key:state['field.'+key].numpy()
    owner=vals('owner');all_rows=np.flatnonzero(owner==6);xyz=vals('xyz')[all_rows].astype(np.float64)
    scales=np.exp(vals('log_scales')[all_rows].astype(np.float64)).clip(saved['config']['scale_min_m'],saved['config']['scale_max_m'])
    basis=Rotation.from_quat(vals('quaternion')[all_rows][:,[1,2,3,0]]).as_matrix()
    body=dict(np.load(base/'body_bank_a06.npz'));raw=dict(np.load(base/'source_candidate_evidence_a06.npz'))
    keep=raw['road_positive'];old_road_count=int(keep.sum())
    road={k:v[keep] for k,v in raw.items()}
    data=json.loads((base/'data/manifest.json').read_text())
    # 旧a6局部池没有road-positive，不等于全局独立实测道路库为空。
    # 用各实测来源时刻的真实a6轨迹，把世界道路证据投回canonical坐标。
    world=dict(np.load(base/'actual_world_road_support.npz'))
    trajectory=dict(np.load(base/'data/trajectory.npz'))
    source=world['source_frame'].astype(np.int64)
    use=(world['support_frames']>=3)&trajectory['valid'][source,6]
    use&=np.asarray([data['frames'][int(i)]['split']=='train' for i in source])
    source=source[use];world_xyz=world['xyz'][use];pose=trajectory['poses'][source,6]
    local=np.einsum('ni,nij->nj',world_xyz-pose[:,:3,3],pose[:,:3,:3])
    augmented=dict(xyz=local,source_frame=source,source_raw_row=world['source_raw_row'][use],source_world=world_xyz)
    road={k:np.concatenate((road[k],augmented[k]),axis=0) for k in augmented}
    for bank in [body,road]:
        if np.any(bank['source_raw_row']%5==0) or any(data['frames'][int(i)]['split']!='train' for i in np.unique(bank['source_frame'])):
            raise ValueError('非BUILD证据进入几何候选')
    bt=cKDTree(body['xyz']);rt=cKDTree(road['xyz']);bd=bt.query(xyz,workers=2)[0];rd=rt.query(xyz,workers=2)[0]
    evidence=dict(np.load(Path(a.audit)/'current_contributions.npz'));ev=evidence['sums'];hits=evidence['hits']
    broad=np.flatnonzero(rd<=3*scales.max(-1))
    counts=dict(actor_points=len(all_rows),body_measurements=len(body['xyz']),
                original_actor_pool_road_measurements=old_road_count,
                global_measured_road_entries=len(local),road_measurements=len(road['xyz']),
                broad_3sigma_road=len(broad),road_repeat=0,mixed_repeat=0,split_separable=0,
                road_only_no_body4sigma=0,road_only_render_body_veto=0,road_only_anchor006=0)
    split_rows=[];directions=[];cuts=[];split_meta=[];road_rows=[];anchors=[];road_meta=[]
    def volume(bank,tree,j,sigma):
        ids=np.asarray(tree.query_ball_point(xyz[j],sigma*float(scales[j].max())),np.int64)
        if not len(ids):return ids
        white=(bank['xyz'][ids]-xyz[j])@basis[j]/scales[j]
        return ids[(white*white).sum(-1)<sigma*sigma]
    start=time.monotonic()
    for j in broad:
        ri=volume(road,rt,j,3.)
        if len(np.unique(road['source_frame'][ri]))<3:continue
        counts['road_repeat']+=1;bi=volume(body,bt,j,3.)
        if len(np.unique(body['source_frame'][bi]))>=3:
            counts['mixed_repeat']+=1
            wb=(body['xyz'][bi]-xyz[j])@basis[j]/scales[j]
            wr=(road['xyz'][ri]-xyz[j])@basis[j]/scales[j]
            normal=wb.mean(0)-wr.mean(0);length=np.linalg.norm(normal)
            if length<.25:continue
            normal/=length;pb=wb@normal;pr=wr@normal;cut=.5*(np.median(pb)+np.median(pr))
            if abs(cut)>1.5 or (pb>cut).mean()<.75 or (pr<cut).mean()<.75:continue
            counts['split_separable']+=1;row=int(all_rows[j]);split_rows.append(row);directions.append(normal);cuts.append(float(cut))
            split_meta.append(dict(row=row,body_source_frames=np.unique(body['source_frame'][bi]).tolist(),
                road_source_frames=np.unique(road['source_frame'][ri]).tolist(),body_right_fraction=float((pb>cut).mean()),road_left_fraction=float((pr<cut).mean()),
                rendered_own_mass=float(ev[row,0]),rendered_road_mass=float(ev[row,3])))
        else:
            if len(volume(body,bt,j,4.)):continue
            counts['road_only_no_body4sigma']+=1;row=int(all_rows[j])
            own_veto=hits[row,0]>=3 and ev[row,0]>=.5
            other_veto=hits[row,2]>=3 and ev[row,2]-ev[row,0]>=.5
            if own_veto or other_veto:
                counts['road_only_render_body_veto']+=1;continue
            nearest=int(ri[np.argmin(np.linalg.norm(road['xyz'][ri]-xyz[j],axis=-1))])
            if np.linalg.norm(road['xyz'][nearest]-xyz[j])>.06:continue
            counts['road_only_anchor006']+=1;road_rows.append(row);anchors.append(road['source_world'][nearest])
            road_meta.append(dict(row=row,source_frame=int(road['source_frame'][nearest]),
                source_raw_row=int(road['source_raw_row'][nearest]),source_world=road['source_world'][nearest].tolist(),
                repeated_source_frames=np.unique(road['source_frame'][ri]).tolist()))
    np.savez_compressed(out/'a6_proposals.npz',split_rows=np.asarray(split_rows,np.int64),
        directions=np.asarray(directions,np.float32).reshape(-1,3),cuts=np.asarray(cuts,np.float32),
        road_rows=np.asarray(road_rows,np.int64),anchors=np.asarray(anchors,np.float32).reshape(-1,3))
    receipt=dict(run_id='R15-A06-MEASURED-3D-PROPOSALS',status='COMPLETE',seconds=time.monotonic()-start,
        GPU_used=False,parameter_updates=0,actual_owner_transfers=0,actual_splits=0,counts=counts,
        split_provenance=split_meta,road_provenance=road_meta,
        evidence_boundary='canonical raw BUILD body and positive road; no new2D labels or height crop; proposals are not applied operations',
        design_change='independent repeated3D evidence can propose candidates even when sparse rendered road mass is0; physical/body protection still required',
        failure_ledger_refs=['V82-F06'],human_verdict=None)
    (out/'a6_geometry.json').write_text(json.dumps(receipt,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in receipt.items() if 'provenance' not in k}),flush=True)


if __name__=='__main__':main()
