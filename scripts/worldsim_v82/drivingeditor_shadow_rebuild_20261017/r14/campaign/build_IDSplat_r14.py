"""IDSplat mask lifting/DBSCAN adapted to fixed GT, fine-part protection and infra.

Not an unmodified IDSplat reproduction. No scene optimization or QUERY selection.
"""
from pathlib import Path
import json,sys,time,os,hashlib
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion
from sklearn.cluster import DBSCAN
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
S=R/'ownership_rebuild_r14';D=S/'idsplat_frontend'
sys.path.insert(0,str(R/'implementation_r14/shadow_rebuild_v3'))
from shadow_rebuild.prepare import transform,voxel_rows,local_geometry
from shadow_rebuild.data import validate_seed
targets=[5,6,12,18,25,26,37]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def merge(v):return {k:np.concatenate([x[k] for x in v]) for k in v[0]}
def write_status(phase,**kw):
    (D/'status.json').write_text(json.dumps(dict(phase=phase,scene_training_updates=0,**kw),indent=2))
def main():
    start=time.monotonic()
    assert (S/'USER_STOP_BC_receipt.json').exists()
    assert not (D/'frontend_complete.json').exists(),'Never replay a completed frontend'
    D.mkdir(exist_ok=True);(D/'pointsets').mkdir(exist_ok=True)
    protocol=json.loads((S/'protocol.json').read_text());assert protocol['run_id']=='ownership_rebuild_r14'
    m=json.loads((S/'data_A_box/manifest.json').read_text());tr=np.load(S/'data_A_box/trajectory.npz');frames=m['frames']
    roadtree=cKDTree(np.load(S/'actual_world_road_support.npz')['xyz'])
    collected={a:[] for a in targets};framelog=[]
    for fi,row in enumerate(frames):
        if row['split']!='train':continue
        fid=row['frame_id'];raw=np.load(S/'raw_measurements'/(fid+'.npz'));sup=np.load(S/'shared_masks_r14'/(fid+'.npz'))
        pixel=raw['pixel'].astype(np.int64);zz=raw['camera_z'];build=raw['BUILD'];eligible=np.flatnonzero(build&raw['in_image'])
        depth=np.full(320*480,np.inf);binid=pixel[eligible,1]*480+pixel[eligible,0];np.minimum.at(depth,binid,zz[eligible])
        visible=np.zeros(len(build),bool);visible[eligible]=zz[eligible]<=depth[binid]+.1
        rgb=np.asarray(Image.open(S/'data_A_box'/row['rgb']).convert('RGB'))/255.
        for a in targets:
            ids=np.flatnonzero(build&(raw['candidate_owner']==a));world=raw['world'][ids];local=transform(world,np.linalg.inv(tr['poses'][fi,a])).astype(np.float32)
            mask=sup['masks'][...,a];known=bool(sup['mask_known'][a]);coremask=binary_erosion(mask,iterations=3)
            direct=np.zeros(len(ids),bool);core=np.zeros(len(ids),bool);v=visible[ids]
            direct[v]=mask[pixel[ids[v],1],pixel[ids[v],0]]&known;core[v]=coremask[pixel[ids[v],1],pixel[ids[v],0]]&known
            colors=np.full((len(ids),3),.5,np.float32);colors[v]=rgb[pixel[ids[v],1],pixel[ids[v],0]]
            labels=np.full(len(ids),-2,np.int32);lift=np.flatnonzero(direct)
            if len(lift):labels[lift]=DBSCAN(eps=.5,min_samples=10,n_jobs=1).fit_predict(world[lift])
            anchored=np.zeros(len(ids),bool)
            for label in np.unique(labels[lift]):
                if label>=0 and core[labels==label].any():anchored|=labels==label
            near_anchor=np.zeros(len(ids),bool)
            if anchored.any():near_anchor=cKDTree(local[anchored]).query(local,workers=1)[0]<=.15
            conflict=roadtree.query(world,workers=1)[0]<.06
            values=dict(xyz=local,rgb=colors,source_world=world.astype(np.float32),source_frame=np.full(len(ids),fi,np.int64),source_raw_row=raw['raw_row'][ids].astype(np.int64),source_candidate_owner=np.full(len(ids),a,np.int64),
                pixel=pixel[ids].astype(np.int16),camera_depth_visible=v,mask_direct=direct,mask_core=core,cluster=labels,cluster_core_anchored=anchored,near_anchored=near_anchor,road_conflict=conflict,sparse_frame=np.full(len(ids),len(lift)<10,bool))
            collected[a].append(values)
            framelog.append(dict(actor=a,frame_id=fid,frame_index=fi,candidates=len(ids),visible_body=int(direct.sum()),core=int(core.sum()),clusters=int(len(set(labels[lift].tolist())-{-1})),anchored=int(anchored.sum()),positive_road_conflicts=int((direct&conflict).sum())))
        write_status('MASK_LIFT_AND_DBSCAN',completed_frames=fi+1,total_frames=49)
    parts={};summary=[]
    for a in targets:
        src=merge(collected[a]);direct=src['mask_direct'];repeat=np.zeros(len(direct),np.int16)
        # Canonical proximity is only repeated semantic surface support. It is
        # NOT independent motion correspondence and is never reported as such.
        for fi in np.unique(src['source_frame'][direct]):
            tree=cKDTree(src['xyz'][direct&(src['source_frame']==fi)])
            repeat+=tree.query(src['xyz'],workers=1)[0]<.08
        support=src['cluster_core_anchored']|src['near_anchored']|(repeat>=2)|src['sparse_frame']
        accepted=direct&support&~src['road_conflict']
        reason=np.full(len(direct),0,np.int8)
        reason[~src['camera_depth_visible']]=1
        reason[src['camera_depth_visible']&~direct]=2
        reason[direct&~support]=3
        reason[direct&src['road_conflict']]=4
        reason[accepted]=5
        src.update(repeated_semantic_frames=repeat,accepted=accepted,evidence_reason=reason)
        assert np.all(src['source_raw_row']%5!=0)
        assert all(frames[int(fi)]['split']=='train' for fi in np.unique(src['source_frame']))
        np.savez_compressed(D/'pointsets'/f'a{a:02d}_all_candidates.npz',**src)
        take=voxel_rows(src['xyz'][accepted],.04);take=np.flatnonzero(accepted)[take]
        parts[a]={key:src[key][take] for key in ['xyz','rgb','source_world','source_frame','source_raw_row','source_candidate_owner']}
        parts[a]['identity_probability']=np.tile([.03,.94,.03],(len(take),1)).astype(np.float32)
        parts[a]['identity_evidence_status']=np.ones(len(take),np.int64)
        conf=np.where(src['mask_core'][take],.95,np.where(repeat[take]>=2,.85,.65)).astype(np.float32)
        np.savez_compressed(D/'pointsets'/f'a{a:02d}_local_measured_points.npz',**parts[a],confidence=conf,evidence_reason=reason[take])
        summary.append(dict(actor=a,candidate_BUILD_returns=len(direct),sameframe_mask_depth_returns=int(direct.sum()),accepted_returns=int(accepted.sum()),voxel_seeds=len(take),uncertain_preserved=int((~accepted).sum()),positive_road_conflict_returns=int((direct&src['road_conflict']).sum()),
                            body_sparse_frame_retained=int((accepted&src['sparse_frame']).sum()),noise_or_boundary_retained=int((accepted&~src['cluster_core_anchored']).sum()),reason_counts={str(i):int((reason==i).sum()) for i in range(6)}))
        print(json.dumps(summary[-1]),flush=True)
    # Keep every background/context seed, geometry, RGB and reference bytewise.
    # Only the admitted seven Actor seed sets change. Rebase neighbor indices.
    A=np.load(S/'data_A_box/seed.npz');out=D/'data_IDSplat';out.mkdir(exist_ok=False);(out/'frames').mkdir()
    chunks=[];offset=0
    for a in sorted(set(A['owner'].tolist())):
        oldids=np.flatnonzero(A['owner']==a)
        if a in targets:
            g=parts[a].copy();xyz=g['xyz'];scales,q,neighbor=local_geometry(xyz,.04)
            g.update(scales=scales,quaternion=q,neighbor=neighbor+offset,owner=np.full(len(xyz),a,np.int64),road=np.zeros(len(xyz),bool),observed_rgb=g['rgb'].copy(),reference_rgb=np.zeros_like(g['rgb']),reference_confidence=np.zeros(len(xyz),np.float32))
        else:
            g={key:A[key][oldids].copy() for key in A.files}
            assert np.array_equal(oldids,np.arange(oldids[0],oldids[0]+len(oldids)))
            assert np.all(A['owner'][g['neighbor']]==a)
            g['neighbor']=g['neighbor']-oldids[0]+offset
        assert set(g)==set(A.files),(a,set(g)^set(A.files))
        chunks.append(g);offset+=len(g['xyz'])
    seed={key:np.concatenate([g[key] for g in chunks]) for key in A.files};validate_seed(seed,len(m['actor_ids']))
    np.savez_compressed(out/'seed.npz',**seed)
    for row in frames:
        for key in ['rgb','supervision']:
            dest=out/row[key];dest.parent.mkdir(parents=True,exist_ok=True);os.link(S/'data_A_box'/row[key],dest)
    os.link(S/'data_A_box/trajectory.npz',out/'trajectory.npz')
    m.update(initialization='adapted official IDSplat mask-lift and DBSCAN; fixed GT; protected fine parts; no random seeds; selected7 only',shared_masks=str(S/'shared_masks_r14'),frontend_protocol=str(D/'protocol.json'))
    (out/'manifest.json').write_text(json.dumps(m,indent=2))
    report=dict(status='POINTSETS_BUILT_NOT_ACCEPTED_NO_SGD',actors=summary,total_gaussians=len(seed['xyz']),all_candidates_preserved=True,
                seconds=time.monotonic()-start,frame_log=framelog,trajectory_sha256=digest(out/'trajectory.npz'),A_trajectory_sha256=digest(S/'data_A_box/trajectory.npz'),other36_and_static_seed_fields_preserved_except_neighbor_offsets=True,
                RGB_and_supervision_hardlinked_unchanged=True,actual_owner_updated=True,QUERY_consumed=False,scene_training_updates=0,
                reason_codes={'0':'unclassified','1':'not BUILD camera depth visible','2':'no sameframe known body mask','3':'unsupported projection outlier','4':'body/positive-world-road conflict remains uncertain','5':'measured Actor seed accepted'})
    (D/'frontend_complete.json').write_text(json.dumps(report,indent=2));write_status('WAITING_INDEPENDENT_DINO_AND_POINT_GATE',seconds=report['seconds'])
    print(json.dumps({k:v for k,v in report.items() if k!='frame_log'}),flush=True)
if __name__=='__main__':main()
