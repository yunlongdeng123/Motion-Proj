from pathlib import Path
import json,sys,zipfile
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');S=R/'ownership_controls_r12_sam3';D=R/'idsplat_frontend_r13';O=D/'pointset_review'
O.mkdir(exist_ok=False);sys.path.insert(0,str(R/'implementation_r12/shadow_rebuild_v3'))
from shadow_rebuild.prepare import transform
targets=[5,6,12,18,25,26,37];samples={5:107,6:105,12:137,18:137,25:137,26:133,37:121}
m=json.loads((R/'data_r2/manifest.json').read_text());tr=np.load(R/'data_r2/trajectory.npz');A=np.load(S/'data_A_box/seed.npz');N=np.load(D/'data_IDSplat/seed.npz');query={a:[] for a in targets};lower={a:[] for a in targets}
for fi,row in enumerate(m['frames']):
    raw=np.load(S/'raw_measurements'/(row['frame_id']+'.npz'));sup=np.load(R/'data_r2'/row['supervision']);ids=np.flatnonzero(raw['QUERY']&raw['in_image']&raw['lidar_projection_frontmost']);pixel=raw['pixel'][ids].astype(int)
    for a in targets:
        mask=sup['masks'][...,a];take=(raw['candidate_owner'][ids]==a)&bool(sup['mask_known'][a])&mask[pixel[:,1],pixel[:,0]]
        if not take.any():continue
        local=transform(raw['world'][ids],np.linalg.inv(tr['poses'][fi,a]));query[a].append(local[take])
        yy=np.where(mask)[0];threshold=yy.min()+.75*(yy.max()-yy.min());lower[a].append(local[take&(pixel[:,1]>=threshold)])
def joined(v):return np.concatenate(v) if v else np.empty((0,3))
stages=[]
for a in targets:
    fi=samples[a]-89;row=m['frames'][fi];src=np.load(D/'pointsets'/f'a{a:02d}_all_candidates.npz');q=joined(query[a]);l=joined(lower[a]);points=src['xyz']
    subsets={'all_OBB_BUILD':np.ones(len(points),bool),'depth_visible':src['camera_depth_visible'],'sameframe_mask_and_depth':src['mask_direct'],'cluster_and_finepart_supported':src['mask_direct']&(src['cluster_core_anchored']|src['near_anchored']|(src['repeated_semantic_frames']>=2)|src['sparse_frame']),'after_road_conflict':src['accepted']}
    rec={'actor':a,'stages':[]}
    for name,take in subsets.items():
        tree=cKDTree(points[take]) if take.any() else None
        def coverage(x):return None if not len(x) else float((tree.query(x,workers=1)[0]<.08).mean()) if tree else 0.
        rec['stages'].append(dict(stage=name,measured_points=int(take.sum()),body_coverage=coverage(q),lowerbody_coverage=coverage(l)))
    stages.append(rec)
    Image.open(S/'data_A_box'/row['rgb']).convert('RGB').save(O/f'a{a:02d}_raw{samples[a]}_RGB.png')
    sup=np.load(S/'shared_sam3_masks_r2'/(row['frame_id']+'.npz'))
    np.savez_compressed(O/f'a{a:02d}_review.npz',A_xyz=A['xyz'][A['owner']==a],A_rgb=A['rgb'][A['owner']==a],D_xyz=N['xyz'][N['owner']==a],D_rgb=N['rgb'][N['owner']==a],
        candidate_xyz=src['xyz'],candidate_frame=src['source_frame'],candidate_world=src['source_world'],candidate_pixel=src['pixel'],accepted=src['accepted'],reason=src['evidence_reason'],mask_direct=src['mask_direct'],road_conflict=src['road_conflict'],
        query_body=q,query_lowerbody=l,sample_mask=sup['masks'][...,a],sample_mask_known=sup['mask_known'][a],sample_pose=tr['poses'][fi,a],world_from_camera=np.asarray(row['world_from_camera']),intrinsic=np.asarray(row['intrinsic']),sample_frame=fi)
(D/'module_coverage_diagnostic.json').write_text(json.dumps({'actors':stages,'posthoc_only':True,'inputs_unchanged':True,'thresholds_unchanged':True,'semantic_proximity_is_not_tracked_motion':True},indent=2))
for name in ['protocol.json','frontend_complete.json','pointset_QUERY_gate.json','module_coverage_diagnostic.json']:(O/name).write_bytes((D/name).read_bytes())
with zipfile.ZipFile(D/'pointset_review.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in O.iterdir():z.write(p,p.name)
print(json.dumps({'stages':stages,'archive':str(D/'pointset_review.zip'),'bytes':(D/'pointset_review.zip').stat().st_size}))
