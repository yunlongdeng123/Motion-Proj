"""Fixed numerical pointset gate; writes evidence only, never starts training."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial import cKDTree
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');S=R/'ownership_controls_r12_sam3';D=R/'idsplat_frontend_r13'
sys.path.insert(0,str(R/'implementation_r12/shadow_rebuild_v3'))
from shadow_rebuild.prepare import transform
m=json.loads((R/'data_r2/manifest.json').read_text());tr=np.load(R/'data_r2/trajectory.npz');targets=[5,6,12,18,25,26,37]
assert (D/'frontend_complete.json').exists()
worldroad=cKDTree(np.load(S/'actual_world_road_support.npz')['xyz'])
def bank():return {a:[] for a in targets}
body=bank();lower=bank();road=bank();counts={a:{s:0 for s in ['train_return_holdout','test_frame']} for a in targets}
for i,row in enumerate(m['frames']):
    d=np.load(S/'raw_measurements'/(row['frame_id']+'.npz'));sup=np.load(R/'data_r2'/row['supervision'])
    ids=np.flatnonzero(d['QUERY']&d['in_image']&d['lidar_projection_frontmost']);pixel=d['pixel'][ids].astype(np.int64);world=d['world'][ids]
    union=sup['masks'].any(-1);roadpix=(worldroad.query(world,workers=1)[0]<.06)&~union[pixel[:,1],pixel[:,0]]
    for a in targets:
        if not tr['valid'][i,a]:continue
        mask=sup['masks'][...,a];take=(d['candidate_owner'][ids]==a)&bool(sup['mask_known'][a])&mask[pixel[:,1],pixel[:,0]]
        local=transform(world,np.linalg.inv(tr['poses'][i,a]));body[a].append(local[take]);counts[a]['test_frame' if row['split']=='test' else 'train_return_holdout']+=int(take.sum())
        if mask.any():
            yy=np.where(mask)[0];threshold=yy.min()+.75*(yy.max()-yy.min());lower[a].append(local[take&(pixel[:,1]>=threshold)])
        road[a].append(local[roadpix&(d['candidate_owner'][ids]==a)])
def join(v):return np.concatenate(v) if v else np.empty((0,3),np.float32)
body={a:join(v) for a,v in body.items()};lower={a:join(v) for a,v in lower.items()};road={a:join(v) for a,v in road.items()}
results=[]
for name,path in [('A_box',S/'data_A_box'),('IDSplat_fixed_GT',D/'data_IDSplat')]:
    seed=np.load(path/'seed.npz');actors=[]
    for a in targets:
        points=seed['xyz'][seed['owner']==a];tree=cKDTree(points) if len(points) else None
        def cov(q):return None if not len(q) else float((tree.query(q,workers=1)[0]<.08).mean()) if tree else 0.
        nearroad=np.full(len(points),np.inf) if not len(road[a]) else cKDTree(road[a]).query(points,workers=1)[0]
        nearbody=np.full(len(points),np.inf) if not len(body[a]) else cKDTree(body[a]).query(points,workers=1)[0]
        actors.append(dict(actor=a,Actor_Gaussians=len(points),heldout_body_points=len(body[a]),heldout_body_coverage8cm=cov(body[a]),heldout_lowerbody_points=len(lower[a]),heldout_lowerbody_coverage8cm=cov(lower[a]),heldout_body_split=counts[a],heldout_positive_road_points=len(road[a]),road_only_seed_proxy=int(((nearroad<.06)&(nearbody>.08)).sum()),mixed_protected_seed_proxy=int(((nearroad<.06)&(nearbody<=.08)).sum())))
    results.append(dict(data=name,actors=actors))
screens=[]
for a,b in zip(results[0]['actors'],results[1]['actors']):
    def ratio(key):return b[key]/a[key] if a[key] is not None and a[key]>0 else None
    br=ratio('heldout_body_coverage8cm');lr=ratio('heldout_lowerbody_coverage8cm');reduction=1-b['road_only_seed_proxy']/a['road_only_seed_proxy'] if a['road_only_seed_proxy'] else None
    screens.append(dict(actor=a['actor'],body_ratio=br,lowerbody_ratio=lr,body_passed=br is not None and br>=.9,lowerbody_passed=lr is not None and lr>=.9,road_reduction=reduction,road_reliable_denominator=a['road_only_seed_proxy']>=30,road_passed=None if a['road_only_seed_proxy']<30 else reduction>=.5))
passed=all(x['body_passed'] and x['lowerbody_passed'] and x['road_passed'] is not False for x in screens)
v=dict(status='NUMERICAL_SCREEN_PASSED_REQUIRES_ROOT_VISUAL_REVIEW' if passed else 'POINTSET_GATE_FAILED_NO_TRAINING',models=results,screen=screens,numerical_passed=passed,root_visual_review_pending=True,scene_training_updates=0,
       evaluation_labels='unchanged original DEVA + actual heldout LiDAR, same A/B gate; not hand-labeled body truth',road_labels='actual heldout returns near unchanged repeated BUILD world-road support, outside original known masks; positive geometric proxy',lowerbody='bottom25percent visible 2D body mask bbox, not precise tire/undercarriage labels',QUERY='already-used development data; never fed to frontend, not new blind test',inputs_or_thresholds_changed_after_QUERY=False)
(D/'pointset_QUERY_gate.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
