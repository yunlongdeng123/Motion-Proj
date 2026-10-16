"""QUERY-only real-return coverage and road-source proxy; never changes BUILD inputs."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial import cKDTree
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');root=r/'ownership_controls_r11';sys.path.insert(0,str(r/'implementation_r11/shadow_rebuild_v3'))
from shadow_rebuild.prepare import transform
m=json.loads((r/'data_r2/manifest.json').read_text());tr=np.load(r/'data_r2/trajectory.npz');targets=[5,6,12,18,25,26,37];body={a:[] for a in targets};lower={a:[] for a in targets};road={a:[] for a in targets};split_counts={a:{s:0 for s in ['train_return_holdout','test_frame']} for a in targets};worldroad=cKDTree(np.load(root/'actual_world_road_support.npz')['xyz'])
for i,row in enumerate(m['frames']):
 d=np.load(root/'raw_measurements'/(row['frame_id']+'.npz'));sup=np.load(r/'data_r2'/row['supervision']);ids=np.flatnonzero(d['QUERY']&d['in_image']&d['lidar_projection_frontmost']);pixel=d['pixel'][ids].astype(np.int64);world=d['world'][ids];onroad=worldroad.query(world,workers=1)[0]<.06;union=sup['masks'].any(-1);roadpix=onroad&~union[pixel[:,1],pixel[:,0]]
 for actor in targets:
  if not tr['valid'][i,actor]:continue
  mask=sup['masks'][...,actor];take=(d['candidate_owner'][ids]==actor)&bool(sup['mask_known'][actor])&mask[pixel[:,1],pixel[:,0]];local=transform(world,np.linalg.inv(tr['poses'][i,actor]));body[actor].append(local[take]);split_counts[actor]['test_frame' if row['split']=='test' else 'train_return_holdout']+=int(take.sum())
  if mask.any():
   yy=np.where(mask)[0];threshold=yy.min()+.75*(yy.max()-yy.min());lower[actor].append(local[take&(pixel[:,1]>=threshold)])
  # Positive world-road surface plus actual heldout measurement, not complement of an unknown Actor mask.
  road[actor].append(local[roadpix&(d['candidate_owner'][ids]==actor)])
def joined(values):return np.concatenate(values) if values else np.empty((0,3),np.float32)
body={a:joined(v) for a,v in body.items()};lower={a:joined(v) for a,v in lower.items()};road={a:joined(v) for a,v in road.items()};results=[]
for name in ['data_A_box','data_B_semantic_motion']:
 seed=np.load(root/name/'seed.npz');actors=[]
 for a in targets:
  points=seed['xyz'][seed['owner']==a];tree=cKDTree(points) if len(points) else None
  def coverage(query):return None if not len(query) else float((tree.query(query,workers=1)[0]<.08).mean()) if tree is not None else 0.
  nearroad=np.full(len(points),np.inf) if not len(road[a]) else cKDTree(road[a]).query(points,workers=1)[0];nearbody=np.full(len(points),np.inf) if not len(body[a]) else cKDTree(body[a]).query(points,workers=1)[0]
  roadonly=(nearroad<.06)&(nearbody>.08);mixed=(nearroad<.06)&(nearbody<=.08)
  actors.append(dict(actor=a,Actor_Gaussians=len(points),heldout_body_points=len(body[a]),heldout_body_coverage8cm=coverage(body[a]),heldout_lowerbody_points=len(lower[a]),heldout_lowerbody_coverage8cm=coverage(lower[a]),
   heldout_body_split=split_counts[a],heldout_positive_road_points=len(road[a]),road_only_seed_proxy=int(roadonly.sum()),mixed_protected_seed_proxy=int(mixed.sum()),road_only_seed_proxy_fraction=float(roadonly.mean()) if len(points) else None))
 results.append(dict(data=name,actors=actors))
value=dict(status='INITIALIZATION_QUERY_REPORT_NOT_AUTOMATIC_ACCEPTANCE',models=results,QUERY='raw101/113/125 actualreturns plus raw rowmod5==0 in other46; excluded from all initialization/depth/tracks/references',
 body_labels='original DEVA visible body + actual LiDAR in candidate box; predicted labels, not manual3D ownership GT',road_labels='heldout actual point near independently built repeated actual world-road support, outside all known original bodies; a geometric positive proxy, not complete road GT',
 lowerbody='bottom25percent of visible2D body bbox, diagnostic only, not precise wheel/undercarriage labels and never a geometry crop',
 metric='8cm Euclidean actual-return point coverage; seed-road proxy requires within6cm actual heldoutroad and >8cm all heldoutbody; mixed points separately protected',
 no_opacity_in_metric=True,initial_opacity_same_all_arms=.6,inputs_changed_after_QUERY=False,human_verdict=None)
(root/'initialization_QUERY_gate.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
