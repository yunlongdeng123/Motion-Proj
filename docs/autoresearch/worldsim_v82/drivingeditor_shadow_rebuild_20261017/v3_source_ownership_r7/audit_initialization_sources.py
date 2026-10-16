"""追溯框内实际LiDAR：区分道路源点、车体正证据与未知；只读、不裁高度。"""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.spatial import cKDTree
import torch
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r6/shadow_rebuild_v3'))
from shadow_rebuild.data import Dataset
from shadow_rebuild.prepare import project,transform
from shadow_rebuild.train import write
from shadow_rebuild.math3d import covariance
torch.set_num_threads(2);d=Dataset(r/'data_r2','cpu',True);actors=[5,6,12,18,25,26,37]
out=r/'initialization_source_audit_r7';out.mkdir(exist_ok=False);start=time.monotonic()
atlas=np.load(r/'road_evidence_r5/actual_lidar_atlas.npz');cert=atlas['certified'];roadxyz=atlas['xyz'];roadframe=atlas['frame_index']
scene=Path(d.manifest['source_scene']);origin=np.loadtxt(scene/'pose'/(d.rows[0]['frame_id']+'.txt'))[:3,3]
banks={a:{key:[] for key in ['body','road','body_frame','road_frame','body_lidar_row','road_lidar_row']} for a in actors}
for j,index in enumerate(d.train_indices):
 row=d.rows[index];fid=row['frame_id'];f=d.frame(index)
 raw=np.fromfile(scene/'velodyne'/(fid+'.bin'),np.float32).reshape(-1,6)[:,:3]
 pose=np.loadtxt(scene/'pose'/(fid+'.txt'));pose[:3,3]-=origin;world=transform(raw,pose)
 ij,z,inimage=project(world,np.array(row['intrinsic']),np.array(row['world_from_camera']),f.depth.shape)
 visible=np.flatnonzero(inimage);yy,xx=ij[visible,1],ij[visible,0]
 visible=visible[(f.depth.numpy()[yy,xx]>0)&(np.abs(f.depth.numpy()[yy,xx]-z[visible])<.15)]
 labels=json.loads((Path(d.manifest['annotations'])/(fid+'.json')).read_text())['laser_labels']
 dimensions={x['id']:np.array([x['box']['length'],x['box']['width'],x['box']['height']]) for x in labels}
 frame_road=np.flatnonzero(cert&(roadframe==index))
 for actor in actors:
  if not bool(f.valid[actor]):continue
  inverse=np.linalg.inv(f.poses[actor].numpy());size=dimensions[d.actor_ids[actor]]
  local=transform(world[visible],inverse);inside=(np.abs(local)<=size/2+.03).all(-1)
  actual_body=f.masks.numpy()[ij[visible,1],ij[visible,0],actor]&bool(f.mask_known[actor])&inside
  b=visible[actual_body];banks[actor]['body'].append(local[actual_body]);banks[actor]['body_frame'].append(np.full(len(b),index,np.int16));banks[actor]['body_lidar_row'].append(b)
  localroad=transform(roadxyz[frame_road],inverse);take=(np.abs(localroad)<=size/2+.03).all(-1);rr=frame_road[take]
  banks[actor]['road'].append(localroad[take]);banks[actor]['road_frame'].append(np.full(len(rr),index,np.int16));banks[actor]['road_lidar_row'].append(atlas['lidar_row'][rr])
 write(out/'progress.json',dict(frames=j+1,total=46,elapsed_seconds=time.monotonic()-start))
 print(json.dumps(dict(frame=fid,done=j+1,total=46)),flush=True)
saved=torch.load(r/'E2_ownership_s0/latest.pt',map_location='cpu',weights_only=False)
if saved['global_step']!=7000:raise ValueError('frozen query changed')
state=saved['state'];owner=state['field.owner'].numpy();xyz=state['field.xyz'].numpy();seed=state['field.seed_xyz'].numpy()
cov=covariance(state['field.quaternion'],state['field.log_scales'].exp()).numpy();records=[]
for actor in actors:
 bank={key:np.concatenate(value) for key,value in banks[actor].items()};np.savez_compressed(out/f'a{actor:02d}_actual_banks.npz',**bank)
 rows=np.flatnonzero(owner==actor);points=xyz[rows];precision=np.linalg.inv(cov[rows])
 def nearest(name):
  values=bank[name]
  if not len(values):return np.full(len(rows),np.inf),np.full(len(rows),np.inf),np.zeros(len(rows),np.int16)
  tree=cKDTree(values);distance,index=tree.query(points,k=min(16,len(values)),workers=1)
  if distance.ndim==1:distance=distance[:,None];index=index[:,None]
  delta=values[index]-points[:,None];mahal=np.einsum('nki,nij,nkj->nk',delta,precision,delta).min(-1)
  neighborhoods=tree.query_ball_point(seed[rows],.06,workers=1)
  frames=np.array([len(np.unique(bank[name+'_frame'][v])) for v in neighborhoods],np.int16)
  return distance[:,0],mahal,frames
 bd,bm,bf=nearest('body');rd,rm,rf=nearest('road')
 # Only nominations here: positive measured source classes, not a height test or zero-visibility label.
 road_only=(rf>=3)&(rd<.06)&(rm<9)&(bm>16)&(bd>.06)
 mixed=(rf>=3)&(rm<9)&(bm<=9)
 source=d.seed['xyz'][d.seed['owner']==actor]
 exactroad=cKDTree(bank['road']).query(source,workers=1)[0] if len(bank['road']) else np.full(len(source),np.inf)
 exactbody=cKDTree(bank['body']).query(source,workers=1)[0] if len(bank['body']) else np.full(len(source),np.inf)
 np.savez_compressed(out/f'a{actor:02d}_gaussian_support.npz',rows=rows,nearest_body_m=bd,body_mahalanobis_squared=bm,body_source_frames=bf,
                     nearest_road_m=rd,road_mahalanobis_squared=rm,road_source_frames=rf,road_only_candidate=road_only,mixed=mixed)
 rec=dict(actor=actor,actual_body_measurements=len(bank['body']),actual_road_measurements_inside_OBB=len(bank['road']),
          original_seed_points=len(source),original_seed_exact_road_source=int((exactroad<2e-5).sum()),original_seed_exact_body_source=int((exactbody<2e-5).sum()),
          trained_gaussians=len(rows),positive_road_only_no_body_support=int(road_only.sum()),both_body_road_volume=int(mixed.sum()),
          no_body_within_4sigma=int((bm>16).sum()))
 records.append(rec);print(json.dumps(rec),flush=True)
write(out/'summary.json',dict(status='COMPLETE_READ_ONLY',global_step=7000,training_updates=0,train_only=True,
      method='actual known-body depth-matched LiDAR inside annotation OBB vs actual repeated-world-road LiDAR; retain source frame and raw LiDAR row; Gaussian3D Mahalanobis support',
      unknown_policy='missing body evidence alone never background; road-only nominations require positive repeated actual road source',
      limitation='nearest16 body/road points approximate volume intersection; nomination only, no relabel or deletion; exact source match is canonical tolerance2e-5m',
      actors=records,elapsed_seconds=time.monotonic()-start))
