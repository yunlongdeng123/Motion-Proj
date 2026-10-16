"""Independent conservative positive-road certificates; raw body masks unchanged."""
from pathlib import Path
import sys,json,os,copy,time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r8/shadow_rebuild_v3'))
from shadow_rebuild.data import Dataset
from shadow_rebuild.prepare import transform
from shadow_rebuild.train import write
d=Dataset(r/'data_r3_ownership','cpu',True);out=r/'data_r4_road_identity';out.mkdir(exist_ok=False);(out/'frames').mkdir()
atlas=np.load(r/'road_evidence_r5/actual_lidar_atlas.npz');source=Path(d.manifest['source_scene']);origin=np.loadtxt(source/'pose'/(d.rows[0]['frame_id']+'.txt'))[:3,3]
roadrows=np.flatnonzero(d.seed['road']);roadxyz=d.seed['xyz'][roadrows];roadtree=cKDTree(roadxyz)
roadnormal=Rotation.from_quat(d.seed['quaternion'][roadrows][:,[1,2,3,0]]).as_matrix()[...,2]
banks={a:np.load(r/'initialization_source_audit_r7'/f'a{a:02d}_actual_banks.npz')['body'] for a in [5,6,12,18,25,26,37]}
bodytrees={a:cKDTree(b) for a,b in banks.items() if len(b)}
for name in ['seed.npz','trajectory.npz']:os.link(d.root/name,out/name)
records=[];start=time.monotonic()
for index,row in enumerate(d.rows):
 fid=row['frame_id'];f=d.frame(index);os.link(d.root/row['rgb'],out/row['rgb'])
 with np.load(d.root/row['supervision']) as old:values={k:old[k] for k in old.files}
 certificate=np.zeros_like(values['road']);stat=dict(frame_id=fid,split=row['split'],before=0,normal_pass=0,after_body_protection=0,pixels=0)
 if row['split']=='train':
  take=np.flatnonzero(atlas['certified']&(atlas['frame_index']==index));points=atlas['xyz'][take]
  raw=np.fromfile(source/'velodyne'/(fid+'.bin'),np.float32).reshape(-1,6)[:,:3];pose=np.loadtxt(source/'pose'/(fid+'.txt'));pose[:3,3]-=origin;world=transform(raw,pose)
  _,near=cKDTree(world).query(points,k=12,workers=1);sample=world[near];delta=sample-sample.mean(1,keepdims=True)
  cov=np.einsum('nki,nkj->nij',delta,delta)/12;eigen,basis=np.linalg.eigh(cov)
  _,anchors=roadtree.query(points,workers=1);normal=basis[:,:,0]
  good=(np.abs((normal*roadnormal[anchors]).sum(-1))>.95)&(np.sqrt(eigen[:,0].clip(0))<.04)&(eigen[:,0]/eigen[:,1:].sum(-1).clip(1e-9)<.05)
  stat.update(before=len(take),normal_pass=int(good.sum()))
  # Positive body3D support from other train times protects contact/tires, even if current mask unknown.
  for actor,tree in bodytrees.items():
   if not bool(f.valid[actor]):continue
   local=transform(points,np.linalg.inv(f.poses[actor].numpy()));distance,_=tree.query(local,workers=1)
   good&=distance>.04
  pix=atlas['uv'][take[good]];certificate[pix[:,1],pix[:,0]]=True
  certificate&=values['road']&(values['depth']>0)&(~values['masks'].any(-1))
  stat.update(after_body_protection=int(good.sum()),pixels=int(certificate.sum()))
 values['road_certified']=certificate;np.savez_compressed(out/row['supervision'],**values);records.append(stat)
 print(json.dumps(stat),flush=True)
manifest=copy.deepcopy(d.manifest);manifest['certified_road_identity']=dict(version='r8',source=str(r/'road_evidence_r5'),train_only=True,
 actual_LiDAR_rows=True,actual_neighbor_PCA=True,protect_other_time_positive_body_m=.04,
 mask_known_unchanged=True,body_masks_unchanged=True,unknown_is_not_background=True)
write(out/'manifest.json',manifest)
summary=dict(status='COMPLETE',certified_pixels=sum(x['pixels'] for x in records),frames=records,training_only=True,
 method='already>=3worldtime actualroad; candidate own measured neighborhood flat normal agrees with measured road, positive-body3D contact protection, actual depth, no known-body overlap',
 limitation='conservative geometric road certificate, not manually annotated road semantics; noncertified unknown remains unknown',elapsed_seconds=time.monotonic()-start)
write(out/'certification.json',summary);Dataset(out,'cpu',True)
print(json.dumps({k:v for k,v in summary.items() if k!='frames'}),flush=True)
