"""Conservative camera visibility rejection using actual body LiDAR; no new body mask."""
from pathlib import Path
import sys,json,os,copy,time
import numpy as np
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r8/shadow_rebuild_v3'))
from shadow_rebuild.data import Dataset
from shadow_rebuild.prepare import project,transform
from shadow_rebuild.train import write
d=Dataset(r/'data_r4_road_identity','cpu',True);original=Dataset(r/'data_r2','cpu',True)
out=r/'data_r5_ray_visible_road';out.mkdir(exist_ok=False);(out/'frames').mkdir();start=time.monotonic()
for name in ['seed.npz','trajectory.npz']:os.link(d.root/name,out/name)
banks={a:np.load(r/'initialization_source_audit_r7'/f'a{a:02d}_actual_banks.npz')['body'] for a in [5,6,12,18,25,26,37]};records=[]
for index,row in enumerate(d.rows):
 f=d.frame(index);h,w=f.depth.shape;front=np.full((h,w),np.inf)
 if row['split']=='train':
  for actor,points in banks.items():
   if not bool(f.valid[actor]):continue
   world=transform(points,f.poses[actor].numpy());ij,z,valid=project(world,np.asarray(row['intrinsic']),np.asarray(row['world_from_camera']),(h,w))
   ij,z=ij[valid],z[valid]
   for dy in [-1,0,1]:
    for dx in [-1,0,1]:
     xx,yy=ij[:,0]+dx,ij[:,1]+dy;ok=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
     np.minimum.at(front,(yy[ok],xx[ok]),z[ok])
 with np.load(d.root/row['supervision']) as old:values={k:old[k] for k in old.files}
 with np.load(original.root/row['supervision']) as raw:originalroad=raw['road']
 blocked=(front+.1<values['depth'])&(values['depth']>0)
 before=int(values['road_certified'].sum());before_road=values['road'].copy()
 if row['split']=='train':
  values['road_certified']&=~blocked
  # Original conservative road labels preserved. Correct only added measured-road projections
  # so known physical body occluders are not accidentally excluded from RGB reconstruction.
  values['road']=originalroad|(values['road']&~blocked)
 os.link(d.root/row['rgb'],out/row['rgb']);np.savez_compressed(out/row['supervision'],**values)
 rec=dict(frame_id=row['frame_id'],split=row['split'],before_certified=before,after_certified=int(values['road_certified'].sum()),
          rejected_camera_occlusion=before-int(values['road_certified'].sum()),added_road_pixels_rejected=int((before_road&~values['road']).sum()))
 records.append(rec);print(json.dumps(rec),flush=True)
manifest=copy.deepcopy(d.manifest);manifest['certified_road_identity']['camera_visibility_guard']=dict(version='r9',
 source='actual train-only positive body LiDAR transported by measured trajectories',pixel_support_radius=1,body_depth_margin_m=.1,
 action='reject potentially camera-occluded road certificates; no synthetic body masks or new body alpha labels')
manifest['ownership_road_evidence']['camera_occluded_added_pixels_rejected']=True
write(out/'manifest.json',manifest);Dataset(out,'cpu',True)
write(out/'visibility_certification.json',dict(status='COMPLETE',train_only=True,training_updates=0,
 before_certified=sum(x['before_certified'] for x in records),after_certified=sum(x['after_certified'] for x in records),
 rejected=sum(x['rejected_camera_occlusion'] for x in records),added_road_pixels_rejected=sum(x['added_road_pixels_rejected'] for x in records),
 body_masks_unchanged=True,mask_known_unchanged=True,original_conservative_road_preserved=True,
 limitation='sparse actual-body point occlusion proxy, conservative1pixel rejection; absence of body points is not an occlusion-free proof',
 frames=records,elapsed_seconds=time.monotonic()-start))
