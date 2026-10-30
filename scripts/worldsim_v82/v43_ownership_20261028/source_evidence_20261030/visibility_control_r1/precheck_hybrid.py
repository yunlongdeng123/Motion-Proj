import sys,json,time,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np,bpy
from mathutils import Vector
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030');out=ROOT/'visibility_control_r1';out.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'code/simulation'));import generate_scene as g
cfg=SimpleNamespace(**json.loads((ROOT/'dataset_r2/dataset_manifest.json').read_text())['parameters'])
scene,cam,car,objects,device=g.build_scene(cfg);geom=g.IndependentGeometry(objects)
original=json.loads((ROOT/'dataset_r2/observations/cameras.json').read_text())['frames'];targets=[]
for f in [0,8,15]:
 a=np.load(ROOT/f'dataset_r2/privileged/original/{f:06d}.npz');b=np.load(ROOT/f'dataset_r2/privileged/references/delete/geometry/{f:06d}.npz')
 mask=(a['semantic_id']==10)&np.isin(b['semantic_id'],[1,2])&b['valid']
 y,x=np.where(mask);targets.append(dict(frame=f,xyz=b['xyz_world'][mask],normal=b['normal_world'][mask],surface=b['surface_id'][mask],owner=b['object_id'][mask],semantic=b['semantic_id'][mask],yx=np.stack([y,x],-1)))
xyz=np.concatenate([t['xyz'] for t in targets]);owners=np.concatenate([t['owner'] for t in targets]);surfaces=np.concatenate([t['surface'] for t in targets]);n=len(xyz)
np.savez_compressed(out/'target_world_points.npz',xyz=xyz,normal=np.concatenate([t['normal'] for t in targets]),surface=surfaces,object_id=owners,semantic=np.concatenate([t['semantic'] for t in targets]),yx=np.concatenate([t['yx'] for t in targets]),anchor_frame=np.concatenate([np.full(len(t['xyz']),t['frame']) for t in targets]))
trajectories=['hybrid8_z3.5','hybrid10_z3.5','hybrid8_z5'];manifest={}
for trajectory in trajectories:
 vis=[];inframes=[];occluders=[];dist_error=[];cams=[];pixelchecks=[]
 for f in range(16):
  g.set_frame(f,cfg,cam,car,'none',True)
  if trajectory!='original' and f not in [0,8,15]:
   radius=10 if trajectory.startswith('hybrid10') else 8;z=5 if trajectory.endswith('z5') else 3.5
   theta=math.pi+2*math.pi*f/16
   cam.location=car.location+Vector((radius*math.cos(theta),radius*math.sin(theta),z))
   cam.rotation_euler=(car.location+Vector((0,0,.5))-cam.location).to_track_quat('-Z','Y').to_euler();bpy.context.view_layer.update()
  cal=g.camera_metadata(f,cfg,cam);cams.append(cal)
  if trajectory=='original':assert np.max(np.abs(np.asarray(cal['camera_to_world'])-np.asarray(original[f]['camera_to_world'])))==0
  c2w=np.asarray(cal['camera_to_world']);w2c=np.asarray(cal['world_to_camera']);k=np.asarray(cal['K']);q=xyz@w2c[:3,:3].T+w2c[:3,3];uv=q[:,:2]/q[:,2:]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
  inside=(q[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<cfg.width)&(uv[:,1]>=0)&(uv[:,1]<cfg.height)
  geom.build_bvh();delta=xyz-c2w[:3,3];ranges=np.linalg.norm(delta,axis=1);directions=delta/ranges[:,None];hit=geom.cast(c2w[:3,3],directions,float(ranges.max()+1));err=np.linalg.norm(hit['xyz_world']-xyz,axis=1)
  visible=inside&(err<=.005)&(hit['object_id']==owners)&np.isin(hit['semantic_id'],[1,2]);vis.append(visible);inframes.append(inside);occluders.append(hit['semantic_id']);dist_error.append(err)
  if trajectory=='original':
   d=np.load(ROOT/f'dataset_r2/privileged/original/{f:06d}.npz');x=np.clip(np.floor(uv[:,0]).astype(int),0,479);y=np.clip(np.floor(uv[:,1]).astype(int),0,319);pixelchecks.append(np.linalg.norm(d['xyz_world'][y,x]-xyz,axis=1))
  print('VIS_FRAME',trajectory,f,int(visible.sum()),flush=True)
 vis=np.array(vis);np.savez_compressed(out/(trajectory+'_visibility.npz'),visible=vis,in_fov=np.array(inframes),firsthit_semantic=np.array(occluders),firsthit_world_error=np.array(dist_error),nearest_pixel_world_error=np.array(pixelchecks))
 g.dump_json(out/(trajectory+'_cameras.json'),{'frames':cams,'coordinate_contract':g.coordinate_contract(),'purpose':'privileged trajectory visibility precheck only; RGB not generated'})
 origins=np.array([c['camera_to_world'] for c in cams])[:,:3,3];centered=origins-origins.mean(0);ev,basis=np.linalg.eigh(centered.T@centered);pc=centered@basis[:,np.argsort(ev)[::-1]];extent=float(abs(pc).max());scale=min(.1,1/extent)
 manifest[trajectory]={'points':n,'visible_any':float(vis.any(0).mean()),'visible_2plus':float((vis.sum(0)>=2).mean()),'visible_counts':[int((vis.sum(0)==i).sum()) for i in range(17)],'camera_origin_PCA_max_abs':extent,'native_scale_estimate':scale,'camera_radius_world_max_center':float(np.linalg.norm(centered,axis=1).max()),'anchors':{str(t['frame']):{'pixels':len(t['xyz']),'visible_any':float(vis[:,sum(len(x['xyz']) for x in targets[:j]):sum(len(x['xyz']) for x in targets[:j+1])].any(0).mean())} for j,t in enumerate(targets)}}
g.dump_json(out/'hybrid_precheck.json',{'schema':'v43_world_surface_visibility_v1','target_definition':'original vehicle first-hit AND independently deleted road/lane first-hit at original anchor camera; world XYZ fixed across all observations','visibility_definition':'point in calibrated camera FOV AND exact original-world BVH firsthit within 5mm and same object and road/lane semantic; surface triangle match not required at shared edges','pixel_sampling_caveat':'continuous projected surface visibility; nearest actual original pixel world discrepancy exported separately; does not guarantee texture resolution or appearance identifiability','train_input':False,'trajectory_selection_scope':'world geometry privileged only diagnostic control; same world/times/16 single cameras; no edited RGB used','trajectories':manifest})
