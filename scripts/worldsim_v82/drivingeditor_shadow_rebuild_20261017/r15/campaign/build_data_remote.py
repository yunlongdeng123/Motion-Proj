"""New dense raw-measurement background, frozen R14 instance frontend and masks."""
from pathlib import Path
import os,sys,json,time,shutil
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');OLD=R/'ownership_rebuild_r14';D=R/'backend_restore_r15';I=R/'implementation_r15/shadow_rebuild_v3'
sys.path.insert(0,str(I))
from shadow_rebuild.prepare import local_geometry,ground_candidates,voxel_rows
from shadow_rebuild.data import Dataset,validate_seed
from shadow_rebuild.config import Config
from shadow_rebuild.references import refresh
from shadow_rebuild.field import GaussianField
from dataclasses import asdict
import torch
torch.set_num_threads(2);start=time.monotonic()
assert not (D/'data').exists(),'Never replace a completed input build'
D.mkdir(exist_ok=True);out=D/'data';out.mkdir();(out/'frames').mkdir();(out/'sky').mkdir()
m=json.loads((OLD/'idsplat_frontend/data_IDSplat/manifest.json').read_text());source=OLD/'idsplat_frontend/data_IDSplat';parts=[];logs=[]
for fi,row in enumerate(m['frames']):
    fid=row['frame_id'];original=Path(m['source_scene'])/'sky/sky_0'/(fid+'.png')
    if not original.exists():raise FileNotFoundError('Original measured/predicted sky input missing: '+str(original))
    sky=Image.open(original).convert('L').resize((480,320),Image.Resampling.NEAREST);sky.save(out/'sky'/(fid+'.png'));row['sky']='sky/'+fid+'.png'
    for key in ['rgb','supervision']:os.link(source/row[key],out/row[key])
    if row['split']!='train':continue
    raw=np.load(OLD/'raw_measurements'/(fid+'.npz'));take=raw['BUILD']&(raw['candidate_owner']==-1)&raw['in_image']&(raw['camera_z']<500);ids=np.flatnonzero(take);pixel=raw['pixel'].astype(np.int64);z=raw['camera_z'];build=np.flatnonzero(raw['BUILD']&raw['in_image']);depth=np.full(320*480,np.inf);bins=pixel[build,1]*480+pixel[build,0];np.minimum.at(depth,bins,z[build]);visible=z[ids]<=depth[pixel[ids,1]*480+pixel[ids,0]]+.1
    rgb=np.asarray(Image.open(source/row['rgb']).convert('RGB'),np.float32)/255;colors=np.full((len(ids),3),.5,np.float32);colors[visible]=rgb[pixel[ids[visible],1],pixel[ids[visible],0]]
    sup=np.load(OLD/'shared_masks_r14'/(fid+'.npz'));body=sup['masks'].any(-1);notbody=~body[pixel[ids,1],pixel[ids,0]];ids,colors=ids[notbody],colors[notbody]
    parts.append(dict(xyz=raw['world'][ids],rgb=colors,source_world=raw['world'][ids],source_frame=np.full(len(ids),fi,np.int64),source_raw_row=raw['raw_row'][ids].astype(np.int64),source_candidate_owner=np.full(len(ids),-1,np.int64)))
    logs.append(dict(frame=fid,static_raw=len(ids),actual_visible_color=int(visible[notbody].sum())))
raw={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};keep=voxel_rows(raw['xyz'],.001);raw={k:v[keep] for k,v in raw.items()};n=len(keep)
if n<500000:raise ValueError('Dense static build unexpectedly undersized: '+str(n))
scales,q,neighbor=local_geometry(raw['xyz'],.015,ground=True);road=ground_candidates(raw['xyz'])
# Original spatial Gaussian initialization: RMS 3-nearest-neighbor distance.
dist,_=cKDTree(raw['xyz']).query(raw['xyz'],k=4,workers=1);sigma=np.sqrt((dist[:,1:]**2).mean(-1)).clip(1e-5,None).astype(np.float32);scales=np.repeat(sigma[:,None],3,1);scales[road,2]=np.maximum(scales[road,2]*.15,.001)
raw.update(scales=scales,quaternion=q,neighbor=neighbor.astype(np.int64),owner=np.full(n,-1,np.int64),road=road,observed_rgb=raw['rgb'].copy(),reference_rgb=np.zeros((n,3),np.float32),reference_confidence=np.zeros(n,np.float32),identity_probability=np.tile([.8,.05,.15],(n,1)).astype(np.float32),identity_evidence_status=np.full(n,2,np.int64))
old=dict(np.load(source/'seed.npz'));actor=np.flatnonzero(old['owner']>=0);actor_map=np.full(len(old['owner']),-1,np.int64);actor_map[actor]=np.arange(len(actor))+n
second={k:v[actor].copy() for k,v in old.items()};second['neighbor']=actor_map[old['neighbor'][actor]]
if (second['neighbor']<n).any():raise ValueError('Actor adjacency points into old static table')
seed={k:np.concatenate((raw[k],second[k])) for k in second}
validate_seed(seed,len(m['actor_ids']));np.savez_compressed(out/'seed.npz',**seed);os.link(source/'trajectory.npz',out/'trajectory.npz')
m.update(initialization='R15 dense actual BUILD static + unchanged measured IDSplat Actor frontend; no learned prior',static_voxel_m=.001,background_seed_count=n,reference_refresh='real observations only; zero-confidence remains unknown',backend='legacy SH3/1024cube restored',parent_input=str(source));(out/'manifest.json').write_text(json.dumps(m,indent=2))
cfg=Config(geometry_steps=30000,material_steps=10000,joint_steps=5000,instance_balanced_supervision=True,outside_fullscene_visibility=True,instance_balanced_visible_identity=True,continuous_ownership=True,train_opacity_stage2=True,densify_interval=100,densify_until=15000,max_gaussians=3000000,legacy_reference_config=str(R.parent/'training_configs/scene0008_089_137_GN11.json'))
centers=np.asarray([r['world_from_camera'] for r in m['frames'] if r['split']=='train'])[:,:3,3];cfg.spatial_lr_scale=float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1);cfg.weights.mask_outside=.5;cfg.weights.road_identity=.5;cfg.validate()
dataset=Dataset(out,'cpu',True);field=GaussianField(seed,cfg);reference=refresh(field,dataset);seed['reference_rgb']=field.reference_rgb.numpy();seed['reference_confidence']=field.reference_confidence.numpy();np.savez_compressed(out/'seed.npz',**seed)
(D/'config.json').write_text(json.dumps(asdict(cfg),indent=2));(D/'input_build.json').write_text(json.dumps(dict(static=n,Actor=len(actor),total=len(seed['xyz']),reference=reference,frames=logs,source_actor_unchanged=True,raw_BUILD_only=True,QUERY_used=False,learned_checkpoint_used=False,seconds=time.monotonic()-start),indent=2));print(json.dumps(dict(status='DENSE_INPUT_BUILT_REQUIRES_PREFLIGHT',static=n,Actor=len(actor),reference=reference,seconds=time.monotonic()-start)))
