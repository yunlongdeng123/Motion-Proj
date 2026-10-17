"""冻结模型全场景 alpha*T 归属审计。仅训练帧形成证据，不修改参数。"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--run',default='E1_balanced_s0');p.add_argument('--step',type=int,default=6000)
p.add_argument('--output',default='ownership_006000_r5');p.add_argument('--road-evidence');p.add_argument('--implementation',default='implementation_r6');p.add_argument('--all-identities',action='store_true');p.add_argument('--data',default='data_r2');p.add_argument('--certified-road',action='store_true');a=p.parse_args()
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(root/a.implementation/'shadow_rebuild_v3'))
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import seed_from_state,write
from shadow_rebuild.raster import cuda_raster
from shadow_rebuild.losses import pixel_valid
torch.set_num_threads(2)
out=root/a.output;out.mkdir(exist_ok=False)
saved=torch.load(root/a.run/'latest.pt',map_location='cpu',weights_only=False)
if saved['global_step']!=a.step:raise ValueError('frozen step mismatch')
cfg=saved['config'].copy();cfg['weights']=LossWeights(**cfg['weights']);cfg=Config(**cfg).validate()
model=SceneModel(seed_from_state(saved['state']),cfg).cuda();model.load_state_dict(saved['state']);model.requires_grad_(False)
d=Dataset(root/a.data,'cuda',True);n=len(model.field.xyz);k=len(d.actor_ids)
eligible=[5,6,12,18,25,26,37]
# Per target: own body, OTHER KNOWN bodies, certified sparse measured road, total full-scene visibility.
sums=np.zeros((n,4),np.float64);hits=np.zeros((n,3),np.uint16)
perframe=[];start=time.monotonic()
identity_sums=np.zeros((n,k),np.float32) if a.all_identities else None
identity_frames=np.zeros((n,k),np.uint8) if a.all_identities else None
for j,index in enumerate(d.train_indices):
 f=d.frame(index)
 with torch.no_grad():g=model.field.evaluate(f.poses,f.valid,f.camera.center)
 # All scene Gaussians participate in transmittance. Feature gradient yields exact accumulated alpha*T.
 # Eight body channels (7 targets + union of known others), one road, one total = capacity10.
 target=f.masks&pixel_valid(f) if a.all_identities else f.masks[...,eligible]&pixel_valid(f)[...,eligible]
 union=(f.masks&pixel_valid(f)).any(-1)
 road=f.road_certified if a.certified_road else (f.road if not a.road_evidence else torch.as_tensor(np.load(root/a.road_evidence/(f.frame_id+'.npz'))['road'],device='cuda'))
 if road is None:raise ValueError('certified road missing')
 measured=(f.depth>0)&road&(~union)
 domains=torch.cat((target,union[...,None],measured[...,None],torch.ones_like(union)[...,None]),-1).float()
 c=target.shape[-1]
 feature=torch.ones(len(g['xyz']),c+3,device='cuda',requires_grad=True)
 image=cuda_raster(f.camera,g['xyz'],g['cov'],g['opacity'],feature,10)
 contribution=torch.autograd.grad((image*domains).sum(),feature)[0].detach()
 rows=g['rows'];owners=g['owner'];own=torch.zeros(len(rows),device='cuda')
 for t,actor in enumerate(range(k) if a.all_identities else eligible):own[owners==actor]=contribution[owners==actor,t]
 other=(contribution[:,c]-own).clamp_min(0)
 evidence=torch.stack((own,other,contribution[:,c+1],contribution[:,c+2]),-1).cpu().numpy()
 rr=rows.cpu().numpy();sums[rr]+=evidence;hits[rr]+=(evidence[:,:3]>.02).astype(np.uint16)
 if a.all_identities:
  identities=contribution[:,:k].cpu().numpy();identity_sums[rr]+=identities;identity_frames[rr]+=(identities>.02).astype(np.uint8)
 np.savez_compressed(out/f"{f.frame_id}.npz",rows=rr,evidence=evidence)
 mass=[]
 for actor in eligible:
  take=owners==actor
  mass.append(dict(actor=actor,mask_known=bool(f.mask_known[actor]),body_pixels=int(f.masks[...,actor].sum()),own_body=float(own[take].sum()),other_body=float(other[take].sum()),
                   measured_road=float(contribution[take,c+1].sum()),visible=float(contribution[take,c+2].sum())))
 perframe.append(dict(frame_id=f.frame_id,split=f.split,road_pixels=int(measured.sum()),actors=mass))
 write(out/'progress.json',dict(frames=j+1,total=len(d.train_indices),elapsed_seconds=time.monotonic()-start))
 print(json.dumps(perframe[-1]),flush=True)
 del image,contribution,feature,g
owner=model.field.owner.cpu().numpy();known=sums[:,:3].sum(-1);ratio=sums[:,2]/np.maximum(known,1e-9)
body=sums[:,0]/np.maximum(known,1e-9);dynamic=np.isin(owner,eligible)
repeated_body=(hits[:,0]>=3)&(sums[:,0]>=.5)
road_dominant=dynamic&(hits[:,2]>=3)&(sums[:,2]>=.5)&(ratio>.8)&(body<.05)&(~repeated_body)
mixed=dynamic&repeated_body&(hits[:,2]>=3)&(sums[:,2]>=.2)
np.savez_compressed(out/'per_gaussian.npz',owner=owner,evidence=sums,frames=hits,road_dominant=road_dominant,mixed=mixed)
if a.all_identities:np.savez_compressed(out/'identity_evidence.npz',mass=identity_sums,frames=identity_frames)
summary=dict(run=a.run,global_step=a.step,parameters_updated=False,train_only=True,data=a.data,road_evidence=('dataset road_certified, actual body camera-occlusion proxy rejection' if a.certified_road else a.road_evidence or 'original guarded sparse road'),occlusion='full scene alpha*T; all static/other actors retained',
 domains=['own known body','other known body','actual LiDAR depth AND road surface pixel','total full-scene visible alpha'],
 thresholds=dict(min_frames=3,min_road_mass=.5,road_ratio=.8,max_body_ratio=.05,repeated_body_mass_priority=.5),
 unknown_policy='zero visibility or no certified semantic support stays unknown; NOT background',
 scope='7 admitted actors only classified; all scene occluders included; road pixels sparse and conservative',
 actors=[dict(actor=x,gaussians=int((owner==x).sum()),road_dominant=int((road_dominant&(owner==x)).sum()),
              mixed=int((mixed&(owner==x)).sum()),evidence=sums[owner==x].sum(0).tolist()) for x in eligible],
 frames=perframe,elapsed_seconds=time.monotonic()-start)
write(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='frames'}),flush=True)
