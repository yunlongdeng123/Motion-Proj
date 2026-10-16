import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
import torch
from surface_model import GeometryShadow,ActorShadow
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
NEW=ROOT/'shadow_sequence_20261016_v2';OLD=ROOT/'shadow_BC_20261016_v1'
torch.set_num_threads(2)
checkpoint=NEW/'joint_v1/shadow_final.pt'
saved=torch.load(checkpoint,map_location='cpu',weights_only=False);s={k[7:]:v for k,v in saved['surface'].items() if k.startswith('shadow.')}
with np.load(OLD/'training_data_v2/training_data.npz') as f:
    scale=float(f['world_scale']);plane=torch.tensor(f['plane_model']).float();pos=torch.tensor(f['poses'][12,:,:3,3]);rot=torch.tensor(f['poses'][12,:,:3,:3]);size=torch.tensor(f['sizes_model'][12]);valid=torch.tensor(f['pose_valid'][12])
with np.load(NEW/'surface_guarded_data/surface.npz') as f:q=torch.tensor(f['xyz'][f['shadow'][12]])
base=ActorShadow(plane,scale);keys=base.state_dict().keys()
m=GeometryShadow(plane,scale,s['part_center'],s['part_size'],s['part_owner'],{k:s[k] for k in keys});m.load_state_dict(s)
weights=torch.linspace(.1,1,len(q))[:,None]
loss=(m.attenuation(q,pos,rot,size,valid)*weights).sum();loss.backward()
rows=[]
for name in ['azimuth','part_shift_raw','part_scale_raw']:
    p=getattr(m,name);index=None if p.ndim==0 else tuple(int(v) for v in np.unravel_index(int(p.grad.abs().argmax()),p.shape))
    analytic=float(p.grad if index is None else p.grad[index]);eps=1e-3
    with torch.no_grad():
        old=p.clone()
        if index is None:p.add_(eps)
        else:p[index]+=eps
        plus=float((m.attenuation(q,pos,rot,size,valid)*weights).sum())
        p.copy_(old)
        if index is None:p.sub_(eps)
        else:p[index]-=eps
        minus=float((m.attenuation(q,pos,rot,size,valid)*weights).sum());p.copy_(old)
    numeric=(plus-minus)/(2*eps);rel=abs(analytic-numeric)/max(1,abs(analytic),abs(numeric))
    rows.append({'parameter':name,'component':index,'analytic':analytic,'central_difference':numeric,'relative_error':rel})
with torch.no_grad():
    a=m.attenuation(q,pos,rot,size,valid)
    shift=m.basis[:,0]*scale*.7
    b=m.attenuation(q+shift,pos+shift,rot,size,valid)
    deleted=m.attenuation(q,pos,rot,size,torch.zeros_like(valid))
    translation=float(abs(a-b).max());identity=bool(torch.equal(deleted,torch.ones_like(deleted)))
    ambient=bool((a>=1-m.direct_raw.sigmoid()-1e-6).all())
result={'status':'PASS' if all(r['relative_error']<.05 and abs(r['analytic'])>1e-5 for r in rows) and translation<2e-5 and identity and ambient else 'REVIEW',
    'finite_difference':rows,'plane_tangent_translation_equivariance_max':translation,'deleted_all_identity':identity,'ambient_floor':ambient,
    'source':str(checkpoint),'queries':len(q),'test_device':'CPU; independent of actual GPU raster gradient smoke'}
(NEW/'shadow_math_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
