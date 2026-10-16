"""Cover local foreground contributors across all49 times, not just raw101."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
import json,math,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from shadow_math import project_gaussians,rotation_matrix
from prepare_e0 import sample
base=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_BC_20261016_v1')
out=base/'selection_union_v1';out.mkdir(exist_ok=False)
torch.set_num_threads(2)
with np.load(base/'selection_v1/selection.npz') as f:old={k:f[k] for k in f.files}
with np.load(base/'training_data_v2/training_data.npz') as f:
    plane=f['plane_model'];scale=float(f['world_scale']);cloud=f['local_lidar'];query=f['xyz_model'];times=f['times'];atlas=f['reference_srgb'].reshape(60,140,3)
cam=json.loads((base/'training_data_v2/camera_metadata.json').read_text())[12]['camera']
cap=torch.load(base.parent/'rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt',map_location='cpu',weights_only=False)['background']
arr=lambda t:t.detach().numpy().astype('float32',copy=False)
N=len(cap[1]);seen=np.zeros(N,bool);road=np.zeros(N,bool);canonical=np.zeros((N,3),np.float32)
minimum=np.full((N,3),np.inf,np.float32);maximum=-minimum.copy();omega=2*math.pi/float(cap[16])
scales=arr(cap[4].exp());rotations=arr(cap[5]);opa=arr(cap[6].sigmoid()).ravel();rotation=rotation_matrix(rotations)
sig=np.linalg.norm(np.einsum('nji,j->ni',rotation,plane[:3])*scales,axis=1)/scale
tree=cKDTree(cloud/scale);view=np.asarray(cam['world_view_transform']);qdepth=(query@view[:3,:3]+view[3,:3])[:,2]
for i,t in enumerate(times):
    xyz=arr(cap[1]+cap[9]*torch.sin((float(t)-cap[7])*omega)/omega)
    minimum=np.minimum(minimum,xyz);maximum=np.maximum(maximum,xyz)
    marginal=arr(torch.exp(-.5*((cap[7]-float(t))/cap[8].exp())**2)).ravel()
    state=dict(xyz=xyz,scales=scales,rotation=rotations,opacity=opa*marginal,colors=np.zeros((N,3),np.float32),mask=(marginal>.05)&(opa*marginal>=1/255))
    pr=project_gaussians(state,cam,True)
    # Conservative kernel/tile overlap also includes formerly occluded points
    # that can become visible after road repair; depth behind the patch is excluded.
    candidate=(pr['rmax'][:,0]>2)&(pr['rmin'][:,0]<12)&(pr['rmax'][:,1]>13)&(pr['rmin'][:,1]<19)&(pr['depth']<qdepth.max()+.2*scale)
    ids=pr['ids'][candidate];fresh=ids[~seen[ids]];canonical[fresh]=xyz[fresh];seen[ids]=True
    height=(xyz[ids]@plane[:3]+plane[3])/scale
    uv=pr['uv'][candidate];inside=(uv[:,0]>=40)&(uv[:,0]<179)&(uv[:,1]>=225)&(uv[:,1]<284)
    ids_road=ids[inside&(abs(height)<.20)&(3*sig[ids]<=.30)]
    if len(ids_road):
        distance,_=tree.query(xyz[ids_road]/scale,k=1,workers=2)
        road[ids_road[distance<.45]]=True
    if i%7==0:print(json.dumps({'frame':i,'seen_candidates':int(seen.sum()),'road_candidates':int(road.sum())}),flush=True)
span=np.linalg.norm(maximum-minimum,axis=1)/scale
road&=span<=.25
road[old['receiver_ids']]=True;seen[old['receiver_ids']]=True;seen[old['artifact_ids']]=True
canonical[old['receiver_ids']]=old['receiver_xyz_anchor'];canonical[old['artifact_ids']]=old['artifact_xyz_anchor']
ri=np.flatnonzero(road);ai=np.flatnonzero(seen&~road)
sys.path.insert(0,str(base.parent/'infra'));from renderer import evaluate_sh_rgb
xyz=torch.from_numpy(canonical[ri]);features=torch.cat([cap[2][ri],cap[3][ri]],1)
colors=arr(evaluate_sh_rgb(features,torch.nn.functional.normalize(xyz-torch.tensor(cam['camera_center']),dim=-1),int(cap[0])))
# New road receivers start from actual accepted clean reference, not inherited dark color.
probe=project_gaussians(dict(xyz=canonical[ri],scales=scales[ri],rotation=rotations[ri],opacity=np.ones(len(ri),np.float32),colors=colors,mask=np.ones(len(ri),bool)),cam,True)
assert np.array_equal(probe['ids'],np.arange(len(ri)))
ref,valid=sample(atlas,probe['uv']-np.array([40,225]));is_old=np.isin(ri,old['receiver_ids']);colors[valid&~is_old]=ref[valid&~is_old].clip(0,1)
np.savez_compressed(out/'selection.npz',receiver_ids=ri,artifact_ids=ai,receiver_xyz_anchor=canonical[ri],artifact_xyz_anchor=canonical[ai],
    receiver_rgb_anchor=colors,receiver_base_opacity=opa[ri],artifact_base_opacity=opa[ai])
receipt={'schema':'all49_local_foreground_coverage_v1','frames':49,'receiver_points':len(ri),'other_opacity_only_points':len(ai),
    'new_road_points_added':0,'original_receiver_points':len(old['receiver_ids']),'original_opacity_points':len(old['artifact_ids']),
    'criterion':'any49 time potentially foreground EWA kernel in local road window; receivers need local LiDAR/plane/extent/limited motion support; other candidates opacity-only',
    'new_receiver_reference_initialization':True,'unknown_other_candidates_are_not_semantically_labelled_as_bad':True,
    'first_visible_state_is_canonical_for_new_candidates':True,'source_checkpoint_unchanged':True}
(out/'selection_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
