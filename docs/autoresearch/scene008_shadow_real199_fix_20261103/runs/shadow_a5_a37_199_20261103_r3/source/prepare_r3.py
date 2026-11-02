"""Recover measured road omitted by the old span filter; exact all-view contributors."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from editor_common import Native,prepare_rasterization,raster,cpu

RUN=Path(__file__).resolve().parent.parent
OLD=RUN.with_name('shadow_a5_a37_199_20261103')

def main():
    start=time.time();dest=RUN/'inputs';dest.mkdir(parents=True,exist_ok=False)
    r=dict(np.load(OLD/'inputs_final/road_reference_199.npz'))
    a=np.load(OLD/'inputs/expanded/evidence.npz')
    assert np.array_equal(r['source_raw'],np.arange(199))
    geom=a['surface_supported']&r['domain']
    strong=r['active'].copy()
    measured=geom&(r['agree_count']>=3)&(r['reference_linear'].min(1)>.012)&(r['reference_linear'].max(1)<.88)
    # Consecutive clean observations are still observations. Keep their provenance
    # separate from the former >=6-frame temporal-span subset.
    ids=np.flatnonzero(measured);tree=cKDTree(r['local_xyz'][ids,:2])
    distance,near=tree.query(r['local_xyz'][:,:2],k=8)
    weight=1/np.maximum(distance,.04)**2;weight/=weight.sum(1,keepdims=True)
    neighbours=r['reference_linear'][ids[near]]
    filled=(neighbours*weight[:,:,None]).sum(1)
    inferred=geom&~measured&(distance[:,0]<=1.2)&((neighbours.max(1)-neighbours.min(1)).max(1)<.10)
    ref=r['reference_linear'].copy();ref[inferred]=filled[inferred]
    active=measured|inferred
    r['original_strong']=strong;r['measured']=measured;r['inferred']=inferred;r['active']=active
    r['reference_linear']=ref
    r['reference_srgb']=np.where(ref<=.0031308,12.92*ref,1.055*np.maximum(ref,1e-10)**(1/2.4)-.055)
    r['fit_weight']=np.where(strong,1.,np.where(measured,.65,np.where(inferred,.2,0))).astype(np.float32)
    np.savez_compressed(dest/'road_reference.npz',**r)
    views=dest/'views';views.mkdir()
    for raw in range(89,138):
        v=dict(np.load(OLD/'inputs_final/views_current'/('%03d.npz'%raw)))
        uid=v['unit_id'];safe=np.maximum(uid,0)
        # domain masks already contain the four-corner measured geometry test.
        geometry=(v['domain_a5']|v['domain_a37'])&(uid>=0)&~v['other_vehicle_guard']
        v['repair_mask']=geometry&active[safe]
        v['fit_weight']=r['fit_weight'][safe]*v['repair_mask']
        v['recovery_reference_linear']=ref[safe]
        v['inferred']=geometry&inferred[safe]
        v['weak_measured']=geometry&measured[safe]&~strong[safe]
        np.savez_compressed(views/('%03d.npz'%raw),**v)
    n=Native();total=len(n.static.get_xyz);scores=torch.zeros(total,device='cuda');rows=[]
    for raw in range(89,138):
        c=n.camera(raw);v=np.load(views/('%03d.npz'%raw));mask=torch.as_tensor(v['repair_mask'],device='cuda')
        kw=prepare_rasterization(c,n.static,n.args);kw['means2D']=torch.zeros_like(kw['means3D'])
        marker=torch.zeros((total,1),device='cuda',requires_grad=True)
        kw['features']=torch.cat([kw['features'],marker],1)
        out=raster(c,n.args,n.static.active_sh_degree,kw,n.bg)
        out['extra'][0,mask].sum().backward()
        contribution=marker.grad[:,0];scores=torch.maximum(scores,contribution.detach())
        rows.append(dict(raw=raw,road_pixels=int(mask.sum()),contributors=int((contribution>1e-4).sum())))
        if raw%10==0:print(json.dumps(rows[-1]),flush=True)
        del out,kw,marker,contribution
    selected=cpu(scores)>1e-4
    np.savez_compressed(dest/'contributors.npz',ids=np.flatnonzero(selected),max_alpha_T_pixel_sum=cpu(scores)[selected])
    receipt=dict(source_frames=list(range(199)),native_frames=list(range(89,138)),
        strong_cells=int(strong.sum()),measured_cells=int(measured.sum()),weak_measured_cells=int((measured&~strong).sum()),
        inferred_cells=int(inferred.sum()),active_cells=int(active.sum()),unknown_cells=int((geom&~active).sum()),
        contributor_rows=int(selected.sum()),selection='exact raster alpha*T > 1e-4 pixel-sum in at least one of all49 road masks; no ownership label',
        weak_policy='>=3 colour-agreeing real views; consecutive observations retained; all199 searched',
        inferred_policy='8-neighbour inverse-distance interpolation, nearest measured<=1.2m, linearRGB range<.10; excluded from factual reference metrics',
        rows=rows,updates=0,parent_unchanged=n.unchanged(),seconds=time.time()-start)
    (dest/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
