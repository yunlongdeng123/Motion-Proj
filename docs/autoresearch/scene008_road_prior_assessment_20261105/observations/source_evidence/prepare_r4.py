"""Correct deleted-actor/other-vehicle guard overlap, retaining full199 provenance."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_dilation,binary_closing
from editor_common import Native,prepare_rasterization,raster,cpu

RUN=Path(__file__).resolve().parent.parent
R3=RUN.with_name('shadow_a5_a37_199_20261103_r3')
R1=RUN.with_name('shadow_a5_a37_199_20261103')

def main():
    start=time.time();dest=RUN/'inputs';dest.mkdir(exist_ok=False);views=dest/'views';views.mkdir()
    r=dict(np.load(R3/'inputs/road_reference.npz'));a=np.load(R1/'inputs/expanded/evidence.npz')
    assert np.array_equal(r['source_raw'],np.arange(199))
    np.savez_compressed(dest/'road_reference.npz',**r)
    records=[]
    for raw in range(89,138):
        v=dict(np.load(R3/'inputs/views'/('%03d.npz'%raw)));old=v['other_vehicle_guard'].copy()
        selected=v['actor_labels']>0
        # The stored "other" edge mask was dilated AFTER subtracting selected
        # labels. It consequently overlaps the very actors being deleted.
        clear=binary_dilation(binary_closing(selected,iterations=3),iterations=8)
        other=old&~clear
        uid=v['unit_id'];safe=np.maximum(uid,0)
        # Keep protection independent of reference eligibility and avoid tiny
        # holes in the four-corner test being mislabeled as preserve-original.
        domain=(uid>=0)&a['surface_supported'][safe]&r['domain'][safe]
        repair=domain&r['active'][safe]&~other
        v['other_vehicle_guard']=other;v['repair_mask']=repair;v['protect_mask']=~domain|other
        v['fit_weight']=r['fit_weight'][safe]*repair
        v['inferred']=repair&r['inferred'][safe]
        v['weak_measured']=repair&r['measured'][safe]&~r['original_strong'][safe]
        np.savez_compressed(views/('%03d.npz'%raw),**v)
        records.append(dict(raw=raw,guard_on_deleted_labels_before=int((old&selected).sum()),
            guard_on_deleted_labels_after=int((other&selected).sum()),removed_guard_pixels=int((old&~other).sum()),repair_pixels=int(repair.sum())))
    n=Native();N=len(n.static.get_xyz);score=torch.zeros(N,device='cuda')
    for raw in range(89,138):
        c=n.camera(raw);v=np.load(views/('%03d.npz'%raw));mask=torch.as_tensor(v['repair_mask'],device='cuda')
        kw=prepare_rasterization(c,n.static,n.args);kw['means2D']=torch.zeros_like(kw['means3D'])
        marker=torch.zeros((N,1),device='cuda',requires_grad=True)
        kw['features']=torch.cat([kw['features'],marker],1);out=raster(c,n.args,n.static.active_sh_degree,kw,n.bg)
        out['extra'][0,mask].sum().backward();score=torch.maximum(score,marker.grad[:,0].detach())
        del kw,out,marker
    ids=np.flatnonzero(cpu(score)>1e-4);np.savez_compressed(dest/'contributors.npz',ids=ids)
    rec=dict(source_frames=199,native_frames=49,contributor_rows=len(ids),
        guard_overlap_deleted_before=sum(x['guard_on_deleted_labels_before'] for x in records),
        guard_overlap_deleted_after=sum(x['guard_on_deleted_labels_after'] for x in records),rows=records,
        road_colour_and_opacity='fixed full199 reference colour and opacity=.99005; no optimization can erase the recovery surface',
        reference_provenance=json.loads((R3/'inputs/receipt.json').read_text()),updates=0,seconds=time.time()-start,parent_unchanged=n.unchanged())
    assert rec['guard_overlap_deleted_after']==0
    (dest/'receipt.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True)

if __name__=='__main__':main()
