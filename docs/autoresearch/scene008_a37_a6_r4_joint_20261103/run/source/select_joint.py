"""Exact R4-joint alpha*T selection over all49 a37/a6 footprints."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_closing,binary_dilation
import r4_runtime as r4
from editor_common import cpu,prepare_rasterization,road_kwargs,merge_rows,raster
RUN=Path(__file__).resolve().parent.parent;OLD=RUN.with_name('shadow_a5_a37_199_20261103_r4');DENSE=RUN.parent/'r15_gn11_stage1_v41_20261023_dense_r7/frames'
r4.RUN=OLD;r4.INPUT=OLD/'inputs'
def main():
    start=time.time();e=r4.Experiment();ck=torch.load(OLD/'formal/checkpoint001000.pt',map_location='cuda',weights_only=False);e.repair.load_state_dict(ck['repair']);N=len(e.n.static.get_xyz);score=torch.zeros(N,device='cuda');rows=[]
    oldids=cpu(e.repair.ids);inold=np.zeros(N,bool);inold[oldids]=True
    for raw in range(89,138):
        labels=np.load(DENSE/('0008%03d.npz'%raw))['actor_labels'];mask=binary_dilation(binary_closing((labels==4)|(labels==5),iterations=2),iterations=5)
        c=e.camera(raw);kw=prepare_rasterization(c,e.n.static,e.n.args);kw['means2D']=torch.zeros_like(kw['means3D']);ids=e.repair.ids
        kw['opacities']=kw['opacities'].index_copy(0,ids,kw['opacities'][ids]*e.repair.gate_raw.sigmoid()[:,None]);marker=torch.zeros((N,1),device='cuda',requires_grad=True)
        kw['features']=torch.cat([kw['features'],marker],1);rr=road_kwargs(c,e.repair.road,e.n.args,kw['shs'].shape[1] if kw['shs'] is not None else 1,precomputed_color=kw['colors_precomp'] is not None)
        rr['features']=torch.cat([rr['features'],torch.zeros_like(rr['features'][:,:1])],1);kw=merge_rows(kw,rr);out=raster(c,e.n.args,e.n.static.active_sh_degree,kw,e.n.bg)
        out['extra'][0,torch.tensor(mask,device='cuda')].sum().backward();g=marker.grad[:,0].detach();score=torch.maximum(score,g)
        a=cpu(g);mass=float(a.sum());rows.append(dict(raw=raw,contributors=int((a>1e-4).sum()),mass_not_in_old_candidate_fraction=float(a[~inold].sum()/max(mass,1e-12))))
        del kw,out,marker,g
    ids=np.flatnonzero(cpu(score)>1e-4);np.savez_compressed(RUN/'joint_contributors.npz',ids=ids,max_alpha_T_pixel_sum=cpu(score)[ids]);receipt=dict(joint_actors=[37,6],frames=49,rows=rows,candidate_rows=len(ids),new_vs_R4=int((~inold[ids]).sum()),updates=0,parent_unchanged=e.unchanged(),seconds=time.time()-start)
    r4.write(RUN/'selection_receipt.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
