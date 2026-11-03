"""R4 a37/a6 source and coverage audit; no parameter updates."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json
import numpy as np
import torch
from scipy.ndimage import binary_dilation,binary_closing
import r4_runtime as r4
from editor_common import cpu,save,prepare_rasterization,raster,road_kwargs,merge_rows

RUN=Path(__file__).resolve().parent.parent
OLD=RUN.with_name('shadow_a5_a37_199_20261103_r4')
DENSE=RUN.parent/'r15_gn11_stage1_v41_20261023_dense_r7/frames'
r4.RUN=OLD;r4.INPUT=OLD/'inputs'

def main():
    dest=RUN/'diagnostic_joint';dest.mkdir(exist_ok=False);e=r4.Experiment();ck=torch.load(OLD/'formal/checkpoint001000.pt',map_location='cuda',weights_only=False);e.repair.load_state_dict(ck['repair'])
    meta=e.meta;summary=dict(actors=meta['actors'],render_includes_dynamic=False,parent_dynamic_rows=len(e.n.dynamic.get_xyz),parent_static_rows=len(e.n.static.get_xyz),rows=[])
    for raw in [89,101,105,113,125,137]:
        d=e.view(raw);dense=np.load(DENSE/('0008%03d.npz'%raw));labels=dense['actor_labels']
        masks={str(a):binary_dilation(binary_closing(labels==meta['actors'].index(a)+1,iterations=2),iterations=3) for a in [6,37]}
        c=e.camera(raw);kw=prepare_rasterization(c,e.n.static,e.n.args);kw['means2D']=torch.zeros_like(kw['means3D'])
        # Keep actual R4 opacity, then use an auxiliary feature gradient to
        # measure real alpha*T row contribution in the actor image footprints.
        ids=e.repair.ids;kw['opacities']=kw['opacities'].index_copy(0,ids,kw['opacities'][ids]*e.repair.gate_raw.sigmoid()[:,None])
        marker=torch.zeros((len(e.n.static.get_xyz),1),device='cuda',requires_grad=True);kw['features']=torch.cat([kw['features'],marker],1)
        rr=road_kwargs(c,e.repair.road,e.n.args,kw['shs'].shape[1] if kw['shs'] is not None else 1,precomputed_color=kw['colors_precomp'] is not None)
        rr['features']=torch.cat([rr['features'],torch.zeros_like(rr['features'][:,:1])],1);kw=merge_rows(kw,rr)
        out=raster(c,e.n.args,e.n.static.active_sh_degree,kw,e.n.bg)
        union=masks['6']|masks['37'];out['extra'][0,torch.tensor(union,device='cuda')].sum().backward();score=cpu(marker.grad[:,0]);means=cpu(kw['means3D'])
        top=np.flatnonzero(score>1e-4);T=np.asarray(meta['world_to_model'])@np.asarray(meta['coordinate_anchor']);inv=np.linalg.inv(T);local=means[top]@inv[:3,:3].T+inv[:3,3]
        with torch.no_grad():fixed=e.render(raw);save(dest/('%03d_R4.png'%raw),fixed['rgb']);save(dest/('%03d_B.png'%raw),e.base(raw))
        a=cpu(fixed['extra'][0]);record=dict(raw=raw,actors={})
        for actor,mask in masks.items():
            uid=d['unit_id'];valid=mask&(uid>=0);safe=np.maximum(uid,0)
            record['actors'][actor]=dict(pixels=int(mask.sum()),road_projection_supported=int(valid.sum()),road_active_cells_pixels=int((valid&e.ref['active'][safe]).sum()),
                road_fit_pixels=int((mask&(d['fit_weight']>0)).sum()),protected_pixels=int((mask&d['protect_mask']).sum()),
                road_alpha_T_mean=float(a[mask].mean()) if mask.any() else None)
            save(dest/('%03d_mask_a%s.png'%(raw,actor)),torch.tensor(mask,dtype=torch.float32,device='cuda').expand(3,-1,-1))
        summary['rows'].append(record)
        np.savez_compressed(dest/('%03d_contributors.npz'%raw),ids=top,score=score[top],local_xyz=local,native_model_xyz=means[top])
        del kw,marker,out,score
    summary['parent_unchanged']=e.unchanged();summary['updates']=0
    r4.write(dest/'receipt.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
