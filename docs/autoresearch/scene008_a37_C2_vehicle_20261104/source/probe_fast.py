"""固定几何/道路的最短原生责任行与读出诊断；所有关闭行仅为oracle。"""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,time
from pathlib import Path
import numpy as np
import torch
from vehicle_runtime import VehicleExperiment,NAMES
from editor_common import save,prepare_rasterization,raster

RUN=Path(__file__).resolve().parent.parent
DEST=RUN/'evidence/fast113'

def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    DEST.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    start=time.monotonic()
    e=VehicleExperiment(dict(actor_subset='all',legacy_replacement='old_c2'))
    raw=113;c=e.old.base.camera(raw)
    with torch.no_grad():
        akw=e.old.actor.kwargs(c,e.old.matrix(raw),e.n.args,e.n.static.get_features.shape[1],e.n.args.convert_SHs_python)
        e.tagged(akw,e.groups)
        alone=raster(c,e.n.args,e.n.static.active_sh_degree,akw,e.n.bg)
        lower=alone['extra'][5]>.10
        v=e.old.view(raw)
        unknown=torch.as_tensor(v['unknown_mask'],device='cuda',dtype=torch.bool)
        domain=lower&unknown
        labels=torch.as_tensor(v['actor_labels'],device='cuda')
        guard=~(alone['alpha'][0]>.10)|(labels==4)|((labels>0)&(labels<4))
        assert domain.any()
        save(DEST/'lower_domain.png',domain[None].float().repeat(3,1,1))
        save(DEST/'actor_alone.png',alone['rgb']+(1-alone['alpha']))
    marker=torch.zeros((len(e.n.static.get_xyz),2),device='cuda',requires_grad=True)
    joint=e.render(raw,False,static_marker=marker)
    val=joint['static_marker'][0,domain].sum()+joint['static_marker'][1,guard].sum()
    weights=torch.autograd.grad(val,marker,only_inputs=True)[0].detach()
    fit,protect=weights[:,0],weights[:,1]
    order=torch.argsort(fit,descending=True)
    cumulative=fit[order].cumsum(0)/fit.sum().clamp_min(1e-9)
    cut=int(torch.searchsorted(cumulative,torch.tensor(.99,device='cuda')))+1
    ids=order[:cut]
    candidate=fit>1e-6
    kw=prepare_rasterization(c,e.n.static,e.n.args)
    inv=torch.linalg.inv(e.old.matrix(raw))
    canonical=kw['means3D']@inv[:3,:3].T+inv[:3,3]
    camdepth=kw['features'][:,0]
    rgb=joint['rgb'].detach()
    summary=dict(raw=raw,domain_pixels=int(domain.sum()),guard_pixels=int(guard.sum()),
        exact_native_PVD_lower_visible_support=True,hidden_domain_is_not_GT=True,
        lower_actor_alone_mean=float(alone['extra'][5,domain].mean()),
        same_scene_components={k:dict(alpha_T_mean=float(joint['component_alpha_T'][i,domain].mean()),depth_model=float(joint['component_depth_sum'][i,domain].sum()/joint['component_alpha_T'][i,domain].sum().clamp_min(1e-8))) for i,k in enumerate(NAMES)},
        responsible_rows_99percent=cut,total_static_contribution=float(fit.sum()),candidate_rows=int(candidate.sum()),
        static_center_canonical_for_top20=canonical[ids[:20]].detach().cpu().tolist(),
        static_depth_model_for_top20=camdepth[ids[:20]].detach().cpu().tolist(),
        static_fit_for_top20=fit[ids[:20]].cpu().tolist(),
        static_guard_for_top20=protect[ids[:20]].cpu().tolist(),
        oracles=[],updates=0)
    np.savez_compressed(DEST/'row_attribution.npz',ids=np.flatnonzero(candidate.cpu().numpy()),
        fit=fit[candidate].cpu().numpy(),guard=protect[candidate].cpu().numpy(),
        canonical_center=canonical[candidate].detach().cpu().numpy(),depth_model=camdepth[candidate].detach().cpu().numpy(),
        top99_ids=ids.cpu().numpy(),domain=domain.cpu().numpy(),guard_mask=guard.cpu().numpy())
    del marker,joint,val
    torch.cuda.empty_cache()
    save(DEST/'old_C2_delete.png',rgb)
    original_ids=e.ids;original_keep=e.keep
    with torch.no_grad():
        for name,selection in [('top99_off',ids),('pure80_off',torch.nonzero(candidate&(fit/(fit+protect+1e-9)>.8)).flatten()),('pure50_off',torch.nonzero(candidate&(fit/(fit+protect+1e-9)>.5)).flatten())]:
            e.ids=selection;e.keep=torch.zeros(len(selection),device='cuda')
            out=e.render(raw,False)
            save(DEST/(name+'.png'),out['rgb'])
            save(DEST/(name+'_lower_alpha.png'),out['component_alpha_T'][5:6].repeat(3,1,1))
            summary['oracles'].append(dict(name=name,rows=len(selection),lower_alpha_T_mean=float(out['component_alpha_T'][5,domain].mean()),
                total_actor_alpha_T_mean=float(out['actor_alpha'][0,domain].mean()),
                guard_rgb_change=float((out['rgb']-rgb).abs()[:,guard].mean())))
        e.ids=original_ids;e.keep=original_keep
    summary.update(seconds=time.monotonic()-start,parent_unchanged=e.unchanged())
    write(DEST/'summary.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
