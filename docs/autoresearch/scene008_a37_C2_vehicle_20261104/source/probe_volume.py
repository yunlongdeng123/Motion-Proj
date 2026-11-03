"""a37物理邻域的冻结零更新诊断；粗体积删除不是对象归属真值。"""
from pathlib import Path
import json,time
import numpy as np
import torch
from vehicle_runtime import VehicleExperiment,NAMES
from editor_common import save,prepare_rasterization,raster

RUN=Path(__file__).resolve().parent.parent

def main():
    dest=RUN/'evidence/volume113';dest.mkdir(exist_ok=False)
    torch.set_num_threads(1);start=time.monotonic()
    e=VehicleExperiment(dict(actor_subset='all',legacy_replacement='old_c2'))
    raw=113;view=e.old.view(raw)
    domain=torch.as_tensor(np.load(RUN/'evidence/fast113/row_attribution.npz')['domain'],device='cuda')
    guard=torch.as_tensor(view['actor_labels']!=5,device='cuda')
    visible=torch.as_tensor(view['visible_evidence_mask'],device='cuda')
    rows=[]
    with torch.no_grad():
        old=e.render(raw,False);native=e.n.render(e.old.base.camera(raw),dynamic=True)['rgb']
        save(dest/'old_C2.png',old['rgb'])
        for family in ['same_C2','missing_only_native']:
            e.config.update(legacy_replacement='old_c2' if family=='same_C2' else 'native')
            e.old.actor.active=torch.ones_like(e.groups,dtype=torch.bool) if family=='same_C2' else e.groups==5
            if family=='missing_only_native':
                e.old.actor.colour_logit.copy_(torch.logit(e.old.actor.initial_colour.clamp(.005,.995)))
            for margin in [0.,.25,.5,.75,1.,1.5]:
                e.config['target_static_volume_margin_lwh_m']=[margin,.10,.10]
                out=e.render(raw,False)
                save(dest/(family+'_x%03d.png'%round(margin*100)),out['rgb'])
                row=dict(family=family,margin_lwh_m=e.config['target_static_volume_margin_lwh_m'],
                    domain_pixels=int(domain.sum()),lower_alpha_T_mean=float(out['component_alpha_T'][5,domain].mean()),
                    actor_alpha_T_mean=float(out['actor_alpha'][0,domain].mean()),
                    static_alpha_T_mean=float(out['component_alpha_T'][0,domain].mean()),
                    outside_target_rgb_change=float((out['rgb']-old['rgb']).abs()[:,guard].mean()),
                    a6_observed_MAE=float((out['rgb']-e.old.truth(raw)).abs()[:,visible].mean()))
                rows.append(row);print(json.dumps(row),flush=True)
        result=dict(rows=rows,updates=0,parent_unchanged=e.unchanged(),seconds=time.monotonic()-start,
            role='diagnostic coarse physical volume; not semantic ownership, not formal accepted method')
        (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
