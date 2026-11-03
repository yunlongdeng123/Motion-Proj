"""局部持久3D替换体积的零更新三锚对照。"""
from pathlib import Path
import json,time
import numpy as np
import torch
from vehicle_runtime import VehicleExperiment
from editor_common import save
from baseline_audit import render_baseline

RUN=Path(__file__).resolve().parent.parent

def main():
    dest=RUN/'evidence/corridor_r1';dest.mkdir(exist_ok=False)
    torch.set_num_threads(1);start=time.monotonic()
    cfg=dict(actor_subset='missing_lower',legacy_replacement='native',colour_source='observed_initial',road_policy='delete_only')
    e=VehicleExperiment(cfg);rows=[]
    with torch.no_grad():
        for raw in [105,113,130]:
            v=e.old.view(raw);vis=torch.as_tensor(v['visible_evidence_mask'],device='cuda',dtype=torch.bool)
            labels=torch.as_tensor(v['actor_labels'],device='cuda');other=(labels>0)&(labels<4)
            unknown=torch.as_tensor(v['unknown_mask'],device='cuda',dtype=torch.bool)
            baseline=render_baseline(e.old.base,raw,False,True)['rgb']
            e.config['replacement_corridor']=False;before=e.render(raw,False)
            e.config['replacement_corridor']=True;after=e.render(raw,False)
            full=e.render(raw,True);native=e.n.render(e.old.base.camera(raw),dynamic=True)['rgb']
            truth=e.old.truth(raw)
            folder=dest/f'{raw}';folder.mkdir()
            for name,img in [('baseline',baseline),('before',before['rgb']),('after',after['rgb']),('noedit',full['rgb']),('native',native),('lower_alpha',after['component_alpha_T'][5:6].repeat(3,1,1))]:save(folder/(name+'.png'),img)
            row=dict(raw=raw,stats=e.corridor_stats[raw],
                lower_unknown_mean_before=float(before['component_alpha_T'][5,unknown].mean()),lower_unknown_mean_after=float(after['component_alpha_T'][5,unknown].mean()),
                visible_MAE_baseline=float((baseline-truth).abs()[:,vis].mean()),visible_MAE_after=float((after['rgb']-truth).abs()[:,vis].mean()),
                other_changed=float((after['rgb']-baseline).abs()[:,other].mean()),
                noedit_native_meanabs=float((full['rgb']-native).abs().mean()),noedit_native_maxabs=float((full['rgb']-native).abs().max()))
            rows.append(row);print(json.dumps(row),flush=True)
    result=dict(config=cfg,rows=rows,updates=0,seconds=time.monotonic()-start,parent_unchanged=e.unchanged(),
        role='diagnostic local3D replacement; corridor not ownership GT; no image-space routing')
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
