"""Fixed-parent, zero-training actor surface counterfactuals for the yellow line."""
import json,os
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import label
import r4_runtime as r4
from editor_common import save,cpu
RUN=Path(__file__).resolve().parent.parent;OLD=RUN.with_name('shadow_a5_a37_199_20261103_r4')
r4.RUN=OLD;r4.INPUT=OLD/'inputs'
def main():
    e=r4.Experiment();ck=torch.load(OLD/'formal/checkpoint001000.pt',map_location='cuda',weights_only=False);e.repair.load_state_dict(ck['repair'])
    dest=RUN/'actor_cause_audit';dest.mkdir(exist_ok=False);rows=[]
    gate=e.repair.gate_raw.detach().clone();active=e.repair.road.active.clone()
    with torch.no_grad():
        for raw in [101,105,113]:
            base=e.base(raw);b=cpu(base).transpose(1,2,0);d=e.view(raw)
            yellow=(b[:,:,0]>1.22*b[:,:,2])&(b[:,:,1]>1.08*b[:,:,2])&(b[:,:,0]>.32)&d['repair_mask']&(d['actor_labels']==0)
            labs,_=label(yellow);sizes=np.bincount(labs.ravel());sizes[0]=0;yellow&=sizes[labs]>=40
            save(dest/('%03d_B.png'%raw),base)
            for gates in ['native','trained_R4']:
                if gates=='native':e.repair.gate_raw.fill_(20.)
                else:e.repair.gate_raw.copy_(gate)
                for actor in ['none','a5','a37','both']:
                    if actor=='none':mask=np.zeros(len(active),bool)
                    elif actor=='both':mask=np.ones(len(active),bool)
                    else:mask=e.ref['domain_'+actor][e.repair.road.ids]
                    e.repair.road.active.copy_(torch.tensor(mask,device='cuda'))
                    out=e.render(raw);rgb=cpu(out['rgb']).transpose(1,2,0);err=np.abs(rgb-b).mean(2);a=cpu(out['extra'][0])
                    rec=dict(raw=raw,gates=gates,surface=actor)
                    for name,region in [('all',np.ones(yellow.shape,bool)),('near',np.indices(yellow.shape)[0]>=260),('far',np.indices(yellow.shape)[0]<260)]:
                        m=yellow&region;rec[name]=dict(samples=int(m.sum()),native_yellow_MAE=float(err[m].mean()) if m.any() else None,road_alpha_T=float(a[m].mean()) if m.any() else None)
                    rows.append(rec);save(dest/('%03d_%s_%s.png'%(raw,gates,actor)),out['rgb'])
            e.repair.road.active.copy_(active);e.repair.gate_raw.copy_(gate)
    result=dict(updates=0,parent_unchanged=e.unchanged(),rows=rows,
        control='native gates=1 isolates forward road surface; trained_R4 holds the same joint gate fixed',
        independent_single_actor_training=False,explicit_smoothing_loss_in_R4=False,
        road_cell_m=.08,tangent_sigma_m=.052,normal_sigma_m=.015,opacity=.99005)
    (dest/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps([x for x in rows if x['raw']==105]),flush=True)
if __name__=='__main__':main()
