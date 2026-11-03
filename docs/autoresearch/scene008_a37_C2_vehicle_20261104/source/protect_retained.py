"""用49帧原生贡献显式保护另外三辆保留车的旧静态支撑。"""
from pathlib import Path
import json,time
import torch,numpy as np
from vehicle_runtime import VehicleExperiment
RUN=Path(__file__).resolve().parent.parent

def main():
    torch.set_num_threads(1);start=time.monotonic()
    e=VehicleExperiment(dict(actor_subset='none',legacy_replacement='native',road_policy='delete_only'))
    count=len(e.n.static.get_xyz);scores=torch.zeros(count,device='cuda');rows=[]
    for raw in range(89,138):
        labels=torch.as_tensor(e.old.view(raw)['actor_labels'],device='cuda')
        mask=(labels>0)&(labels<4)
        marker=torch.zeros((count,1),device='cuda',requires_grad=True)
        out=e.render(raw,False,static_marker=marker)
        flow=torch.autograd.grad(out['static_marker'][0,mask].sum(),marker,only_inputs=True)[0][:,0].detach()
        scores=torch.maximum(scores,flow);rows.append(dict(raw=raw,other_pixels=int(mask.sum()),contribution=float(flow.sum())))
        del out,flow,marker
    ids=torch.nonzero(scores>1e-3).flatten()
    np.savez_compressed(RUN/'inputs/retained_static_protection.npz',ids=ids.cpu().numpy(),scores=scores[ids].cpu().numpy())
    result=dict(rows=len(ids),threshold_native_alpha_pixel_sum=.001,source='maximum original static alphaT on reviewed visible other-car labels1..3 over49 frames',
        optimizer_updates=0,parent_unchanged=e.unchanged(),seconds=time.monotonic()-start,frames=rows)
    (RUN/'inputs/retained_static_protection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='frames'}),flush=True)

if __name__=='__main__':main()
