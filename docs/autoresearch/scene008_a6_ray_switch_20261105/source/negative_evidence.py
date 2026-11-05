"""Separate certified empty rays from unknown non-target support in the clipping controls."""
import numpy as np
import torch
from switch import RUN,Trial,RAWS,write
from ray_support import metric
from render_pvd import OBS


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    trial=Trial()
    trial.parents()
    with np.load(RUN/'switch.npz',allow_pickle=False) as f:
        accepted=torch.as_tensor(f['selected'],device='cuda')
    rows=[]
    for raw in RAWS:
        old=trial.original_pair[raw]
        new=trial.readout(raw,accepted)
        all_rows=trial.readout(raw,trial.base_options['selected'])
        with np.load(OBS/f'{raw:03d}.npz',allow_pickle=False) as f:
            free=torch.as_tensor(f['free'][0],device='cuda')
        outside=trial.protection[raw]['outside']
        target=trial.evidence[raw]['mask']
        assert not (free&target).any()
        verified=free&outside
        remain=outside&~free
        dims=dict(raw=raw,known_empty=int(free.sum()),empty_outside=int(verified.sum()),variants={})
        for name,pkg in [('previous',old),('accepted',new),('all_switched',all_rows)]:
            delta=(pkg['rgb']-trial.raw38[raw]['rgb']).abs().mean(0)
            dims['variants'][name]=dict(empty_alpha=metric(pkg['alpha'][0],free),
                                        supported_empty_alpha=metric(pkg['alpha'][0],verified),
                                        outside_empty_rgb=metric(delta,verified),
                                        outside_remaining_rgb=metric(delta,remain))
        rows.append(dims)
    pooled={}
    for name in rows[0]['variants']:
        pooled[name]={}
        for key in rows[0]['variants'][name]:
            r=[row['variants'][name][key] for row in rows]
            n=sum(x['count'] for x in r)
            pooled[name][key]=dict(pixels=n,mean=sum(x['sum'] for x in r)/n if n else None)
    write(RUN/'negative_evidence.json',dict(rows=rows,pooled=pooled,training_updates=0,
                                           note='Diagnostic only. No change to the accepted protection mask.'))
    print(pooled,flush=True)


if __name__=='__main__':
    main()
