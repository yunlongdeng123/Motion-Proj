"""Replay every saved joint group on the prespecified mechanism probes."""
import hashlib
import numpy as np
import torch

from coupled import RUN,Coupled,group_mask,read,write,local_metrics,seam,compare,PRIMARY
from run_coupled import measure


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    n=read(RUN/'groups.json')['qualified_count']
    checked=[]
    diagnostic=RUN/'diagnostic'
    diagnostic.mkdir()
    max_parity={k:0. for k in ('rgb','alpha','depth_sum','extra')}
    ledger_hash={}
    for rank in range(n):
        path=RUN/f'group_ledger_{rank}.npz'
        with np.load(path,allow_pickle=False) as f:
            ledger={k:f[k] for k in f.files}
        e=Coupled(ledger)
        gamma=read(RUN/f'joint_group{rank}_fit.json')['choices'][str(PRIMARY)]['gamma']
        expected=read(RUN/f'joint_group{rank}_evaluation.json')
        arms={'parent':(0.,0.),'retirement':(PRIMARY,0.),'proxy_only':(0.,gamma),'coupled':(PRIMARY,gamma)}
        unit=e.proxy_strength.clone()
        for raw in (105,113,130):
            mask=group_mask(e,raw)
            for state in ('black','keep','delete'):
                b=e.bundle(raw,state)
                baseline=b.render(0.,0.)
                reference,_=b.coverage(0.,0.,mask)
                e.proxy_strength=unit*unit.new_tensor(gamma)
                physical=e.merged(raw,state,True).base()
                e.proxy_strength=unit
                p=e.parity(physical,b.render(0.,gamma))
                for k,v in p.items():
                    max_parity[k]=max(max_parity[k],v)
                assert p['depth_sum']<5e-5 and max(p[k] for k in ('rgb','alpha','extra'))<4e-6,p
                for arm,(s,g) in arms.items():
                    out=b.render(s,g);coverage,_=b.coverage(s,g,mask)
                    m=measure(e,raw,out,mask,coverage,reference,baseline)
                    local=local_metrics(e,raw,state,baseline,out,mask)
                    old=expected['rows'][raw-89]['states'][state][arm]
                    assert m==old['group'] and local==old['local'],(rank,raw,state,arm)
                    if state=='black':
                        actual=e.tr.measure(raw,out)
                        assert actual['metrics']==old['scene'] and actual['violations']==old['failures']
                    else:
                        values,issues=compare(e,raw,state,baseline,out,seam(e,raw,state))
                        assert values==old['scene'] and issues==old['failures']
                    if state=='black':
                        from depth_order import save
                        visible=e.tr.raw(raw)['obs']
                        spatial,_=b.coverage(s,g,visible)
                        save(diagnostic/f'{rank}_{raw:03d}_{arm}_coverage.png',spatial[None].expand(3,-1,-1))
                        baseline_spatial,_=b.coverage(0.,0.,visible)
                        change=(spatial-baseline_spatial)/.05
                        heat=torch.stack((change.clamp(0,1),torch.zeros_like(change),(-change).clamp(0,1)))
                        save(diagnostic/f'{rank}_{raw:03d}_{arm}_change.png',heat)
                    checked.append([rank,raw,state,arm])
        assert e.tr.exp.unchanged()
        ledger_hash[str(rank)]=sha(path)
    write(RUN/'verification.json',dict(status='EXECUTION_PASS',groups=n,checked=checked,
          saved_ledger_exact_replay=True,physical_proxy_opacity_maxabs=max_parity,
          ledger_sha256=ledger_hash,all49_states=['black','keep','delete'],training_updates=0))


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        main()
