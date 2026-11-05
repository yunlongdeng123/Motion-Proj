"""Real full-domain no-op/operator/reference smoke before qualification."""
import torch
from consistency import RUN,Consistency,Window,write

def main():
    e=Consistency();rows=[]
    assert int(e.tr.lower.sum())==431
    assert sum(int(e.tr.evidence[r]['mask'].sum()) for r in range(89,138))==26229
    for raw in (105,113,130):
        reference=e.reference(raw)
        assert torch.isfinite(reference[e.tr.evidence[raw]['mask']]).all()
        for state in ('black','keep','delete'):
            w=e.frame(raw,state)
            before=w.trace.base();noop=w.render(e.zero)
            parity=e.parity(before,noop)
            assert not any(parity.values()),(raw,state,parity)
            expected=Window.render(w,torch.full((len(e.boxes),),.5,device='cuda'))
            after=w.render(e.zero+.5)
            constant=e.parity(expected,after)
            assert constant['depth_sum']<5e-5 and max(constant[k] for k in ('rgb','alpha','extra'))<4e-6,constant
            if state=='black':assert e.tr.measure(raw,noop)==e.prior['black'][raw-89]
            rows.append(dict(raw=raw,state=state,no_op=parity,constant_crop=constant,
                center_feature_maxabs_m=w.center_feature_maxabs_m,partition=w.partition_max,
                affected_hits=len(w.indices),fraction_max=float(w.fraction.max()) if len(w.fraction) else 0.))
            print('CONSISTENCY_SMOKE',raw,state,constant,w.center_feature_maxabs_m,flush=True)
            del w
    assert e.tr.exp.unchanged()
    write(RUN/'smoke.json',dict(status='PASS',rows=rows,actual_gpu_renderer=True,
        full_target_pixels=26229,lower_PVD_rows=431,native_rows=len(e.zero),training_updates=0))

if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():main()
