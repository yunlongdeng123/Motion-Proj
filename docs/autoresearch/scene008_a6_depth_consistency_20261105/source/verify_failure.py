"""Independently replay the sole qualified-arm guard failure at raw97."""
import numpy as np
import torch
from consistency import RUN,Consistency,read,write,save
from evaluate_consistency import ARMS,STATES,fields,extras
from heuristic_evaluate import seam
from street_metrics import compare
from depth_order import metric

def main():
    e=Consistency();expected=read(RUN/'evaluation.json')
    with np.load(RUN/'ledger.npz',allow_pickle=False) as f:ledger={k:f[k] for k in f.files}
    coefficients=fields(e,ledger);records=[]
    for state in STATES:
        w=e.frame(97,state);before=w.render(e.zero)
        edge=seam(e,97,state) if state!='black' else None
        for arm,strength in coefficients.items():
            out=w.render(strength);old=expected['rows'][8]['states'][state][arm]
            if state=='black':
                m=e.tr.measure(97,out);scene=m['metrics'];issues=m['violations']
            else:scene,issues=compare(e,97,state,before,out,edge)
            local,extra=extras(e,97,state,before,out)
            assert scene==old['scene'] and local==old['local'] and list(issues)+extra==old['issues']
            q=e.tr.evidence[97]['mask']
            assert metric(out['extra'][1 if state=='black' else 3],q)==old['lower']
            records.append(dict(raw=97,state=state,arm=arm,issues=list(issues)+extra))
            if state!='black':save(RUN/'render'/f'097_{state}_{arm}.png',out['rgb'])
        assert not any(e.parity(w.render(e.zero),w.trace.base()).values())
    assert e.tr.exp.unchanged()
    write(RUN/'failure_verification.json',dict(status='PASS',records=records,
        qualified_failure_exact_replay=True,source_models_unchanged=True))

if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():main()
