"""Actual renderer, save/reload and evaluation entry smoke for full basis R."""
import time
import numpy as np
import torch
from full_basis import RUN,FullBasis,parity,write
from evaluate_basis import measure

def main():
    e=FullBasis();records=[];start=time.monotonic()
    for raw in (105,113,130,97):
        t=time.monotonic();packet,audit=e.prepare(raw)
        restored=e.load(raw)
        for key in packet:assert np.array_equal(packet[key],restored[key]),key
        for state in ('black','keep','delete'):
            parent=e.merged(raw,state,False).base()
            identity=e.render_basis(raw,state)
            identity_error=parity(parent,identity,state)
            assert not any(identity_error.values()),(raw,state,identity_error)
            basis=e.render_basis(raw,state,packet)
            replay=e.render_basis(raw,state,restored)
            equality=parity(basis,replay,state)
            assert not any(equality.values()),(raw,state,equality)
            assert all(torch.isfinite(basis[k]).all() for k in ('rgb','alpha','depth_sum','extra'))
            measured=measure(e,raw,state,parent,basis)
            if state=='black':
                group=basis['extra'][0]+basis['extra'][2]+basis['extra'][3]
                assert float((group-basis['alpha'][0]).max())<5e-6
                quality=e.tr.measure(raw,basis)['violations']
            else:quality=measured['issues']
            records.append(dict(raw=raw,state=state,identity=identity_error,restored=equality,
                parent_rgb_change=float((parent['rgb']-basis['rgb']).abs().mean()),
                quality_issues=quality))
        print('BASIS_SMOKE',raw,round(time.monotonic()-t,3),audit['candidate_primitives'],
              audit['rewritten_primitives'],len(audit['unsplit']),len(audit['numerical_errors']),flush=True)
    assert e.tr.exp.unchanged()
    write(RUN/'smoke.json',dict(status='ENGINEERING_PASS',actual_gpu=True,records=records,
        parent_identity_exact=True,basis_saved_state_exact=True,training_updates=0,
        elapsed_seconds=time.monotonic()-start,quality_is_not_smoke_acceptance=True))

if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():main()
