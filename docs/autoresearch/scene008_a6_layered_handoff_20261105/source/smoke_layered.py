"""Real renderer smoke: persistent partition, parent parity and saved-state entry."""
import gc
import hashlib
import numpy as np
import torch
from layered import RUN,Layered,core,write,fixed_group_mask

def main():
    records=[];hashes={}
    for rank in range(3):
        p=RUN/f'group_ledger_{rank}.npz'
        hashes[str(rank)]=hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p,allow_pickle=False) as f:ledger={k:f[k] for k in f.files}
        e=Layered(ledger)
        write(RUN/f'group{rank}_geometry.json',e.audit())
        for raw in (105,113,130):
            for state in ('black','keep','delete'):
                trace=e.merged(raw,state,True)
                mass={};parity={}
                for layer in ('near','same','far','full'):
                    e.set_layer(layer);b=core.Bundle(e,trace)
                    mass[layer]=b.fraction
                    out=b.render(0.,0.)
                    base=e.merged(raw,state,False).base()
                    parity[layer]=e.parity(base,out)
                    assert not any(parity[layer].values()),(rank,raw,state,layer,parity[layer])
                    assert all(torch.isfinite(out[k]).all() for k in ('rgb','alpha','depth_sum','extra'))
                    del b
                error=float((mass['near']+mass['same']+mass['far']-mass['full']).abs().max())
                assert error<3e-4,(rank,raw,state,error)
                records.append(dict(group=rank,raw=raw,state=state,mass_partition_maxabs=error,no_op=parity))
                print('LAYER_SMOKE',rank,raw,state,error,flush=True)
        assert e.tr.exp.unchanged() and hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(rank)]
        del e;gc.collect();torch.cuda.empty_cache()
    write(RUN/'smoke.json',dict(status='PASS',records=records,ledger_sha256=hashes,
         actual_gpu_renderer=True,partition_error_limit=3e-4,all_no_op_exact=True,training_updates=0))

if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():main()
