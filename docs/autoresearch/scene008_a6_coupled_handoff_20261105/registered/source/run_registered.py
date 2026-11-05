"""Registered support handoff, with the same group geometry and held domains."""
import gc
import hashlib
import numpy as np
import torch
from registered import RUN,SOURCE,Registered,fit_root,fit_joint,run_coupled,read,write,PRIMARY


def main():
    groups=read(RUN/'groups.json')
    outcomes=[]
    for rank in range(groups['qualified_count']):
        path=RUN/f'group_ledger_{rank}.npz'
        with np.load(path,allow_pickle=False) as f:
            ledger={k:f[k] for k in f.files}
        e=Registered(ledger)
        fit_root.fit_root(e,rank)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        choices=fit_joint.fit_joint(e,rank,digest)
        outcomes.append(run_coupled.evaluate(e,rank,choices,prefix='joint_'))
        assert e.tr.exp.unchanged()
        del e;gc.collect();torch.cuda.empty_cache()
    write(RUN/'joint_summary.json',dict(status='EXECUTION_PASS',groups=len(outcomes),all_groups_tested=True,
          accepted_groups=[r['rank'] for r in outcomes if r['method_accepted']],selection_used_heldout=False,
          source_registration='BUILD actual contribution linkage with canonical normal bounds',
          retirement=PRIMARY,training_updates=0))


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        main()
