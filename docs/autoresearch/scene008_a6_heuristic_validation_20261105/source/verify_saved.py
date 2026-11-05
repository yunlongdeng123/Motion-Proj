"""Reopen the saved heuristic assets and replay the probe frames."""
import hashlib
import numpy as np
import torch

from rescue import RUN, HELD, Probe, read, write
from positive import Positive
from heuristic_evaluate import pair, panel_mask, local_metrics, seam
from street_metrics import compare


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    checked=[]
    ledgers={}
    for label, factory in (('rescue',Probe),('positive',Positive)):
        expected=read(RUN/f'{label}_rgb_evaluation.json')
        ledger_path=RUN/f'{label}_ledger.npz'
        assert expected['ledger_sha256']==sha(ledger_path)
        with np.load(ledger_path,allow_pickle=False) as f:
            ledger={k:f[k] for k in f.files}
        assert not np.isin(ledger['source'],HELD).any()
        assert not np.isin(ledger['build_witness_raw'],HELD).any()
        assert ledger['build_witness_mask'].shape[1]==len(ledger['xyz'])
        exp=factory()
        exp.load_proxy(ledger)
        for raw in (105,113,130):
            mask=panel_mask(exp,raw)
            for state in ('black','keep','delete'):
                outputs=pair(exp,raw,state)
                base=outputs['parent']
                for arm,pkg in outputs.items():
                    old=expected['rows'][raw-89]['states'][state][arm]
                    local=local_metrics(exp,raw,state,base,pkg,mask)
                    assert local==old['local'],(label,raw,state,arm,'local')
                    if state=='black':
                        r=exp.measure_volume(raw,pkg)
                        assert r['metrics']==old['metrics'] and r['violations']==old['failures']
                    else:
                        m,f=compare(exp,raw,state,base,pkg,seam(exp,raw,state))
                        assert m==old['metrics'] and f==old['failures']
                    checked.append([label,raw,state,arm])
        assert exp.tr.exp.unchanged()
        ledgers[label]=dict(rows=len(ledger['xyz']),sha256=sha(ledger_path))
    raw=read(RUN/'first_return/summary.json')
    assert raw['status']=='PASS' and len(raw['rows'])==199 and raw['first_return_only']
    assert len(list((RUN/'first_return/witnesses').glob('*.npz')))==199
    write(RUN/'verification.json',dict(status='EXECUTION_PASS',method_accepted=False,
          checked=checked,exact_replay=True,ledger=ledgers,heldout_not_used_for_proxy=True,
          first_return_frames=199,all49_evaluation_states=['black','keep','delete'],
          note='Independent saved-ledger replay covers the three prespecified probe views. Metrics include all 49.'))


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        main()
