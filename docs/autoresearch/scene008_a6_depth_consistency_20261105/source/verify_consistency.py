"""Sparse-ledger reconstruction and independent saved-state GPU replay."""
import hashlib
import numpy as np
import torch
from consistency import RUN,Consistency,BUILD,HELD,read,write
from evaluate_consistency import ARMS,STATES,fields,extras
from register_consistency import qualify,frame_masses
from heuristic_evaluate import seam
from street_metrics import compare
from depth_order import metric

def main():
    path=RUN/'ledger.npz';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as f:ledger={k:f[k] for k in f.files}
    contract=read(RUN/'contract.json')
    assert str(ledger['contract_sha256'])==hashlib.sha256((RUN/'contract.json').read_bytes()).hexdigest()
    masses=np.zeros_like(ledger['masses']);maxframe=np.zeros_like(masses)
    framesC=np.zeros_like(ledger['framesC']);framesCM=np.zeros_like(ledger['framesCM'])
    for raw in BUILD:
        with np.load(RUN/'registration_frames'/f'{raw:03d}.npz',allow_pickle=False) as f:
            current=np.zeros_like(masses);current[:,f['ids']]=f['masses']
            assert np.array_equal(f['native_row_id'],ledger['native_row_id'][f['ids']])
            assert int(f['raw'])==raw
        masses+=current;np.maximum(maxframe,current,out=maxframe)
        framesC+=current[1]>0;framesCM+=current[2]>0
    assert np.array_equal(masses,ledger['masses']) and np.array_equal(maxframe,ledger['max_frame_mass'])
    assert np.array_equal(framesC,ledger['framesC']) and np.array_equal(framesCM,ledger['framesCM'])
    selection=qualify(masses,framesC,framesCM,contract['qualification'])
    for key in ('eligible_center','eligible_dual'):assert np.array_equal(selection[key],ledger[key])
    assert selection['match_coefficient']==float(ledger['match_coefficient'])
    expected=read(RUN/'evaluation.json');e=Consistency();coeff=fields(e,ledger)
    selected_path=RUN/'selected_state.npz'
    with np.load(selected_path,allow_pickle=False) as f:
        chosen=e.zero.new_tensor(f['strength']);adopted=bool(f['adopted'])
        assert np.array_equal(f['native_row_id'],ledger['native_row_id']) and str(f['ledger_sha256'])==digest
    assert adopted==expected['adopted']
    assert torch.equal(chosen,coeff['dual'] if adopted else coeff['parent'])
    for arm in ARMS:
        with np.load(RUN/f'{arm}_state.npz',allow_pickle=False) as f:
            assert np.array_equal(f['native_row_id'],ledger['native_row_id']) and np.array_equal(f['boxes'],ledger['boxes'])
            assert np.array_equal(f['strength'],coeff[arm].cpu().numpy()) and str(f['ledger_sha256'])==digest
    actual_dose={arm:float((ledger['masses'][0]*coeff[arm].cpu().numpy().astype(np.float64)).sum()) for arm in ARMS}
    dose_gap=abs(actual_dose['dual']-actual_dose['center_matched'])
    assert dose_gap<=1e-6+1e-7*actual_dose['dual']
    rebuild=[]
    for raw in (106,114,129):
        actual,_=frame_masses(e,raw)
        with np.load(RUN/'registration_frames'/f'{raw:03d}.npz',allow_pickle=False) as f:
            old=np.zeros_like(actual);old[:,f['ids']]=f['masses']
        diff=float(np.abs(actual-old).max());assert diff<1e-9,(raw,diff)
        rebuild.append(dict(raw=raw,mass_maxabs=diff))
    checks=[];chosen_checks=[]
    for raw in (105,113,130):
        q=e.tr.evidence[raw]['mask'];den=e.tr.evidence[raw]['lower_alpha']
        for state in STATES:
            w=e.frame(raw,state);before=w.render(e.zero)
            edge=seam(e,raw,state) if state!='black' else None
            for arm,strength in coeff.items():
                out=w.render(strength);old=expected['rows'][raw-89]['states'][state][arm]
                if state=='black':
                    measured=e.tr.measure(raw,out);scene=measured['metrics'];issues=list(measured['violations'])
                else:scene,issues=compare(e,raw,state,before,out,edge)
                local,additional=extras(e,raw,state,before,out)
                assert scene==old['scene'] and local==old['local'] and list(issues)+additional==old['issues']
                lower=out['extra'][1 if state=='black' else 3]
                assert metric(lower,q)==old['lower'] and metric(den,q)==old['isolated']
                checks.append([raw,state,arm])
            out=w.render(chosen)
            reference=w.render(coeff['dual']) if adopted else w.trace.base()
            parity=e.parity(out,reference);assert not any(parity.values()),parity
            chosen_checks.append(dict(raw=raw,state=state,selected_parity=parity))
            del w
    assert e.tr.exp.unchanged() and hashlib.sha256(path.read_bytes()).hexdigest()==digest
    write(RUN/'verification.json',dict(status='PASS',sparse_BUILD_frames_exact=40,
        registration_rebuilds=rebuild,checks=checks,selected_state=chosen_checks,ledger_sha256=digest,
        source_models_unchanged=True,actual_first_order_dose=actual_dose,dose_matched_abs_gap=dose_gap,
        primary_candidate='dual',adopted=adopted,training_updates=0))
    print('CONSISTENCY_VERIFIED',len(checks),dose_gap,adopted,flush=True)

if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():main()
