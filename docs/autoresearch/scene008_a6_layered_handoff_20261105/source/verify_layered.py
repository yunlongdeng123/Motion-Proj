"""Reload fitted and selected states; verify output and depth guard receipts."""
import gc
import hashlib
import numpy as np
import torch
from layered import RUN,Layered,core,fixed_group_mask as group_mask,read,write,PRIMARY
from run_coupled import measure
from coupled import local_metrics,seam,compare
from run_layered import non_target_depth

def main():
    checked=[];selected=[];opacity_max={k:0. for k in ('rgb','alpha','depth_sum','extra')}
    for rank in range(3):
        path=RUN/f'group_ledger_{rank}.npz';digest=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as f:ledger={k:f[k] for k in f.files}
        e=Layered(ledger)
        for layer in ('near','same','far'):
            e.set_layer(layer)
            expected=read(RUN/f'{layer}_group{rank}_evaluation.json')
            gamma=expected['selected_gamma']
            state_path=RUN/f'{layer}_group{rank}_selected.npz'
            accepted=expected['method_accepted']
            np.savez_compressed(state_path,retirement=np.float32(PRIMARY if accepted else 0.),
                gamma=np.full(len(ledger['xyz']),gamma if accepted else 0.,np.float32),
                layer=np.asarray(layer),group_ledger_sha256=np.asarray(digest),candidate_accepted=np.bool_(accepted))
            with np.load(state_path,allow_pickle=False) as f:
                s=float(f['retirement']);g=f['gamma'].copy()
                assert str(f['group_ledger_sha256'])==digest and str(f['layer'])==layer
            arm_params={'parent':(0.,0.),'retirement':(PRIMARY,0.),'proxy_only':(0.,gamma),'coupled':(PRIMARY,gamma)}
            unit=e.proxy_strength.clone()
            for raw in (105,113,130):
                m=group_mask(e,raw)
                for state in ('black','keep','delete'):
                    b=e.bundle(raw,state);before=b.render(0.,0.);reference,_=b.coverage(0.,0.,m)
                    e.proxy_strength=unit*gamma
                    physical=e.merged(raw,state,True).base();e.proxy_strength=unit
                    values=e.parity(physical,b.render(0.,gamma))
                    for key,value in values.items():opacity_max[key]=max(opacity_max[key],value)
                    assert values['depth_sum']<5e-5 and max(values[k] for k in ('rgb','alpha','extra'))<4e-6
                    for arm,(retire,strength) in arm_params.items():
                        out=b.render(retire,strength);coverage,_=b.coverage(retire,strength,m)
                        actual=measure(e,raw,out,m,coverage,reference,before)
                        old=expected['rows'][raw-89]['states'][state][arm]
                        assert actual==old['group'] and local_metrics(e,raw,state,before,out,m)==old['local']
                        if state=='black':
                            a=e.tr.measure(raw,out)
                            assert a['metrics']==old['scene'] and a['violations']==old['failures']
                        else:
                            metrics,issues=compare(e,raw,state,before,out,seam(e,raw,state))
                            assert metrics==old['scene'] and issues==old['failures']
                        checked.append([rank,layer,raw,state,arm])
                    coupled=b.render(PRIMARY,gamma)
                    depth=non_target_depth(e,raw,before,coupled,m)
                    old=next(r for r in expected['non_target_depth'] if r['raw']==raw and r['state']==state)
                    assert all(depth[k]==old[k] for k in depth)
                    out=b.render(s,g)
                    baseline=b.render(PRIMARY,gamma) if accepted else before
                    parity=e.parity(out,baseline)
                    assert max(parity.values())<5e-7,parity
                    if not accepted:
                        native=e.merged(raw,state,False).base()
                        assert not any(e.parity(out,native).values())
                    selected.append(dict(group=rank,layer=layer,raw=raw,state=state,
                        candidate_accepted=accepted,selected_ledger=str(state_path),parity=parity))
                    del b
            print('LAYER_REPLAY',rank,layer,accepted,flush=True)
        assert e.tr.exp.unchanged() and hashlib.sha256(path.read_bytes()).hexdigest()==digest
        del e;gc.collect();torch.cuda.empty_cache()
    write(RUN/'verification.json',dict(status='PASS',checked=checked,selected=selected,
        exact_saved_metric_replay=True,physical_proxy_opacity_maxabs=opacity_max,
        saved_selected_state_reloaded=True,training_updates=0,
        selection_scope='coefficients BUILD-only; HELD veto participates in asset selection',
        depth_scope='strict parent target checks black; non-target depth checks fixed a6 measurements outside group, all three states; other components use inherited RGB guards'))

if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():main()
