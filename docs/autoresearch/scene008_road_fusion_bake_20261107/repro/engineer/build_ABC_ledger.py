"""Append the single posthoc C control to the immutable A/B accounting."""
from pathlib import Path
import argparse
from collections import Counter
import numpy as np
from shared_fit_common import RUN,BUILD,load,read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--with-replay',action='store_true');args=p.parse_args()
    engineer=RUN/'engineer';base_path=engineer/'pipeline_compute_ledger.json';base=read(base_path)
    directory=engineer/'material_fit_edgeblend_v1';branch=directory/'edgeblend';records=[]
    for phase in ('smoke','restore','formal'):
        path=branch/phase/'receipt.json';r=read(path)
        records.append(dict(role='posthoc_edgeblend_'+phase,forwards=r['forwards'],backwards=r['backwards'],
            optimizer_steps=r['steps_executed']+(2 if phase=='smoke' else 0),formal_steps=400 if phase=='formal' else 0,
            worker_window_seconds=r['seconds'],receipt=str(path),peak_allocated=r['peak_allocated_bytes'],peak_reserved=r['peak_reserved_bytes']))
    formal=read(branch/'formal/receipt.json');assert formal['formal_steps']==400 and formal['forwards']==800 and formal['backwards']==800
    order=[row['raw'] for row in formal['records']];assert Counter(order)==Counter({raw:10 for raw in BUILD})
    direct=read(engineer/'material_fit_v1/direct/formal/receipt.json');assert order==[r['raw'] for r in direct['records']]
    q=load(directory/'qualified_parents.npz');m=load(branch/'material_override.npz')
    assert np.array_equal(m['colours_srgb'][~q['eligible']],q['initial_colours_srgb'][~q['eligible']])
    restore=read(branch/'restore/nextstep_numeric_details.json');assert all(restore['checkpoint_restore_before_next_steps'].values())
    replay=None
    if args.with_replay:
        folder=engineer/'canonical_edgeblend_v1/parity';r=read(folder/'receipt.json')
        checks=[]
        for raw in (105,113,128,130,137):
            for state in ('keep','delete'):
                actual=load(folder/f'{raw:03d}'/f'{state}_float.npz');expected=load(RUN/'results/edgeblend49'/f'{raw:03d}'/f'edgeblend_{state}.npz')
                for key,a in actual.items():
                    b=expected[key];exact=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes();assert exact,(raw,state,key)
                    checks.append(dict(raw=raw,state=state,field=key,shape_dtype_bytes_exact=exact))
        assert r['forwards']==10 and r['all_float_exact'] and r['all_PNG_byte_exact']
        records.append(dict(role='posthoc_C_fresh_canonical_replay',forwards=10,backwards=0,optimizer_steps=0,formal_steps=0,
            worker_window_seconds=r['seconds'],receipt=str(folder/'receipt.json'),peak_allocated=r['peak_allocated_bytes'],peak_reserved=r['peak_reserved_bytes']))
        replay=dict(forwards=10,float_shape_dtype_bytes_exact=True,PNG_byte_exact=True,checks=checks)
    result=dict(schema='ABC_ROAD_FUSION_WRITEBACK_ACCOUNTING_V1',immutable_AB_ledger_sha256=sha(base_path),AB_ledger=base,
        posthoc_C_records=records,total_native_forwards=base['additional_native_forwards']+sum(r['forwards'] for r in records),
        total_native_backwards=base['native_backwards']+sum(r['backwards'] for r in records),
        total_reported_worker_windows_seconds=base['reported_worker_windows_seconds']+sum(r['worker_window_seconds'] for r in records),
        formal_optimizer_steps=1200,smoke_and_disposable_optimizer_steps=24,
        C_material=dict(path=str(branch/'material_override.npz'),sha256=sha(branch/'material_override.npz'),eligible_parents=2053,
            unqualified_parents_byte_frozen=8045,actual_changed_parents=int((m['colours_srgb']!=m['initial_colours_srgb']).any(1).sum()),
            current_fit_contract_sha256=sha(directory/'fit_contract.json')),
        C_restore=restore,C_replay=replay,qualification_reused_GPU_forwards=0,
        method_roles=dict(AB='same190-source information comparison, only Poisson fusion changes',
            C='single posthoc fixed10pixel edgeblend, strong original RGB anchoring, not isolated Poisson gain'),
        legacy_static_GPU_forwards=0,legacy_static_optimizer_steps=0,original_parameter_updates=0,geometry_updates=0,opacity_updates=0,
        root_evaluation_forwards_separate=490,root_evaluation_scope='294 A/B plus196 singleC, excluded from above counts',
        held_scope='C HELD target includes query O for construction diagnostics only; native RGB is not conditioned on O, evaluator may load it',
        adoption_claim=False,human_verdict=None,AutoDL_shutdown=False)
    path=engineer/('pipeline_compute_ledger_ABC.json' if args.with_replay else 'pipeline_compute_ledger_ABC_pre_replay.json');write(path,result)
    print(dict(path=str(path),sha256=sha(path),forwards=result['total_native_forwards'],backwards=result['total_native_backwards'],
        worker_windows_seconds=result['total_reported_worker_windows_seconds'],C_material=result['C_material']))


if __name__=='__main__':main()
