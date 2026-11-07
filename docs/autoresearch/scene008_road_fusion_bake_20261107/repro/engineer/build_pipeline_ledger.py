"""CPU-only state/compute/material accounting; no duplicate full49 evaluation."""
from pathlib import Path
import argparse
import json
from collections import Counter
import numpy as np
from shared_fit_common import OUT,RUN,BUILD,load,read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--with-replay',action='store_true');args=p.parse_args()
    records=[]
    q=read(OUT/'qualification_receipt.json')
    records.append(dict(role='shared_column_qualification',forwards=q['forwards'],backwards=q['backwards'],optimizer_steps=0,
        worker_window_seconds=q['seconds'],receipt=str(OUT/'qualification_receipt.json'),peak_allocated=q['peak_allocated_bytes'],peak_reserved=q['peak_reserved_bytes']))
    roots=[('engineering_placeholder',RUN/'engineer/smoke_engineering_placeholder/direct'),('scientific_direct',OUT/'direct'),('scientific_fusion',OUT/'fusion')]
    restores=[]
    for role,branch in roots:
        for phase in ('smoke','restore'):
            receipt=read(branch/phase/'receipt.json')
            optimizer_steps=receipt['steps_executed']+(2 if phase=='smoke' else 0)
            records.append(dict(role=role+'_'+phase,forwards=receipt['forwards'],backwards=receipt['backwards'],optimizer_steps=optimizer_steps,
                formal_steps=0,worker_window_seconds=receipt['seconds'],receipt=str(branch/phase/'receipt.json'),
                peak_allocated=receipt['peak_allocated_bytes'],peak_reserved=receipt['peak_reserved_bytes']))
        details=read(branch/'restore/nextstep_numeric_details.json')
        restores.append(dict(role=role,checkpoint_loaded_all_exact=all(details['checkpoint_restore_before_next_steps'].values()),
            disposable_next_state_all_exact=details['new_nextstep_tree_exact'],
            differences=[entry for entry in details['leaves'] if not entry['exact']]))
    materials=[];orders={}
    for variant in ('direct','fusion'):
        branch=OUT/variant;r=read(branch/'formal/receipt.json');m=load(branch/'material_override.npz')
        assert r['formal_steps']==400 and r['forwards']==800 and r['backwards']==800
        order=[row['raw'] for row in r['records']];assert Counter(order)==Counter({raw:10 for raw in BUILD});orders[variant]=order
        assert np.array_equal(m['colours_srgb'][~m['eligible']],m['initial_colours_srgb'][~m['eligible']])
        records.append(dict(role=variant+'_formal',forwards=r['forwards'],backwards=r['backwards'],optimizer_steps=400,formal_steps=400,
            worker_window_seconds=r['seconds'],receipt=str(branch/'formal/receipt.json'),
            peak_allocated=r['peak_allocated_bytes'],peak_reserved=r['peak_reserved_bytes']))
        delta=m['colours_srgb'].astype(float)-m['initial_colours_srgb'].astype(float)
        bake=read(RUN/f'engineer/canonical_{variant}_v1/cpu_bake_verification.json')
        assert all(bake['unchanged_baseline_fields'].values()) and all(bake['all_payload_reload_checks'].values())
        materials.append(dict(variant=variant,path=str(branch/'material_override.npz'),sha256=sha(branch/'material_override.npz'),
            eligible_parents=int(m['eligible'].sum()),frozen_zero_mass_parents=int((~m['eligible']).sum()),
            changed_parents=int((delta!=0).any(1).sum()),colour_delta_maxabs=float(np.abs(delta).max()),
            initial_zero_mass_rows_byte_exact=True,zero_mass_rows_absent_from_Adam=True,
            colour_min=float(m['colours_srgb'].min()),colour_max=float(m['colours_srgb'].max()),
            canonical_asset=bake['asset_sha256'],canonical_manifest=bake['scene_manifest_sha256'],
            unchanged_nonmaterial_field_count=len(bake['unchanged_baseline_fields'])))
    assert orders['direct']==orders['fusion']
    replays=[]
    if args.with_replay:
        for variant in ('direct','fusion'):
            folder=RUN/f'engineer/canonical_{variant}_v1/parity';r=read(folder/'receipt.json')
            byte_checks=[]
            for raw in (105,113,128,130,137):
                for state in ('keep','delete'):
                    actual=load(folder/f'{raw:03d}'/f'{state}_float.npz')
                    expected=load(RUN/'results/full49'/f'{raw:03d}'/f'{variant}_{state}.npz')
                    for key,value in actual.items():
                        ref=expected[key];exact=value.dtype==ref.dtype and value.shape==ref.shape and value.tobytes()==ref.tobytes()
                        byte_checks.append(dict(raw=raw,state=state,field=key,shape_dtype_bytes_exact=exact));assert exact,(variant,raw,state,key)
            assert r['forwards']==10 and r['all_float_exact'] and r['all_PNG_byte_exact']
            records.append(dict(role=variant+'_fresh_canonical_replay',forwards=10,backwards=0,optimizer_steps=0,
                worker_window_seconds=r['seconds'],receipt=str(folder/'receipt.json'),peak_allocated=r['peak_allocated_bytes'],peak_reserved=r['peak_reserved_bytes']))
            replays.append(dict(variant=variant,forwards=10,all_float_shape_dtype_bytes_exact=all(x['shape_dtype_bytes_exact'] for x in byte_checks),
                all_PNG_byte_exact=True,field_checks=byte_checks))
    result=dict(schema='ROAD_SHARED_FUSION_BAKE_PIPELINE_STATE_COMPUTE_LEDGER_V1',records=records,
        additional_native_forwards=sum(x['forwards'] for x in records),native_backwards=sum(x['backwards'] for x in records),
        formal_optimizer_steps=800,smoke_and_disposable_optimizer_steps=18,
        reported_worker_windows_seconds=sum(x['worker_window_seconds'] for x in records),
        timing_definition='reported GPU-enabled worker windows include runtime/data/output work; not CUDA-event kernel timing',
        maximum_allocated_bytes=max(x['peak_allocated'] for x in records),maximum_reserved_bytes=max(x['peak_reserved'] for x in records),
        root294F_full49_scope='separate root-owned evaluation, excluded from above; full49 geometry assertions belong to root receipt',
        geometry_updates=0,opacity_updates=0,original_parameter_updates=0,source_information_matched=True,
        formal_frame_order_identical=True,formal_frame_counts={str(raw):10 for raw in BUILD},restoration=restores,materials=materials,replays=replays,
        target_manifest_sha256=read(OUT/'fit_contract.json')['target_manifest_sha256'],fit_contract_sha256=sha(OUT/'fit_contract.json'),
        capacity_probe='atlas_target_capacity.json, CPU-only, independent-pixel relaxation not a 3D feasibility certificate',
        method_limit='Only2053 qualified parent colours train;8045 remain initial bytes. This is not full-atlas writeback success.',
        adoption_claim=False,human_verdict=None,AutoDL_shutdown=False)
    target=RUN/'engineer'/('pipeline_compute_ledger.json' if args.with_replay else 'pipeline_compute_ledger_pre_replay.json');write(target,result)
    print(json.dumps(dict(path=str(target),sha256=sha(target),forwards=result['additional_native_forwards'],backwards=result['native_backwards'],
        formal_steps=800,worker_windows_seconds=result['reported_worker_windows_seconds'],materials=materials)))


if __name__=='__main__':main()
