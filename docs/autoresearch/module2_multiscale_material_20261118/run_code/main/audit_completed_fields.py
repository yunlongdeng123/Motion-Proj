"""完整正式field/原始ledger CPU审计与源清单；无GPU、无优化、无默认scene写回。"""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse,gc,hashlib,json
import numpy as np
import torch

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
G=P.parent/'road_capacity_gpu_20261118_r1'
STAGES=['A200','L800','B500','ABC800']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def same(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def tensor_hash(value):
    a=value.detach().numpy()
    h=hashlib.sha256(json.dumps(dict(shape=a.shape,dtype=str(a.dtype))).encode())
    if a.ndim==0:h.update(np.ascontiguousarray(a).data)
    else:
        for i in range(0,len(a),1024):h.update(np.ascontiguousarray(a[i:i+1024]).data)
    return h.hexdigest()
def hashes(path):
    d=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    s=d['state']
    if set(k for k in s if k in ['low','high'])!={'low','high'}:
        raise ValueError('missing shared trainable fields')
    if any(not torch.isfinite(s[k]).all() for k in ['low','high']):
        raise ValueError('nonfinite learned fields')
    result={k:tensor_hash(v) for k,v in s.items() if k not in ['low','high']}
    setting=d['settings']
    del s,d;gc.collect()
    return result,setting

def main(run):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU-only audit requires CUDA hidden')
    torch.set_num_threads(1)
    run=Path(run)
    output=run/'field_and_ledger_audit.json'
    manifest=run/'source_manifest_final.json'
    if output.exists() or manifest.exists():raise FileExistsError('preserve completed audit')
    frozen=json.loads((run/'frozen_contract.json').read_text())
    actual={path:sha(path)==expected for path,expected in frozen['binding'].items()}
    if not all(actual.values()):raise AssertionError('frozen input/source changed')
    parent=Path(frozen['method']['coverage_parent'])
    with np.load(P/'method_scope_r1/scope.npz') as s:
        rows=s['active_world_rows'].copy();eligible=s['eligible'].copy()
    with np.load(parent/'road.npz') as r:
        world_mask=np.zeros(len(r['active']),bool);world_mask[rows]=eligible
    initial,setting=hashes(P/'cpu_exact_r1/field_initial.pt')
    mask_keys={'high_parameter_mask','high_output_mask','low_parameter_mask'}
    reference=None;stages={}
    for stage in STAGES:
        asset=run/stage/'asset';checks={}
        complete=json.loads((run/stage/'stage_complete.json').read_text())
        checks['stage_completed']=complete['completed'] is True
        with np.load(parent/'road.npz',allow_pickle=False) as before,np.load(asset/'road.npz',allow_pickle=False) as after:
            for key in before.files:
                b,a=before[key],after[key]
                checks['road_'+key+'_exact']=same(b[~world_mask],a[~world_mask]) if key=='reference_srgb' else same(b,a)
            checks['active_world_order_exact']=np.array_equal(np.flatnonzero(after['active']),rows)
            effective=after['reference_srgb'].copy()
        with np.load(parent/'provenance.npz',allow_pickle=False) as before,np.load(asset/'provenance.npz',allow_pickle=False) as after:
            for key in before.files:
                if key!='effective_row_rgb':checks['prior_'+key+'_bytes_exact']=same(before[key],after[key])
            checks['derived_effective_row_rgb']=same(after['effective_row_rgb'],effective)
            checks['capacity_effective_rgb']=same(after['capacity_effective_rgb'],effective)
            checks['capacity_color_eligible_exact']=same(after['capacity_color_eligible'],world_mask)
            with np.load(parent/'road.npz') as road:
                checks['capacity_source_prior_exact_parent']=same(after['capacity_source_prior_reference_srgb'],road['reference_srgb'])
        buffers,settings=hashes(asset/'world_field.pt')
        checks['all_field_settings_fixed']=settings==setting
        checks['all_source_kernel_buffers_exact_initial']=all(buffers[k]==v for k,v in initial.items() if k not in mask_keys)
        if reference is None:reference=buffers
        checks['all_field_buffers_and_parameter_masks_frozen']=buffers==reference
        receipt=json.loads((asset/'field_receipt.json').read_text())
        checks['field_authority_hash']=sha(asset/'world_field.pt')==receipt['field_sha256']
        checks['derived_cache_hash']=sha(asset/'road.npz')==receipt['derived_road_sha256']
        evaluation=json.loads((run/stage/'rendered/evaluation.json').read_text())
        checks['full_K77266']=sum(r['K_pixels'] for r in evaluation['rows'])==77266
        checks['all_native_K_exact']=all(r['K_RGB_max']==0 for r in evaluation['rows'])
        checks['all_native_alpha_exact']=all(r['alpha_exact_parent'] for r in evaluation['rows'])
        if not all(checks.values()):raise AssertionError(stage+': '+str([k for k,v in checks.items() if not v]))
        stages[stage]=dict(status='PASS',checks=checks,field_buffer_sha256=buffers)
    result=json.loads((run/'results.json').read_text())
    checks=dict(formal1400=result['formal_physical_updates']==1400,
        no_formal_replay=result['actual_optimizer_updates_all_attempts']==1400 and result['engineering_replay_updates']==0,
        future_sampler_trace_exact=result['future_trace_exact_shared'] is True,
        B_low_and_AdamL_exact=result['drift']['B_low_and_AdamL_frozen_exact'] is True,
        native_forward_accounting_1490=result['native_calls_training_context']+result['additional_public_reload_calls']==1490,
        no_production_promotion=result['production_promotion'] is False)
    if not all(checks.values()):raise AssertionError('budget/identity check failed')
    output.write_text(json.dumps(dict(schema='multiscale_formal_allfields_ledger_CPU_audit.v1',status='PASS',
        GPU_executed=False,native_calls=0,optimizer_updates=0,stages=stages,checks=checks,
        all_original_observed_source_known_unknown_ledger_immutable=True,
        only_original_provenance_field_permitted_to_update='effective_row_rgb (derived shader cache)',
        all_nonRGB_road_fields_exact_parent=True,all_serialized_quad_buffers_exact_prevalidated_source=True,
        frozen_input_bindings=actual,quality_acceptance_not_inferred=True),indent=2)+'\n')
    groups=dict(core=['world_material_field.py','road_chart_sampling.py'],
        native=['train_world_material.py','field_native_io.py'],
        formal=['run_material_pair.py'],
        engineering=['run_material_lifecycle.py','complete_lifecycle_readonly.py','prepare_real_field.py',
            'qualify_field_numerics.py','probe_multiscale_native.py','probe_shared_field.py'],
        cpu_audits=['audit_source_kernel_impact.py','summarize_field_components.py','audit_completed_fields.py'],
        tests=['test_world_material_field.py','test_optimizer_restore.py'])
    sources={str(P/'code'/name):sha(P/'code'/name) for names in groups.values() for name in names}
    sources[str(G/'code/road_capacity_native.py')]=sha(G/'code/road_capacity_native.py')
    manifest.write_text(json.dumps(dict(schema='multiscale_scientific_source_manifest.v1',source_frozen=True,
        source_groups=groups,source_sha256=sources,scientific_run=str(run),
        formal_contract_sha256=sha(P/'execution_contract.json'),frozen_run_contract_sha256=sha(run/'frozen_contract.json'),
        source_chart_helper_QA_owned=True,root_publication_copy_may_use_new_module_names=True,
        authoritative_scientific_run_source_is_original_bytes=True,default_scene_promoted=False),indent=2)+'\n')
    print(json.dumps(dict(status='PASS',stages={k:len(v['checks']) for k,v in stages.items()},
        budget_checks=checks,output=str(output),source_manifest=str(manifest))),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=P/'formal_r1')
    main(ap.parse_args().run)

