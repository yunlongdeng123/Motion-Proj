"""Freeze CPU-only T2 HF input and honest readiness/accounting; never launches execution."""
from pathlib import Path
import json, hashlib, tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/trellis_hf'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    selected=ROOT/'seed_r5_nodal_b034'
    receipt=json.loads((selected/'receipt.json').read_text())
    audits={name:json.loads((ROOT/name/'receipt.json').read_text())for name in ['reader_audit_r5','runtime_audit_r5']}
    if receipt['strict_faithful_gate'] is not False or receipt['exploratory_seed_valid'] is not True:
        raise ValueError('do not silently reinterpret the accepted bounded exploratory role')
    with np.load(selected/'H_seed.npz')as z:
        seed=z['H_seed_latent'];shape=list(seed.shape)
    entry=dict(schema='T2_HF_CPU_frozen_initialization_entry.v1',task_id='MODULE2-TEACHER-DETAIL-CPU-20261119',
        role='bounded_projected_HF_initialization_only',selected_seed='seed_r5_nodal_b034/H_seed.npz',
        selected_seed_remote=REMOTE+'/seed_r5_nodal_b034/H_seed.npz',H_parameter_key='H_seed_latent',
        H_shape=shape,H_dtype=str(seed.dtype),seed_sha256=sha(selected/'H_seed.npz'),
        strict_faithful_gate=False,exploratory_seed_valid=True,primary_four_arms_depend_on_T2=False,
        model_H_bound=.34,effective_HF_initialization_cap=.02,
        actual_HF_RMS=receipt['actual']['RGB_RMS'],actual_HF_abs_max=receipt['actual']['abs_max'],
        retained_target_energy=receipt['retained_signal_energy_ratio'],cosine_to_original_HF_target=receipt['cosine'],
        source_and_mask_ledger=receipt['parent_ledger_path'],source_and_mask_ledger_sha256=receipt['parent_ledger_sha256'],
        common_base_buffer_sha256=receipt['common_base_buffer_sha256'],
        initialization_contract='same teacherH+pixel (or explicitly matched average) reader and same A200; replace only H at B entry; H Adam zero; no extra loss, same B300/C300/trace',
        base_low_teacherRGB_coverage_original_ledger_modified=False,
        target_artifact='seed_r3_b034/H_seed.npz::desired_effective_H (frozen target only, NOT selected initialization)',
        signal_limit='partial projection; original faithful criterion fails, not a full-T2 representation or capacity test',
        future_required=['actual A200/Adam/RNG/sampler lifecycle','authorized native K/basis/reader preflight','real-scene smoke before formal'],
        CPU_checks_only=True,GPU_authorized=False,GPU_entry_permitted=False,auto_launch=False,
        new_inference_calls=0,native_renderer_calls=0,model_training_updates=0,power_operations=0)
    write(ROOT/'seed_entry.json',entry)
    cases=[]
    for name in ['seed_lsmr_r1_b034','seed_lsmr_r2_b034','seed_lsmr_r3_continuous_b034',
                 'seed_lsmr_r4_continuous_20k_b034','seed_r5_nodal_b034']:
        p=ROOT/name/'receipt.json';r=json.loads(p.read_text())
        cases.append(dict(path=name,CPU_seconds=r['CPU_seconds'],solver_iterations=sum(x['iterations']for x in r['solver']),
                          solver_stop_codes=[x['stop_code']for x in r['solver']],status=r['status']))
    write(ROOT/'receipts/engineering_accounting.json',dict(schema='T2_HF_CPU_encoding_accounting.v1',cases=cases,
        total_Krylov_iterations=sum(x['solver_iterations']for x in cases),
        total_projection_CPU_seconds=sum(x['CPU_seconds']for x in cases),
        original_eroded_candidates_preserved=['seed_r2_b034','seed_r3_b034'],
        source_preparation_CPU_seconds=4.343,tests_passed=7,
        preparation_failures=['float32 reduction used unnormalized sum instead of DC mean; checker corrected',
                              'mH dtype/layout roundtrip mismatch corrected before final source',
                              'partial SCP dependency attempt had no input file and zero solver iterations',
                              'first continuous RHS channel-shape error before solver; log preserved'],
        software_semantic_failures_preserved=['extra any-corner quarter veto introduced unregistered erosion',
                                             'extra independent window target differed from fixed nodal H runtime reconstruction'],
        not_scene_or_photo_training=True,model_training_updates=0,new_inference_calls=0,native_calls=0,GPU_operations=0,power_operations=0))
    included=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(x in ('__pycache__','.pytest_cache')for x in rel.parts)or p.suffix=='.tar':continue
        if rel.as_posix()=='extraction_r1/physical_grid.npz':continue
        if p.name=='delivery_manifest.json':continue
        included.append(p)
    manifest=dict(schema='T2_HF_CPU_delivery_manifest.v1',role=entry['role'],strict_faithful_gate=False,
        exploratory_seed_valid=True,GPU_authorized=False,auto_launch=False,
        files={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))for p in included},
        local_dense_extraction_retained=dict(path='extraction_r1/physical_grid.npz',bytes=(ROOT/'extraction_r1/physical_grid.npz').stat().st_size,
                                             sha256=sha(ROOT/'extraction_r1/physical_grid.npz')))
    write(ROOT/'delivery_manifest.json',manifest)
    included.append(ROOT/'delivery_manifest.json')
    archive=ROOT/'trellis_hf_cpu_delivery.tar'
    if archive.exists():raise FileExistsError(archive)
    with tarfile.open(archive,'w')as tar:
        for p in included:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(included),
        seed_entry=str(ROOT/'seed_entry.json'),strict_faithful_gate=False,exploratory_seed_valid=True,GPU_entry_permitted=False)))


if __name__=='__main__':main()
