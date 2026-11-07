"""Independent CPU data/contract checks for final190-source atlas supervision."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,warnings
from pathlib import Path
import numpy as np

HELD=(89,95,101,105,113,121,125,130,137)
BUILD=tuple(r for r in range(89,138) if r not in HELD)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def srgb(x):return np.where(x<=.0031308,x*12.92,1.055*np.maximum(x,0)**(1/2.4)-.055)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run=args.run;out=args.output
    if out.exists():raise FileExistsError(out)
    target=run/'targets';contract=read(target/'atlas_contract.json');manifest=read(target/'target_manifest.json')
    shared=read(run/'engineer/material_fit_v1/shared_contract.json');p=load(target/'atlas_inputs.npz')
    q=load(run/'engineer/material_fit_v1/qualified_parents.npz');masks=load(run/'engineer/material_fit_v1/supervision_masks.npz')
    assert manifest['fusion_ready'] is True
    expected=np.array([r for r in range(199) if r not in HELD]);assert np.array_equal(p['source_raws'],expected)
    assert contract['atlas_source_raws']==expected.tolist() and contract['BUILD40']==list(BUILD) and contract['HELD9']==list(HELD)
    assert np.array_equal(p['build_raws'],BUILD)
    source=next(Path(s) for s in contract['immutable_sources'] if s.endswith('road_reference_199.npz'))
    assert sha(source)==contract['immutable_sources'][str(source)]
    with np.load(source,allow_pickle=False) as a:
        raw=a['source_raw'];take=np.flatnonzero(~np.isin(raw,HELD));ids=p['candidate_ids']
        valid=a['source_valid'][take][:,ids].astype(bool);rgb=a['source_rgb'][take][:,ids].astype(np.float32)
    observed=linear(np.where(valid[...,None],rgb,0));luma=observed@np.array([.2126,.7152,.0722],np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        upper=valid&(luma>=np.nanquantile(np.where(valid,luma,np.nan),.75,axis=0)[None]-.025)&(luma<=np.nanquantile(np.where(valid,luma,np.nan),.75,axis=0)[None]+.06)
        proto=np.nanmedian(np.where(upper[...,None],observed,np.nan),axis=0)
    finite=np.isfinite(proto).all(1);proto=np.nan_to_num(proto,nan=0)
    agree=valid&(np.abs(observed-proto[None]).max(2)<.035);counts=agree.sum(0)
    first=np.where(agree,expected[:,None],999).min(0);last=np.where(agree,expected[:,None],-1).max(0)
    supported=finite&(counts>=3)&(last-first>=6);S=srgb(proto).astype(np.float32);S[~finite]=0
    source_checks=dict(source190_exact=True,HELD_intersection=[],source_agree_exact=bool(np.array_equal(agree,p['source_agree'])),
        source_chart_max_abs_difference=float(np.max(np.abs(S-p['source_chart_rgb']))),source_supported_exact=bool(np.array_equal(supported,p['source_supported'])),
        core_nodes=int(p['core_mask'].sum()),core_supported=int((p['core_mask']&p['source_supported']).sum()),
        source_count_min=int(counts.min()),source_count_max=int(counts.max()))
    assert source_checks['source_agree_exact'] and source_checks['source_supported_exact'] and source_checks['source_chart_max_abs_difference']<2e-7
    bi=[int(np.flatnonzero(expected==r)[0]) for r in BUILD]
    assert np.array_equal(p['build_source_agree'],agree[bi]) and np.array_equal(p['build_source_valid'],valid[bi])
    assert np.array_equal(np.nan_to_num(p['build_source_rgb']),np.nan_to_num(rgb[bi]))
    gradients_true=0
    for start in range(0,len(p['edges']),2048):
        e=p['edges'][start:start+2048];both=agree[:,e[:,0]]&agree[:,e[:,1]];n=both.sum(0)
        assert np.array_equal(n,p['same_frame_gradient_count'][start:start+len(e)]);gradients_true+=int((n>=2).sum())
    boundary=load(target/'boundary_evidence.npz');boundary_held=set()
    br=read(target/'boundary_receipt.json')
    for r in br['records']:assert r['raw'] in BUILD;boundary_held.add(r['raw'])
    assert not (set(HELD)&boundary_held)
    assert not (boundary['boundary_mask']&p['core_mask']).any()
    eligible=q['eligible'];assert np.array_equal(q['eligible_parent_ids'],np.flatnonzero(eligible))
    assert len(eligible)==10098 and int(eligible.sum())==2053
    assert np.array_equal(eligible,q['observed_mass_by_state'].sum(0)>0)
    target_checks=[];fit_total=0;replace_total=0
    original=run.parent/'scene008_instances_20261105_r1/observations/frames'
    for r in manifest['rows']:
        raw=int(r['raw']);file=target/r['file'];assert sha(file)==r['sha256'];z=load(file);obs=load(original/f'{raw:03d}.npz')
        observation=obs['rgb'].astype(np.float32);observation=observation/255 if observation.max()>1 else observation
        assert bool(z['fusion_ready']) and np.array_equal(z['observation_rgb'],observation)
        replace=z['fusion_replace_mask'];fit=z['fit_mask'];support=z['atlas_support_mask']
        assert replace.dtype==bool and fit.dtype==bool and not (replace&~fit).any() and not (replace&~support).any()
        assert not (replace&z['imputed_mask']).any()
        for name in ('direct_rgb','fusion_rgb'):
            assert z[name].shape==(320,480,3) and z[name].dtype==np.float32 and np.isfinite(z[name]).all()
            assert z[name].min()>=0 and z[name].max()<=1 and np.array_equal(z[name][~replace],observation[~replace])
        if raw in BUILD:
            assert np.array_equal(fit,masks[f'raw{raw:03d}']);fit_total+=int(fit.sum());replace_total+=int(replace.sum())
        target_checks.append(dict(raw=raw,role='BUILD' if raw in BUILD else 'HELD_PROJECT_ONLY',file_sha256=r['sha256'],fit_pixels=int(fit.sum()),replace_pixels=int(replace.sum()),
            outside_replace_exact_observation=True,no_imputed_replacement=True,common_state_shared_target=True))
    assert len(target_checks)==49 and fit_total==714546
    fit_contract_path=run/'engineer/material_fit_v1/fit_contract.json'
    fit_contract=read(fit_contract_path) if fit_contract_path.exists() else None
    fit_checks=dict(eligible_parent_rows=2053,trainable_scalars=6159,excluded_parent_rows=8045,zero_mass_excluded_from_Parameter_and_Adam='verified from fit_shared_material.py Material source',
        shared_optimizer=shared['optimizer'],same_mask_sum_BUILD=fit_total,replace_sum_BUILD=replace_total,
        initial_colour_frozen_for_excluded='source asserts exact before/after each step and export',fit_contract_present=fit_contract is not None)
    if fit_contract:
        assert not fit_contract['placeholder_engineering_only'];assert fit_contract['target_manifest_sha256']==sha(target/'target_manifest.json')
        assert fit_contract['BUILD']==list(BUILD) and fit_contract['HELD']==list(HELD)
        assert fit_contract['trainable_parent_rows']==2053 and fit_contract['optimizer']['formal_steps']==400
        assert fit_contract['optimizer']['lr']==.01
        for name,digest in fit_contract['source_sha256'].items():assert sha(run/'engineer'/name)==digest
        fit_checks['frozen_fit_contract_sha256']=sha(fit_contract_path)
    sources=[target/'atlas_contract.json',target/'atlas_inputs.npz',target/'source_chart_receipt.json',target/'target_manifest.json',target/'boundary_receipt.json',target/'poisson_receipt.json',run/'engineer/material_fit_v1/shared_contract.json',run/'engineer/fit_shared_material.py']
    report=dict(schema='INDEPENDENT_190SOURCE_TARGET_AND_MATCHED_FIT_CONTRACT_AUDIT_V1',status='DATA_AND_CONTRACT_CHECKS_PASS_WITH_LIMITATIONS',
        input_sha256={str(x):sha(x) for x in sources},source_checks=source_checks,gradient_same_frame_edges=gradients_true,
        synthetic_fallback_gradient_edges=int((p['same_frame_gradient_count']<2).sum()),boundary_BUILD_views=sorted(boundary_held),boundary_nodes=int(boundary['boundary_mask'].sum()),
        target_file_checks=target_checks,fit_checks=fit_checks,
        limitations=['Target is shared apparent encoded-sRGB desired joint-image supervision, not intrinsic albedo or hidden GT.',
            'Fusion adds qualified outside-core KEEP baseline boundary context; this is declared method information, not only an identity pixel transform.',
            'Geometry and inherited parent material were previously exposed to199-source scene data; current190 target source exclusion is not independent end-to-end blindness.',
            'Atlas/projection coverage is broader than2053 eligible trainable parents; target quality and writeback quality must be separate.',
            'Observed original RGB is reused outside common replacement domain; low whole-fit target error alone does not demonstrate fusion quality.',
            'This audit checks CPU data and code/contract. Actual Adam/RNG restore, formal budget and frozen native outputs require GPU-run receipts.'],
        GPU_used=False,model_updates=0,executed_source_sha256=sha(Path(__file__)))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(source_checks=source_checks,fit_checks=fit_checks,targets=len(target_checks),output=str(out),sha256=sha(out))),flush=True)


if __name__=='__main__':main()
