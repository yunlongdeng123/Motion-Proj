"""固定GH15核的CPU来源分解；不改变训练资格或分母，不导入Torch/native。"""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse,hashlib,json,time
import numpy as np

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')

def load(path,keys=None):
    with np.load(path,allow_pickle=False) as d:
        return {k:d[k].copy() for k in (d.files if keys is None else keys)}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('explicit CPU-only visibility required')
    output=Path(output)
    if output.exists():raise FileExistsError('preserve source audit')
    started=time.monotonic()
    initial=P/'cpu_exact_r1/initial_row_queries.npz'
    source=P/'cpu_exact_r1/A0.npz'
    geometry=P/'chart_plan_candidate_r1/geometry_plan.npz'
    uv_path=P/'chart_plan_candidate_r1/sample_uv_f64.npy'
    scope_path=P/'method_scope_r1/scope.npz'
    stage_scope=P/'stage0_native_r1/scope.npz'
    a=load(source,['valid','source_effective'])
    g=load(geometry,['active_world_rows','geometric_sample_valid','weights'])
    s=load(scope_path);q=load(initial);mass=load(stage_scope)
    rows=g['active_world_rows'];n=len(rows)
    for d in [s,q,mass]:
        if not np.array_equal(rows,d['active_world_rows']):
            raise ValueError('world row order differs')
    uv=np.load(uv_path,mmap_mode='r')
    H,W=a['valid'].shape
    by_source=np.zeros((n,4),np.float64)
    geo_invalid=np.zeros(n,np.float64)
    chart_unknown=np.zeros(n,np.float64)
    weights=g['weights'][None,:]
    for start in range(0,n,1024):
        stop=min(start+1024,n)
        point=np.asarray(uv[start:stop]);geometric=g['geometric_sample_valid'][start:stop]
        x,y=point[...,0],point[...,1]
        ix=np.floor(x).astype(np.int64);iy=np.floor(y).astype(np.int64)
        dx=x-ix;dy=y-iy
        valid_mass=np.zeros(x.shape,np.float64)
        for oy,ox,b in [(0,0,(1-dx)*(1-dy)),(0,1,dx*(1-dy)),(1,0,(1-dx)*dy),(1,1,dx*dy)]:
            xx,yy=ix+ox,iy+oy
            inside=(xx>=0)&(xx<W)&(yy>=0)&(yy<H)&geometric
            cx=np.clip(xx,0,W-1);cy=np.clip(yy,0,H-1)
            legal=a['valid'][cy,cx]&inside
            value=b*legal
            valid_mass+=value
            src=a['source_effective'][cy,cx]
            for code in (1,2,3):
                by_source[start:stop,code]+=(weights*value*(src==code)).sum(1)
        geo_invalid[start:stop]=(weights*~geometric).sum(1)
        chart_unknown[start:stop]=(weights*geometric*(1-valid_mass)).sum(1)
    reconstructed=by_source.sum(1)
    error=float(np.abs(reconstructed-q['kernel_valid_mass']).max())
    if error>1e-10:raise AssertionError('source decomposition differs from actual field mass: '+str(error))
    if np.max(np.abs(reconstructed+geo_invalid+chart_unknown-1))>1e-10:
        raise AssertionError('kernel mass not conserved')
    available=q['row_material_available'].astype(bool)
    low=reconstructed<.9
    stats=[]
    for region in ['A','C','S']:
        m=mass['actual_'+region+'_mass'];hit=m>0;denom=float(m.sum())
        if denom<=0:raise ValueError('empty native region mass')
        stats.append(dict(region=region,actual_road_only_contributing_rows=int(hit.sum()),
            unavailable_readonly_rows=int((hit&~available).sum()),
            kernel_mass_below_point9_rows=int((hit&low).sum()),
            eligible_rows=int((hit&s['eligible']).sum()),
            eligible_mass_below_point9_rows=int((hit&s['eligible']&low).sum()),
            readonly_rows=int((hit&~s['eligible']).sum()),
            max_over_two_frame_native_region_mass_sum=denom,
            unavailable_native_maxmass_fraction=float(m[~available].sum()/denom),
            low_kernel_mass_native_maxmass_fraction=float(m[low].sum()/denom),
            all_rows_base_query_including_readonly_counterfactual_source_mean={
                str(code):float(np.sum(m*np.divide(by_source[:,code],reconstructed,
                    out=np.zeros(n),where=reconstructed>1e-12))/denom) for code in [1,2,3]},
            eligible_field_base_source_native_maxmass_mean={
                str(code):float(np.sum(m*s['eligible']*np.divide(by_source[:,code],reconstructed,
                    out=np.zeros(n),where=reconstructed>1e-12))/np.sum(m*s['eligible'])) for code in [1,2,3]}))
    output.mkdir(parents=True)
    np.savez_compressed(output/'row_source_kernel.npz',active_world_rows=rows,
        source_valid_mass=by_source,geometry_invalid_mass=geo_invalid,
        chart_unknown_mass=chart_unknown,kernel_valid_mass=reconstructed,
        row_material_available=available,eligible=s['eligible'])
    receipt=dict(schema='fixed_GH15_source_kernel_native_mass_CPU_audit.v1',GPU_executed=False,
        native_calls=0,optimizer_updates=0,active_rows=n,eligible_rows=int(s['eligible'].sum()),
        unavailable_readonly_rows=int((~available).sum()),
        kernel_source_decomposition_max_error=error,
        source_codes={'0':'unavailable, separately geometry/unknown','1':'measured prior','2':'retrieved prior','3':'frozen generated or B32 pseudo prior'},
        region_rows=stats,
        statistic_boundary='native masses are actual road-only alphaT from frozen masks; saved row mass is max across frames86/110, not their sum. Weighted summaries are aggregate row-max diagnostics, not per-pixel fractions or joint visibility. Only eligible_field_base_source means describe consumed field baseF; readonly all_rows query proportions are counterfactual and not their actual legacy shader source. Learned L/H terms are separate.',
        fixed_denominators=dict(S_supported=3131,S_full=3187,S_thin=56),
        mask_or_qualification_changed=False,valid_mass_point9_is_numeric_stratum_not_training_gate=True,
        mass_quadrature_not_certified_confidence=True,
        inputs={str(path):sha(path) for path in [Path(__file__),source,geometry,uv_path,scope_path,stage_scope,initial]},
        elapsed_seconds=time.monotonic()-started)
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    main(ap.parse_args().output)

