"""仅由已经保存的浮点结果恢复 camera02 receipt；不重解、不加载 GPU。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import load_npz


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def same(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)


def spectral(a,b,mask):
    rows=[]
    for sigma in (1.2,5/3,3.):
        aa=(a-gaussian_filter(a,(sigma,sigma,0)))[mask].ravel()
        bb=(b-gaussian_filter(b,(sigma,sigma,0)))[mask].ravel()
        an,bn=np.linalg.norm(aa),np.linalg.norm(bb)
        rows.append(dict(sigma_native_px=sigma,HF_cosine=float(aa@bb/(an*bn)) if an*bn>1e-20 else None,
                         HF_RMS_ratio=float(bn/an) if an>1e-20 else None))
    return rows


def metric(a,b,mask):
    return dict(score_n=int(mask.sum()),RGB_MAE=float(abs(a-b)[mask].mean()),spectral=spectral(a,b,mask))


def margin(mask):
    ys,xs=np.nonzero(mask)
    return int(min(ys.min(),xs.min(),mask.shape[0]-1-ys.max(),mask.shape[1]-1-xs.max()))


def roundtrip(intrinsic):
    fx,fy,cx,cy,k1,k2,p1,p2,k3=intrinsic
    yy,xx=np.mgrid[:1280,:1920].astype(np.float64)
    x=(xx+.5-cx)/fx;y=(yy+.5-cy)/fy;ux=x.copy();uy=y.copy()
    for _ in range(12):
        rr=ux*ux+uy*uy;rad=1+k1*rr+k2*rr*rr+k3*rr*rr*rr
        tx=2*p1*ux*uy+p2*(rr+2*ux*ux);ty=p1*(rr+2*uy*uy)+2*p2*ux*uy
        ux=(x-tx)/rad;uy=(y-ty)/rad
    rr=ux*ux+uy*uy;rad=1+k1*rr+k2*rr*rr+k3*rr*rr*rr
    bx=fx*(ux*rad+2*p1*ux*uy+p2*(rr+2*ux*ux))+cx-.5
    by=fy*(uy*rad+p1*(rr+2*uy*uy)+2*p2*ux*uy)+cy-.5
    return float(np.hypot(bx-xx,by-yy).max())


def write_new(path,value):
    if path.exists():raise FileExistsError('Preserve existing receipt: '+str(path))
    temp=path.with_suffix('.cpu_recovery.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=True,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def main():
    started=time.monotonic();C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
    R=C/'road_source_fidelity_20261122_r2';out=R/'camera_corrected_02_115'
    prior=R/'fixed_color_operator_02_115/patch_02';bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
    source=load(C/'road_gaussian_transfer_20261122_r1/source_selection/candidates/patch_02.npz')
    repaired=load(out/'source_camera_corrected_RGB.npz');state=load(out/'linear_state.npz')
    fields={k:same(v,repaired[k]) for k,v in source.items() if k!='reference_srgb'}
    halo=same(source['reference_srgb'][~source['core_mask']],repaired['reference_srgb'][~source['core_mask']])
    assert all(fields.values()) and halo
    assert same(repaired['reference_srgb'][source['core_mask']],state['fitted_core_RGB'])
    solver=[dict(channel=0,status=1,success=True,iterations=25,cost=.029605109463717638,optimality=5.744305445685789e-8),
            dict(channel=1,status=1,success=True,iterations=21,cost=.027022336385525453,optimality=3.248092433932458e-8),
            dict(channel=2,status=1,success=True,iterations=23,cost=.02752535547821603,optimality=2.361498183674798e-8)]
    A=load_npz(prior/'native_color_operator.npz')[state['reused_operator_row_selection']]
    x=state['fitted_core_RGB'].astype(float);y=state['target_minus_fixed'].astype(float);x0=state['direct_core_RGB'].astype(float)
    m,n=A.shape;weight=1e-5*m/n
    costs=.5*((A@x-y)**2).sum(0)+.5*weight*((x-x0)**2).sum(0)
    cost_deltas=[float(abs(costs[c]-solver[c]['cost'])) for c in range(3)]
    assert max(cost_deltas)<1e-7
    evaluations=[];checks=[];cache={}
    for raw in (86,110,115,117):
        folder=out/f'{raw:03d}';f=load(folder/'camera_float.npz');old=load(prior/f'{raw:03d}'/'source_float.npz')
        assert np.array_equal(f['alpha'],old['alpha'])
        pinmask=f['score_mask'];rawmask=f['raw_mask']
        pm,rm=margin(pinmask),margin(rawmask)
        assert pm>=12 and rm>=12,'Full image HF cannot be recovered from insufficient crop support'
        pin={};raw_metrics={}
        for name in ('original','uncorrected_solve','camera_corrected_solve'):
            pin[name]=metric(f['observation_pinhole'],f[name+'_pinhole'],pinmask)
            raw_metrics[name]=metric(f['observation_raw'],f[name+'_raw'],rawmask)
        obs=json.loads((bank/f'observations/{raw:03d}.json').read_text());intrinsic=tuple(obs['camera_intrinsic_Brown'])
        if intrinsic not in cache:cache[intrinsic]=roundtrip(intrinsic)
        evaluations.append(dict(raw=raw,role='REFERENCE_FIT' if raw==115 else 'QUERY_not_fit',metrics=pin,
            raw_camera_metrics=raw_metrics,Brown_inverse_roundtrip_max_px=cache[intrinsic],output=str(folder),
            bbox_xyxy=[int(q) for q in f['bbox_xyxy']]))
        check=dict(raw=raw,alpha_exact_vs_prior_native=True,pinhole_score_min_crop_margin=pm,raw_score_min_crop_margin=rm,
                   Gaussian_default_max_radius_px=12,HF_full_image_equivalence_by_crop_support=True)
        if raw==115:
            full=state['score_mask'];x0b,y0b,x1b,y1b=f['bbox_xyxy'];assert np.array_equal(full[y0b:y1b,x0b:x1b],pinmask)
            predicted=A@x+state['fixed_core_image']
            check['matrix_vs_saved_camera_corrected_native_max_abs']=float(abs(predicted-f['camera_corrected_solve_pinhole'][pinmask]).max())
        checks.append(check)
    receipt=dict(complete=True,patch='patch_02',reference=115,solver=solver,evaluations=evaluations,
        nativeF=12,nativeB=0,RGB_linear_solves=3,gradient_optimizer_updates=0,source_non_RGB_fields_exact=True,halo_RGB_exact=True,
        new_diffusion_calls=0,target_migration=False,original_asset_modified=False,
        recovered_from_saved_float_after_json_type_failure=True,
        recovery=dict(CPU_only=True,extra_nativeF=0,extra_nativeB=0,extra_RGB_linear_solves=0,extra_optimizer_updates=0,
                      source_non_RGB_field_checks=fields,alpha_and_crop_checks=checks,
                      solver_status_source='Parent supplied actual solver stdout; cost independently checked from saved float32 state',
                      solver_cost_abs_differences_from_saved_float32=cost_deltas,
                      metrics_source='All RGB/HF metrics independently recomputed from saved camera_float.npz',
                      seconds=time.monotonic()-started))
    broken_code=R/'code/camera_corrected_source_before_json_fix.py'
    lines=broken_code.read_text().splitlines()
    loc=[i+1 for i,line in enumerate(lines) if 'bbox_xyxy=list(v[' in line]
    try:json.dumps([np.int64(1)])
    except TypeError as error:reproduced_error=str(error)
    trace=dict(schema='camera02_receipt_recovery_trace.v1',failure='TypeError: Object of type int64 is not JSON serializable',
        observed_failure_source='Parent reports final JSON serialization failure after all 12 native forwards and saved frames',
        offending_code=str(broken_code),offending_code_lines=loc,reproduced_CPU_error=reproduced_error,
        cause='bbox_xyxy=list(v[\'bbox\']) retains NumPy int64 scalars; h.write JSON default has no NumPy encoder',
        remedy='Recompute metrics from already saved float arrays and serialize bbox with Python int(q)',
        no_model_or_linear_state_replay=True,new_native_calls=0,new_linear_solves=0,source_assets_modified=False,
        recovered_from_saved_float_after_json_type_failure=True,receipt=str(out/'receipt.json'))
    write_new(out/'recovery_trace.json',trace)
    write_new(out/'receipt.json',receipt)
    print(json.dumps(dict(receipt=receipt,recovery_trace=trace),ensure_ascii=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
