"""最终只读资产/浮点验收；只写本审计目录 aggregate_cpu_audit.json。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.sparse import load_npz


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def exact(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)


def support(mask):
    yy,xx=np.nonzero(mask)
    return int(min(yy.min(),xx.min(),mask.shape[0]-1-yy.max(),mask.shape[1]-1-xx.max()))


def metrics(a,b,mask):
    out=[]
    for sigma in (1.2,5/3,3.):
        aa=(a-gaussian_filter(a,(sigma,sigma,0)))[mask].ravel()
        bb=(b-gaussian_filter(b,(sigma,sigma,0)))[mask].ravel()
        an,bn=np.linalg.norm(aa),np.linalg.norm(bb)
        out.append(dict(sigma_native_px=sigma,HF_cosine=float(aa@bb/(an*bn)),HF_RMS_ratio=float(bn/an)))
    return dict(score_n=int(mask.sum()),RGB_MAE=float(abs(a-b)[mask].mean()),spectral=out)


def main():
    started=time.monotonic();directory=Path(__file__).resolve().parent
    R=directory.parent;C=R.parent
    bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
    raw_provenance=json.loads((directory/'raw_jpeg_provenance_results.json').read_text())
    assert raw_provenance['all_selected_bank_PNGs_are_unwarped_JPEG_decodes']
    source_manifest=json.loads((C/'road_gaussian_transfer_20261122_r2_isometric/source_selection/manifest.json').read_text())
    source_paths={e['candidate_id']:e['packet'] for e in source_manifest['candidates']}
    audits=[];raw_frames=set()
    for patch,ref,run,prior in [('patch_02',115,'camera_corrected_02_115','fixed_color_operator_02_115'),
                              ('patch_00',137,'camera_corrected_00_137','fixed_color_operator_00_137')]:
        folder=R/run;receipt=json.loads((folder/'receipt.json').read_text());old=R/prior/patch
        assert receipt['complete'] and receipt['patch']==patch and receipt['reference']==ref
        recovered=bool(receipt.get('recovered_from_saved_float_after_json_type_failure',False))
        assert recovered==(patch=='patch_02')
        source=load(source_paths[patch]);corrected=load(folder/'source_camera_corrected_RGB.npz');state=load(folder/'linear_state.npz')
        noncolor={k:exact(v,corrected[k]) for k,v in source.items() if k!='reference_srgb'}
        halo=exact(source['reference_srgb'][~source['core_mask']],corrected['reference_srgb'][~source['core_mask']])
        core_exact=exact(corrected['reference_srgb'][source['core_mask']],state['fitted_core_RGB'])
        assert all(noncolor.values()) and halo and core_exact
        selection=state['reused_operator_row_selection'];oldstate=load(old/'linear_state.npz')
        assert np.array_equal(selection,state['score_mask'][oldstate['score_mask']])
        A=load_npz(old/'native_color_operator.npz')[selection].astype(np.float64)
        x=state['fitted_core_RGB'].astype(float);y=state['target_minus_fixed'].astype(float);x0=state['direct_core_RGB'].astype(float)
        m,n=A.shape;weight=1e-5*m/n
        costs=.5*((A@x-y)**2).sum(0)+.5*weight*((x-x0)**2).sum(0)
        cost_delta=[float(abs(costs[c]-receipt['solver'][c]['cost'])) for c in range(3)]
        assert max(cost_delta)<1e-7
        audit=dict(patch=patch,run=run,reference=ref,complete=True,
            receipt_recovered_from_saved_float=recovered,all_non_RGB_fields_exact=all(noncolor.values()),
            non_RGB_fields=noncolor,halo_RGB_exact=halo,saved_core_RGB_equals_linear_state=core_exact,
            operator_row_selection_exact=True,operator_shape=[m,n],solver_cost_abs_difference_from_saved_float32=cost_delta,
            nativeF=receipt['nativeF'],nativeB=receipt['nativeB'],RGB_linear_solves=receipt['RGB_linear_solves'],frames=[])
        for e in receipt['evaluations']:
            raw=e['raw'];raw_frames.add(raw);f=load(folder/f'{raw:03d}'/'camera_float.npz');prior_float=load(old/f'{raw:03d}'/'source_float.npz')
            alpha_exact=exact(f['alpha'],prior_float['alpha']);assert alpha_exact
            x0b,y0b,x1b,y1b=[int(q) for q in f['bbox_xyxy']]
            observation=json.loads((bank/f'observations/{raw:03d}.json').read_text())
            photo=np.asarray(Image.open(bank/observation['image_path']).convert('RGB'),np.float32)/255
            raw_pixels_exact=exact(f['observation_raw'],photo[y0b:y1b,x0b:x1b]);assert raw_pixels_exact
            pinmask=f['score_mask'];rawmask=f['raw_mask'];pm,rm=support(pinmask),support(rawmask)
            assert pm>=12 and rm>=12
            pin={};raw_metrics={};differences=[]
            for name in ('original','uncorrected_solve','camera_corrected_solve'):
                pin[name]=metrics(f['observation_pinhole'],f[name+'_pinhole'],pinmask)
                raw_metrics[name]=metrics(f['observation_raw'],f[name+'_raw'],rawmask)
                for collection,new in [('metrics',pin[name]),('raw_camera_metrics',raw_metrics[name])]:
                    saved=e[collection][name]
                    assert new['score_n']==saved['score_n']
                    differences.append(abs(new['RGB_MAE']-saved['RGB_MAE']))
                    differences.extend(abs(a['HF_cosine']-b['HF_cosine']) for a,b in zip(new['spectral'],saved['spectral']))
            max_metric_error=float(max(differences));assert max_metric_error<1e-5
            fr=dict(raw=raw,role=e['role'],alpha_exact_vs_original_measured_operator_run=alpha_exact,
                    raw_observation_crop_exact_bank_RGB=raw_pixels_exact,min_score_crop_margin_px=pm,
                    min_raw_score_crop_margin_px=rm,max_Gaussian_radius_px=12,HF_crop_support_complete=True,
                    receipt_RGB_MAE_and_HF_cosine_max_recompute_delta=max_metric_error,
                    pinhole_metrics=pin,raw_Brown_metrics=raw_metrics)
            if raw==ref:
                assert np.array_equal(state['score_mask'][y0b:y1b,x0b:x1b],pinmask)
                matrix_delta=float(abs(A@x+state['fixed_core_image']-f['camera_corrected_solve_pinhole'][pinmask]).max())
                assert matrix_delta<3e-6;fr['reused_operator_vs_saved_native_max_abs']=matrix_delta
            audit['frames'].append(fr)
        audits.append(audit)
    assert raw_frames.issubset(set(raw_provenance['frames']))
    run_names=['chain_probe_initial','fixed_color_operator_02_115','fixed_color_operator_00_137','native_static_02_115',
               'native_static_contributors_02_115','camera_corrected_02_115','camera_corrected_00_137']
    receipts=[json.loads((R/name/'receipt.json').read_text()) for name in run_names]
    counts=dict(nativeF=sum(r['nativeF'] for r in receipts),nativeB=sum(r.get('nativeB',0) for r in receipts),
        RGB_source_fit_channel_solves=sum(r.get('RGB_linear_solves',0) for r in receipts),
        gradient_optimizer_updates=sum(r.get('gradient_optimizer_updates',r.get('optimizer_updates',0)) for r in receipts),
        contexts=len(receipts))
    assert counts==dict(nativeF=918,nativeB=1,RGB_source_fit_channel_solves=12,gradient_optimizer_updates=0,contexts=7)
    session_path=R/'runtime/native_session_log.jsonl'
    sessions=[json.loads(line) for line in session_path.read_text().splitlines() if line.strip()]
    session_F=sum(s['native_context_F'] for s in sessions)
    assert session_F==918
    output=dict(schema='source_fidelity_final_aggregate_CPU_audit.v1',all_checks_passed=True,
        source_runs=audits,raw_JPEG_provenance=dict(evidence='raw_jpeg_provenance_results.json',frames=raw_provenance['frames'],
        all_selected_raw_JPEG_equals_bank_PNG=True,all_saved_raw_observation_crops_exact=True),
        parent_actual_native_totals=counts,native_session_log_total_F=session_F,
        parent_source_RGB_solves_are_separate_from_CPU_precision_replay_channel_solves=6,
        this_aggregate_new_nativeF=0,this_aggregate_new_nativeB=0,this_aggregate_linear_solves=0,
        this_aggregate_optimizer_updates=0,model_and_linear_state_assets_modified=False,
        HF_strict_small_support_evidence='solver_matrix_cpu_results.json: rows[].HF_support_sensitivity; 205px inner core is not whole-source evidence',
        seconds=time.monotonic()-started)
    target=directory/'aggregate_cpu_audit.json';temp=directory/'aggregate_cpu_audit.tmp'
    temp.write_text(json.dumps(output,ensure_ascii=True,indent=2,allow_nan=False)+'\n');temp.replace(target)
    print(json.dumps(output,ensure_ascii=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
