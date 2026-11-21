"""Independent CPU checks of completed source-only RGB refit."""
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from audit_source_quality import BASE, OUT, load, sha, measure
from audit_native1920 import difference_of_gaussians

RUN = OUT.parent/'source_rgb_refit'


def metrics(data):
    photo = data['observation'].astype(np.float64)
    gs = data['isolated_rgb'].astype(np.float64)
    mask = data['score_mask']
    result = measure(photo, gs, mask, sigmas=[1.2, 5/3, 3.])
    a, b = photo[mask], gs[mask]
    result['photo_mean_RGB'] = a.mean(0).tolist()
    result['GS_mean_RGB'] = b.mean(0).tolist()
    result['mean_RGB_error'] = (b.mean(0)-a.mean(0)).tolist()
    result['centered_RGB_MSE'] = float(np.mean(((b-b.mean(0))-(a-a.mean(0)))**2))
    result['MSE'] = float(np.mean((a-b)**2))
    result['DoG_physical_bands'] = difference_of_gaussians(photo, gs, mask)
    return result


def main():
    global RUN
    parser=argparse.ArgumentParser()
    parser.add_argument('--refit-directory',default='source_rgb_refit')
    parser.add_argument('--require-photo-legal',action='store_true')
    args=parser.parse_args()
    RUN=OUT.parent/args.refit_directory
    receipt = json.loads((RUN/'receipt.json').read_text())
    assert receipt['complete'] is True
    original_path = BASE/'source_selection/candidates/patch_00.npz'
    final_path = RUN/'source_patch_00_refit.npz'
    old, new = load(original_path), load(final_path)
    assert set(old) == set(new)
    frozen = {key: bool(np.array_equal(old[key], new[key])) for key in old if key != 'reference_srgb'}
    core = old['core_mask']
    frozen['halo_RGB_exact'] = bool(np.array_equal(old['reference_srgb'][~core],new['reference_srgb'][~core]))
    frozen['RGB_finite_in_01'] = bool(np.isfinite(new['reference_srgb']).all()
        and new['reference_srgb'].min() >= 0 and new['reference_srgb'].max() <= 1)
    plan=json.loads((OUT.parent/'source_view_ranking/source00_probe_plan.json').read_text())
    photo_records={row['raw']:row for row in plan['records']}
    cases=[]
    for initial_row in receipt['original']:
        raw=initial_row['raw']
        before=load(RUN/f'original/{raw:03d}/core_float.npz')
        after=load(RUN/f'refit/{raw:03d}/core_float.npz')
        same={key: bool(np.array_equal(before[key], after[key])) for key in ['observation','alpha','score_mask','core']}
        assert all(same.values())
        if raw in [86,110]:
            independent=load(OUT.parent/f'native1920/patch_00/{raw:03d}/core_float.npz')
            same['before_matches_independent_native_capture']=all(np.array_equal(before[key], independent[key])
                for key in ['observation','isolated_rgb','alpha','core'])
            assert same['before_matches_independent_native_capture']
            same['score_is_subset_of_original_native_score']=bool((~before['score_mask']|independent['score_mask']).all())
        photo_legal=np.asarray(Image.open(photo_records[raw]['core_photo_valid_mask']))>0
        if photo_legal.shape!=before['score_mask'].shape:
            x0,y0,x1,y1=initial_row['bbox_xyxy'];photo_legal=photo_legal[y0:y1,x0:x1]
        expected=binary_erosion(before['core'],iterations=5)&(before['alpha']>=.95)&photo_legal
        qualification=dict(qualified_mask_exact=bool(np.array_equal(before['score_mask'],expected)),
            actual_score_pixels=int(before['score_mask'].sum()),expected_qualified_pixels=int(expected.sum()),
            score_pixels_outside_photo_legal=int((before['score_mask']&~photo_legal).sum()),
            photo_legal_path=photo_records[raw]['core_photo_valid_mask'])
        if args.require_photo_legal:assert qualification['qualified_mask_exact'],qualification
        a,b=metrics(before),metrics(after)
        frequency=[]
        for left,right in zip(a['bands'],b['bands']):
            frequency.append(dict(sigma_native1920_px=left['highpass_gaussian_sigma_px'],
                sigma_in_1152_px=left['highpass_gaussian_sigma_px']*.6,
                correlation_before=left['cosine'],correlation_after=right['cosine'],
                RMS_ratio_before=left['GS_to_photo_RMS'],RMS_ratio_after=right['GS_to_photo_RMS'],
                HF_residual_RMS_before=left['residual_RMS'],HF_residual_RMS_after=right['residual_RMS'],
                correlation_improved=right['cosine']>left['cosine'],
                energy_ratio_closer_to_one=bool(abs(np.log(right['GS_to_photo_RMS']))<abs(np.log(left['GS_to_photo_RMS'])))))
        cases.append(dict(raw=raw,role=initial_row['role'],fixed_input_checks=same,qualification=qualification,
            before=a,after=b,MAE_improved=b['RGB_MAE']<a['RGB_MAE'],spectral_comparison=frequency))
    query=[row for row in cases if row['raw'] in receipt['contract']['query_frames']]
    outcome=dict(query_all_MAE_improved=all(row['MAE_improved'] for row in query),
        query_all_correlation_improved=all(all(b['correlation_improved'] for b in row['spectral_comparison']) for row in query),
        query_all_energy_ratios_closer_to_one=all(all(b['energy_ratio_closer_to_one'] for b in row['spectral_comparison']) for row in query),
        source_RGB_refit_quality_gate_passed=False,
        text='Fixed-source RGB refit lowers color error in some views but does not recover corresponding fine texture in the query views; it is not accepted for target transfer.',
        full_3D_source_reconstruction_tested=False,target_transfer_executed=False,
        no_further_training_in_this_control=True,new_diffusion_calls=0,human_verdict=None,production_promoted=False)
    source_asset=Path(str(old['source_asset']))
    parent_road_hash=sha(source_asset/'road.npz')
    expected=json.loads((BASE/'appearance/receipt.json').read_text())['binding']['files'][str(source_asset/'road.npz')]
    frozen['original_parent_road_SHA256_exact']=parent_road_hash==expected
    result=dict(schema='source_RGB_refit_independent_CPU_audit.v2',CPU_only=True,GPU_calls=0,
        refit_directory=str(RUN),photo_legal_required=args.require_photo_legal,
        photo_qualification_passed=all(row['qualification']['qualified_mask_exact'] for row in cases),
        frozen_parameters_passed=all(frozen.values()),frozen_checks=frozen,
        actual_core_RGB_parameters=int(core.sum()*3),cases=cases,outcome=outcome,
        resource_scope=dict(formal_source_updates=400,discarded_source_updates=6,
            nativeF=receipt['nativeF'],nativeB=receipt['nativeB'],target_updates=0,new_diffusion_calls=0,
            note='source-only control; separate from earlier target seam-fit400 and native1920 diagnostic8F'),
        evidence=[dict(path=str(p),sha256=sha(p)) for p in [RUN/'receipt.json',original_path,final_path]],
        original_parent_road_SHA256=parent_road_hash,
        source_role='fixed-geometry/sigma/opacity core RGB optimization, not a new 3D Gaussian source reconstruction',
        query_scope='86/110/173 not fitted in this source update; all were part of historical BUILD, and137/173 have almost identical camera positions')
    path=OUT/(RUN.name+'_independent_cpu.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(path=str(path),freeze_passed=result['frozen_parameters_passed'],outcome=outcome,
        query=[dict(raw=q['raw'],MAE_before=q['before']['RGB_MAE'],MAE_after=q['after']['RGB_MAE'],spectral=q['spectral_comparison']) for q in query]),indent=2))


if __name__=='__main__':
    main()
