"""只读R4真核心RGBA与旧support画廊；颜色统计不用于删点，不调用GPU。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import json,time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,map_coordinates


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def read(path):return json.loads(Path(path).read_text())


def exact(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)


def stats(x):
    a=np.asarray(x).ravel()
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),p99_9=float(np.quantile(a,.999)),max=float(a.max())) if a.size else dict(n=0)


def color_rows(rgb):
    span=np.ptp(rgb,axis=1)
    return dict(rows=len(rgb),RGB_min=float(rgb.min()),RGB_max=float(rgb.max()),RGB_mean=rgb.mean(0).tolist(),
        channel_span=stats(span),point_span_gt_0p18=int((span>.18).sum()),
        point_any_channel_lt_0p02_or_gt_0p98=int(((rgb<.02)|(rgb>.98)).any(1).sum()),
        note='Descriptive point colors only; inverse-rendering colors may blend to neutral pixels. No deletion criterion.')


def chroma_counts(rgb,mask,threshold):
    span=np.ptp(rgb,axis=2)
    return dict(pixels=int(mask.sum()),high_chroma=int((mask&(span>threshold)).sum()),chroma=stats(span[mask]))


def alpha_bands(effective,composite,alpha,physical,threshold):
    result=[]
    for lo,hi in [(0.,.01),(.01,.1),(.1,.5),(.5,.95),(.95,1.000001)]:
        m=(alpha>=lo)&(alpha<hi)
        effective_high=np.ptp(effective,axis=2)>threshold
        display_high=np.ptp(composite,axis=2)>threshold
        result.append(dict(alpha_interval=[lo,hi],pixels=int(m.sum()),physical_core_pixels=int((m&physical).sum()),
            outside_physical_core_pixels=int((m&~physical).sum()),
            effective_high_chroma_inside_physical_core=int((m&physical&effective_high).sum()),
            effective_high_chroma_outside_physical_core=int((m&~physical&effective_high).sum()),
            display_high_chroma_inside_physical_core=int((m&physical&display_high).sum()),
            display_high_chroma_outside_physical_core=int((m&~physical&display_high).sum()),
            effective_RGB_over_alpha=chroma_counts(effective,m,threshold),
            neutral_background_display=chroma_counts(composite,m,threshold)))
    return result


def footprint_classes(effective,display,alpha,physical,score,threshold):
    eroded=binary_erosion(physical,iterations=3);edge=physical&~eroded
    masks=dict(official_eroded_high_alpha_score=score,
        physical_core_alpha_ge_0p95=physical&(alpha>=.95),
        physical_edge_alpha_ge_0p95=edge&(alpha>=.95),
        physical_edge_alpha_0p05_to_0p95=edge&(alpha>=.05)&(alpha<.95),
        eroded_core_alpha_0p05_to_0p95=eroded&(alpha>=.05)&(alpha<.95),
        physical_core_alpha_0p05_to_0p95=physical&(alpha>=.05)&(alpha<.95),
        outside_core_alpha_0p05_to_0p95=~physical&(alpha>=.05)&(alpha<.95),
        all_component_alpha_0p05_to_0p95=(alpha>=.05)&(alpha<.95),
        outside_core_alpha_ge_0p95=~physical&(alpha>=.95),
        all_visible_alpha_ge_0p05=alpha>=.05)
    return {name:dict(effective_RGB_over_alpha=chroma_counts(effective,m,threshold),
        actual_or_reconstructed_neutral_display=chroma_counts(display,m,threshold)) for name,m in masks.items()}


def photo_crop(raw,bbox,bank):
    obs=read(bank/f'observations/{raw:03d}.json');rgb=np.asarray(Image.open(bank/obs['image_path']).convert('RGB'),np.float32)/255
    x0,y0,x1,y1=bbox;yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float64)
    fx,fy,cx,cy,k1,k2,p1,p2,k3=obs['camera_intrinsic_Brown'];x=(xx+.5-cx)/fx;y=(yy+.5-cy)/fy;r=x*x+y*y
    radial=1+k1*r+k2*r*r+k3*r*r*r
    u=fx*(x*radial+2*p1*x*y+p2*(r+2*x*x))+cx-.5
    v=fy*(y*radial+p1*(r+2*y*y)+2*p2*x*y)+cy-.5
    return np.stack([map_coordinates(rgb[:,:,c],[v,u],order=1,mode='nearest',prefilter=False) for c in range(3)],2)


def main():
    started=time.monotonic();C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=C/'road_clean_core_transfer_20261122_r4'
    out=R/'cpu_audit';out.mkdir(parents=True,exist_ok=True)
    probe=R/'source_core_probe';receipt=read(probe/'receipt.json');assert receipt['complete'] and receipt['nativeF']==42 and receipt['nativeB']==0
    parent=C/'road_surface_residual_20261121_r2/appearance_fit/asset';road=load(parent/'road.npz')
    old=C/'road_source_fidelity_20261122_r2';r3=C/'road_source_multiframe_20261122_r3'
    bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
    sources={p:load(C/f'road_gaussian_transfer_20261122_r1/source_selection/candidates/{p}.npz') for p in ('patch_00','patch_02')}
    transfers=[]
    for run in ('road_gaussian_transfer_20261122_r1','road_gaussian_transfer_20261122_r2_isometric'):
        root=C/run;chosen=read(root/'proposal/source_decision.json')['chosen'];source=load(root/f'source_selection/candidates/{chosen}.npz')
        proposal=load(root/'proposal/transferred_rows.npz');donor=proposal['source_row_id'];ids=source['world_row_ids'];position=np.searchsorted(ids,donor)
        assert np.array_equal(ids[position],donor)
        only_core=bool(source['core_mask'][position].all());rgb_exact=exact(proposal['reference_srgb'],source['reference_srgb'][position])
        assert only_core and rgb_exact
        transfers.append(dict(run=run,chosen=chosen,transferred_rows=len(donor),unique_donor_rows=len(np.unique(donor)),
            all_rows_from_source_core=True,copied_halo_rows=0,RGB_exact_source_rows=True,
            source='Actual saved proposal source_row_id and reference_srgb, independently CPU read'))
    variants=[];reference_by_patch={}
    for variant in receipt['variants']:
        label=variant['label'];patch=variant['patch'];source=sources[patch];core=source['core_mask'];ids=source['world_row_ids'][core]
        material=load(variant['material_input']);component=load(variant['component'])
        noncolor={k:exact(v,material[k]) for k,v in source.items() if k!='reference_srgb'}
        assert all(noncolor.values())
        checks=dict(active_all_true=bool(component['active'].all()),core_IDs_exact=exact(component['source_world_row_ids'],ids),
            model_xyz_exact_parent=exact(component['model_xyz'],road['model_xyz'][ids]),
            normal_exact_source=exact(component['normal_world'],source['normal_world'][core]),
            sigma_exact_source=exact(component['gaussian_sigma_m'],source['gaussian_sigma_m'][core]),
            opacity_exact_source=exact(component['road_opacity'],source['road_opacity'][core]),
            RGB_exact_material_core=exact(component['reference_srgb'],material['reference_srgb'][core]),
            model_transform_exact_parent=exact(component['model_from_world'],road['model_from_world']))
        assert all(checks.values()) and len(component['active'])==(1598 if patch=='patch_00' else 900)
        record=dict(label=label,patch=patch,component_core_rows=len(ids),halo_exported_rows=0,
            noncolor_material_fields_exact=all(noncolor.values()),component_CPU_field_checks=checks,
            point_color_groups=dict(core=color_rows(material['reference_srgb'][core]),halo=color_rows(material['reference_srgb'][~core]),
                full_support=color_rows(material['reference_srgb'])),views=[])
        for view in variant['views']:
            raw=view['raw'];folder=probe/label/f'{raw:03d}';f=load(folder/'core_float.npz');alpha=f['alpha'];score=f['score_mask'];physical=f['physical_core']
            expected_mask=binary_erosion(physical,iterations=3)&(alpha>=.95);assert np.array_equal(score,expected_mask)
            independent_photo=photo_crop(raw,[int(x) for x in f['bbox_xyxy']],bank)
            photo_error=float(abs(independent_photo-f['photo_pinhole']).max());assert photo_error<1e-7
            effective=f['core_black']/np.maximum(alpha[:,:,None],1e-8)
            expected_grey=f['core_black']+.32*(1-alpha[:,:,None]);grey_error=float(abs(expected_grey-f['core_grey']).max());assert grey_error<1e-6
            png=np.asarray(Image.open(folder/'core_grey.png').convert('RGB'))
            assert np.array_equal(png,np.rint(np.clip(f['core_grey'],0,1)*255).astype(np.uint8))
            threshold=float(np.quantile(np.ptp(f['photo_pinhole'][score],axis=1),.999)+.10)
            anomalous=int((np.ptp(effective[score],axis=1)>threshold).sum())
            threshold_delta=abs(threshold-view['high_chroma_threshold'])
            assert threshold_delta<1e-7 and anomalous==view['high_chroma_pixels'] and score.sum()==view['score_n']
            key=(patch,raw)
            if key not in reference_by_patch:reference_by_patch[key]=(alpha.copy(),score.copy())
            else:assert np.array_equal(alpha,reference_by_patch[key][0]) and np.array_equal(score,reference_by_patch[key][1])
            vx=dict(raw=raw,score_n=int(score.sum()),physical_core_n=int(physical.sum()),
                eroded_core_n=int(binary_erosion(physical,iterations=3).sum()),
                high_chroma_threshold=threshold,high_chroma_pixels=anomalous,high_chroma_fraction=anomalous/int(score.sum()),
                threshold_receipt_abs_delta=threshold_delta,
                neutral_background_native_vs_CPU_RGBA_max_abs=grey_error,photo_Brown_sampling_max_abs=photo_error,
                PNG_matches_entire_unmasked_native_grey_crop=True,
                alpha_and_score_identical_between_same_patch_material_variants=True,
                kernel_pixels_outside_physical_core_alpha_gt_0p01=int((~physical&(alpha>.01)).sum()),
                native_core_alpha_bands=alpha_bands(effective,f['core_grey'],alpha,physical,threshold),
                native_core_footprint_classes=footprint_classes(effective,f['core_grey'],alpha,physical,score,threshold),
                native_receipt_RoadRows_reload_RGB_max_abs=view['reload_RGB_max_abs'],
                native_receipt_parent_core_baseline_max_abs=view['component_vs_parent_core_RGB_max_abs'])
            if label.startswith('source00'):
                legacy_path=old/f'camera_corrected_00_137/{raw:03d}/camera_float.npz'
                rgb_key='original_pinhole' if label=='source00_original' else 'camera_corrected_solve_pinhole'
                alpha_key='alpha'
            else:
                group='initial' if label=='source02_original' else 'eight_frames/final_reloaded'
                legacy_path=r3/f'paired400_seed0/{group}/{raw:03d}/source_float.npz';rgb_key='render_pinhole_canonical';alpha_key='alpha'
            legacy=load(legacy_path);a=legacy[alpha_key];rgb=legacy[rgb_key]
            oldbox=legacy['bbox_xyxy'];newbox=f['bbox_xyxy'];lo=np.maximum(oldbox[:2],newbox[:2]);hi=np.minimum(oldbox[2:],newbox[2:])
            oldcrop=np.s_[lo[1]-oldbox[1]:hi[1]-oldbox[1],lo[0]-oldbox[0]:hi[0]-oldbox[0]]
            newcrop=np.s_[lo[1]-newbox[1]:hi[1]-newbox[1],lo[0]-newbox[0]:hi[0]-newbox[0]]
            m=score[newcrop]&(a[oldcrop]>=.95)
            reconstructed_grey=rgb[oldcrop]*a[oldcrop][:,:,None]+.32*(1-a[oldcrop][:,:,None])
            vx['legacy_support_comparison']=dict(path=str(legacy_path),effective_RGB_field=rgb_key,
                common_high_alpha_physical_core_pixels=int(m.sum()),
                support_effective_high_chroma=chroma_counts(rgb[oldcrop],m,threshold),
                core_only_effective_high_chroma=chroma_counts(effective[newcrop],m,threshold),
                support_low_alpha_normalization_bands=alpha_bands(rgb[oldcrop],reconstructed_grey,a[oldcrop],physical[newcrop],threshold),
                support_footprint_classes=footprint_classes(rgb[oldcrop],reconstructed_grey,a[oldcrop],physical[newcrop],m,threshold),
                support_grey_image_is_CPU_reconstruction=True,
                common_core_effective_RGB_MAE=float(abs(rgb[oldcrop]-effective[newcrop])[m].mean()),
                explanation='Legacy gallery uses support rows and RGB/alpha everywhere. Neutral-grey reconstruction attenuates low-alpha tails. Current core_grey is an actual native grey-background render.')
            record['views'].append(vx)
        variants.append(record)
    result=dict(schema='clean_core_source_independent_CPU_audit.v1',complete=True,all_field_and_metric_checks_passed=True,
        historical_transfer_source_proof=transfers,variants=variants,
        parent_actual_source_probe_nativeF=42,parent_actual_source_probe_nativeB=0,
        this_audit_new_nativeF=0,this_audit_new_nativeB=0,optimizer_updates=0,source_fields_modified=False,
        point_color_deletion_or_filtering_applied=False,
        conclusion='Original00/02 eroded high-alpha metric ROI has no high-chroma pixels under this diagnostic. This does not mean the whole core-only Gaussian component is clean: boundary and outer moderate-alpha tails are retained and measured separately. Neither point colors nor gallery tails are a point-deletion rule.',
        seconds=time.monotonic()-started)
    (out/'source_core_cpu_audit.json').write_text(json.dumps(result,ensure_ascii=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
