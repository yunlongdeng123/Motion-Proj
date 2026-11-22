"""One CPU material diagnostic: canonical field versus high-alpha local GS.

No photo-domain gain, low-alpha unpremultiplication, GPU, solve or new render.
Native sRGB-valued alpha compositing is not a linear-light material model.
"""
from pathlib import Path
import argparse,csv,time
import numpy as np
from scipy.ndimage import gaussian_filter,distance_transform_edt
from local_gaussian_asset import read,write,bank,sha
from local_opacity_gate import pin
from fit_new_local_rgb import field_sample


def eotf(x):
    return np.where(x<=.04045,x/12.92,((np.maximum(x,0)+.055)/1.055)**2.4)


def highpass(image,mask,sigma):
    support=gaussian_filter(mask.astype(np.float64),sigma=sigma)
    source=np.where(mask[...,None],image,0)
    smooth=gaussian_filter(source,sigma=(sigma,sigma,0))/np.maximum(support[...,None],1e-30)
    return image-smooth


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;start=time.time()
    assert not a.output.exists();a.output.mkdir(parents=True)
    intrinsic=r/'asset_intrinsic_coverage_v4_v1';surface=r/'timewindow17_requalified_surface_review_v2_replay1'
    receipt=read(intrinsic/'receipt.json');sr=read(surface/'receipt.json');cameras=read(intrinsic/'camera_timestamp_audit.json')
    assert receipt['complete'] and receipt['nativeF']==5 and receipt['joint_local_alphaT_not_used']
    assert receipt['geometry_sha256']==sr['geometry_sha256'] and receipt['source_field_sha256']==sr['final_atlas_sha256']
    parent=read(receipt['asset_manifest']['path']);fieldpath=Path(parent['source']['field']['path'])
    assert sha(fieldpath)==receipt['source_field_sha256']
    f=bank(fieldpath);legal=f['permission_mask']&(f['source_kind']>0)&np.isfinite(f['atlas_linear']).all(-1)
    origin=np.asarray(parent['source'].get('uv_origin_global',[0,0]),float);rows=[]
    for raw in receipt['frames']:
        rp=intrinsic/f'{raw:03d}/local_render.npz';sp=surface/f'{raw:03d}/surface_review.npz'
        rr=next(x for x in receipt['rows'] if x['raw']==raw);assert sha(rp)==rr['render']['sha256']
        image=bank(rp);surf=bank(sp);alpha=image['alpha'];opaque=alpha>=.95
        sampled,valid=field_sample(f['atlas_linear'],surf['atlas_uv_global']-origin,legal)
        mask=opaque&surf['shown']&surf['projection_valid']&valid
        assert mask.any() and np.isfinite(sampled[mask]).all()
        # Do not unpremultiply any low-alpha pixel, even for visualization.
        straight=np.zeros((*alpha.shape,3),np.float64)
        straight[opaque]=image['rgb'].transpose(1,2,0)[opaque]/alpha[opaque,None]
        canonical=eotf(straight);reference=sampled.astype(np.float64)
        delta=canonical[mask]-reference[mask];dist=distance_transform_edt(mask);hf=[]
        for sigma in [1.,2.]:
            core=mask&(dist>3*sigma)
            assert core.any()
            h1=highpass(canonical,mask,sigma)[core];h2=highpass(reference,mask,sigma)[core]
            x=h1-h1.mean(0);y=h2-h2.mean(0);den=np.sqrt((x*x).sum()*(y*y).sum())
            nr=float(np.sqrt(np.mean(h1*h1)));fr=float(np.sqrt(np.mean(h2*h2)))
            hf.append(dict(image_sigma_pixels=sigma,edge_exclusion_distance_pixels=3*sigma,core_pixels=int(core.sum()),
                native_HF_RMS=nr,field_HF_RMS=fr,RMS_ratio_native_over_field=nr/fr if fr else None,
                centred_channel_HF_correlation=float((x*y).sum()/den) if den else None,
                native_minus_field_HF_RMS=float(np.sqrt(np.mean((h1-h2)**2)))))
        cam=next(x for x in cameras['rows'] if x['raw']==raw);assert cam['K_max_abs_vs_R7']==0
        row=dict(raw=raw,intrinsic_alpha_ge95_pixels=int(opaque.sum()),surface_shown_pixels=int(surf['shown'].sum()),
            fixed_intersection_pixels=int(mask.sum()),high_alpha_excluded_from_intersection=int((opaque&~mask).sum()),
            canonical_linear_MAE=float(abs(delta).mean()),pixel_mean_abs_error_p95=float(np.quantile(abs(delta).mean(1),.95)),
            mean_colour_bias_native_minus_field=delta.mean(0).tolist(),mean_native_canonical=canonical[mask].mean(0).tolist(),
            mean_field_canonical=reference[mask].mean(0).tolist(),
            core_straight_sRGB_max_below40_255=int((mask&(straight.max(-1)<40/255)).sum()),
            core_max_absolute_channel_error=float(abs(delta).max()),HF=hf,
            K_max_abs_vs_R7=cam['K_max_abs_vs_R7'],native_world_to_camera_max_abs_vs_R7=cam['world_to_camera_max_abs_vs_R7'],
            intrinsic_render=pin(rp),surface_projection=pin(sp))
        rows.append(row);print({k:row[k] for k in ['raw','fixed_intersection_pixels','canonical_linear_MAE','mean_colour_bias_native_minus_field','core_straight_sRGB_max_below40_255']},flush=True)
    summary=dict(schema='R10.intrinsic_GS_canonical_material_diagnostic.v1',complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,
        intrinsic_receipt=pin(intrinsic/'receipt.json'),surface_receipt=pin(surface/'receipt.json'),source_field=pin(fieldpath),
        geometry_sha256=receipt['geometry_sha256'],source_field_sha256=receipt['source_field_sha256'],rows=rows,
        comparison='EOTF(native premultiplied sRGB / intrinsic alpha) versus atlas_linear sampled at saved surface global UV',
        alpha_threshold=.95,primary_domain='fixed intersection: alpha>=.95 and surface.shown and projection_valid and all-four legal finite field',
        photo_exposure_gain_applied=False,source_field_already_canonical=True,
        native_compositing_domain='sRGB-valued alpha compositing, not linear-light material or recovered reflectance',
        camera_alignment='same immutable five-view pinhole K and world geometry; native float32 world-to-camera residual is reported, not claimed bitexact',
        texture_metrics='image-space sigma1/2px normalized Gaussian highpass; secondary common core excludes 3*sigma image boundary; no metric-centimetre bandwidth claim',
        error_contains=['Gaussian spatial filtering and footprint mixing','sRGB compositing versus direct field sampling','surface/GS sampling and native camera floating precision'],
        original_native_or_actor_occlusion_included=False,asset_strict_true_same_world_RGB_anchor_cells=0,
        full_requested_region_quality_not_inferred_from_intersection=True,seconds=time.time()-start)
    write(a.output/'metrics.json',summary)
    with (a.output/'small_table.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        out=csv.writer(handle);out.writerow(['raw','fixed_pixels','canonical_linear_MAE','bias_R','bias_G','bias_B','HF1px_RMSratio','HF1px_corr','HF2px_RMSratio','HF2px_corr','dark_core_pixels'])
        for x in rows:out.writerow([x['raw'],x['fixed_intersection_pixels'],x['canonical_linear_MAE'],*x['mean_colour_bias_native_minus_field'],x['HF'][0]['RMS_ratio_native_over_field'],x['HF'][0]['centred_channel_HF_correlation'],x['HF'][1]['RMS_ratio_native_over_field'],x['HF'][1]['centred_channel_HF_correlation'],x['core_straight_sRGB_max_below40_255']])


if __name__=='__main__':main()
