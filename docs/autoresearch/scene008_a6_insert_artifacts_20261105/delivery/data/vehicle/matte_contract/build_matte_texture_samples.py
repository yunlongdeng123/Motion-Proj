"""All47544 chart samples for candidate-specific alpha/premult texture contract."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105')
DEST=RUN.parent/'scene008_a6_insert_artifacts_20261105/vehicle/matte_contract'
SOURCE=RUN/'output_adapter/projective_surface'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    with np.load(SOURCE/'reference_chart.npz') as z:chart={k:z[k].copy() for k in z.files}
    with np.load(DEST/'matte_contract.npz') as z:matte={k:z[k].copy() for k in z.files}
    rgb=np.asarray(Image.open(DEST/'teacher_rgb_512.png').convert('RGB')).astype(np.float64)/255
    alpha=np.asarray(matte['alpha'],np.float64)
    assert rgb.shape==(512,512,3) and alpha.shape==(512,512)
    uv=chart['generated_xy'];coords=[uv[:,1],uv[:,0]]
    a=map_coordinates(alpha,coords,order=1,mode='constant',cval=0,prefilter=False)
    premul=np.column_stack([map_coordinates(rgb[...,j]*alpha,coords,order=1,mode='constant',cval=0,prefilter=False) for j in range(3)])
    valid=a>1e-6
    unpremul=np.zeros_like(premul);unpremul[valid]=premul[valid]/a[valid,None]
    nearest=np.floor(uv+.5).astype(int);in_bounds=((nearest>=0)&(nearest<512)).all(1)
    clipped=np.clip(nearest,[0,0],[511,511])
    allowed=in_bounds&matte['foreground'][clipped[:,1],clipped[:,0]]
    real=chart['color_real'].astype(bool)
    effective_alpha=a.copy();effective_alpha[real]=1
    effective_rgb=unpremul.copy();effective_rgb[real]=chart['color_rgb'][real]
    effective_allowed=allowed|real
    effective_valid=valid|real
    assert np.array_equal(effective_rgb[real].astype(np.float32),chart['color_rgb'][real])
    M=len(uv);assert M==47544
    np.savez_compressed(DEST/'matte_texture_samples.npz',
        reference_raw=np.int64(106),original_hw=np.array([1280,1920]),generated_hw=np.array([512,512]),
        chart_frame_pixels=chart['pixels'],chart_pixel_xy=chart['pixel_xy'],generated_xy=uv,
        original_asset_row=np.arange(2778,2778+M,dtype=np.int64),
        sampled_alpha=a.astype(np.float32),sampled_premultiplied_RGB=premul.astype(np.float32),
        sampled_unpremultiplied_RGB=unpremul.astype(np.float32),sampled_alpha_valid=valid,
        matte_allowed_nearest=allowed,real106_alwayskeep=real,
        effective_alpha_real_priority=effective_alpha.astype(np.float32),
        effective_RGB_real_priority=effective_rgb.astype(np.float32),
        effective_allowed_real_priority=effective_allowed,effective_valid_real_priority=effective_valid,
        prototype_initial_RGB=chart['color_rgb'],prototype_color_generated=chart['color_generated'],
        prototype_color_prior_fallback=chart['color_prior_fallback'],
        highres_to_generated_pixel_center=chart['highres_to_generated_pixel_center'],
        reference_pose=chart['reference_pose'],reference_world_view=chart['reference_world_view'],
        reference_full_projection=chart['reference_full_projection'],
        source_RGB_sha256=np.asarray(sha(DEST/'teacher_rgb_512.png')),
        source_matte_sha256=np.asarray(sha(DEST/'car_matte_512.png')))
    dark=chart['color_rgb'].max(1)<16/255
    receipt=dict(status='PASS_ALL_CHART_TEXTURE_CONTRACT',chart_rows=M,reference_raw=106,
        reference_split='BUILD',input_teacher='same hosted strongv2, byte exact decodedRGB',
        source_RGB_sha256=sha(DEST/'teacher_rgb_512.png'),source_matte_sha256=sha(DEST/'car_matte_512.png'),
        source_chart_sha256=sha(SOURCE/'reference_chart.npz'),
        samples_sha256=sha(DEST/'matte_texture_samples.npz'),
        alpha_zero=int((a==0).sum()),alpha_partial=int(((a>0)&(a<1)).sum()),alpha_one=int((a==1).sum()),
        nearest_foreground_allowed=int(allowed.sum()),real106_priority_rows=int(real.sum()),
        dark_chart_rows_before=int(dark.sum()),
        dark_chart_outside_alpha_zero=int((dark&(a==0)&~real).sum()),
        dark_chart_foreground_alpha_gt099=int((dark&(a>.99)).sum()),
        semantics=dict(sampled_alpha='Bilinear SAM binary-foreground coverage, not a brightness test',
            sampled_unpremultiplied_RGB='Bilinear(RGB*alpha)/Bilinear(alpha), zero only where alpha<=1e-6',
            matte_allowed_nearest='Strict candidate foreground at nearest512pixel, recommended prior/query support mask',
            effective_allowed_real_priority='matte_allowed_nearest OR real106_alwayskeep; use to filter continuous-surface prior/query',
            effective_valid_real_priority='sampled_alpha_valid OR real106_alwayskeep; fractional silhouette support for actual alpha sampling',
            effective_RGB_real_priority='Real106 original HR RGB takes precedence; elsewhere alpha-corrected same512 strongv2',
            prototype_initial_RGB='Original pre40fit chart colours, for the new depth branch initialization',
            original_asset_row='Row index2778..50321 in original50322-row projective asset;47544 chart samples in exact original order'),
        model_training_updates=0,PVD_updates=0,depth_unchanged=True,colour_fit_steps=0)
    (DEST/'texture_samples_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
