"""Fixed topology contract: fill closed SAM holes, preserve exterior gaps.

No SAM rerun, no dilation, no area/radius thresholds or search. Rawv1 retained.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_fill_holes,label,map_coordinates,binary_erosion

RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105')
ROOT=RUN.parent/'scene008_a6_insert_artifacts_20261105/vehicle/matte_contract'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    with np.load(ROOT/'matte_contract.npz') as z:raw={k:z[k].copy() for k in z.files}
    with np.load(ROOT/'matte_texture_samples.npz') as z:sample={k:z[k].copy() for k in z.files}
    sam=raw['raw_sam_mask'].astype(bool);known=raw['known_protected_mask'].astype(bool)
    filled_sam=binary_fill_holes(sam)
    matte=filled_sam|known;raw_matte=sam|known
    holes=filled_sam&~sam;added=matte&~raw_matte
    labels,_=label(~sam);outside_labels=np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]])
    exterior=(labels!=0)&np.isin(labels,outside_labels)
    assert not filled_sam[exterior].any()
    assert np.all(matte[known]) and np.all(matte[raw_matte])
    rgb=np.asarray(Image.open(ROOT/'teacher_rgb_512.png').convert('RGB')).astype(np.float64)/255
    alpha=matte.astype(np.float64)
    Image.fromarray(matte.astype(np.uint8)*255).save(ROOT/'car_matte_filled_v2_512.png')
    Image.fromarray(holes.astype(np.uint8)*255).save(ROOT/'filled_holes_v2_512.png')
    np.savez_compressed(ROOT/'matte_contract_filled_v2.npz',alpha=alpha.astype(np.float32),foreground=matte,
        raw_sam_mask=sam,filled_sam_mask=filled_sam,known_protected_mask=known,
        raw_v1_foreground=raw_matte,closed_holes_filled=holes,added_to_v1_foreground=added,
        source_RGB_sha256=raw['source_RGB_sha256'])
    uv=sample['generated_xy'];coords=[uv[:,1],uv[:,0]]
    a=map_coordinates(alpha,coords,order=1,mode='constant',cval=0,prefilter=False)
    premul=np.column_stack([map_coordinates(rgb[...,j]*alpha,coords,order=1,mode='constant',cval=0,prefilter=False) for j in range(3)])
    valid=a>1e-6;unpremul=np.zeros_like(premul);unpremul[valid]=premul[valid]/a[valid,None]
    near=np.floor(uv+.5).astype(int);inbounds=((near>=0)&(near<512)).all(1);near=np.clip(near,[0,0],[511,511])
    allowed=inbounds&matte[near[:,1],near[:,0]]
    real=sample['real106_alwayskeep'];effective_rgb=unpremul.copy();effective_rgb[real]=sample['prototype_initial_RGB'][real]
    effective_alpha=a.copy();effective_alpha[real]=1
    previous_alpha=sample['sampled_alpha'].copy();previous_allowed=sample['effective_allowed_real_priority'].copy()
    sample.update(sampled_alpha=a.astype(np.float32),sampled_premultiplied_RGB=premul.astype(np.float32),
        sampled_unpremultiplied_RGB=unpremul.astype(np.float32),sampled_alpha_valid=valid,
        matte_allowed_nearest=allowed,effective_alpha_real_priority=effective_alpha.astype(np.float32),
        effective_RGB_real_priority=effective_rgb.astype(np.float32),
        effective_allowed_real_priority=allowed|real,effective_valid_real_priority=valid|real,
        source_matte_sha256=np.asarray(sha(ROOT/'car_matte_filled_v2_512.png')),
        matte_contract_version=np.asarray('filled_v2'),source_raw_SAM_sha256=np.asarray(sha(ROOT/'sam_raw_mask_512.png')))
    assert np.array_equal(sample['effective_RGB_real_priority'][real],sample['prototype_initial_RGB'][real])
    np.savez_compressed(ROOT/'matte_texture_samples_filled_v2.npz',**sample)
    overlay=(rgb*255+.5).astype(np.uint8);overlay[holes]=[255,200,0]
    edge=matte&~binary_erosion(matte);overlay[edge]=[0,255,70]
    Image.fromarray(overlay).save(ROOT/'filled_topology_overlay_v2_512.png')
    panel=Image.new('RGB',(2048,542),(237,243,247));draw=ImageDraw.Draw(panel)
    for i,(im,title) in enumerate([(sam,'raw SAM /untouched'),(raw_matte,'v1 SAM +known'),(matte,'v2 fill closed holes +known'),(overlay,'yellow filled /green boundary')]):
        if im.dtype==bool:im=im.astype(np.uint8)*255
        panel.paste(Image.fromarray(im).convert('RGB'),(i*512,30));draw.text((i*512+6,8),title,fill=(12,25,38))
    panel.save(ROOT/'filled_v2_review_panel.png')
    receipt=dict(status='PASS_FIXED_TOPOLOGY_FILLED_V2',version='filled_v2',SAM_predict_calls_added=0,
        model_updates=0,operation='scipy.ndimage.binary_fill_holes(raw_SAM), then OR known mask',
        no_dilation=True,no_radius_or_area_threshold=True,no_grid_or_scale_search=True,
        raw_SAM_mask_unchanged=True,raw_v1_contract_unchanged=True,known_always_priority=True,
        closed_raw_SAM_holes_filled_pixels=int(holes.sum()),new_matte_pixels_added_vs_v1=int(added.sum()),
        exterior_connected_background_preserved_before_known_union=True,
        matte_pixels=int(matte.sum()),known_pixels=int(known.sum()),
        chart_rows=len(a),chart_alpha_changed_vs_v1=int((a.astype(np.float32)!=previous_alpha).sum()),
        extra_chart_prior_allowed_vs_v1=int(((allowed|real)&~previous_allowed).sum()),
        effective_allowed_chart_rows=int((allowed|real).sum()),alpha_zero=int((a==0).sum()),
        source_RGB_sha256=sha(ROOT/'teacher_rgb_512.png'),
        matte_sha256=sha(ROOT/'car_matte_filled_v2_512.png'),
        samples_sha256=sha(ROOT/'matte_texture_samples_filled_v2.npz'),
        final_surface_instruction='Use filled_v2 effective_allowed_real_priority for prior/query; real106 RGB takes precedence; restart from prototype colours then40BUILD recalibrate')
    (ROOT/'filled_v2_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
