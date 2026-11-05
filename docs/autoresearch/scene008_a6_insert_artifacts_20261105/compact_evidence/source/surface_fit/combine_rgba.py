"""New combined asset: fixed C2 geometry, approved matte RGB and alpha.

Generated opacity is .99*sampled_alpha; real106 keeps .99. No RGB-darkness
threshold, free opacity fitting, pose change, or old-formal SH initialization.
"""
import argparse,json
from pathlib import Path
import numpy as np
from fit_surface import bank,sha,same,geometry_hash,provenance_hash,write,ART


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--depth-fit',default=str(ART/'surface_fit/masked_depth_v1'))
    p.add_argument('--matte',default=str(ART/'vehicle/matte_contract/matte_texture_samples_filled_v2.npz'))
    p.add_argument('--output',default=str(ART/'surface_fit/combined_v1'))
    args=p.parse_args();root=Path(args.depth_fit);dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    path=root/'smooth_depth_matte_initialization.npz';a=bank(path);post=bank(root/'posterior.npz');m=bank(args.matte)
    original_rows=post['query_original_asset_rows'];chart=a['representation_kind']==1;real=a['resample_rgb_real']
    qindex=original_rows[chart]-int((~chart).sum())
    assert np.array_equal(m['original_asset_row'][qindex],original_rows[chart])
    assert np.array_equal(m['real106_alwayskeep'][qindex],real[chart])
    alpha=m['sampled_alpha'][qindex].astype(np.float32)
    assert np.isfinite(alpha).all() and ((alpha>=0)&(alpha<=1)).all()
    keep=np.ones(len(chart),bool);keep[np.flatnonzero(chart)]=real[chart]|(alpha>0)
    n=len(chart);out={k:(v[keep].copy() if v.ndim and v.shape[0]==n else v.copy()) for k,v in a.items()}
    idx=np.flatnonzero(chart);target_opacity=np.float32(.99)*alpha
    target_opacity[real[chart]]=np.float32(.99)
    opacity=a['opacity'].copy();opacity[idx]=target_opacity;out['opacity']=opacity[keep]
    newchart=out['representation_kind']==1;newreal=out['resample_rgb_real']
    assert (out['opacity'][newchart]>0).all()
    assert (out['opacity'][newreal]==np.float32(.99)).all()
    for key in ['canonical_sh','rgb','colour_inferred','xyz','normal','sigma','thickness','resample_source_row','resample_source_parent']:
        assert same(out[key],a[key][keep]),key
    assert same(out['opacity'][~newchart],a['opacity'][~chart])
    out['geometry_hash']=np.asarray(geometry_hash(out));out['provenance_hash']=np.asarray(provenance_hash(out))
    meta=json.loads(str(a['metadata_json']));meta['combined_RGBA_contract']=dict(
        source_depth_asset=str(path),source_depth_asset_sha256=sha(path),matte_samples=str(args.matte),matte_sha256=sha(args.matte),
        generated_opacity='.99 * sampled_alpha',real106_opacity=.99,zero_alpha_nonreal_not_born=True,
        RGB_source='Approved matte premultiplied/unpremultiplied samples; reference real106 coefficients protected',
        old_formal_SH_used=False,geometry_changed_from_old_prior=True,no_free_camera_pose_scale_or_opacity_optimization=True,
        PVD_training_updates=0,colour_optimization_updates=0)
    meta['geometry_hash']=str(out['geometry_hash']);meta['provenance_hash']=str(out['provenance_hash'])
    out['metadata_json']=np.asarray(json.dumps(meta,ensure_ascii=False,sort_keys=True,allow_nan=False))
    asset=dest/'combined_initial.npz';np.savez_compressed(asset,**out)
    receipt=dict(status='PASS_COMBINED_RGBA_INITIALIZATION',asset=str(asset),asset_sha256=sha(asset),
        geometry_hash=str(out['geometry_hash']),rows=len(out['xyz']),retained_source_rows=int((~newchart).sum()),
        chart_rows=int(newchart.sum()),real106_rows=int(newreal.sum()),
        zero_alpha_nonreal_removed=int((~keep).sum()),partial_alpha_chart_rows=int(((alpha>0)&(alpha<1)&~real[chart]).sum()),
        generated_opacity_changed=True,RGB_and_XYZ_unchanged_from_matched_matte_depth_initialization=True,
        unknown_future_fit_columns_must_keep_this_initialization=True,contract=meta['combined_RGBA_contract'])
    np.savez_compressed(dest/'row_mapping.npz',source_depth_rows=np.flatnonzero(keep),original_projective_asset_rows=original_rows[keep],
                        chart_alpha=alpha,source_depth_rows_keep=keep)
    write(dest/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
