"""Actual local-only native coverage; not joint local alphaT. Zero updates."""
from pathlib import Path
import argparse,time,shutil
import numpy as np
from local_gaussian_asset import load_local_asset,read,write,sha
from local_joint_adapter import LocalJointAdapter
from render_new_local_zero import pin,png,npy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    asset=load_local_asset(a.asset,verify_parent_pins=True);m=asset.manifest
    assert int(asset.field['active'].sum())==142187 and len(asset.field['active'])==175302
    assert m['source']['field']['sha256']=='45b4b9d752f1584019e6687008de489de071a81907ab2c3c1b61b20964f64d32'
    assert m['geometry']['npz']['sha256']=='0d62e87abddd3017a40b06d80c485d331c0feb5c13a433be7f50f6933cda269b'
    assert m['frames']==[86,110,134,179,198]
    if not a.execute:
        print(dict(prepared=True,CPU_only=True,asset=pin(a.asset/'manifest.json'),active_rows=142187,total_rows=175302,
            expected_nativeF=5,nativeB=0,updates=0,intrinsic_local_only=True));return
    assert not a.output.exists();a.output.mkdir(parents=True);start=time.time();rt=None;rows=[];error=None
    before={k:v.copy() for k,v in asset.field.items()};initial_sha=sha(a.asset/'manifest.json')
    try:
        rt=LocalJointAdapter(a.asset,a.output/'native',enable_gpu=True)
        assert not rt.gate_ids.numel() and m['native_gate']['mode']=='identity_no_retirement'
        assert npy(rt.scene._background).shape==(3,) and (npy(rt.scene._background)==0).all()
        for raw in m['frames']:
            with rt.torch.no_grad():out=rt.render(raw,'local')
            rgb=npy(out['rgb']);alpha=npy(out['alpha']).squeeze();depth_sum=npy(out['depth_sum_m']).squeeze()
            assert rgb.shape==(3,768,1152) and alpha.shape==(768,1152)
            assert np.isfinite(rgb).all() and np.isfinite(alpha).all() and np.isfinite(depth_sum).all()
            assert (alpha>=0).all() and (alpha<=1).all()
            valid=alpha>1e-8;depth=np.zeros_like(depth_sum);depth[valid]=depth_sum[valid]/alpha[valid]
            folder=a.output/f'{raw:03d}';folder.mkdir()
            np.savez_compressed(folder/'local_render.npz',rgb=rgb,alpha=alpha,depth_sum_m=depth_sum,
                depth_expected_m=depth,depth_valid=valid,covered_ge95=alpha>=.95)
            # Also store alpha alone for an immutable downstream coverage pin.
            np.save(folder/'intrinsic_alpha.npy',alpha,allow_pickle=False)
            png(folder/'local_rgb.png',rgb);png(folder/'local_alpha.png',alpha)
            png(folder/'local_depth_0_60m.png',np.where(valid,np.clip(depth/60,0,1),0))
            row=dict(raw=raw,render=pin(folder/'local_render.npz'),intrinsic_alpha=pin(folder/'intrinsic_alpha.npy'),
                alpha_dtype=str(alpha.dtype),alpha_shape=list(alpha.shape),alpha_min=float(alpha.min()),alpha_max=float(alpha.max()),
                positive_alpha_pixels=int((alpha>0).sum()),intrinsic_alpha_ge95_pixels=int((alpha>=.95).sum()),
                depth_valid_pixels=int(valid.sum()),depth_min_valid_m=float(depth[valid].min()),depth_max_valid_m=float(depth[valid].max()),
                local_alphaT_vs_intrinsic_alpha_max_abs=float(abs(npy(out['local_road_alpha'])-alpha).max()),
                actual_component='local',environment_added=False,background_RGB=[0,0,0])
            write(folder/'metrics.json',row);rows.append(row);print(dict(raw=raw,F=rt.calls,covered=row['intrinsic_alpha_ge95_pixels']),flush=True)
        rt.assert_fixed()
        for key,value in before.items():assert value.dtype==asset.field[key].dtype and value.tobytes()==asset.field[key].tobytes()
        assert sha(a.asset/'manifest.json')==initial_sha and rt.calls==5
        write(a.output/'camera_timestamp_audit.json',dict(rows=rt.camera_audit))
        code=a.output/'executed_code';code.mkdir()
        for path in [Path(__file__),Path(__file__).with_name('local_joint_adapter.py'),Path(__file__).with_name('local_gaussian_asset.py')]:shutil.copy2(path,code/path.name)
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        write(a.output/'receipt.json',dict(schema='R10.actual_intrinsic_new_local_coverage.v1',complete=error is None,error=error,
            asset_manifest=pin(a.asset/'manifest.json'),source_field_sha256=m['source']['field']['sha256'],geometry_sha256=m['geometry']['npz']['sha256'],
            local_field=pin(a.asset/'local_field.npz'),frames=m['frames'],active_rows=142187,total_rows=175302,
            actual_component='local_only',intrinsic_alpha_from_actual_native_output=True,joint_local_alphaT_not_used=True,
            target_free_loader=True,training_targets_read=False,all_asset_fields_unchanged=error is None,
            original_native_and_actor_parameters_unmodified=True,nativeF=rt.calls if rt else 0,nativeB=0,updates=0,
            new_diffusion_calls=0,source_or_geometry_changed=False,rows=rows,depth_preview_linear_range_m=[0,60],
            zero_depth_is_invalid_sentinel_when_depth_valid_false=True,seconds=time.time()-start,whole_scene_quality_accepted=False))


if __name__=='__main__':main()
