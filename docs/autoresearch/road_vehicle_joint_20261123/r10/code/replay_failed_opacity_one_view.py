"""One target-free native replay of the two-update failed opacity state.

Only --execute constructs CUDA. Readout comparisons run after the one actual
joint raster and never affect loading, Gaussian fields, gates or rendering.
"""
from pathlib import Path
import argparse,time,shutil
import numpy as np
from local_gaussian_asset import read,write,bank,sha
from local_opacity_gate import load_opacity_sidecar,pin
from fit_new_local_rgb import arrays,png


def pgm(path,array):
    a=np.asarray(array)
    assert a.ndim==2 and a.dtype==np.uint16
    Path(path).write_bytes(('P5\n%d %d\n65535\n'%(a.shape[1],a.shape[0])).encode('ascii')+a.astype('>u2').tobytes())


def stats(mask,da,drgb,original,current,zero,shown,halo):
    count=int(mask.sum())
    if not count:return dict(pixels=0)
    absolute=np.where(mask,abs(da),-1);y,x=np.unravel_index(absolute.argmax(),da.shape)
    excess=mask&(abs(da)>.5/255);rgbexcess=mask&(abs(drgb).max(0)>1/255)
    partition={}
    for name,region in [('direct_surface_shown',shown),('halo8_only',halo),('outside_halo8',~(shown|halo))]:
        selected=excess&region
        partition[name]=dict(pixels=int(selected.sum()),alpha_increased=int((selected&(da>0)).sum()),alpha_decreased=int((selected&(da<0)).sum()))
    assert sum(v['pixels'] for v in partition.values())==int(excess.sum())
    return dict(pixels=count,alpha_increased_exact=int((mask&(da>0)).sum()),alpha_decreased_exact=int((mask&(da<0)).sum()),
        alpha_identical=int((mask&(da==0)).sum()),alpha_increased_gt1e6=int((mask&(da>1e-6)).sum()),alpha_decreased_gt1e6=int((mask&(da< -1e-6)).sum()),
        max_abs_alpha_delta=float(abs(da[mask]).max()),signed_alpha_delta_at_abs_max=float(da[y,x]),
        max_abs_alpha_location_xy=[int(x),int(y)],original_alpha_at_max=float(original[y,x]),zero_alpha_at_max=float(zero[y,x]),
        replay_alpha_at_max=float(current[y,x]),max_pixel_direct_surface_shown=bool(shown[y,x]),max_pixel_halo8=bool(halo[y,x]),
        alpha_bound=.5/255,alpha_bound_exceeded_pixels=int(excess.sum()),
        alpha_exceed_increase=int((excess&(da>0)).sum()),alpha_exceed_decrease=int((excess&(da<0)).sum()),
        max_abs_RGB_delta=float(abs(drgb[:,mask]).max()),RGB_bound=1/255,RGB_bound_exceeded_pixels=int(rgbexcess.sum()),
        new_alpha95_holes_from_original_ge95=int((mask&(original>=.95)&(current<.95)).sum()),
        alpha_excess_support_partition=partition)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['sidecar','zero','baseline-measurement','masks','surface-review','output']:
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--raw',type=int,default=86);p.add_argument('--execute',action='store_true');a=p.parse_args()
    m=read(a.sidecar/'manifest.json');zr=read(a.zero/'receipt.json');mr=read(a.masks/'receipt.json');sr=read(a.surface_review/'receipt.json')
    assert m['complete'] and m['failed_unaccepted_diagnostic_only'] and m['training_record']['actual_optimizer_updates']==2
    assert zr['complete'] and mr['complete'] and sr['complete']
    assert zr['asset_manifest']['sha256']==mr['parent_asset_manifest']['sha256']==m['parent_asset_manifest']['sha256']
    assert sr['final_atlas_sha256']==m['source_field_sha256']
    if not a.execute:
        print(dict(prepared=True,raw=a.raw,diagnostic_existing_updates=2,expected_nativeF=1,nativeB=0,updates=0));return
    assert not a.output.exists();a.output.mkdir(parents=True);start=time.time();rt=None;error=None;summary=None
    asset_sha=sha(a.sidecar/'opacity_gate.npz')
    try:
        rt=load_opacity_sidecar(a.sidecar,a.output/'native',enable_gpu=True)
        with rt.torch.no_grad():value=arrays(rt,a.raw)
        assert rt.calls==1
        rt.assert_opacity_only();assert sha(a.sidecar/'opacity_gate.npz')==asset_sha
        np.savez_compressed(a.output/'joint_render_2updates_diagnostic.npz',**value)
        for name,key in [('shared','rgb'),('alpha','alpha'),('local_road_alphaT','local_road_alphaT')]:
            png(a.output/(name+'_2updates_diagnostic.png'),value[key])
        # References and masks are read after the target-free actual render.
        old=bank(a.baseline_measurement/f'{a.raw:03d}/baseline_joint.npz');zero=bank(a.zero/f'{a.raw:03d}/joint_render.npz')
        domain=bank(a.masks/f'{a.raw:03d}/scope_masks.npz');surface=bank(a.surface_review/f'{a.raw:03d}/surface_review.npz')
        from scipy.ndimage import distance_transform_edt
        shown=surface['shown'].astype(bool);halo=(distance_transform_edt(~shown)<=8)&~shown
        outside=domain['outside_allowed_edit'];protect=domain['K_protected']|domain['actor_protected'];band=domain['full_original_band']
        differences={};comparisons={};scales={}
        for name,ref in [('original_baseline',old),('v4zero',zero)]:
            da=value['alpha']-ref['alpha'];drgb=value['rgb']-ref['rgb']
            differences[name+'_alpha_delta']=da;differences[name+'_RGB_delta']=drgb
            if 'depth_expected_m' in ref:differences[name+'_depth_expected_m_delta']=value['depth_expected_m']-ref['depth_expected_m']
            comparisons[name]=dict(outside=stats(outside,da,drgb,ref['alpha'],value['alpha'],zero['alpha'],shown,halo),
                full_original_band=stats(band,da,drgb,ref['alpha'],value['alpha'],zero['alpha'],shown,halo),
                K_car_RGBA_byteexact=value['rgb'][:,protect].tobytes()==ref['rgb'][:,protect].tobytes() and value['alpha'][protect].tobytes()==ref['alpha'][protect].tobytes())
            scale=max(float(abs(da).max()),1e-12);signed=np.rint(np.clip(.5+.5*da/scale,0,1)*65535).astype(np.uint16)
            pgm(a.output/(name+'_signed_alpha_delta.pgm'),signed)
            rgbmax=abs(drgb).max(0);rgbscale=max(float(rgbmax.max()),1e-12)
            pgm(a.output/(name+'_RGB_abs_max_delta.pgm'),np.rint(np.clip(rgbmax/rgbscale,0,1)*65535).astype(np.uint16))
            scales[name]=dict(signed_alpha_pgm_zero_code=32768,signed_alpha_pgm_scale=scale,RGB_abs_max_pgm_scale=rgbscale,
                PGM_bits=16,PGM_endian='big',exact_float_arrays='differences_and_support.npz')
        np.savez_compressed(a.output/'differences_and_support.npz',**differences,outside_allowed_edit=outside,
            full_original_band=band,K_car=protect,direct_surface_shown=shown,halo8_only=halo)
        # Display only: green increases, magenta decreases; exact floats above.
        da=differences['original_baseline_alpha_delta'];scale=max(float(abs(da[outside]).max()),1e-12)
        display=np.zeros((*da.shape,3),np.float32);positive=np.clip(da/scale,0,1);negative=np.clip(-da/scale,0,1)
        display[...,1]=positive;display[...,0]=negative;display[...,2]=negative
        png(a.output/'original_baseline_signed_alpha_delta_display.png',display)
        png(a.output/'outside_alpha_excess_mask.png',outside&(abs(da)>.5/255))
        summary=dict(complete=True,scope='one_actual_replay_of_failed_two_update_state',raw=a.raw,
            existing_smoke_updates_in_loaded_state=2,additional_updates=0,additional_nativeF=1,nativeB=0,
            target_free_loader=True,targets_read_by_loader=False,references_read_after_actual_render=True,
            sidecar=pin(a.sidecar/'manifest.json'),gate_sha256_unchanged=asset_sha,zero_receipt=pin(a.zero/'receipt.json'),
            baseline_receipt=pin(a.baseline_measurement/'receipt.json'),mask_receipt=pin(a.masks/'receipt.json'),direct_surface_receipt=pin(a.surface_review/'receipt.json'),
            halo_definition='Euclidean image distance <=8 pixels from actual direct_surface_shown, excluding shown itself; disjoint complement is outside_halo8',
            comparisons=comparisons,PGM_scales=scales,failed_RGBA_now_saved=True,raw_RGB_alpha_depth=pin(a.output/'joint_render_2updates_diagnostic.npz'),
            numeric_thresholds_changed=False,new_method_or_formal_optimization=False,quality_decision='NOT_ACCEPTED')
        write(a.output/'diagnostic.json',summary);write(a.output/'camera_timestamp_audit.json',dict(rows=rt.camera_audit))
        copied=a.output/'executed_code';copied.mkdir();shutil.copy2(__file__,copied/Path(__file__).name)
        print(dict(complete=True,diagnostic=pin(a.output/'diagnostic.json'),outside=comparisons['original_baseline']['outside']),flush=True)
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        write(a.output/'receipt.json',dict(schema='R10.failed_opacity_one_view_actual_replay.v1',complete=error is None,error=error,
            nativeF=rt.calls if rt else 0,nativeB=0,updates=0,loaded_state_updates=2,new_diffusion_calls=0,
            actual_raw=a.raw,source_field_sha256=m['source_field_sha256'],gate_sha256=asset_sha,
            seconds=time.time()-start,whole_scene_quality_accepted=False))


if __name__=='__main__':main()
