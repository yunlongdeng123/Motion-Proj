"""Actual current-joint row alphaT inside/outside a fixed semantic edit scope.

Default CPU preparation only. All native/road/local/car parameters stay fixed.
The full image is a disjoint partition of inside and outside edit masks; car
and K masks are separately measured overlapping protected diagnostic regions.
"""
from pathlib import Path
import argparse,time
import numpy as np
from local_gaussian_asset import ACTOR_SHA,bank,read,write,sha,load_local_asset
from local_joint_adapter import LocalJointAdapter
from render_new_local_zero import pin,npy,png

REGIONS=['full_image','outside_allowed_edit','actor_protected','K_protected','inside_allowed_edit','outside_replacement','inside_replacement','single_pixel_sentinel']

def masks(a,cfg,asset):
    receipt_path=a.masks/'receipt.json'
    surface_receipt=read(a.surface_review/'receipt.json')
    assert surface_receipt['complete'] and surface_receipt['final_atlas_sha256']==asset.manifest['source']['field']['sha256']
    assert surface_receipt['geometry_sha256']==asset.manifest['geometry']['npz']['sha256']
    if not a.masks.exists():
        assert not a.execute,'Prepare the frozen masks on CPU before GPU execution'
        a.masks.mkdir(parents=True);rows=[]
        for raw in cfg['frames']:
            ep=Path(cfg['fixed_eval'])/f'{raw:03d}/fixed_eval.npz';mp=Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz'
            ev=bank(ep);m=bank(mp);K=ev['K_protected'].astype(bool);band=ev['full_original_band'].astype(bool)
            assert np.array_equal(K,m['protected'])
            if raw in [86,110]:
                actor=m['actor_visible'].astype(bool);assert actor.any();kind='opened_strict_real_V'
            else:actor=m['actor_visible_proxy'].astype(bool);assert actor.any();kind='actor_visible_proxy_no_heldout_V_read'
            inside=band&~K&~actor;outside=~inside;full=np.ones_like(inside)
            surface_path=a.surface_review/f'{raw:03d}/surface_review.npz';surface=bank(surface_path)
            shown=surface['shown'].astype(bool);replacement=inside&shown;outside_replacement=~replacement
            y,x=np.nonzero(replacement);assert len(x)
            distance=(x-(replacement.shape[1]-1)/2)**2+(y-(replacement.shape[0]-1)/2)**2
            chosen=int(np.argmin(distance));sentinel=np.zeros_like(replacement);sentinel[y[chosen],x[chosen]]=True
            assert not (inside&outside).any() and np.array_equal(inside|outside,full)
            assert not (replacement&outside_replacement).any() and np.array_equal(replacement|outside_replacement,full)
            folder=a.masks/f'{raw:03d}';folder.mkdir()
            np.savez_compressed(folder/'scope_masks.npz',full_image=full,outside_allowed_edit=outside,actor_protected=actor,
                K_protected=K,inside_allowed_edit=inside,full_original_band=band,direct_surface_shown=shown,
                outside_replacement=outside_replacement,inside_replacement=replacement,single_pixel_sentinel=sentinel)
            preview=np.repeat(inside[...,None],3,axis=-1).astype(np.float32);preview[actor]=[.3,.5,.9];preview[K]=[.8,.3,.4];png(folder/'allowed_edit_mask.png',preview)
            rows.append(dict(raw=raw,car_mask_kind=kind,full_original_band_pixels=int(band.sum()),inside_allowed_edit_pixels=int(inside.sum()),
                outside_allowed_edit_pixels=int(outside.sum()),actor_protected_pixels=int(actor.sum()),K_pixels=int(K.sum()),
                inside_replacement_pixels=int(replacement.sum()),outside_replacement_pixels=int(outside_replacement.sum()),
                exact_inside_outside_disjoint_full_partition=True,exact_replacement_partition=True,
                sentinel_xy=[int(x[chosen]),int(y[chosen])],sentinel_selection='nearest image-centre insideReplacement pixel, row-major tie break; no RGB or alpha selection',
                masks=pin(folder/'scope_masks.npz'),fixed_eval=pin(ep),rectified_masks=pin(mp),direct_surface_projection=pin(surface_path)))
        write(receipt_path,dict(schema='R10.fixed_semantic_local_edit_masks.v3',complete=True,CPU_only=True,
            parent_asset_manifest=pin(a.asset/'manifest.json'),measurement_contract=pin(a.measurement_contract),
            fixed_eval_receipt=pin(Path(cfg['fixed_eval'])/'receipt.json'),direct_surface_receipt=pin(a.surface_review/'receipt.json'),engineering_contract=pin(a.engineering_contract),frames=cfg['frames'],rows=rows,
            definition='inside = original full band minus K minus real V for86/110 or actor proxy for134/179/198; outside is full-image complement',
            replacement_definition='inside_allowed_edit intersect direct_surface_shown (fixed geometry/material availability); current Gaussian alpha not used',
            source_colour_used_for_masks=False,current_Gaussian_alpha_used_for_masks=False,broad_canonical_alpha_not_used_as_car_mask=True,heldout_real_V_not_read=True,GPU_forward=0,GPU_backward=0,updates=0))
    r=read(receipt_path);assert r['complete'] and r['parent_asset_manifest']['sha256']==sha(a.asset/'manifest.json')
    assert r['measurement_contract']['sha256']==sha(a.measurement_contract) and r['frames']==cfg['frames']
    assert r['direct_surface_receipt']['sha256']==sha(a.surface_review/'receipt.json')
    assert r['engineering_contract']['sha256']==sha(a.engineering_contract)
    for row in r['rows']:
        for key in ['masks','fixed_eval','rectified_masks','direct_surface_projection']:assert sha(row[key]['path'])==row[key]['sha256']
        z=bank(row['masks']['path']);assert np.array_equal(z['full_image'],z['inside_allowed_edit']|z['outside_allowed_edit'])
        assert not (z['inside_allowed_edit']&z['outside_allowed_edit']).any()
        assert not (z['inside_allowed_edit']&(z['actor_protected']|z['K_protected'])).any()
        assert np.array_equal(z['inside_replacement'],z['inside_allowed_edit']&z['direct_surface_shown'])
        assert np.array_equal(z['full_image'],z['inside_replacement']|z['outside_replacement']) and not (z['inside_replacement']&z['outside_replacement']).any()
        assert z['single_pixel_sentinel'].sum()==1 and not (z['single_pixel_sentinel']&~z['inside_replacement']).any()
    return r

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['asset','measurement-contract','zero','surface-review','engineering-contract','masks','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    asset=load_local_asset(a.asset,verify_parent_pins=True);cfg=read(a.measurement_contract);zreceipt=read(a.zero/'receipt.json')
    assert asset.manifest['actor_parent_sha256']==cfg['accepted_actor_sha256']==ACTOR_SHA
    assert asset.manifest['native_gate']['mode']=='identity_no_retirement' and not len(asset.native_gate['global_row_ids'])
    assert asset.manifest['frames']==cfg['frames']==[86,110,134,179,198]
    assert zreceipt['complete'] and zreceipt['asset_manifest']['sha256']==sha(a.asset/'manifest.json')
    engineering=read(a.engineering_contract)
    assert engineering['complete'] and engineering['root_authorized_engineering_replay'] is True
    assert engineering['aggregate_pixel_multiplier']==2e-5 and engineering['aggregate_absolute_floor']==.01
    assert engineering['sentinel_absolute_tolerance']==1e-5 and engineering['constant_one_alpha_tolerance']==3e-6
    assert engineering['per_row_partition_relative_scale']==.001 and engineering['strict_zero_eligibility_not_relaxed'] is True
    scope=masks(a,cfg,asset)
    if not a.execute:
        print(dict(prepared=True,regions=REGIONS,feature_columns=[7,8,9,10,11,12,13,14],expectedF=5,expectedB=5,nativeF=0,nativeB=0,updates=0,
            mask_receipt_sha256=sha(a.masks/'receipt.json'),counts={r['raw']:[r['inside_allowed_edit_pixels'],r['inside_replacement_pixels']] for r in scope['rows']}));return
    import torch
    assert not a.output.exists();a.output.mkdir(parents=True);started=time.time();rt=None;B=0;attempts=0;error=None;rows=[]
    try:
        rt=LocalJointAdapter(a.asset,a.output/'native',enable_gpu=True);proof=read(rt.cfg['scene_proof']);native_end=proof['topology']['static_rows']+proof['topology']['dynamic_rows']
        for raw in cfg['frames']:
            z=bank(a.masks/f'{raw:03d}/scope_masks.npz');ref=bank(a.zero/f'{raw:03d}/joint_render.npz');folder=a.output/f'{raw:03d}';folder.mkdir()
            packet=rt.packet(raw);context,group,owner,gids,offset=rt.context[raw];nc=len(gids)
            columns=list(range(rt.local_channel+1,rt.local_channel+9));assert columns==[7,8,9,10,11,12,13,14]
            feature=packet['features'].detach().clone();feature[:,columns]=1;feature.requires_grad_(True);packet=dict(packet,features=feature)
            cam=rt.cameras[raw];attempts+=1
            out=rt.scene._bindings().raster(cam,rt.scene.pipe,0,packet,rt.scene._background);rt.base.calls+=1
            rt.base.trace.append(dict(call=rt.calls,raw=raw,new_local_asset=True,component='semantic_edit_footprint_VJP',single_joint_native_raster=True))
            env=rt.scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            rgb=npy(out['rgb']+(1-out['alpha'])*env);alpha=npy(out['alpha']).squeeze()
            assert rgb.dtype==ref['rgb'].dtype and rgb.tobytes()==ref['rgb'].tobytes(),'Measurement changed new0 RGB'
            assert alpha.dtype==ref['alpha'].dtype and alpha.tobytes()==ref['alpha'].tobytes(),'Measurement changed new0 alpha'
            loss=sum((out['extra'][column-2]*torch.as_tensor(z[name],device='cuda')).sum() for column,name in zip(columns,REGIONS))
            grad=torch.autograd.grad(loss,feature)[0][:,columns];B+=1;assert torch.isfinite(grad).all() and float(grad.min())>=0
            masses=npy(grad).astype(np.float32);conservation=[]
            # Preserve every region's unrounded float32 row gradient before
            # applying aggregate engineering checks, including a failed frame.
            np.savez_compressed(folder/'raw_feature_alphaT_vjp.npz',
                **{'alphaT_'+name:masses[:,j] for j,name in enumerate(REGIONS)},
                context_global_row_ids=npy(gids),context_owner_ids=npy(owner),native_context=npy(group==0),
                context_prefix_rows=np.asarray(nc),all_packet_rows=np.asarray(len(masses)),ROI_names=np.asarray(REGIONS),feature_columns=np.asarray(columns))
            for j,(column,name) in enumerate(zip(columns,REGIONS)):
                direct=float(alpha[z[name]].astype(np.float64).sum());total=float(masses[:,j].astype(np.float64).sum())
                one_error=float(abs(npy(out['extra'][column-2])-alpha).max())
                bound=engineering['sentinel_absolute_tolerance'] if name=='single_pixel_sentinel' else max(engineering['aggregate_absolute_floor'],engineering['aggregate_pixel_multiplier']*z[name].sum())
                assert one_error<engineering['constant_one_alpha_tolerance'] and abs(total-direct)<=bound,(raw,name,total,direct,one_error,bound)
                conservation.append(dict(region=name,pixels=int(z[name].sum()),all_packet_row_alphaT_mass=total,direct_alpha_mass=direct,
                    absolute_sum_error=abs(total-direct),relative_sum_error=abs(total-direct)/max(abs(direct),1e-12),
                    registered_absolute_bound=float(bound),constant_one_feature_alpha_max_abs=one_error))
            global_ids=npy(gids);native=npy(group==0);owners=npy(owner).astype(np.int32)
            oldguard=torch.isin(gids,rt.base.preserve)
            partition=masses[:nc,0].astype(np.float64)-masses[:nc,1].astype(np.float64)-masses[:nc,4].astype(np.float64)
            replacement_partition=masses[:nc,0].astype(np.float64)-masses[:nc,5].astype(np.float64)-masses[:nc,6].astype(np.float64)
            row_bound=engineering['per_row_partition_relative_scale']*(1+masses[:nc,0].astype(np.float64))
            assert (abs(partition)<=row_bound).all() and (abs(replacement_partition)<=row_bound).all(),'Per-row partition exceeds registered FP32 scale'
            np.savez_compressed(folder/'all_context_rows.npz',global_row_ids=global_ids,owner_ids=owners,native_context=native,
                old_permanent_guard=npy(oldguard),ROI_alphaT_mass=masses[:nc],ROI_names=np.asarray(REGIONS),
                full_minus_outside_minus_inside_float_roundoff=partition,
                full_minus_outsideReplacement_minus_insideReplacement_float_roundoff=replacement_partition,frames=np.asarray([raw]))
            assert np.array_equal(global_ids,np.unique(global_ids)) and (global_ids[native]<native_end).all()
            local=npy(out['extra'][rt.local_channel-2]);road=npy(out['extra'][0])
            np.savez_compressed(folder/'measurement_joint.npz',rgb=rgb,alpha=alpha,depth_expected_m=npy(out['depth_sum']).squeeze()/rt.meter_to_model/np.maximum(alpha,1e-8),
                road_alphaT=road,local_road_alphaT=local,existing_road_alphaT=road-local,actor_alphaT=npy(out['extra'][1]),context_alphaT=npy(out['extra'][2]))
            record=dict(raw=raw,all_packet_rows=len(masses),all_context_rows=nc,native_context_rows=int(native.sum()),native_global_row_count=native_end,
                packet=pin(folder/'all_context_rows.npz'),scope_masks=pin(a.masks/f'{raw:03d}/scope_masks.npz'),
                raw_region_vjp=pin(folder/'raw_feature_alphaT_vjp.npz'),
                full_RGB_alpha_byteexact_new0=True,actual_single_depth_sort=True,conservation=conservation,
                exact_pixel_partition_inside_outside=True,protected_masks_are_overlapping_diagnostics_not_disjoint_partition=True,
                per_context_row_partition_roundoff_abs_max=float(abs(partition).max()),
                per_context_row_partition_roundoff_abs_sum=float(abs(partition).sum()),
                per_context_row_replacement_partition_roundoff_abs_max=float(abs(replacement_partition).max()),
                per_context_row_replacement_partition_roundoff_abs_sum=float(abs(replacement_partition).sum()),
                per_context_row_partition_max_fraction_registered_bound=float(np.maximum(abs(partition),abs(replacement_partition)).__truediv__(row_bound).max()),
                roundoff_is_not_a_tolerance_for_strict_zero_selection=True,
                non_context_alphaT_mass=masses[nc:].astype(np.float64).sum(0).tolist())
            write(folder/'receipt.json',record);rows.append(record);print(dict(raw=raw,nativeF=rt.calls,nativeB=B,native_rows=int(native.sum())),flush=True)
            del packet,feature,out,grad,loss,masses
        rt.assert_fixed();write(a.output/'camera_timestamp_audit.json',dict(rows=rt.camera_audit))
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        write(a.output/'receipt.json',dict(schema='R10.actual_semantic_edit_footprint_alphaT_VJP.v3',complete=error is None,error=error,
            native_F_attempts=attempts,actual_nativeF=rt.calls if rt else 0,nativeF=rt.calls if rt else 0,nativeB=B,updates=0,
            asset_manifest=pin(a.asset/'manifest.json'),zero_receipt=pin(a.zero/'receipt.json'),mask_receipt=pin(a.masks/'receipt.json'),
            baseline_contract_sha256=sha(cfg['baseline_contract']),measurement_contract=pin(a.measurement_contract),
            engineering_contract=pin(a.engineering_contract),ROI_names=REGIONS,feature_columns=[7,8,9,10,11,12,13,14],frames=cfg['frames'],rows=rows,
            original_vehicle_SHA=ACTOR_SHA,all_native_source_parameters_unchanged=True,all_native_retained=True,old_road400_frozen=True,local_asset_unchanged=True,
            source_colour_used_for_selection=False,broad_canonical_alpha_not_used_as_car_mask=True,heldout_real_V_not_read=True,
            existing_formal_updates=400,new_local_optimizer_updates=0,new_diffusion_calls=0,new_semantic_calls=0,
            scope_or_quality_approved_by_this_measurement=False,production_gate_created=False,seconds=time.time()-started))

if __name__=='__main__':main()
