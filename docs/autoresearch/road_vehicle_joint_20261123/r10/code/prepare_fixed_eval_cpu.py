"""Freeze original underbody bands, target validity partitions and loss weights.

No colour-selected main domain, constant-Z fill, GPU or parameter update.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image
from r10_common import contracts,read,load,write,sha,source_gate,all4_and_sample,srgb_to_linear_numpy,linear_to_srgb_numpy


def png(path,value):Image.fromarray(np.rint(np.clip(value,0,1)*255).astype(np.uint8)).save(path)
def pin(path):return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg,base=contracts(a.contract);gate=source_gate(cfg);a.output.mkdir(parents=True,exist_ok=False)
    f=load(cfg['source']['field']);atlas_valid=np.isfinite(f['atlas_linear']).all(-1);real=f['real_locks'].astype(np.float32);tau=np.clip(f['effective_teacher_blend'],0,1)
    source=np.stack([real,(1-real)*(1-tau),(1-real)*tau],-1)
    grid=load(Path(cfg['camera_cache']).parent/'shared_grid_geometry.npz')
    base_xyz=np.empty((grid['grid_world_row'].size,3),np.float64);base_xyz[grid['grid_world_row'].ravel()]=grid['world_xyz'].reshape(-1,3)
    base_valid=np.empty(grid['grid_world_row'].size,bool);base_valid[grid['grid_world_row'].ravel()]=grid['geometry_valid'].ravel()
    gains={int(x['raw']):float(x['gain']) for x in read(cfg['exposure_gains'])['records']};rows=[]
    old_receipt=read(Path(cfg['old_group_evidence'])/'receipt.json');old_by={x['raw']:x for x in old_receipt['rows']}
    for raw in cfg['frames']:
        cache_path=Path(cfg['camera_cache'])/f'{raw:03d}/view_cache.npz';v=load(cache_path);shape=v['geometry_valid'].shape
        mask_path=Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz';m=load(mask_path)
        ref_path=Path(cfg['actor_alpha_reference'])/f'{raw:03d}/component_render.npz';component=load(ref_path);ca=component['actor_alpha'].squeeze()
        yy,xx=np.nonzero(m['actor_visible_proxy']);bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
        box=[max(bbox[0]-30,0),max(bbox[3]-12,0),min(bbox[2]+30,shape[1]),min(bbox[3]+100,shape[0])]
        if raw in old_by:assert box==old_by[raw]['fixed_relative_lower_edge_ROI_xyxy']
        band=np.zeros(shape,bool);band[box[1]:box[3],box[0]:box[2]]=True
        visible=m.get('actor_visible',m['actor_visible_proxy']);body=binary_dilation((ca>cfg['body_alpha_threshold'])|visible,iterations=cfg['body_dilation_pixels'])
        K=m['protected'].astype(bool);ground_band=band&~body&~K&(ca<=cfg['low_canonical_alpha_max'])
        source_map_path=Path(str(v['source_map_path']));source_map=load(source_map_path)
        folder=a.output/f'{raw:03d}';folder.mkdir()
        source_valid=source_map['geometry_valid'];ids=source_map['triangle_world_rows'][source_valid];bary=source_map['triangle_weights'][source_valid]
        assert ids.min()>=0 and ids.max()<len(base_xyz)
        xyz=np.einsum('ni,nij->nj',bary,base_xyz[ids]);assert abs(xyz-source_map['world_xyz'][source_valid]).max()<1e-8
        camera=xyz@v['world_to_camera'][:3,:3].T+v['world_to_camera'][:3,3];q=camera@v['camera_K'].T
        uv_check=q[:,:2]/q[:,2,None]-.5;iy,ix=np.nonzero(source_valid);reprojection=float(abs(uv_check-np.stack([ix,iy],-1)).max());assert reprojection<1e-6
        corner_valid=np.zeros(shape,bool);corner_valid[source_valid]=base_valid[ids].all(1)
        old_certificate_path=Path(cfg['surface_geometry_certificate'])/f'{raw:03d}/surface.npz'
        if old_certificate_path.exists():assert np.array_equal(corner_valid,load(old_certificate_path)['triangle_all_corners_geometry_valid'])
        certificate_path=folder/'mesh_source_certificate.npz'
        np.savez_compressed(certificate_path,triangle_all_corners_geometry_valid=corner_valid,source_geometry_valid=source_valid,
            source_map_sha256=np.asarray(sha(source_map_path)),reprojection_max_abs=np.asarray(reprojection))
        UV=v['atlas_uv_map'];F,uv4=all4_and_sample(f['atlas_linear'],UV)
        flag=lambda array:all4_and_sample(np.zeros_like(array,dtype=np.float32),UV,array)[1]
        flags=dict(Brown_RGB_valid=m['RGB_valid'].astype(bool),cache_geometry_valid=v['geometry_valid'],cache_projection_valid=v['projection_valid'],
            triangle_all_corners_geometry_valid=corner_valid,world_XYZ_finite=np.isfinite(source_map['world_xyz']).all(-1),
            UV_four_neighbour_in_bounds=uv4,grid_geometry_four_valid=flag(grid['geometry_valid']),field_permission_four_valid=flag(f['permission_mask']),
            source_kind_four_known=flag(f['source_kind']>0),field_RGB_four_finite=flag(atlas_valid))
        certified=ground_band.copy()
        for value in flags.values():certified&=value
        # A disjoint reason map keeps the FULL original band denominator.
        reason=np.zeros(shape,np.uint8);remaining=band.copy();legend={}
        conditions=[('protected_original_vehicle_body',~body),('protected_other_K',~K),('low_canonical_alpha',ca<=cfg['low_canonical_alpha_max'])]+list(flags.items())
        for index,(name,ok) in enumerate(conditions,1):
            failed=remaining&~ok;reason[failed]=index;remaining&=ok;legend[str(index)]='fails_'+name
        success_code=len(conditions)+1;reason[remaining]=success_code;legend[str(success_code)]='certified_fixed_ground'
        assert np.array_equal(remaining,certified) and int((reason>0).sum())==int(band.sum())
        sw,_=all4_and_sample(source,UV);weight=sw@np.asarray([cfg['source_weights'][k] for k in ['same_world','similar_reference','teacher_continuous']],np.float32)
        weight=np.where(certified,weight,0).astype(np.float32)
        diagnostics={};diagnostic_meta={}
        if raw in old_by:
            packet_path=Path(cfg['old_group_evidence'])/f'CPU_attribution_v1/{raw:03d}/ROI_attribution.npz';packet=load(packet_path)
            for name in ['red_diagnostic','blue_cyan_diagnostic','dark_context_diagnostic']:
                value=np.zeros(shape,bool);value[box[1]:box[3],box[0]:box[2]]=packet[name];diagnostics[name]=value
            diagnostic_meta=dict(source=pin(packet_path),fixed_from_original_readout=True,thresholds_never_reapplied=True)
        else:
            diagnostics={name:np.zeros(shape,bool) for name in ['red_diagnostic','blue_cyan_diagnostic','dark_context_diagnostic']}
            diagnostic_meta=dict(available=False,empty_means_not_measured_not_clean=True)
        payload=dict(full_original_band=band,body_protected=body,K_protected=K,ground_band_before_support=ground_band,ground_eval=certified,
            target_common_linear=F,weight=weight,source_weights=sw,exposure_gain=np.asarray(gains[raw]),world_xyz=source_map['world_xyz'],
            atlas_uv=UV,reason=reason,**flags,**diagnostics)
        np.savez_compressed(folder/'fixed_eval.npz',**payload)
        image=np.rint(np.clip(v['photo_pinhole_srgb'],0,1)*255).astype(np.uint8);overlay=image.copy()
        overlay[ground_band]=np.rint(.5*overlay[ground_band]+.5*np.array([220,155,40])).astype(np.uint8)
        overlay[certified]=np.rint(.35*overlay[certified]+.65*np.array([40,210,150])).astype(np.uint8)
        png(folder/'fixed_band_support_overlay.png',overlay/255);png(folder/'ground_eval.png',certified.astype(float));png(folder/'whole_band.png',band.astype(float))
        bounds={}
        if raw in old_by:
            groups=load(Path(cfg['old_group_evidence'])/f'{raw:03d}/rgb_groups.npz');other=groups['group_RGB_premult'][3]
            assert float(other.min())>=-2e-5
            low=srgb_to_linear_numpy(np.maximum(other-2e-5,0))*gains[raw];excess=np.maximum(low-F.transpose(2,0,1),0)
            for name,domain in dict(certified_ground=certified,**diagnostics).items():
                known=domain&certified;w=weight[known];total=float(w.sum())
                bounds[name]=dict(full_pixels=int(domain.sum()),known_target_pixels=int(known.sum()),unknown_target_pixels=int((domain&~certified).sum()),
                    unavoidable_positive_channel_samples=int((excess[:,known]>0).sum()),
                    old_colour_only_weighted_MSE_lower_bound=float(((excess[:,known]**2).mean(0)*w).sum()/total) if total else None)
        row=dict(raw=raw,proxy_bbox=bbox,full_original_band_xyxy=box,full_original_band_pixels=int(band.sum()),ground_band_before_support_pixels=int(ground_band.sum()),
            certified_ground_eval_pixels=int(certified.sum()),unknown_ground_target_pixels=int((ground_band&~certified).sum()),
            independent_flag_counts_within_full_band={k:int((band&x).sum()) for k,x in flags.items()},
            disjoint_support_partition={legend[str(k)]:int((reason==k).sum()) for k in range(1,success_code+1)},
            reason_legend=legend,diagnostic_baseline=diagnostic_meta,nonnegative_frozen_other_bound=bounds,
            bound_applies_to_old_colour_only_path_not_opacity_retirement=True,roundoff_margin_encoded_RGB=2e-5,
            source_cache=pin(cache_path),source_map=pin(source_map_path),geometry_certificate=pin(certificate_path),masks=pin(mask_path),actor_alpha_geometry_reference=pin(ref_path),
            fixed_eval=pin(folder/'fixed_eval.npz'),body_visible_evidence='strict V' if 'actor_visible' in m else 'instance proxy, not sealed V')
        write(folder/'receipt.json',row);rows.append(row)
    result=dict(schema='R10.immutable_fixed_underbody_eval.v1',complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,
        contract=pin(a.contract),baseline_contract=pin(cfg['baseline_contract']),source_gate=gate,source_field_sha256=cfg['source']['sha256'],rows=rows,
        ground_band_definition=cfg['ground_band_definition'],full_denominators_retained=True,unknown_target_not_scored_as_zero=True,
        primary_loss='fixed weighted MSE: linear(actual joint RGB)*fixed gain vs immutable F; ground_eval only; full band, all support failures and original diagnostic masks always reported',
        fixed_eval_not_random_train_loss=True,raw_geometry_not_replaced_with_constant_Z=True,
        budget=dict(existing_formal=400,total_ceiling=2000,next_gate_phase_max=600,next_cumulative_stop=1000,smoke_updates_separate=13),
        vehicle_original_asset_frozen=True,optimization_enabled=False,stage='ready_for_explicit_root_scheduled_five_view_measurement')
    write(a.output/'receipt.json',result);write(a.output/'fixed_eval_contract_v1.json',result)
    print(dict(complete=True,receipt=str(a.output/'receipt.json'),sha256=sha(a.output/'receipt.json'),rows=[dict(raw=x['raw'],band=x['full_original_band_pixels'],certified=x['certified_ground_eval_pixels']) for x in rows]))


if __name__=='__main__':main()
