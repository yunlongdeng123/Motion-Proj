"""零步完整核心单stamp的只读几何/材质/原生RGBA复核；不删点、不改mask或融合。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import json,time,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
from audit_core_sources_cpu import load,read,exact,footprint_classes,alpha_bands


def hash_matches(path,expected):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()==expected


def covariance(sigma,basis):return (basis*sigma[:,None,:]**2)@basis.transpose(0,2,1)


def main():
    started=time.monotonic();C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=C/'road_clean_core_transfer_20261122_r4'
    out=R/'cpu_audit';probe=read(R/'source_core_probe/receipt.json');receipt=read(R/'single_stamp_zero/receipt.json')
    contract=read(R/'single_stamp_zero/contract.json');target_record=read(R/'target_audit/target_candidates.json')['recommended']
    target_frame=np.asarray(target_record['target_frame']['frame']);target_origin=np.asarray(target_record['target_frame']['origin'])
    assert receipt['complete'] and receipt['nativeF']==65 and receipt['nativeB']==0
    assert all(receipt[k] is False for k in ['parent_asset_modified','parent_retirement','parent_stamp_compositing'])
    assert contract['optimizer_updates']==0 and contract['new_diffusion_calls']==0
    assert not contract['target_material_is_same_position_observation']
    assert contract['material_source']=='similar_road_inference'
    parent=C/'road_surface_residual_20261121_r2/appearance_fit/asset';authority=read(parent/'authority_receipt.json')
    binding=dict(road=hash_matches(parent/'road.npz',authority['road_SHA256']),
                 provenance=hash_matches(parent/'provenance.npz',authority['provenance_SHA256']))
    assert all(binding.values())
    source_records={r['label']:r for r in probe['variants']};pair_frames={};variants=[]
    for variant in receipt['variants']:
        label=variant['label'];patch=variant['patch'];folder=R/'single_stamp_zero'/label
        component=load(folder/'road.npz');trace=load(folder/'transfer_trace.npz');provenance=read(folder/'provenance.json')
        original=load(C/f'road_gaussian_transfer_20261122_r1/source_selection/candidates/{patch}.npz')
        source=load(source_records[label]['component']);core=original['core_mask'];source_ids=original['world_row_ids'][core]
        assert len(component['active'])==len(source_ids) and component['active'].all()
        assert exact(component['source_world_row_ids'],source_ids) and exact(trace['source_world_row_ids'],source_ids)
        assert len(np.unique(source_ids))==len(source_ids)
        field_checks={k:exact(component[k],source[k]) for k in ['reference_srgb','gaussian_sigma_m','road_opacity','model_from_world','meter_to_model']}
        assert all(field_checks.values()) and exact(trace['source_RGB'],source['reference_srgb'])
        assert np.array_equal(trace['source_world_xyz'],original['world_xyz'][core])
        assert np.array_equal(trace['target_frame'],target_frame) and np.array_equal(trace['target_origin'],target_origin)
        rot=target_frame@trace['source_frame'].T;assert np.max(abs(rot-trace['rotation_world']))<1e-14
        assert np.max(abs(rot@rot.T-np.eye(3)))<1e-12 and abs(np.linalg.det(rot)-1)<1e-12
        expected=(trace['source_world_xyz']-trace['source_origin'])@rot.T+target_origin
        mean_error=float(abs(expected-trace['target_world_xyz']).max());assert mean_error<1e-9
        distance_error=float(abs(pdist(trace['source_world_xyz'])-pdist(trace['target_world_xyz'])).max());assert distance_error<1e-9
        T=component['model_from_world'];expected_model=trace['target_world_xyz']@T[:3,:3].T+T[:3,3]
        model_error=float(abs(expected_model-component['model_xyz']).max());assert model_error<1e-12
        sigma=component['gaussian_sigma_m'];assert np.array_equal(sigma[:,0],sigma[:,1])
        # 按旧RoadRows实际旋转约定重建源世界basis；原world normal与近似similarity矩阵有约1e-11的单位化差。
        scale=float(component['meter_to_model']);model_rotation=T[:3,:3]/scale
        n=original['normal_world'][core]@T[:3,:3].T/scale;n/=np.linalg.norm(n,axis=1)[:,None]
        model_q=np.c_[1+n[:,2],-n[:,1],n[:,0],np.zeros(len(n))]
        singular=np.linalg.norm(model_q,axis=1)<1e-8;model_q[singular]=[0,1,0,0]
        model_q/=np.linalg.norm(model_q,axis=1)[:,None]
        model_basis=Rotation.from_quat(model_q[:,[1,2,3,0]]).as_matrix()
        source_basis=Rotation.from_matrix(model_rotation.T[None]@model_basis).as_matrix()
        normal_error=float(abs(source_basis[:,:,2]@rot.T-component['normal_world']).max());assert normal_error<1e-12
        raw_normal_error=float(abs(original['normal_world'][core]@rot.T-component['normal_world']).max())
        source_cov=covariance(sigma,source_basis)
        expected_cov=rot[None]@source_cov@rot.T[None]
        q=component['gaussian_rotation_world_wxyz'];basis=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        actual_cov=covariance(sigma,basis)
        cov_error=float(abs(expected_cov-trace['covariance_world']).max())
        quaternion_cov_error=float(abs(actual_cov-trace['covariance_world']).max())
        assert cov_error<1e-16 and quaternion_cov_error<1e-16 and component['gaussian_rotation_world_override'].all()
        assert provenance['material_source']=='similar_road_inference' and provenance['source_halo_rows']==0 and provenance['scale']==1.
        views=[]
        for view in variant['views']:
            raw=view['raw'];f=load(folder/f'{raw:03d}'/'target_float.npz');a=f['alpha'];p=f['physical_core'];m=f['score_mask']
            assert np.array_equal(m,binary_erosion(p,iterations=3)&(a>=.95))
            effective=f['target_black']/np.maximum(a[:,:,None],1e-8)
            grey_error=float(abs(f['target_black']+.32*(1-a[:,:,None])-f['target_grey']).max());assert grey_error<1e-6
            n_high=int((np.ptp(effective[m],axis=1)>.15).sum());coverage=float((p&(a>=.95)).sum()/p.sum())
            assert n_high==view['high_chroma_pixels'] and int(m.sum())==view['opaque_core_pixels']
            assert int(p.sum())==view['physical_core_pixels'] and abs(coverage-view['alpha95_fraction'])<1e-12
            image=np.asarray(Image.open(folder/f'{raw:03d}'/'target_grey.png').convert('RGB'))
            expected_png=np.rint(np.clip(f['target_grey'],0,1)*255).astype(np.uint8)
            assert np.array_equal(image,expected_png)
            full=np.asarray(Image.open(folder/f'{raw:03d}'/'target_full_grey.png').convert('RGB'))
            x0,y0,x1,y1=[int(x) for x in f['bbox_xyxy']];assert np.array_equal(full[y0:y1,x0:x1],image)
            outside=full.copy();outside[y0:y1,x0:x1]=82
            outside_nonbackground=int((outside!=82).any(2).sum())
            if (patch,raw) not in pair_frames:pair_frames[patch,raw]=(a.copy(),m.copy(),f['parent_road_context'].copy())
            else:
                aa,mm,context=pair_frames[patch,raw];assert exact(a,aa) and exact(m,mm) and exact(f['parent_road_context'],context)
            eroded=binary_erosion(p,iterations=3)
            views.append(dict(raw=raw,physical_core_pixels=int(p.sum()),physical_core_alpha_ge95_pixels=int((p&(a>=.95)).sum()),
                eroded_physical_core_base_pixels=int(eroded.sum()),opaque_eroded_core_pixels=int(m.sum()),
                eroded_core_alpha95_fraction=float(m.sum()/eroded.sum()),
                physical_core_alpha95_fraction=coverage,high_chroma_threshold=.15,high_chroma_pixels=n_high,
                high_chroma_fraction=n_high/int(m.sum()),neutral_grey_RGBA_formula_max_abs=grey_error,
                entire_native_grey_PNG_matches_float=True,full_frame_matches_crop=True,
                full_frame_nonbackground_pixels_outside_saved_crop=outside_nonbackground,
                alpha_metric_mask_and_parent_context_identical_between_same_patch_material_variants=True,
                footprint_classes=footprint_classes(effective,f['target_grey'],a,p,m,.15),
                alpha_bands=alpha_bands(effective,f['target_grey'],a,p,.15),
                native_receipt_RoadRows_reload_RGB_max_abs=view['reload_RGB_max_abs']))
        variants.append(dict(label=label,patch=patch,all_source_core_rows_preserved=True,core_rows=len(source_ids),halo_rows=0,
            source_RGB_sigma_opacity_exact=True,field_checks=field_checks,rigid_rotation_det=float(np.linalg.det(rot)),
            rigid_mean_recompute_max_abs_m=mean_error,all_pairwise_distance_max_abs_m=distance_error,
            model_coordinates_max_abs=model_error,rotated_native_legacy_normal_max_abs=normal_error,
            rotated_raw_world_normal_normalization_delta=raw_normal_error,
            independent_world_covariance_max_abs=cov_error,quaternion_covariance_max_abs=quaternion_cov_error,
            material_source='similar_road_inference',unrelated_parent_context_rows_in_component=0,parent_retirement=False,
            world_support_clipping_applied=False,parent_component_compositing=False,views=views))
    tail_chroma=any(view['footprint_classes']['all_component_alpha_0p05_to_0p95']['actual_or_reconstructed_neutral_display']['high_chroma']>0
                    for variant in variants for view in variant['views'])
    result=dict(schema='single_core_stamp_independent_CPU_audit.v1',complete=True,all_checks_passed=True,
        variants=variants,common_target_origin=target_origin.tolist(),target_frame_matches_selected_candidate=True,
        parent_asset_matches_existing_authority_binding=binding,parent_material_not_retired_or_composited=True,
        source_core_rows_not_clipped_or_pruned=True,source_core_component_retains_diagnostic_colored_boundary_tails=tail_chroma,
        point_color_based_deletion_applied=False,world_support_crop_or_boundary_fusion_tested=False,
        parent_actual_nativeF=65,parent_actual_nativeB=0,this_audit_new_nativeF=0,this_audit_new_nativeB=0,
        optimizer_updates=0,new_diffusion_calls=0,seconds=time.monotonic()-started)
    (out/'single_stamp_cpu_audit.json').write_text(json.dumps(result,ensure_ascii=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
