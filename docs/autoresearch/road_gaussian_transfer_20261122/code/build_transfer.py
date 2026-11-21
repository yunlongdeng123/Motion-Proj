"""表面高斯块拼接：共同米制映射、完整协方差和单块归属。"""
from native import *
from fixed_surface import FixedRoadSurface,optical_ownership
from gaussian_patch_transport import transport_patch,merge_transported_patches,deduplicate_centers
from drivingeditor_dynamic_edit.v44_factory import _legacy_road_world_quaternions
from scipy.spatial import cKDTree
from rigid_patch import RoadFrames,transport_rigid

def main():
    start=time.monotonic();out=ROOT/'proposal';out.mkdir(exist_ok=True)
    surface=FixedRoadSurface.load(ROOT/'target/surface.npz');target=read(ROOT/'target/receipt.json')
    quality=read(ROOT/'source_native/receipt.json')['quality'];scores=[]
    for candidate in sorted({x['patch'] for x in quality}):
        rows=[x for x in quality if x['patch']==candidate and x['score_n']>=100 and x['RGB_MAE'] is not None and x['RGB_MAE']<.05 and x['HF_cosine'] is not None]
        if not rows:continue
        weight=np.array([x['score_n'] for x in rows],float);weight/=weight.sum()
        score=float(sum(w*(x['HF_cosine']-2*x['RGB_MAE']-.1*(1-x['alpha95_fraction'])) for w,x in zip(weight,rows)))
        scores.append(dict(candidate=candidate,score=score,qualified_views=len(rows)))
    assert scores,'原生照片资格没有可迁移源块'
    chosen=max(scores,key=lambda x:(x['score'],x['candidate']))['candidate']
    write(out/'source_decision.json',dict(chosen=chosen,candidates=scores,selection='maximum photo-HF agreement minus RGB/coverage error; no target hidden RGB'))
    source=load(ROOT/'source_selection/candidates'/f'{chosen}.npz');core=source['core_mask']
    r=load(PARENT/'road.npz');p=load(PARENT/'provenance.npz');rowflags=load(ROOT/'target/rows.npz')
    forward=surface.matrix[:3,0];forward=forward/np.linalg.norm(forward)
    frames=RoadFrames(p['world_xyz'],r['normal_world'],rowflags['geometric_valid'],forward)
    # halo只用于原块隔离核验；重叠core的内侧归属保证不迁入未核验halo杂物。
    ids=source['world_row_ids'][core];origin=source['source_center_world'];frame=source['source_tangent_frame_world'][:3,:3]
    q=_legacy_road_world_quaternions(source['normal_world'][core],source['model_from_world'],
        float(load(PARENT/'road.npz')['meter_to_model']))
    side=float(np.min(source['core_size_m']));stride=side*.75
    lo=np.asarray(target['target_xy_min']);hi=np.asarray(target['target_xy_max'])
    xs=np.arange(np.floor(lo[0]/stride)*stride,hi[0]+stride,stride)
    ys=np.arange(np.floor(lo[1]/stride)*stride,hi[1]+stride,stride)
    occupied=surface.world[surface.target,:2];tree=cKDTree(occupied)
    patches=[];placements=[];costs=[];serial=0
    for y in ys:
        for x in xs:
            if not tree.query_ball_point([x,y],side*.72):continue
            target_frame=frames.at([x,y],side*.75)
            if target_frame is None:continue
            moved=transport_rigid(source['world_xyz'][core],source['gaussian_sigma_m'][core],q,
                source['reference_srgb'][core],source['road_opacity'][core],ids,
                source_origin=origin,source_frame=frame,target=target_frame,patch_id=serial,source_asset_id=str(PARENT))
            keep=surface.domain(moved['world_xyz'][:,:2])
            if not keep.any():continue
            for key,value in list(moved.items()):
                if key!='trace' and isinstance(value,np.ndarray) and value.ndim and len(value)==len(keep):moved[key]=value[keep]
            # 块中心偏好+Potts连贯性，仅选择完整高斯行，输出不做颜色平均。
            costs.append(.002*np.sum((moved['source_uv']/(side*.5))**2,axis=1))
            patches.append(moved);placements.append(dict(patch_id=serial,source=chosen,target_center_xy=[float(x),float(y)],
                target_origin=target_frame['origin'].tolist(),target_frame=target_frame['frame'].tolist(),
                rotation_world=(target_frame['frame']@frame.T).tolist(),target_plane_inliers=target_frame['inliers'],
                target_plane_samples=target_frame['samples'],target_plane_residual_p95_m=target_frame['inlier_height_residual_p95'],
                rotation_rad=0.,scale=1.,candidate_rows=len(moved['world_xyz'])))
            serial+=1
    merged=merge_transported_patches(patches,cell_size=.04,unary_cost=np.concatenate(costs),
        ownership_coordinates=np.concatenate([p['target_uv'] for p in patches]),grid_origin=lo,
        smoothness=.003,max_iterations=4)
    duplicate=deduplicate_centers(merged['world_xyz'],tolerance=.0001)
    keep=duplicate['keep_mask'];trace=merged.pop('trace')
    for key,value in list(merged.items()):
        if isinstance(value,np.ndarray) and value.ndim and len(value)==len(keep):merged[key]=value[keep]
    nearest_distance,nearest=cKDTree(p['world_xyz'][:,:2]).query(merged['world_xyz'][:,:2])
    merged['minimum_model_depth']=r['minimum_model_depth'][nearest]
    width=max(.08,4*float(np.median(source['gaussian_sigma_m'][core,:2])))
    new_weight=surface.blend_weight(merged['world_xyz'][:,:2],width)
    old_weight=surface.blend_weight(p['world_xyz'][:,:2],width)*rowflags['replace_allowed']
    # 正规源核心原值单列，零步只在边界按光学厚度分配ownership。
    merged['source_road_opacity']=merged['road_opacity'].copy();merged['ownership_weight']=new_weight.astype(np.float32)
    merged['road_opacity']=optical_ownership(merged['road_opacity'],new_weight).astype(np.float32)
    np.savez_compressed(out/'transferred_rows.npz',**merged)
    np.savez_compressed(out/'parent_ownership.npz',transfer_weight=old_weight.astype(np.float32),
        parent_opacity=optical_ownership(r['road_opacity'],1-old_weight).astype(np.float32),
        parent_active=r['active']&(old_weight<1-1e-9),parent_retired=r['active']&(old_weight>=1-1e-9),
        parent_modified=(old_weight>0)&r['active'])
    ownership=trace['ownership']
    np.savez_compressed(out/'ownership.npz',cell_keys=ownership['cell_keys'],cell_patch_id=ownership['cell_patch_id'],
        cell_size=ownership['cell_size'],grid_origin=ownership['grid_origin'],energy_history=ownership['energy_history'])
    write(out/'placements.json',dict(placements=placements,source_asset=str(PARENT),source_patch=chosen,
        core_gaussians=len(ids),halo_used_for_source_qualification_only=True,parameter_transfer='rigid μ/RΣRᵀ, source sigma/RGB/opacity and internal3D layout exact; robust existing-road frame'))
    write(out/'receipt.json',dict(complete=True,patch_instances=len(patches),candidate_rows=int(sum(len(x['world_xyz']) for x in patches)),
        migrated_rows=len(merged['world_xyz']),duplicate_centers_removed=int((~keep).sum()),
        parent_retired=int((r['active']&(old_weight>=1-1e-9)).sum()),parent_transition=int(((old_weight>0)&(old_weight<1-1e-9)&r['active']).sum()),
        seam_width_m=width,core_RGB_exact=True,core_opacity_exact=True,RGB_spatial_variation_preserved=True,
        mean_mapping_scale=1.,covariance_max_reconstruction_error=float(max(t['covariance_reconstruction_max_abs'].max() for t in trace['patches'])),
        ownership_energy=ownership['energy_history'].tolist(),ownership_method=ownership['method'],new_diffusion_calls=0,
        optimizer_updates=0,GPU_calls=0,seconds=time.monotonic()-start,requires_actual_K_gate=True,transport_mode='locally_rigid_surface_registration'))
    print(json.dumps(read(out/'receipt.json')),flush=True)

if __name__=='__main__':main()
