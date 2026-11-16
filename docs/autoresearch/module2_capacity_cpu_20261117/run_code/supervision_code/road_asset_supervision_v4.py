"""CPU-only candidate routing: real A, physical hole C and target-shadow S.

Frozen instance-associated probabilities provide provisional routing evidence,
not shadow alpha or hidden-road GT. No training, renderer or model is imported.
"""
import hashlib
import numpy as np


def pool_native(value, output_hw, reduction):
    a=np.asarray(value);h,w=output_hw
    if a.ndim!=2 or a.shape[0]%h or a.shape[1]%w:
        raise ValueError('native and output grids must have an exact integer relation')
    blocks=a.reshape(h,a.shape[0]//h,w,a.shape[1]//w)
    if reduction=='any':return blocks.any((1,3))
    if reduction=='all':return blocks.all((1,3))
    if reduction=='min':return blocks.min((1,3))
    if reduction=='max':return blocks.max((1,3))
    if reduction=='mean':return blocks.mean((1,3))
    raise ValueError(reduction)


def rgb_identity(a):
    a=np.ascontiguousarray(a)
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[-1]!=3:raise ValueError('native RGB must be uint8 HWC')
    return hashlib.sha256(b'rgb.uint8.v1\0'+np.asarray(a.shape,'<i8').tobytes()+a.tobytes()).hexdigest()


def mask_identity(a):
    a=np.asarray(a)
    if a.dtype!=bool or a.ndim!=2:raise ValueError('native target must be bool HW')
    return hashlib.sha256(b'mask.bool.v1\0'+np.asarray(a.shape,'<i8').tobytes()+np.packbits(a,bitorder='little').tobytes()).hexdigest()


def verify_shadow_identity(metadata, observation, target):
    if metadata.get('status')!='ready' or metadata.get('fixture_only',False):raise ValueError('shadow prediction is unavailable or a fixture')
    if metadata.get('observation_sha256')!=rgb_identity(observation):raise ValueError('shadow belongs to another observation')
    if metadata.get('target_sha256')!=mask_identity(target):raise ValueError('shadow belongs to another target instance')


def other_shadow_guard(packet, metadata, probability, low):
    """Select the explicit pair index, then verify the exact selected soft map.

    Other pairs are conservatively protected at the inherited low threshold;
    their membership is predicted and has no real-scene recall certificate.
    """
    maps=np.asarray(packet['soft_masks']);classes=np.asarray(packet['classes']);ids=np.asarray(packet['association_ids'])
    diag=metadata['diagnostics'];pairs=diag['official_instance_pairs'];index=diag['selected_pair_index']
    if type(index)is not int or not 0<=index<len(pairs):raise ValueError('invalid selected pair index')
    selected=pairs[index];member=int(selected['shadow_member'])
    if maps.ndim!=3 or maps.shape[1:]!=probability.shape or classes.shape!=(len(maps),) or ids.shape!=classes.shape:
        raise ValueError('invalid paired-instance packet')
    if not np.array_equal(maps[member],probability):raise ValueError('selected pair is not the frozen target shadow field')
    guard=np.zeros(probability.shape,bool);records=[]
    for i,pair in enumerate(pairs):
        obj=int(pair['matched_target_member']);shadow=int(pair['shadow_member']);association=int(pair['association_id'])
        if ids[obj]!=association or ids[shadow]!=association or classes[obj]!=pair['target_class_id'] or classes[shadow]!=pair['shadow_class_id']:
            raise ValueError('paired-instance identity mismatch')
        if i!=index:guard |= maps[shadow]>=low
        records.append(dict(association_id=association,object_member=obj,shadow_member=shadow,target_pair=i==index))
    return guard,dict(selected_pair_index=index,selected_association_id=int(selected['association_id']),selected_shadow_member=member,pairs=records)


def candidate_arrays(old, roles, probability, uncertainty, valid, metadata, other_shadow, policy, teacher_write, shadow_weight_mode=None):
    shape=np.asarray(probability).shape;outshape=old['K'].shape
    def native(name):
        a=np.asarray(roles[name])
        if a.dtype!=bool or a.shape!=shape:raise ValueError(name+': expected native bool mask')
        return a
    for value in [probability,uncertainty]:
        if np.asarray(value).shape!=shape or not np.isfinite(value).all() or np.min(value)<0 or np.max(value)>1:
            raise ValueError('invalid shadow soft map')
    if valid.dtype!=bool or valid.shape!=shape or other_shadow.dtype!=bool or other_shadow.shape!=shape:raise ValueError('invalid shadow validity/other-shadow grid')
    if teacher_write.dtype!=bool or teacher_write.shape!=shape:raise ValueError('original raw teacher write domain must be native bool')
    if shadow_weight_mode not in [None,'binary','model_score']:raise ValueError('unknown provisional shadow weighting')
    association=float(metadata['diagnostics']['association_uncertainty'])
    if not np.isfinite(association) or not 0<=association<=1:raise ValueError('invalid association uncertainty')
    high=valid&(probability>=policy['high'])&native('road_observed')&~native('foreground_possible')&~native('K')&~native('physical_unknown')
    if not np.array_equal(high,native('shadow_high')):raise ValueError('frozen high-shadow routing disagrees with its policy/field')
    # W_output is the original explicit write authorization, never its larger H_input.
    interior=(high&native('W_output')&~native('shadow_uncertain')&~native('shadow_boundary_guard')
        &(uncertainty<=policy['max_write_uncertainty']))
    if association>policy['max_write_uncertainty']:interior[:]=False
    safe=interior&~other_shadow&teacher_write
    possible=pool_native(native('shadow_possible'),outshape,'any')
    uncertain=pool_native(native('shadow_uncertain')|native('shadow_boundary_guard'),outshape,'any')
    other=pool_native(other_shadow,outshape,'any')
    full=pool_native(safe,outshape,'all')
    geom=np.asarray(old['geometry_valid'],bool);retained=np.asarray(old['retained_projection_guard'],bool);K=old['K'].copy()
    available=geom&~retained&~K
    S=full&available
    stable=np.asarray(old['fixed_high_alpha'],bool)
    # A remains real appearance, including non-target shadows; no A pixel may
    # simultaneously demand removal of a predicted target shadow/boundary.
    A=np.asarray(old['A_visible'],bool)&~(possible|uncertain)
    # Conflicting non-target shadow predictions are preserved, not erased with C.
    C=np.asarray(old['C_teacher_hole'],bool)&~other
    Aphoto=A&stable;Cphoto=C&stable;Sphoto=S&stable
    score=pool_native(probability,outshape,'min')*(1-pool_native(uncertainty,outshape,'max'))*(1-association)
    score=np.clip(score,0,1).astype(np.float32)
    qscore=np.zeros(outshape,np.float32) if shadow_weight_mode is None else np.ones(outshape,np.float32) if shadow_weight_mode=='binary' else score
    result={k:np.asarray(v).copy() for k,v in old.items()}
    for name in ['A_mask','C_mask','q_A','q_C','A_visible','C_teacher_hole','coverage_A','coverage_C','coverage_mask','q_A_coverage','q_C_coverage','outside']:
        result['legacy_r3_'+name]=old[name].copy()
    result.update(A_mask=Aphoto,C_mask=Cphoto,q_A=Aphoto.astype(np.float32),q_C=Cphoto.astype(np.float32),
        A_visible=A,C_teacher_hole=C,coverage_A=A,coverage_C=C,coverage_S=S,coverage_mask=A|C|S,
        q_A_coverage=A.astype(np.float32),q_C_coverage=C.astype(np.float32),q_S_coverage=qscore*S,
        outside=~(A|C|S|K),C_hole=C,C_hole_photo=Cphoto,q_C_hole=Cphoto.astype(np.float32),
        C_teacher_only=C&~old['F_control_valid'],C_teacher_only_photo=Cphoto&~old['F_control_valid'],
        C_common_Fcontrol=old['C_mask'].copy(),S_target_shadow=S,S_target_shadow_photo=Sphoto,
        S_target_shadow_possible=possible,S_target_shadow_boundary_unknown=uncertain,
        S_other_shadow_guard=other,S_target_shadow_full_native_footprint=full,
        S_no_geometry=full&~geom,S_projection_guarded=full&(retained|K),
        S_target_other_conflict=pool_native(interior&other_shadow,outshape,'any'),
        S_teacher_missing=pool_native(interior&~teacher_write,outshape,'any'),
        C_other_shadow_conflict=old['C_teacher_hole']&other,
        S_model_score=score,q_S=qscore*Sphoto,q_S_binary_eligible=Sphoto.astype(np.float32),
        S_weight_mode=np.asarray('pending_review' if shadow_weight_mode is None else shadow_weight_mode),
        target_S_srgb=old['target_C_srgb'].copy(),source_candidate_only=np.asarray(True),
        photo_reference=np.asarray('frozen r3 initial S2 alpha; new GPU initialization must rebind/verify before formal'))
    for x,y in [(A,C),(A,S),(C,S),(A|C|S,K),(result['outside'],A|C|S|K)]:
        if (x&y).any():raise ValueError('candidate domains conflict')
    if (result['q_S'][~Sphoto]!=0).any() or (result['q_S'][uncertain]!=0).any():raise ValueError('shadow boundary received loss weight')
    audit=dict(native_target_shadow_possible=int(native('shadow_possible').sum()),
        native_target_shadow_interior=int(interior.sum()),native_target_other_shadow_overlap=int((interior&other_shadow).sum()),
        native_safe_shadow_interior=int(safe.sum()),target_shadow_possible_pixels=int(possible.sum()),
        native_shadow_outside_original_teacher_write=int((interior&~teacher_write).sum()),
        target_shadow_full_footprint_pixels=int(full.sum()),S_target_shadow_pixels=int(S.sum()),S_photo_pixels=int(Sphoto.sum()),
        S_no_geometry=int((full&~geom).sum()),S_projection_guarded=int((full&(retained|K)).sum()),
        S_initial_thin_excluded=int((S&~stable).sum()),S_q_sum=float(result['q_S'].sum()),
        shadow_boundary_unknown_pixels=int(uncertain.sum()),other_shadow_predicted_guard_pixels=int(other.sum()),
        C_other_shadow_conflict_pixels=int((old['C_teacher_hole']&other).sum()),
        A_kept_real_non_target_shadow_pixels=int((A&other).sum()),A_removed_target_shadow_conflict=int((old['A_visible']&~A).sum()),
        A_photo_pixels=int(Aphoto.sum()),C_photo_pixels=int(Cphoto.sum()),C_geometry_pixels=int(C.sum()),
        C_color_added_by_removing_F0_requirement=int((Cphoto&~old['C_mask']).sum()),outside_pixels=int(result['outside'].sum()),
        original_K_preserved_exactly=np.array_equal(K,old['K']),association_uncertainty=association,
        shadow_weight_mode='pending_review' if shadow_weight_mode is None else shadow_weight_mode,
        shadow_weights_calibrated=False,shadow_boundary_as_GT=False,formal_updates=0,GPU_operations=0)
    return result,audit


def previous_mask_audit(old,roles):
    shape=old['K'].shape;rows={}
    for name in ['shadow_possible','shadow_high','shadow_uncertain','shadow_boundary_guard','material_contaminated_effect','remaining_effect']:
        anymask=pool_native(roles[name],shape,'any');full=pool_native(roles[name],shape,'all')
        rows[name]=dict(native_pixels=int(roles[name].sum()),renderer_any_pixels=int(anymask.sum()),renderer_all_pixels=int(full.sum()),
            **{k:int((anymask&old[k]).sum()) for k in ['A_mask','C_mask','coverage_mask','outside','K','geometry_valid','fixed_high_alpha']})
    rows['C_effect']=dict(renderer_any_pixels=int(old['C_effect'].sum()),
        **{k:int((old['C_effect']&old[k]).sum()) for k in ['A_mask','C_mask','coverage_mask','outside','K','geometry_valid','fixed_high_alpha']})
    return rows


def scope_audit(scopes,road,prior):
    active=np.flatnonzero(road['active'])
    if not np.array_equal(active,scopes['active_world_rows']):raise ValueError('scope world rows disagree with original asset')
    p=(prior['surface_supported']&prior['legal_write']&~prior['guarded_world_rows']&(road['minimum_model_depth']<=0))[active]&~scopes['conservative_K_guard']
    A=scopes['actual_A_mass']>0;C=scopes['actual_C_mass']>0;coverage=scopes['actual_coverage_mass']>0
    expected=dict(physical_legal=p,color=p&(A|(C&~prior['known'][active])),opacity=p&coverage,
        scale=p&coverage&(scopes['scale_upper_factor']>1.000001),A_seen=A,C_seen=C)
    checks={key:bool(np.array_equal(scopes[key],value)) for key,value in expected.items()}
    return dict(checks=checks,all_checks_passed=all(checks.values()),
        counts={key:int(scopes[key].sum()) for key in ['color','opacity','scale','A_seen','C_seen','physical_legal','conservative_K_guard']},
        formula='actual road-only alphaT contribution mass>0 plus physical legal support and frozen world/tile-expanded retained guards; C-only known rows frozen',
        new_shadow_actual_contribution_known=False,new_shadow_GPU_eligibility_validated=False)


def world_domain_audit(data,mask,world_xyz,camera):
    y,x=np.where(mask);world=data['ground_world_xyz'][y,x];tri=data['triangle_world_rows'][y,x];weight=data['triangle_weights'][y,x]
    valid=np.isfinite(world).all(1)&np.all((tri>=0)&(tri<len(world_xyz)),1)&np.isfinite(weight).all(1)&np.all(weight>=0,1)&(np.abs(weight.sum(1)-1)<1e-6)
    error=np.full(len(y),np.nan);reprojection=error.copy()
    if valid.any():
        error[valid]=np.linalg.norm((world_xyz[tri[valid]]*weight[valid,:,None]).sum(1)-world[valid],axis=1)
        intrinsic,extrinsic=camera;pc=world[valid]@extrinsic[:3,:3].T+extrinsic[:3,3]
        uv=pc@intrinsic.T;uv=uv[:,:2]/uv[:,2:3]-.5
        reprojection[valid]=np.linalg.norm(uv-np.c_[x[valid],y[valid]],axis=1)
    return dict(pixels=len(y),valid_triangle_world_pixels=int(valid.sum()),invalid_triangle_world_pixels=int((~valid).sum()),
        barycentric_world_error_max_m=float(np.nanmax(error)) if valid.any() else None,
        original_camera_reprojection_error_max_pixels=float(np.nanmax(reprojection)) if valid.any() else None,
        mapped_unique_world_rows=int(np.unique(tri[valid][weight[valid]>1e-7]).size) if valid.any() else 0,
        mapped_rows_are_not_renderer_contributors=True)


def crossview(data,cameras,source,target,source_role,target_role,max_world_error=.05):
    roles={'A':('A_mask','target_A_srgb'),'C':('C_hole_photo','target_C_srgb'),'S':('S_target_shadow_photo','target_S_srgb')}
    sm,key_s=roles[source_role];tm,key_t=roles[target_role]
    s,t=data[source],data[target];mask_s=s[sm];mask_t=t[tm];height,width=mask_s.shape
    y,x=np.where(mask_s);world=s['ground_world_xyz'][y,x];intrinsic,camera=cameras[target]
    pc=world@camera[:3,:3].T+camera[:3,3];projected=pc@intrinsic.T
    with np.errstate(invalid='ignore',divide='ignore'):uv=projected[:,:2]/projected[:,2:3]-.5
    finite=np.isfinite(uv).all(1)&np.isfinite(world).all(1)&(pc[:,2]>.2)
    lo=np.floor(np.nan_to_num(uv,nan=-1e9,posinf=1e9,neginf=-1e9)).astype(np.int64)
    inside=finite&(lo[:,0]>=0)&(lo[:,1]>=0)&(lo[:,0]+1<width)&(lo[:,1]+1<height)
    chosen=np.flatnonzero(inside);low=lo[chosen];fx,fy=(uv[chosen]-low).T
    weights=np.stack([(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy],1)
    xx=low[:,0,None]+[0,1,0,1];yy=low[:,1,None]+[0,0,1,1]
    role_valid=np.all(mask_t[yy,xx],1)
    otherworld=(t['ground_world_xyz'][yy,xx]*weights[:,:,None]).sum(1)
    err=np.linalg.norm(otherworld-world[chosen],axis=1)
    reliable=role_valid&np.isfinite(err)&(err<=max_world_error)
    selected=chosen[reliable];weights=weights[reliable];xx=xx[reliable];yy=yy[reliable];err=err[reliable]
    rgb_t=(t[key_t].transpose(1,2,0)[yy,xx]*weights[:,:,None]).sum(1)
    rgb_s=s[key_s][:,y[selected],x[selected]].T;delta=rgb_s-rgb_t
    row=dict(source_raw=source,target_raw=target,source_role=source_role,target_role=target_role,
        source_pixels=int(mask_s.sum()),project_inside_pixels=int(inside.sum()),target_four_neighbor_role_pixels=int(role_valid.sum()),
        reliable_same_world_pixels=len(selected),missing_correspondence_pixels=int(mask_s.sum())-len(selected),
        sRGB_MAE=float(np.abs(delta).mean()) if len(selected) else None,
        sRGB_signed_mean=delta.mean(0).tolist() if len(selected) else None,
        absolute_channel_mean_quantiles=np.quantile(np.abs(delta).mean(1),[0,.5,.9,.95,1]).tolist() if len(selected) else None,
        max_channel_gt_one_byte_pixels=int((np.abs(delta).max(1)>1/255).sum()) if len(selected) else 0)
    arrays=dict(source_pixel_yx=np.c_[y[selected],x[selected]],world_xyz=world[selected],target_uv_index=uv[selected],
        source_srgb=rgb_s,target_srgb_bilinear=rgb_t,delta_srgb=delta,world_interpolation_error_m=err)
    return row,arrays
