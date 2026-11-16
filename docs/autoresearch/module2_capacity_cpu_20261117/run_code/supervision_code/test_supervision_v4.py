"""Meaningful CPU routing tests for the new, non-formal candidate API."""
import copy
import numpy as np
import pytest
from road_asset_supervision_v4 import candidate_arrays,other_shadow_guard,verify_shadow_identity,rgb_identity,mask_identity,crossview


def fixture():
    h,w=32,48;shape=(8,12);p=np.full((h,w),.001,np.float32);u=np.full((h,w),.01,np.float32)
    physical=np.zeros((h,w),bool);physical[:8,:8]=True
    S=np.zeros((h,w),bool);S[16:24,20:28]=True;p[S]=.95
    A=np.zeros(shape,bool);A[3,:4]=True;C=np.zeros(shape,bool);C[:2,:2]=True
    yes=np.ones(shape,bool);no=np.zeros(shape,bool);native_no=np.zeros((h,w),bool)
    old=dict(K=no.copy(),retained_projection_guard=no.copy(),geometry_valid=yes.copy(),fixed_high_alpha=yes.copy(),
        A_mask=A.copy(),C_mask=C.copy(),q_A=A.astype(np.float32),q_C=C.astype(np.float32),A_visible=A.copy(),C_teacher_hole=C.copy(),
        coverage_A=A.copy(),coverage_C=C.copy(),coverage_mask=A|C,q_A_coverage=A.astype(np.float32),q_C_coverage=C.astype(np.float32),
        outside=~(A|C),F_control_valid=yes.copy(),target_C_srgb=np.ones((3,*shape),np.float32)*.3)
    roles=dict(road_observed=~physical,foreground_possible=native_no.copy(),K=native_no.copy(),physical_unknown=physical,
        shadow_high=S.copy(),shadow_possible=S.copy(),W_output=S|physical,shadow_uncertain=native_no.copy(),shadow_boundary_guard=native_no.copy())
    metadata=dict(diagnostics=dict(association_uncertainty=.2))
    policy=dict(high=.6,low=.1,max_write_uncertainty=.35)
    return old,roles,p,u,np.ones((h,w),bool),metadata,native_no.copy(),policy,np.ones((h,w),bool)


def test_shadow_routing_has_no_opposing_outside_or_A_and_preserves_sources():
    inputs=fixture();saved=copy.deepcopy(inputs)
    out,audit=candidate_arrays(*inputs,shadow_weight_mode='binary')
    S=out['S_target_shadow'];assert S.sum()==4
    assert not (S&out['outside']).any() and not (S&out['A_mask']).any()
    assert out['q_S'][S].min()==1 and audit['native_shadow_outside_original_teacher_write']==0
    for key,value in inputs[0].items():assert np.array_equal(value,saved[0][key])
    assert np.array_equal(out['K'],inputs[0]['K'])


def test_missing_teacher_and_uncertain_boundary_cannot_receive_shadow_loss():
    inputs=list(fixture());inputs[-1][16:20,20:24]=False
    inputs[1]['shadow_uncertain'][20:24,20:24]=True
    out,_=candidate_arrays(*inputs,shadow_weight_mode='binary')
    assert out['S_teacher_missing'][4,5] and out['q_S'][4,5]==0 and out['outside'][4,5]
    assert out['S_target_shadow_boundary_unknown'][5,5] and out['q_S'][5,5]==0 and out['outside'][5,5]
    assert out['S_target_shadow'].sum()==2


def test_other_vehicle_shadow_conflict_is_preserved_while_real_A_is_retained():
    inputs=list(fixture());inputs[6][16:20,20:24]=True
    inputs[6][0:4,0:4]=True;inputs[6][12:16,0:4]=True
    out,audit=candidate_arrays(*inputs,shadow_weight_mode='binary')
    assert out['S_target_other_conflict'][4,5] and out['q_S'][4,5]==0
    assert out['C_other_shadow_conflict'][0,0] and not out['C_mask'][0,0]
    assert out['A_mask'][3,0] and audit['A_kept_real_non_target_shadow_pixels']==1


def test_color_hole_no_longer_requires_F0_color_and_pending_weight_is_zero():
    inputs=list(fixture());inputs[0]['F_control_valid'][0,0]=False;inputs[0]['C_mask'][0,0]=False;inputs[0]['q_C'][0,0]=0
    out,_=candidate_arrays(*inputs)
    assert out['C_hole_photo'][0,0] and out['C_teacher_only_photo'][0,0]
    assert not out['q_S'].any() and out['S_weight_mode'].item()=='pending_review'


def test_pair_selection_uses_list_index_and_exact_map_identity():
    maps=np.zeros((4,2,2),np.float32);maps[1]=.2;maps[3]=.9
    packet=dict(soft_masks=maps,classes=np.array([0,1,0,1]),association_ids=np.array([1,1,2,2]))
    meta=dict(diagnostics=dict(selected_pair_index=1,official_instance_pairs=[
        dict(association_id=1,matched_target_member=0,shadow_member=1,target_class_id=0,shadow_class_id=1),
        dict(association_id=2,matched_target_member=2,shadow_member=3,target_class_id=0,shadow_class_id=1)]))
    other,receipt=other_shadow_guard(packet,meta,maps[3],.1)
    assert receipt['selected_shadow_member']==3 and other.all()
    with pytest.raises(ValueError,match='not the frozen'):other_shadow_guard(packet,meta,maps[1],.1)


def test_shadow_observation_and_target_are_bound_before_routing():
    rgb=np.zeros((4,5,3),np.uint8);target=np.zeros((4,5),bool);target[1,1]=True
    metadata=dict(status='ready',fixture_only=False,observation_sha256=rgb_identity(rgb),target_sha256=mask_identity(target))
    verify_shadow_identity(metadata,rgb,target)
    changed=target.copy();changed[2,2]=True
    with pytest.raises(ValueError,match='another target'):verify_shadow_identity(metadata,rgb,changed)


def test_same_world_conflict_report_does_not_modify_targets_or_masks():
    y,x=np.mgrid[:3,:3];world=np.stack([x+.5,y+.5,np.ones((3,3))],2)
    A=np.ones((3,3),bool);data={raw:dict(A_mask=A.copy(),target_A_srgb=np.full((3,3,3),value),ground_world_xyz=world.copy()) for raw,value in [(86,.2),(110,.3)]}
    camera=(np.eye(3),np.eye(4));row,arrays=crossview(data,{86:camera,110:camera},86,110,'A','A')
    assert row['reliable_same_world_pixels']==4 and np.isclose(row['sRGB_MAE'],.1)
    assert arrays['source_pixel_yx'].shape==(4,2) and data[86]['A_mask'].all()


def test_bitwise_geometry_identity_handles_frozen_invalid_nan_points():
    from verify_v4_candidate import bitwise_equal
    points=np.array([[np.nan,np.nan,np.nan],[1.,2.,3.]],np.float64)
    assert bitwise_equal(points,points.copy())
    changed=points.copy();changed[1,0]+=1e-12
    assert not bitwise_equal(points,changed)
