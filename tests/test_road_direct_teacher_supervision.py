import numpy as np
import pytest
from drivingeditor_dynamic_edit.road_asset_supervision import direct_teacher_arrays


def inputs(alpha=.99):
    h=w=8;W=np.zeros((h,w),bool);W[:,4:]=True;A=~W
    observed=np.full((h,w,3),80,np.uint8);teacher=np.full((h,w,3),120,np.uint8)
    roles=dict(W_guard=W,H_B=W,K=np.zeros_like(W),physical_unknown=W.copy(),C_mat=A,visible_support=A)
    geometry=dict(surface_valid=np.ones((2,2),bool),retained_guard=np.zeros((2,2),bool))
    F=dict(target_rgb_srgb=np.full((3,2,2),.4,np.float32),legal_color=np.ones((2,2),bool))
    return observed,teacher,roles,A,geometry,F,np.full((2,2),alpha,np.float32)


def build(data,**kw):return direct_teacher_arrays(*data,output_wh=(2,2),**kw)


def test_real_A_and_physical_C_do_not_overlap_or_follow_decoder_context():
    data=inputs();z=build(data)
    assert not (z['A_mask']&z['C_mask']).any()
    assert z['A_mask'].sum()==2 and z['C_mask'].sum()==2
    assert np.all(z['target_A_srgb']==np.float32(80/255))
    assert np.all(z['target_C_srgb']==np.float32(120/255))
    assert z['q_A'].sum()==2 and z['q_C'].sum()==2


def test_low_initial_alpha_keeps_live_coverage_and_excludes_outside_regularizer():
    z=build(inputs(alpha=.5))
    assert z['A_mask'].sum()==z['C_mask'].sum()==0
    assert z['q_A'].sum()==z['q_C'].sum()==0
    assert z['coverage_A'].sum()==z['coverage_C'].sum()==2
    assert z['q_A_coverage'].sum()==z['q_C_coverage'].sum()==2
    assert not (z['coverage_mask']&z['outside']).any()
    assert not z['outside'].any()


def test_shadow_appearance_effect_is_not_main_physical_hole():
    data=inputs();data[2]['physical_unknown'][:]=False
    z=build(data)
    assert not z['C_mask'].any() and not z['coverage_C'].any()
    assert z['C_effect'].sum()==2


def test_F_unknown_never_becomes_zero_control_teacher():
    data=inputs();data[5]['legal_color'][:,1]=False;data[5]['target_rgb_srgb'][:,:,1]=0
    z=build(data)
    assert not z['C_mask'].any()
    assert z['coverage_C'].sum()==2 and z['q_C_coverage'].sum()==2
    assert not (z['C_teacher_only']&z['outside']).any()
    assert z['C_teacher_only'].sum()==2
    assert z['C_teacher_hole'].sum()==2


def test_native_full_4x4_footprint_is_required():
    data=inputs();data[2]['physical_unknown'][0,4]=False
    z=build(data)
    assert z['q_C_native_footprint'][0,1]==pytest.approx(15/16)
    assert z['C_full_hole'][0,1] and not z['C_mask'][0,1]


def test_retained_K_and_projection_guard_are_excluded():
    data=inputs();data[4]['retained_guard'][0,1]=True
    z=build(data)
    assert not z['C_mask'][0,1] and not z['coverage_C'][0,1]
    data=inputs();data[2]['K'][0,4]=True
    with pytest.raises(ValueError,match='冲突'):build(data)


def test_diagnostic_frames_cannot_train_or_set_confidence_weights():
    z=build(inputs(),training_allowed=False)
    assert not bool(z['training_allowed'])
    assert not z['A_mask'].any() and not z['C_mask'].any() and not z['coverage_mask'].any()
    assert not z['q_A'].any() and not z['q_C'].any() and not z['q_C_coverage'].any()


def test_wrong_role_type_and_noninteger_resampling_are_rejected():
    data=inputs();data[2]['W_guard']=data[2]['W_guard'].astype(np.float32)
    with pytest.raises(ValueError,match='bool'):build(data)
    data=list(inputs());data[-1]=np.ones((2,3),np.float32)
    with pytest.raises(ValueError,match='整倍数'):direct_teacher_arrays(*data,output_wh=(3,2))
