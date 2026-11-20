import json
import numpy as np
import pytest
from scipy import ndimage as ndi

from road_detail_metrics import (DetailConfig,ResolutionSpec,TileLayout,evaluate_detail,
                                 assert_comparable,rank_cases)


def run(pred,teacher,context=None,score=None,**kwargs):
    h,w=teacher.shape[:2]
    return evaluate_detail(pred,teacher,context_mask=np.ones((h,w),bool)if context is None else context,
        score_mask=np.ones((h,w),bool)if score is None else score,
        resolution=ResolutionSpec(w,h,'synthetic_input_pixels'),tile_layout=TileLayout(32,32),
        teacher_role='synthetic_known',**kwargs)


def line_image(size=128):
    image=np.full((size,size,3),.3)
    image[:,size//2-1]=.8
    return image


def test_identity_is_exact_and_all_fixed_tiles_retained():
    rng=np.random.default_rng(101);target=rng.uniform(.2,.8,(96,96,3))
    result=run(target,target)
    for scale in result['scales']:
        assert scale['core']['HF_MAE']==0
        assert scale['core']['HF_cosine']==pytest.approx(1)
        assert scale['tile_aggregate']['total_fixed_tiles']==9
        assert len(scale['tiles'])==9
    for threshold in result['edges']['threshold_curves']:
        assert threshold['curves'][0]['precision']==1
        assert threshold['curves'][0]['recall']==1
    json.dumps(result,allow_nan=False)
    assert 'pass'not in result and 'score'not in result


def test_erased_thin_line_low_color_MAE_but_missing_HF_and_edges():
    target=line_image();flat=np.full_like(target,.3)
    assert np.abs(flat-target).mean()<.005
    result=run(flat,target)
    core=result['scales'][0]['core']
    assert core['status']=='prediction_zero_reference_nonzero'
    assert core['HF_relative_RMSE']> .99
    assert core['HF_cosine']is None
    assert result['edges']['threshold_curves'][0]['curves'][0]['recall']==0
    aggregate=result['scales'][0]['tile_aggregate']
    assert aggregate['reference_energy_weighted_tile_HF_MAE']>aggregate['macro']['HF_MAE']['mean']
    assert aggregate['reference_detail_strata'][0]['eligible_tiles']>0
    assert any(tile['bounds_y0_y1_x0_x1'][2]in(32,64)for tile in aggregate['worst_tiles'])


def test_blurring_can_keep_low_MAE_while_harming_details():
    target=line_image();blur=ndi.gaussian_filter(target,sigma=(2.,2.,0),mode='nearest')
    assert abs(blur-target).mean()<.01
    result=run(blur,target)
    assert result['scales'][0]['core']['HF_relative_RMSE']>.5
    assert result['edges']['threshold_curves'][-1]['curves'][0]['recall']<.5


def test_equal_energy_wrong_phase_is_not_correspondence():
    yy,xx=np.indices((128,128))
    teacher=np.repeat((.4+.15*np.cos(2*np.pi*xx/16))[...,None],3,-1)
    pred=np.repeat((.4+.15*np.sin(2*np.pi*xx/16))[...,None],3,-1)
    result=run(pred,teacher)
    hf=result['scales'][0]['core']
    assert .8<hf['HF_energy_ratio']<1.2
    assert abs(hf['HF_cosine'])<.2
    assert hf['HF_relative_RMSE']>1.2
    assert result['edges']['threshold_curves'][0]['curves'][0]['recall']<.2


def test_teacher_flat_spurious_HF_is_reported_without_fake_cosine():
    yy,xx=np.indices((96,96));teacher=np.full((96,96,3),.4)
    pred=teacher+(.02*np.cos(2*np.pi*xx/6))[...,None]
    result=run(pred,teacher)
    core=result['scales'][0]['core']
    assert core['status']=='reference_zero_energy'
    assert core['HF_cosine']is None
    assert result['scales'][0]['tile_aggregate']['reference_energy_weighted_tile_HF_MAE']is None
    assert result['scales'][0]['teacher_flat_spurious'][0]['predicted_HF_RMS']>.005
    edges=result['edges']['threshold_curves'][0]
    assert edges['reference_edges']==0 and edges['predicted_edges']>0
    assert edges['curves'][0]['precision']==0


def test_outside_black_white_noise_or_nan_cannot_pollute_domain():
    rng=np.random.default_rng(101);target=rng.uniform(.2,.7,(96,96,3));pred=target+.02*np.sin(np.indices((96,96))[1])[...,None]
    context=np.zeros((96,96),bool);context[9:87,7:89]=1;context[40:48,40:48]=0
    score=context.copy()
    original=run(pred,target,context,score)
    for fill in (0.,1.,np.nan):
        p=pred.copy();t=target.copy();p[~context]=fill;t[~context]=fill
        changed=run(p,t,context,score)
        assert changed==original


def test_empty_strict_core_is_NA_and_boundaries_remain_counted():
    teacher=np.full((64,64,3),.4);context=np.zeros((64,64),bool);context[:,31]=1
    result=run(teacher,teacher,context,context)
    for scale in result['scales']:
        assert scale['core_pixels']==0 and scale['core']['HF_MAE']is None
        assert scale['boundary_pixels']==64
        assert scale['tile_aggregate']['scored_but_no_core_tiles']>0
    assert result['edges']['context_core_pixels']==0
    json.dumps(result,allow_nan=False)


def test_wrong_edge_direction_has_large_normal_error():
    target=np.full((96,96,3),.3);target[:,48:]=.8
    pred=np.full_like(target,.3);pred[48:]=.8
    result=run(pred,target)
    metric=result['edges']['threshold_curves'][0]['curves'][2]
    assert metric['predicted_normal_error_degrees']['count']>0
    assert metric['predicted_normal_error_degrees']['mean']>80


def test_explicit_structure_needs_verified_role_and_unknown_normal_is_NA():
    teacher=np.full((64,64,3),.4);pred=teacher.copy();pred[:,32:]=.8
    edges=np.zeros((64,64),bool);edges[:,32]=1
    with pytest.raises(ValueError):run(pred,teacher,edge_reference=dict(mask=edges,role='generated_lane_ground_truth'))
    result=run(pred,teacher,edge_reference=dict(mask=edges,role='human_verified_structure'))
    assert result['edges']['reference_role']=='human_verified_structure'
    assert result['edges']['threshold_curves'][0]['curves'][0]['orientation_curves'][0]['recall']is None
    assert not result['edges']['semantic_lane_geometry_truth_claim']


def test_varying_shared_alpha_has_no_material_HF_after_stable_evaluation():
    yy,xx=np.indices((96,96));teacher=np.full((96,96,3),.4)
    alpha=.975+.025*np.cos(2*np.pi*xx/8)
    raster=alpha[...,None]*teacher
    legacy=run(raster,alpha[...,None]*teacher,color_space='premult_srgb')
    material=run(raster/alpha[...,None],teacher,color_space='stable_alpha_unpremult_srgb',
                 input_conditioning=dict(common_road_alpha_min=.95,frozen_common_mask=True))
    assert legacy['scales'][0]['core']['HF_MAE']==0
    assert legacy['scales'][0]['core']['reference_HF_RMS']>.001
    assert material['scales'][0]['core']['HF_MAE']<1e-15
    assert material['scales'][0]['core']['reference_HF_RMS']<1e-15
    with pytest.raises(ValueError):assert_comparable([legacy,material])


def test_pixel_coordinate_spaces_cannot_be_pooled_and_case_rank_keeps_NA():
    teacher=line_image(96);good=run(teacher,teacher);bad=run(np.full_like(teacher,.3),teacher)
    incompatible=dict(good,resolution=dict(good['resolution'],pixel_space='native_teacher_pixels'))
    with pytest.raises(ValueError):assert_comparable([good,incompatible])
    ranking=rank_cases({'good':good,'erased':bad},metric_path=('scales',0,'core','HF_relative_RMSE'),higher_is_worse=True)
    assert ranking['cases'][0]['case_id']=='erased'and ranking['excluded_cases']==0
    assert good['tile_layout']['origin_x']==good['tile_layout']['origin_y']==0


def test_correct_one_pixel_match_crossing_fixed_tile_boundary_is_preserved():
    teacher=np.full((64,64,3),.3);teacher[:,32:]=.8
    pred=np.full_like(teacher,.3);pred[:,33:]=.8
    result=run(pred,teacher)
    row=result['edges']['threshold_curves'][0]
    assert row['curves'][1]['precision']==row['curves'][1]['recall']==1
    # Teacher NMS at x31 and prediction NMS at x32 lie in different fixed tiles.
    for tile in row['tiles']:
        metric=tile['metrics'];curve=metric['curves'][1]
        if metric['reference_edges']:assert curve['recall']==1
        if metric['predicted_edges']:assert curve['precision']==1


def test_inverted_paint_can_match_unsigned_edges_but_signed_HF_rejects():
    yy,xx=np.indices((96,96));wave=.15*np.cos(2*np.pi*xx/16)
    target=np.repeat((.4+wave)[...,None],3,-1);pred=np.repeat((.4-wave)[...,None],3,-1)
    result=run(pred,target)
    hf=result['scales'][0]['core']
    assert hf['HF_energy_ratio']==pytest.approx(1)
    assert hf['HF_cosine']<-.999 and hf['HF_relative_RMSE']>1.99
    spatial=result['edges']['threshold_curves'][0]['curves'][1]
    assert spatial['precision']>.9 and spatial['recall']>.9
