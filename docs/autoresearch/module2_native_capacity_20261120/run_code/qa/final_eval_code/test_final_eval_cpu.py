"""CPU-only negative controls for endpoint interpretation and packet identity."""
from pathlib import Path
import numpy as np
import pytest
from eval_native_endpoints import validate_packet,validate_endpoint,signed_sobel_error,visibility_counts,full_source_summary,metric_module,evaluate_region,fixed_edge_scalars
from ideal_endpoint_metrics import paint_metrics

def packet(shape=(15,23)):
    return dict(road_rgb=np.zeros((3,*shape),np.float32),joint_rgb=np.zeros((3,*shape),np.float32),
        road_alpha=np.ones(shape,np.float32),joint_alpha=np.ones(shape,np.float32),K=np.zeros(shape,bool))

def test_packet_is_actual_chw_and_exact_native_size_not_resized():
    p=packet();validate_packet(p,(15,23))
    with pytest.raises(ValueError):validate_packet(p)
    p['road_rgb']=p['road_rgb'].transpose(1,2,0)
    with pytest.raises(ValueError):validate_packet(p,(15,23))

def test_signed_sobel_masks_outside_context_before_kernel():
    rng=np.random.default_rng(101);p=rng.random((3,20,25));t=p.copy();mask=np.zeros((20,25),bool);mask[4:17,5:20]=True
    t[:,~mask]=1e8
    assert signed_sobel_error(p,t,mask,mask)['RMSE']==0
    tiny=np.zeros_like(mask);tiny[5,5]=True
    assert signed_sobel_error(p,t,tiny,tiny)['status']=='N/A_no_strict_core'

def test_static_visibility_is_separate_from_road_alpha_and_missing_is_na():
    m=np.ones((3,4),bool);a=np.ones_like(m,float)
    assert visibility_counts(m,a,None)['status'].startswith('N/A')
    v=dict(joint_road_alphaT=np.zeros_like(a),static_alphaT=np.ones_like(a)*.7)
    z=visibility_counts(m,a,v)
    assert z['high_road_alpha_but_joint_road_below05']==12 and z['of_these_static_alphaT_ge50']==12
    assert not z['causal_static_removal_claim']

def test_zero_alpha_does_not_make_material_recovery_pass():
    p=packet();p['road_alpha'][:]=0;t=np.full_like(p['road_rgb'],.4);m=np.ones_like(p['K'])
    z=full_source_summary(p,t,m,m,m,None)
    assert z['legacy_premult_P_vs_alphaT']['MAE']==0
    assert z['opaque_road_P_vs_teacher']['MAE']==pytest.approx(.4)
    assert z['gap_below_01']==m.size

def test_known_ideal_paint_has_unit_contrast_identity_and_width_zero_error():
    shape=(28,35);f=np.zeros(shape);f[4:25,13:19]=1.;a=np.full((3,*shape),.3);w=np.ones(shape);b=np.full_like(a,.07)
    P=b+w[None]*(a+.6*f[None]);m=np.ones(shape,bool);dash=np.zeros(shape,int)
    z=paint_metrics(P,readonly=b,weight=w,asphalt=a,paint_rgb=[.9]*3,paint_fraction=f,qualified_mask=m,dash_ids=dash)
    assert z['contrast_retention']==pytest.approx(1.) and z['edge_F1_1px']==1.
    assert z['line_center_error_max_px']==pytest.approx(0.,abs=1e-12)
    assert z['halfheight_width']['absolute_error_px']['max']==0 and z['old_combined_detail_gate'] is True

def test_weak_eligible_weight_cannot_masquerade_as_ideal_capability():
    p=packet();m=np.ones(p['K'].shape,bool)
    with pytest.raises(ValueError):paint_metrics(p['road_rgb'],readonly=p['road_rgb'],weight=np.full(m.shape,.2),asphalt=p['road_rgb'],paint_rgb=[.9]*3,paint_fraction=np.zeros(m.shape),qualified_mask=m)

def frozen_metric_path():
    local=Path(__file__).resolve().parents[3]/'road_quality_bench_20261120/metrics/road_detail_metrics.py'
    remote=Path(__file__).resolve().parents[3]/'quality_bench_20261120_r1/metrics/road_detail_metrics.py'
    return local if local.exists()else remote

def test_frozen_metric_zero_core_is_na_and_held_small_block_is_not_blind():
    metric=metric_module(frozen_metric_path());p=packet((18,24));t=np.full_like(p['road_rgb'],.3);p['road_rgb']=t.copy()
    score=np.ones((18,24),bool);split=dict(fixed_alpha=p['road_alpha'],A_held=score,A_held_context=score)
    src=dict(F0={'held':{'prior_exposure':'preexposed'}},held_blocks=1)
    r=evaluate_region(metric,p,t,split,'A','held',src)
    assert r['RGB_stable']['MAE']==0 and r['signed_Sobel_error']['RMSE']==0
    assert r['scales'][-1]['core']['pixels']==0 and r['scales'][-1]['core']['HF_cosine'] is None
    assert r['interpretation_status'].startswith('N/A_') and r['prior_exposure']['blind_test'] is False
    s=fixed_edge_scalars(r);assert s['edge_reference_n']==0 and s['edge_F1'] is None

def test_edge_f1_empty_prediction_retains_failure_and_empty_reference_is_na():
    def r(nref,npred,p,rec):return dict(edges=dict(context_core_pixels=10,threshold_curves=[dict(strength_threshold=2/255,reference_edges=nref,predicted_edges=npred,curves=[dict(tolerance_pixels=1.,precision=p,recall=rec)])]))
    assert fixed_edge_scalars(r(5,0,None,0.))['edge_F1']==0
    assert fixed_edge_scalars(r(0,5,0.,None))['edge_F1'] is None

def test_endpoint_metadata_cannot_choose_a_different_shared_step_or_extend_oracle():
    protocol=dict(arms=dict(G_RGB=dict(type='shared_field',steps=1200),G_ORACLE=dict(type='box',max_accepted_steps=2000)))
    r=dict(completed=True,arm='G_RGB',extra=dict(logical_steps=1200))
    assert validate_endpoint(r,'G_RGB',protocol)['steps']==1200
    r['extra']['logical_steps']=1100
    with pytest.raises(ValueError):validate_endpoint(r,'G_RGB',protocol)
    r=dict(completed=True,arm='G_ORACLE',extra=dict(accepted_steps=4000,continued=True))
    assert validate_endpoint(r,'G_ORACLE',protocol)['steps']==4000
    r['extra']['accepted_steps']=4001
    with pytest.raises(ValueError):validate_endpoint(r,'G_ORACLE',protocol)
