import numpy as np
from readonly_floor_audit import interval_floor,floor_summary,mse_ratio,category_overlay

def test_pointwise_box_interval_is_a_relaxation_of_shared_positive_gaussian_colors():
    rng=np.random.default_rng(101);w=rng.uniform(0,.08,(8,5,7));W=w.sum(0);b=rng.uniform(0,.15,(3,5,7));ci=rng.random((8,3));P=b+np.einsum('nhw,nc->chw',w,ci)
    a=np.ones((5,7));T=rng.random(P.shape);m=np.ones(a.shape,bool);f=interval_floor(T,b,W,a);z=floor_summary(f,m)
    assert np.mean((P-T)**2)>=z['premult_MSE_lower_proxy']-1e-15
    assert np.min(P-b)>=0 and np.max(P-b-W[None])<=0

def test_zero_floor_ratio_is_na_and_fixed_context_overlay_partitions_full_source():
    shape=(3,5);b=np.zeros((3,*shape));W=np.ones(shape);T=np.full_like(b,.5);a=np.ones(shape);m=np.ones(shape,bool)
    z=floor_summary(interval_floor(T,b,W,a),m);r=mse_ratio(T,T,a,z,m)
    assert z['premult_MSE_lower_proxy']==0 and r['premult_actual_over_floor']is None
    seen=m.copy();seen[:,2:]=False;held=m.copy();held[:,:2]=False;held[:,-1]=False
    overlay,other=category_overlay(T,m,seen,held)
    assert other[:,-1].all()and not np.any(seen&held)and np.array_equal(seen|held|other,m)

def test_impossible_readonly_color_reports_floor_without_hf_or_quality_claim():
    b=np.full((3,2,2),.6);W=np.full((2,2),.1);a=np.ones((2,2));T=np.full_like(b,.2);m=np.ones((2,2),bool)
    z=floor_summary(interval_floor(T,b,W,a),m)
    assert np.isclose(z['premult_MAE_lower_proxy'],.4)and np.isclose(z['premult_MSE_lower_proxy'],.16)
    assert z['pixels_target_below_readonly']==4 and z['pixels_target_above_readonly_plus_We']==0
