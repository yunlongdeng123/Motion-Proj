import numpy as np
from audit_real_pair_independent import premult_mae,frequency_metrics,excess_gate,byte_equal


def test_premult_target_is_alpha_times_teacher_and_domain_normalized():
    alpha=np.full((2,2),.5);teacher=np.full((3,2,2),.4);pred=np.full((3,2,2),.1)
    mask=np.array([[True,False],[True,False]]);q=np.array([[1.,0.],[.25,0.]])
    assert np.isclose(premult_mae(pred,alpha,teacher,mask,q),.1)


def test_second_difference_axes_detect_horizontal_stripes():
    rgb=np.repeat(np.array([0.,1.,0.,1.,0.])[:,None],5,axis=1)[None].repeat(3,axis=0)
    result=frequency_metrics(rgb,np.zeros_like(rgb),np.ones((5,5),bool),np.ones((3,5),bool),np.ones((5,3),bool))
    assert result['E_vertical']==2 and result['E_horizontal']==0 and result['RGB_delta_energy']==.4


def test_positive_excess_half_gate_and_nonpositive_NA():
    assert excess_gate(.2,.1,.05)['status']=='NOT_APPLICABLE'
    assert excess_gate(.1,.3,.199999)['status']=='PASS'
    assert excess_gate(.1,.3,.200001)['status']=='FAIL'


def test_frozen_nan_geometry_is_compared_by_dtype_shape_and_bytes():
    a=np.array([[np.nan,0.],[1.,2.]],np.float64)
    assert byte_equal(a,a.copy()) and not byte_equal(a,a.astype(np.float32))
