import numpy as np
from audit_ideal_capacity_independent import weighted_f,range_floor,fw_numeric,level_status


def test_fixed_q_denominator_and_half_squared_objective():
    p=np.zeros((3,2,2));t=np.ones_like(p);q=np.array([[1.,0.],[1.,0.]])
    p[:,:,1]=123.
    assert weighted_f(p,t,q)==.5


def test_interval_floor_uses_both_fixed_b_and_box_upper():
    target=np.array([[[0.,1.]],[[0.,1.]],[[0.,1.]]]);q=np.ones((1,2))
    assert np.isclose(range_floor(target,np.full_like(target,.2),np.full_like(target,.8),q),.02)


def test_FW_proxy_is_not_strict_negative_certificate():
    d=fw_numeric(np.array([.2,.8]),np.array([1.,-1.]),.5)
    assert np.isclose(d['FW_gap_numeric_proxy'],.4) and np.isclose(d['lower_numeric_proxy'],.1)
    assert d['negative_certificate_allowed'] is False


def test_registered_error_levels_need_detail_and_ignore_stress_failure():
    assert not level_status(.012,2/255,True)['joint_feasible_witness']
    assert level_status(.012,4/255,True)['joint_feasible_witness']
    assert not level_status(.012,4/255,False)['joint_feasible_witness']
    assert level_status(.001,1/255,True,stress=True)['status']=='diagnostic_stress_only'
