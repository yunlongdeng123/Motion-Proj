import numpy as np
import pytest
from validate_asset_data import pixel_gates

def sample():
    y=np.full((576,1024,3),88,np.uint8);x=y.copy();h=np.zeros((576,1024),bool);h[200:260,420:520]=True
    infl=np.zeros_like(h);infl[202:258,422:518]=True;x[infl]=119
    return y,x,infl,h,{}

def test_complete_actor_erasure():
    a=sample();assert pixel_gates(*a)['synthetic_RGB_leak']==0

def test_one_pixel_rgb_leak_is_rejected():
    a=sample();a[1][199,450]=180
    with pytest.raises(AssertionError,match='RGB'):pixel_gates(*a)

def test_feather_influence_outside_hole_is_rejected():
    a=sample();a[2][199,450]=True
    with pytest.raises(AssertionError,match='influence'):pixel_gates(*a)

def test_ego_region_is_rejected():
    a=sample();a[3][530,450]=True
    with pytest.raises(AssertionError,match='ego'):pixel_gates(*a)

def test_protected_visible_evidence_required():
    a=sample();b=a[3].copy();a[-1]['rear']=b
    with pytest.raises(AssertionError,match='exhausted'):pixel_gates(*a)
