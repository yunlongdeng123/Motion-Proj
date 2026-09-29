"""几何边界与条件合同的有意义回归。"""
import numpy as np
from geometry_factory import projection,footprint,ground_orientation
from build_pairs import runlen

def test_near_plane_intersection_not_silently_lost():
    frame={'_w2c':np.eye(4),'intrinsics_1024':[[800,0,512],[0,800,288],[0,0,1]]}
    a={'translation':[0,0,.7],'rotation':[1,0,0,0],'size':[2,4,2]}
    assert projection(a,frame) is None
    p=projection(a,frame,clip_near=True)
    assert p is not None and np.isfinite(p['box']).all() and p['near_depth']>=.5-1e-9

def test_footprint_clearance_uses_vehicle_dimensions():
    a={'translation':[0,0,1],'rotation':[1,0,0,0],'size':[2,4,2]}
    b=a|{'translation':[4.2,0,1]}
    assert abs(footprint(a).distance(footprint(b))-.2)<1e-6
    assert footprint(a).intersects(footprint(a|{'translation':[3.5,0,1]}))

def test_slope_alignment_and_run_length():
    r=ground_orientation(.7,[.05,-.08,.2]);assert np.allclose(r.T@r,np.eye(3)) and abs(np.linalg.det(r)-1)<1e-6
    assert np.allclose(r[:,2],np.array([-.05,.08,1])/np.linalg.norm([-.05,.08,1]))
    assert runlen([True]*6+[False]+[True]*9)==9
