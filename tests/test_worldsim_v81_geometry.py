import numpy as np
from motion_proj.worldsim_v81.geometry import transform,apply,project,reference_filter,depth_metrics,interaction_interval,remove_boxes

def test_camera_z_is_not_ego_ray_distance():
    T=transform({'rotation':[1,0,0,0],'translation':[2,0,0]});K=np.array([[100,0,50],[0,100,40],[0,0,1]])
    xyz=np.array([[3.,4.,10.]]);uv,z,c=project(apply(xyz,T),T,K)
    np.testing.assert_allclose(uv,[[80,80]]);np.testing.assert_allclose(z,[10])
    assert np.linalg.norm(apply(xyz,T)[0])!=z[0]

def test_reference_requires_disjoint_scan_support_and_rejects_edge():
    uv=np.array([[10,10],[10,10],[20,20],[20,20],[30,30],[30,30]])
    z=np.array([10,10.1,10,20,10,10.1]);src=np.array([0,1,0,1,0,0])
    np.testing.assert_array_equal(reference_filter(uv,z,src,100,100),[0,1])

def test_missing_predictions_keep_denominator():
    m=depth_metrics([10,np.nan,0],[10,10,10]);assert m['support']==3 and m['valid']==1 and m['coverage']==1/3
    assert depth_metrics([np.nan],[10])['absrel'] is None

def test_scene_bootstrap_does_not_count_pixels_as_logs():
    rows=[{'log':'one','cohort':c,'error':1} for c in ['C00','C10','C01','C11'] for _ in range(1000)]
    assert interaction_interval(rows)['status']=='INSUFFICIENT_MATCHED_LOGS'
    rows=[{'log':str(log),'cohort':c,'error':v} for log in range(4) for c,v in zip(['C00','C10','C01','C11'],[1,2,3,8])]
    r=interaction_interval(rows);assert r['interaction']==4 and r['ci95']==[4,4]

def test_nuscenes_box_width_length_axes():
    b={'rotation':[1,0,0,0],'translation':[0,0,0],'size':[2,6,2]}
    points=np.array([[2,0,0],[0,2,0],[4,0,0]])
    np.testing.assert_array_equal(remove_boxes(points,[b],0),points[1:])
