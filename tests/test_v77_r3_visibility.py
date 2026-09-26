import pathlib,sys,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts/worldsim_v77'))
from r3_visibility import segment_box_occlusion,ground_fit,close_box_to_ground,core_reference_gate
def test_ray_box_blocks_only_foreground_segment():
    pose=np.eye(4);size=np.array([2.,2.,2.]);q=np.array([[3,0,0],[0,0,0],[-2,0,0],[3,3,0]])
    assert segment_box_occlusion(np.array([-3.,0,0]),q,pose,size).tolist()==[True,True,False,False]
def test_parallel_ray_outside_slab_is_clear():
    assert not segment_box_occlusion(np.array([-3.,2,0]),np.array([[3.,2,0]]),np.eye(4),[2,2,2])[0]
def test_query_on_front_face_is_not_occluded_before_endpoint():
    assert not segment_box_occlusion(np.array([-3.,0,0]),np.array([[-1.,0,0]]),np.eye(4),[2,2,2])[0]
def test_ground_plane_ignores_above_ground_outliers():
    rng=np.random.default_rng(1);xy=rng.uniform(-4,4,(1000,2));z=.03*xy[:,0]-.02*xy[:,1]-.8+rng.normal(0,.005,1000);z[:200]+=.3
    coef,report=ground_fit(np.c_[xy,z],-.8)
    assert np.allclose(coef,[.03,-.02,-.8],atol=.005) and report['inlier_points']>=750

def test_ground_closure_blocks_false_background_and_preserves_box_top():
    p=np.eye(4);p[2,3]=1.5;s=np.array([4.,2.,2.])
    camera=np.array([-8.,0.,1.]);query=np.array([[0.,0.,0.]])
    assert not segment_box_occlusion(camera,query,p,s)[0]
    p2,s2=close_box_to_ground(p,s,np.eye(4),[0.,0.,0.])
    assert segment_box_occlusion(camera,query,p2,s2)[0]
    assert np.isclose(p2[2,3]+s2[2]/2,p[2,3]+s[2]/2)
    assert np.isclose(p2[2,3]-s2[2]/2,-.1)

def test_disjoint_visual_and_lidar_evidence_does_not_admit_copy():
    gate=core_reference_gate([True,True],[1,0],[False,True])
    assert gate['joint_candidate_queries']==0 and not gate['copy_as_observed_background_admitted']
    assert gate['status']=='no_jointly_supported_core_reference'
