import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'gpu'))
import unittest
import numpy as np
from evidence import classify,conditional_depth,evaluate_arrays,Evidence,GATES

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(branch='ordinary',weight=.1)
        self.s=dict(road=dict(supported=True,z_m=10,uncertainty_m=.3,current_visible=True,confirmed_sources=3),vehicle=None)
        self.d=dict(interval_m=[5,6],screen_mahalanobis2=1)
    def test_free_space(self):self.assertEqual(classify(self.row,self.s,self.d)['label'],'geometry_conflict')
    def test_broad_ellipsoid_not_center_judgement(self):
        self.d['interval_m']=[5,11];self.assertEqual(classify(self.row,self.s,self.d)['label'],'unknown')
    def test_occluded_not_illegal(self):
        self.s['road']['current_visible']=False;self.assertEqual(classify(self.row,self.s,self.d)['label'],'unknown')
    def test_single_source_unknown(self):
        self.s['road']['confirmed_sources']=1;self.assertEqual(classify(self.row,self.s,self.d)['label'],'unknown')
    def test_legal_road_behind_car(self):
        self.s['vehicle']=dict(far_m=8);self.s['road']['current_visible']=False;self.d['interval_m']=[9.8,10.1]
        self.assertEqual(classify(self.row,self.s,self.d)['label'],'legal_occlusion')
    def test_non_target_not_auto_deleted(self):
        self.s['non_target_occluder']=True;self.assertEqual(classify(self.row,self.s,self.d)['label'],'unknown')
    def test_surface_color_copy(self):
        self.s=dict(vehicle=dict(far_m=8),target_mask=True,lidar_m=5.5,lidar_uncertainty_m=.6,color_L1=.01)
        self.assertEqual(classify(self.row,self.s,self.d)['reason'],'static_near_measured_vehicle_surface_and_vehicle_color')
    def test_color_alone_not_evidence(self):
        self.s=dict(vehicle=dict(far_m=8),target_mask=True,color_L1=0)
        self.assertEqual(classify(self.row,self.s,self.d)['label'],'unknown')
    def test_outside_ellipse_unknown(self):
        self.d['screen_mahalanobis2']=10;self.assertEqual(classify(self.row,self.s,self.d)['label'],'unknown')
    def test_no_contribution(self):
        self.row['weight']=0;self.assertEqual(classify(self.row,self.s,self.d)['label'],'unknown')
    def test_covariance_interval(self):
        row=dict(quaternion=[1,0,0,0],sigma_m=[.1,.1,.2],model_xyz=[0,0,10],camera_z_m=10,cov2=[100,0,100],uv=[240,160])
        v=conditional_depth(row,np.eye(4),np.array([[500,0,240],[0,500,160],[0,0,1]]),[240,160],1,False)
        self.assertAlmostEqual(v['mean_m'],10);self.assertGreater(v['interval_m'][1],10)
    def test_fixed_denominator_and_harm(self):
        b=dict(rgb=np.array([[.3]*3,[0]*3]),depth=np.array([[10.],[10.]]),alpha=np.ones((2,1)),road=np.zeros((2,1)))
        a={k:v.copy() for k,v in b.items()};a['rgb'][0]=.1;a['rgb'][1]=.2
        v=evaluate_arrays(b,a,np.zeros((2,3)),np.array([10,10]),np.ones(2,bool),np.array([False,True]),[1,2])
        self.assertAlmostEqual(v['valid']['delta_L_valid'],0);self.assertEqual(v['normal_control']['RGB_harmed_count'],1)
    def test_empty_unknown_not_zero(self):
        b={k:np.zeros((0,c)) for k,c in [('rgb',3),('depth',1),('alpha',1),('road',1)]}
        v=evaluate_arrays(b,b,np.empty((0,3)),np.empty(0),np.zeros(0,bool),np.zeros(0,bool),[])
        self.assertIsNone(v['valid']['delta_L_valid'])
    def fixture(self):
        e=Evidence.__new__(Evidence);e.u=dict(source_uv=np.full((49,1,2),10.5),base_weight=np.ones((49,1)),source_rgb=np.full((49,1,3),.1),unit_ids=np.array([7]),is_query=np.ones(1,bool))
        for g in GATES:e.u['gate_'+g]=np.ones((49,1),bool)
        f=dict(mask=np.zeros((320,480),bool),z=np.array([10.]))
        e.frame=lambda raw:f;e.measured=lambda f,uv:(np.array([10.]),np.array([True]),np.array([0.]))
        b=dict(rgb=np.full((3,320,480),.1),depth=np.full((1,320,480),10.),alpha=np.ones((1,320,480)),road=np.zeros((1,320,480)))
        return e,f,b,{k:v.copy() for k,v in b.items()}
    def test_same_world_sample_and_fixed_count(self):
        e,f,b,a=self.fixture();a['rgb'][:]=.2;v=e.evaluate_cf(102,b,a,[0,0])
        self.assertEqual(v['valid']['unit_ids'],[7]);self.assertAlmostEqual(v['valid']['delta_L_valid'],-.1)
    def test_nonfinite_after_rejected_not_dropped(self):
        e,f,b,a=self.fixture();a['rgb'][:,10,10]=np.nan
        with self.assertRaises(FloatingPointError):e.evaluate_cf(102,b,a,[0,0])
    def test_occlusion_rejects_hidden_RGB_truth(self):
        e,f,b,a=self.fixture();f['mask'][10,10]=True;v=e.evaluate_cf(102,b,a,[0,0])
        self.assertEqual(v['valid']['count'],0);self.assertIsNone(v['valid']['delta_L_valid']);self.assertIsNone(v['unobserved_query']['RGB_loss'])

if __name__=='__main__':unittest.main()
