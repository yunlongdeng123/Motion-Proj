"""固定道路点来源的几何/可见性契约；CPU真实数组，无训练或CUDA。"""
import unittest
import numpy as np
from .v44_road_observation import (project,measured_patch_evidence,certify_parent_sources,
    inherit_child_sources,LocalRoadSurface,select_reference,infer_asphalt_neighbours)


class RoadObservationTests(unittest.TestCase):
    def setUp(self):
        self.K=np.array([[100.,0,50],[0,100.,50],[0,0,1.]])
        self.C=np.eye(4)
        self.cloud=np.array([[-.4,-.4,10],[.4,-.4,10],[-.4,.4,10],[.4,.4,10]])
        self.query=np.array([[.10,.05,10.]])
        self.normal=np.array([[0.,0,1.]])

    def leaf(self,query=None,cloud=None,ground=None,normal=None):
        q=self.query if query is None else query;c=self.cloud if cloud is None else cloud
        uv,_=project(q,self.K,self.C)
        return measured_patch_evidence(c,np.ones(len(c),bool) if ground is None else ground,self.K,self.C,q,uv,
            self.normal if normal is None else normal,image_shape=(100,100))

    def test_supported_triangle_has_correct_depth(self):
        valid,depth,tri,conflict=self.leaf()
        self.assertTrue(valid[0]);self.assertFalse(conflict[0]);self.assertAlmostEqual(depth[0],10.)
        self.assertEqual(len(np.unique(tri[0])),3)

    def test_closer_non_ground_return_vetoes_background(self):
        q=np.array([[0.,0.,10.]]);cloud=np.vstack([self.cloud,[0.,0.,5.]])
        valid,_,_,conflict=self.leaf(query=q,cloud=cloud,ground=np.array([1,1,1,1,0],bool))
        self.assertTrue(conflict[0]);self.assertFalse(valid[0])

    def test_wrong_depth_or_normal_is_not_certified(self):
        self.assertFalse(self.leaf(query=np.array([[.1,.05,10.4]]))[0][0])
        self.assertFalse(self.leaf(normal=np.array([[1.,0,0.]]))[0][0])

    def test_annotated_foreground_box_blocks_parent_source(self):
        args=dict(lidar_xyz=self.cloud,lidar_ground=np.ones(4,bool),K=self.K,world_to_camera=self.C,
            query_xyz=self.query,query_normals=self.normal,foreground_guard=np.zeros((100,100),bool),surface_supported=np.ones(1,bool))
        self.assertTrue(certify_parent_sources(**args)['valid'][0])
        result=certify_parent_sources(**args,box_centers=np.array([[0.,0,5.]]),
            box_halves=np.array([[.5,.5,.5]]),box_axes=np.eye(3)[None])
        self.assertTrue(result['box_blocked'][0]);self.assertFalse(result['valid'][0])

    def test_child_inherits_logical_parent_and_rechecks_guard(self):
        guard=np.zeros((8,8),bool);guard[2,2]=True
        result=inherit_child_sources(np.array([1,0],bool),np.array([0,0,1,-1]),
            np.array([[1.,1.],[2.,2.],[3.,3.],[4.,4.]]),np.full(4,10.),guard)
        np.testing.assert_array_equal(result,[True,False,False,False])

    def test_local_surface_support_is_not_plane_extrapolation(self):
        xy=np.array([[-.2,-.2],[.2,-.2],[-.2,.2],[.2,.2]])
        points=np.c_[xy,xy@np.array([.1,.2])+1.]
        surface=LocalRoadSurface(points,np.array([.1,.2,1.]))
        z,support=surface.height(np.array([[0.,0.],[2.,2.]]))
        self.assertAlmostEqual(z[0],1.);np.testing.assert_array_equal(support,[True,False])
        n=np.array([-.1,-.2,1.]);n/=np.linalg.norm(n)
        np.testing.assert_allclose(surface.normal(np.array([[0.,0.]]))[0],n,atol=1e-10)

    def test_three_agreeing_bright_observations_are_required(self):
        rgb=np.array([[[.2,.2,.2]],[[.2,.2,.2]],[[.2,.2,.2]],[[.01,.01,.01]]],np.float32)
        result=select_reference(rgb,np.ones((4,1),bool),np.arange(1,5.)[:,None],chunk_size=1)
        self.assertTrue(result['real'][0]);self.assertEqual(result['chosen_index'][0],2)
        self.assertFalse(result['source_agree'][3,0])
        self.assertFalse(select_reference(rgb[:2],np.ones((2,1),bool),np.ones((2,1)))['real'][0])

    def test_inferred_is_separate_and_nearest_distance_is_bounded(self):
        xy=np.c_[np.r_[np.arange(8.),-.5,-.7],np.zeros(10)]
        xyz=np.c_[xy,np.zeros(10)];real=np.r_[np.ones(8,bool),False,False]
        rgb=np.zeros((10,3));rgb[:8]=.1
        result=infer_asphalt_neighbours(xyz,rgb,real)
        self.assertTrue(result['inferred'][8]);self.assertFalse(result['inferred'][9])
        self.assertEqual(result['confidence'][8],.25);self.assertFalse(real[8])
        np.testing.assert_allclose(result['reference_linear'][8],.1)
        np.testing.assert_array_equal(rgb[8],np.zeros(3))


if __name__=='__main__':unittest.main()
