import unittest
import numpy as np
from audit_math import in_region,region_grid_keys,grid_membership,projected_mask,pixel_metrics,horizontal_footprints


class AuditMathTest(unittest.TestCase):
    def test_rotated_world_selection(self):
        c,s=np.cos(.7),np.sin(.7);r=dict(center_xy=[12.,3.],axes_xy=[[c,-s],[s,c]],half_xy=[2.,1.])
        points=np.array([[0,0],[1.9,.9],[2.1,0]])@np.array(r['axes_xy']).T+r['center_xy']
        np.testing.assert_array_equal(in_region(points,r),[True,True,False])

    def test_common_grid_does_not_drop_non_candidate_cells(self):
        r=dict(center_xy=[1.,0.],axes_xy=[[1,0],[0,1]],half_xy=[.4,.3])
        keys=region_grid_keys(r);xy=(keys+.5)*.2
        self.assertTrue(in_region(xy,r).all());self.assertGreater(len(keys),0)
        result=grid_membership(xy,keys[:-1]);self.assertEqual(result[-1],-1)
        self.assertEqual(np.count_nonzero(result>=0),len(keys)-1)

    def test_footprint_uses_anisotropic_covariance(self):
        probes=np.array([[0.,0.],[.25,0],[0,.25],[1,1]])
        mask=horizontal_footprints(probes,np.zeros((1,2)),np.array([[[.01,0],[0,.0001]]]))
        np.testing.assert_array_equal(mask,[True,True,False,False])

    def test_pixel_denominator_keeps_zero_alpha(self):
        alpha=np.array([[0.,1.],[.2,.8]],np.float32);depth=np.ones_like(alpha)*12
        r=pixel_metrics(alpha,depth,np.ones_like(alpha,bool),np.ones_like(alpha))
        self.assertEqual(r['pixels'],4);self.assertAlmostEqual(r['alpha_mean'],.5)
        self.assertEqual(r['coverage50'],.5);self.assertEqual(r['capacity95'],1.)

    def test_no_projected_region_is_unknown_not_zero(self):
        result=pixel_metrics(np.zeros((2,2)),np.zeros((2,2)),np.zeros((2,2),bool))
        self.assertEqual(result['pixels'],0);self.assertIsNone(result['alpha_mean'])

    def test_behind_camera_quad_does_not_fill_screen(self):
        corners=np.array([[[0,0,-1],[1,0,-1],[1,1,-1],[0,1,-1]]],float)
        mask,n=projected_mask(corners,np.eye(4),np.eye(3),20,20)
        self.assertFalse(mask.any());self.assertEqual(n,1)


if __name__=='__main__':unittest.main()
