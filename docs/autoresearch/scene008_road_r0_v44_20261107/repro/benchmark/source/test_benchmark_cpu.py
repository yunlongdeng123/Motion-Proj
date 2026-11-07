"""Meaningful leakage/mask invariance checks; no model or network required."""
import sys
sys.dont_write_bytecode=True
import unittest
import numpy as np
import road_patch_benchmark as b


class BenchmarkCPU(unittest.TestCase):
    def test_target_colour_intervention_is_exact(self):
        rng=np.random.default_rng(12)
        rgb=rng.integers(0,256,(128,192,3),dtype=np.uint8)
        rectangles=[[20,45,89,110],[100,50,133,90]]
        image,valid,invalid=b.blind_native_image(rgb,rectangles,(32,48))
        poisoned=rgb.copy();poisoned[invalid]=rng.integers(0,256,(int(invalid.sum()),3),dtype=np.uint8)
        second,v2,_=b.blind_native_image(poisoned,rectangles,(32,48))
        np.testing.assert_array_equal(image,second)
        np.testing.assert_array_equal(valid,v2)

    def test_Brown_interval_contains_projected_prism_points(self):
        rng=np.random.default_rng(5)
        patch={"guard_xy_bounds_local_m":[-1.,-.5,1.,.5],"context_z_interval_m":[10.,12.]}
        intrinsic=[1000.,1000.,960.,640.,.0414,-.3349,.0005,-.00093,0.]
        w=np.eye(4)
        rect=b.projected_prism_rectangle(patch,w,intrinsic,1920,1280,padding=0)
        p=rng.uniform([-1.,-.5,10.],[1.,.5,12.],size=(30000,3))
        x,y=p[:,0]/p[:,2],p[:,1]/p[:,2];r=x*x+y*y
        radial=1+intrinsic[4]*r+intrinsic[5]*r*r
        xd=x*radial+2*intrinsic[6]*x*y+intrinsic[7]*(r+2*x*x)
        yd=y*radial+intrinsic[6]*(r+2*y*y)+2*intrinsic[7]*x*y
        u,v=1000*xd+959.5,1000*yd+639.5
        self.assertTrue(((u>=rect[0])&(u<rect[2])&(v>=rect[1])&(v<rect[3])).all())

    def test_near_plane_crossing_masks_entire_image(self):
        patch={"guard_xy_bounds_local_m":[-1.,-.5,1.,.5],"context_z_interval_m":[-.2,12.]}
        self.assertEqual(b.projected_prism_rectangle(patch,np.eye(4),[1000.,1000.,960.,640.,0,0,0,0,0],1920,1280),[0,0,1920,1280])

    def test_geometry_prism_removes_guard_not_just_center(self):
        patch={"guard_xy_bounds_local_m":[-.75,-.75,1.75,1.75],"context_z_interval_m":[-.5,.5]}
        xyz=np.array([[0.,0.,0.],[-.5,-.5,0.],[2.,0.,0.],[0.,0.,1.]])
        np.testing.assert_array_equal(b.in_prisms(xyz,[patch]),[True,True,False,False])


if __name__=="__main__":
    unittest.main()
