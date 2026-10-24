import unittest
import numpy as np
from cpu_plane_propagation import project,rays,intersect,pair_patch,verify,refine_planes

class GeometryTests(unittest.TestCase):
    def frame(self):
        yy,xx=np.mgrid[:320,:480]
        rgb=np.stack([xx/480,yy/320,(xx+yy)/800],2)
        return dict(K=np.array([[100.,0,240],[0,100,160],[0,0,1]]),w2l=np.eye(4),rgb=rgb,
                    measured_valid=np.ones((320,480),bool),depth=np.full((320,480),5.),patch_clear=np.ones((320,480),bool))
    def test_half_pixel_roundtrip(self):
        f=self.frame();uv=np.array([[255.,240.],[256.,219.]])
        o,r=rays(f,uv);x=o+5*r;actual,z=project(x,f)
        np.testing.assert_allclose(actual,uv,atol=1e-10);np.testing.assert_allclose(z,5.)
    def test_plane_intersection_and_mask(self):
        f=self.frame();uv=np.array([[255.,240.]])
        o,r=rays(f,uv);n=np.array([[0.,0.,1.]]);x,t=intersect(o,r,np.array([[0.,0.,5.]]),n)
        np.testing.assert_allclose(t,5.);self.assertTrue(pair_patch(x,n,f,f)[0][0])
        g=self.frame();g['patch_clear'][240,255]=False
        self.assertFalse(pair_patch(x,n,f,g)[0][0])
    def test_hidden_target_rgb_never_used(self):
        fs=[self.frame() for _ in range(4)];uv=np.array([[255.,240.]])
        o,r=rays(fs[3],uv);x=o+5*r;n=np.array([[0.,0.,1.]])
        a=verify(x,n,np.ones(1,bool),3,uv,fs,np.array([True,False,True,False]))
        fs[3]['rgb'][:]=[1,0,1]
        b=verify(x,n,np.ones(1,bool),3,uv,fs,np.array([True,False,True,False]))
        self.assertTrue(a['accepted'][0]);np.testing.assert_array_equal(a['source'],[[0,2,1]])
        np.testing.assert_array_equal(a['color'],b['color'])
        fs[0]['depth'][:]=4.
        c=verify(x,n,np.ones(1,bool),3,uv,fs,np.array([True,False,True,False]))
        self.assertFalse(c['accepted'][0])
    def test_source_depth_refinement_without_validation_leak(self):
        fs=[self.frame() for _ in range(4)];uv=np.array([[255.,240.]])
        o,r=rays(fs[3],uv);x=o+5.19*r;n=np.array([[0.,0.,1.]])
        fit=np.array([True,False,True,False])
        actual=refine_planes(x,n,np.ones(1,bool),3,uv,fs,fit)
        np.testing.assert_allclose(actual[:,2],5.,atol=1e-8)
        fs[1]['depth'][:]=10.;fs[3]['rgb'][:]=1.
        again=refine_planes(x,n,np.ones(1,bool),3,uv,fs,fit)
        np.testing.assert_allclose(actual,again,atol=1e-8)

if __name__=='__main__':unittest.main()
