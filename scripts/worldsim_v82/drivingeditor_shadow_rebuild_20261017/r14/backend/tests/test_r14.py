import unittest
from types import SimpleNamespace
import numpy as np
import torch
from shadow_rebuild.mask_fusion import fuse_regions
from shadow_rebuild.losses import mask_terms
from shadow_rebuild.competition import competition_terms
from shadow_rebuild.config import Config
from shadow_rebuild.tracing import SunAmbient


class RegionalMask(unittest.TestCase):
    def test_local_overlap_does_not_replace_entire_instances(self):
        old=np.zeros((12,18,2),bool);sam=old.copy();old[2:9,2:9,0]=True;sam[2:9,3:10,0]=True;old[6:11,9:15,1]=True;sam[...,1]=old[...,1]
        e=np.zeros_like(old);e[3:8,5:8,0]=True;e[7:10,11:14,1]=True
        r=fuse_regions(old,np.ones(2,bool),sam,np.ones(2,bool),e,np.zeros((12,18),bool))
        self.assertTrue(r['masks'][4,6,0]);self.assertTrue(r['masks'][8,12,1]);self.assertFalse(r['mask_valid'][7,9].any());self.assertFalse((r['masks'].sum(-1)>1).any())

    def test_source_order_symmetric_and_known_empty_has_no_priority(self):
        a=np.zeros((10,10,1),bool);b=a.copy();b[2:8,2:8,0]=True;e=np.zeros_like(a);e[4:7,4:7,0]=True;road=np.zeros((10,10),bool)
        x=fuse_regions(a,np.ones(1,bool),b,np.ones(1,bool),e,road);y=fuse_regions(b,np.ones(1,bool),a,np.ones(1,bool),e,road)
        np.testing.assert_array_equal(x['masks'],y['masks']);np.testing.assert_array_equal(x['mask_valid'],y['mask_valid']);self.assertTrue(x['masks'][5,5,0])

    def test_unknown_pixels_have_no_inside_outside_or_identity_gradient(self):
        m=torch.zeros(3,4,1,dtype=torch.bool);m[0,0]=True;v=torch.ones_like(m);v[1,1]=False;f=SimpleNamespace(masks=m,mask_known=torch.tensor([True]),mask_valid=v)
        x=torch.full(m.shape,.5,requires_grad=True);sum(mask_terms({'raw_actor_alpha':x,'visible_labels':x},f,True,True,True)).backward()
        self.assertEqual(float(x.grad[1,1]),0);self.assertGreater(float(x.grad[2,2]),0);self.assertLess(float(x.grad[0,0]),0)


class GaussianCompetition(unittest.TestCase):
    def package(self,static_depth):
        sa=torch.full((2,2),.9,requires_grad=True);da=torch.full((2,2),.8,requires_grad=True)
        return {'alpha':sa+da*0,'static_isolated_alpha':sa,'dynamic_isolated_alpha':da,'static_isolated_depth':torch.full((2,2),static_depth),'dynamic_isolated_depth':torch.full((2,2),3.),'visible_labels':da[...,None]*.5}
    def frame(self):return SimpleNamespace(masks=torch.ones(2,2,1,dtype=torch.bool),mask_known=torch.tensor([True]),mask_valid=None,road_certified=torch.zeros(2,2,dtype=torch.bool))
    def test_duplicate_same_surface_penalizes_wrong_static_not_supported_body(self):
        c=Config(continuous_ownership=True);p=self.package(3.);v=competition_terms(p,self.frame(),c,1000)['ownership_occupancy'];v.backward()
        self.assertGreater(float(p['static_isolated_alpha'].grad.sum()),0);self.assertTrue(p['dynamic_isolated_alpha'].grad is None or float(p['dynamic_isolated_alpha'].grad.abs().sum())==0)
    def test_real_road_behind_car_is_not_duplicate_occupancy(self):
        c=Config(continuous_ownership=True);v=competition_terms(self.package(6.),self.frame(),c,1000);self.assertEqual(float(v['ownership_occupancy']),0)


class PoseShadow(unittest.TestCase):
    def test_same_time_pose_translation_and_delete_change_geometric_rho(self):
        c=Config(sun_samples=1,sun_elevation_degrees=60.,sun_azimuth_degrees=0.)
        light=SunAmbient(c);n=9;x=torch.linspace(-1.,1.,n);pos=torch.stack((x,torch.zeros(n),torch.zeros(n)),-1);normal=torch.tensor([0.,0.,1.]).expand(n,3);albedo=torch.ones(n,3)*.5;rec=torch.ones(n)
        direction=light.direction().detach();mu=direction[None]*.6
        def geo(means,owner):return dict(xyz=means,cov=torch.eye(3)[None]*.05**2,opacity=torch.tensor([[.9]]),owner=torch.tensor([owner]),minimum_scale=torch.tensor([.05]))
        g=geo(mu,0);c0,rho,vis=light.unshadowed_and_rho(pos,normal,albedo,g,rec);moved=geo(mu+torch.tensor([[.75,0.,0.]]),0);_,r2,_=light.unshadowed_and_rho(pos,normal,albedo,moved,rec)
        self.assertNotEqual(int(rho.mean(-1).argmax()),int(r2.mean(-1).argmax()));self.assertGreater(float(rho.max()),.01)
        _,removed,_=light.unshadowed_and_rho(pos,normal,albedo,geo(mu,-1),rec);self.assertEqual(float(removed.max()),0)
        shaded,_=light.shade(pos,normal,albedo,g,rec);torch.testing.assert_close(shaded,c0*(1-rho));self.assertTrue(bool((shaded<=c0+1e-7).all()))

if __name__=='__main__':unittest.main()
