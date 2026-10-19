"""Behavioral CPU regression tests; these do not certify CUDA renderer or quality."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np

# Use already installed CPU torch and bundled missing pure-Python dependencies.
bundled=Path('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
if bundled.exists(): sys.path.append(str(bundled))
import torch
from static_partition import (StaticV2Mixin, route_depth, observed_rgb_loss,
                              vehicle_volume_opacity, independent_road_backward)
from world_road import (RoadSurface, query_from_pixels, project, observations,
                        fuse, fill_from_donor, ray_box_blocked, select_asphalt_donors)
from road_objective import road_objective


class NativeStub:
    def __init__(self):
        self._xyz=torch.nn.Parameter(torch.tensor([[0.,0.,0.],[1.,0.,1.]]))
        self._velocity=torch.nn.Parameter(torch.ones(2,3))
        self._t=torch.nn.Parameter(torch.ones(2,1))
        self._scaling_t=torch.nn.Parameter(torch.zeros(2,1))
        self._opacity=torch.nn.Parameter(torch.zeros(2,1))
        self.optimizer=None
        self.xyz_gradient_accum=torch.zeros(2,1);self.denom=torch.zeros(2,1)
        self.t_gradient_accum=torch.zeros(2,1)
    def get_xyz_SHM(self,t,times=1): return self._xyz+self._velocity*t
    def get_marginal_t(self,t): return torch.exp(-(self._t-t)**2)
    def densify_and_prune(self):
        self.t_grad=True;self.no_time_split=False
        self._velocity=torch.nn.Parameter(torch.ones(2,3))
        self._t=torch.nn.Parameter(torch.ones(2,1))
        self._scaling_t=torch.nn.Parameter(torch.ones(2,1))

    def update_learning_rate(self,step):
        for g in self.optimizer.param_groups:g['lr']=.123


class Static(StaticV2Mixin,NativeStub): pass


class Cases(unittest.TestCase):
    def test_static_time_invariance_dynamic_unchanged(self):
        a,b=Static(),NativeStub();a.enforce_static()
        for t in [-100,0,1,100]:
            self.assertTrue(torch.equal(a.get_xyz_SHM(t),a._xyz))
            self.assertTrue(torch.equal(a.get_marginal_t(t),torch.ones(2,1)))
        self.assertFalse(torch.equal(b.get_xyz_SHM(0),b.get_xyz_SHM(1)))
        a.densify_and_prune();self.assertFalse(a._t.requires_grad)
        self.assertEqual(a._velocity.count_nonzero().item(),0)
        self.assertFalse(a.t_grad);self.assertTrue(a.no_time_split)

    def test_body_rgb_no_direct_static_gradient(self):
        pred=torch.zeros(3,8,8,requires_grad=True);gt=torch.ones_like(pred)
        mask=torch.zeros(1,8,8);mask[:,2:6,2:6]=1
        loss=observed_rgb_loss(pred,gt,mask,lambda a,b:1-(a-b).square().mean())
        loss.backward();self.assertEqual(pred.grad[:,2:6,2:6].count_nonzero().item(),0)
        self.assertGreater(pred.grad[:,0,:].abs().sum().item(),0)

    def test_depth_body_uses_only_dynamic_graph(self):
        static=torch.ones(1,2,2,requires_grad=True)
        dynamic=torch.full((1,2,2),2.,requires_grad=True)
        mask=torch.tensor([[[1,0],[0,1]]])
        route_depth(static,dynamic,torch.ones_like(dynamic),mask).sum().backward()
        self.assertTrue(torch.equal(static.grad,1-mask));self.assertTrue(torch.equal(dynamic.grad,mask))

    def test_vehicle_exclusion_protects_road_and_far_background(self):
        xyz=torch.tensor([[0.,0.,1.],[0.,0.,0.],[0.,0.,5.]],requires_grad=True)
        opa=torch.full((3,1),0.5,requires_grad=True)
        loss=vehicle_volume_opacity(xyz,opa,torch.tensor([[0.,0.,1.]]),
            torch.eye(3)[None],torch.tensor([[1.,1.,1.]]),torch.tensor([0.,0.,1.]),0.,0.2,0.1)
        loss.backward();self.assertGreater(opa.grad[0].item(),0)
        self.assertEqual(opa.grad[1].item(),0);self.assertEqual(opa.grad[2].item(),0)
        self.assertIsNone(xyz.grad)

    def test_independent_objective_does_not_update_dynamic_or_environment(self):
        s=torch.tensor(1.,requires_grad=True);d=torch.tensor(2.,requires_grad=True);e=torch.tensor(3.,requires_grad=True)
        independent_road_backward(s+2*d+3*e,10*s+100*d+100*e,[s])
        self.assertEqual(s.grad.item(),11);self.assertEqual(d.grad.item(),2);self.assertEqual(e.grad.item(),3)

    def test_unknown_road_zero_gradient(self):
        rgb=torch.ones(3,4,4,requires_grad=True);a=torch.ones(1,4,4)
        target=dict(rgb=torch.zeros_like(rgb),depth=a,confidence=a,provenance=torch.zeros_like(a))
        l,_=road_objective(rgb,a,a,target);l.backward();self.assertEqual(rgb.grad.count_nonzero().item(),0)

    def test_world_ray_roundtrip_and_surface_support(self):
        grid=np.array([[x,y,5.] for x in np.arange(-2,2.1,.2) for y in np.arange(-2,2.1,.2)])
        surface=RoadSurface(np.eye(4),np.array([0.,0.,5.]),grid)
        K=np.array([[100.,0,50],[0,100.,50],[0,0,1]])
        pixels=np.array([[50.,50.],[60.,50.],[50.,40.]])
        xyz,valid,support=query_from_pixels(pixels,K,np.eye(4),surface)
        uv,_,inside=project(xyz,K,np.eye(4),(100,100))
        self.assertTrue(valid.all() and support.all() and inside.all())
        np.testing.assert_allclose(uv,pixels,atol=1e-8)

    def test_parallel_ray_box_intersection(self):
        box=dict(center=[0,0,2],axes=np.eye(3),half=[.5,.5,.5])
        result=ray_box_blocked(np.array([[0,0,5.],[2,0,5.],[0,0,1.]]),np.zeros(3),[box])
        np.testing.assert_array_equal(result,[True,False,False])

    def test_visibility_rejects_occlusion_and_missing_depth(self):
        frame=dict(rgb=np.full((100,100,3),128,np.uint8),K=np.array([[100.,0,50],[0,100.,50],[0,0,1]]),
                   world_to_camera=np.eye(4),occluder_mask=np.zeros((100,100),bool),
                   road_mask=np.ones((100,100),bool),depth_m=np.full((100,100),5.),
                   depth_valid=np.ones((100,100),bool),quality=np.ones((100,100)),boxes=[])
        config=dict(texel_m=.1,mask_margin_px=2,depth_tolerance_m=.3,min_cosine=.1,min_area_px=.1,min_quality=.2)
        xyz=np.array([[0.,0.,5.]]);normals=np.array([[0.,0.,1.]])
        self.assertGreater(observations(xyz,normals,np.array([True]),frame,config)[1][0],0)
        frame['depth_valid'][:]=False
        self.assertEqual(observations(xyz,normals,np.array([True]),frame,config)[1][0],0)
        frame['depth_valid'][:]=True;frame['occluder_mask'][50,50]=True
        self.assertEqual(observations(xyz,normals,np.array([True]),frame,config)[1][0],0)

    def test_fusion_rejects_outlier_no_data_unknown_donor_not_evidence(self):
        colors=np.array([[[.5,.5,.5],[.1,.1,.1]],[[.51,.5,.5],[.2,.2,.2]],[[1,0,0],[.2,.2,.2]]])
        weights=np.array([[1,0],[1,0],[1,0]])
        result=fuse(colors,weights,[131,132,133])
        np.testing.assert_array_equal(result['provenance'],[1,0])
        self.assertEqual(result['source_count'][0],2)
        out=fill_from_donor(np.array([[0,0],[1,0]]),result,np.array([[2,0]]),np.array([[.2,.2,.2]]),True)
        np.testing.assert_array_equal(out['provenance'],[1,2])
        np.testing.assert_array_equal(out['rgb'][0],result['rgb'][0])
        self.assertEqual(out['source_count'][1],0)
        self.assertEqual(fill_from_donor(np.array([[0,0],[1,0]]),result,np.array([[2,0]]),np.array([[.2]*3]),False)['provenance'][1],0)

    def test_no_duplicate_source_views(self):
        with self.assertRaises(ValueError): fuse(np.ones((2,1,3)),np.ones((2,1)),[131,131])

    def test_scheduler_cannot_reenable_static_time(self):
        a=Static()
        a.optimizer=torch.optim.Adam([{'params':[a._t],'name':'t'},
                                      {'params':[a._xyz],'name':'xyz'}],lr=.1)
        a.update_learning_rate(2)
        self.assertEqual(a.optimizer.param_groups[0]['lr'],0)
        self.assertGreater(a.optimizer.param_groups[1]['lr'],0)
        self.assertFalse(a._t.requires_grad)

    def test_donor_filter_excludes_markings_occluders_and_unsupported_road(self):
        rgb=np.full((60,80,3),90,np.uint8)
        rgb[10:50,25:28]=255;rgb[10:50,40:44]=[230,210,20]
        occ=np.zeros((60,80),bool);occ[30:35,15:20]=True
        support=np.ones((60,80),bool);support[40:50,60:70]=False
        accepted,regions,marking=select_asphalt_donors(rgb,[[[5,5],[74,5],[74,54],[5,54]]],occ,support)
        self.assertTrue(accepted[20,15]);self.assertFalse(accepted[10:50,25:28].any())
        self.assertFalse(accepted[10:50,40:44].any());self.assertFalse(accepted[occ].any())
        self.assertFalse(accepted[~support].any());self.assertFalse(accepted[0].any())

    def test_single_or_conflicting_real_candidate_is_not_prior_filled(self):
        colors=np.array([[[.2]*3,[0.]*3],[[.2]*3,[1.]*3]])
        result=fuse(colors,np.array([[1,1],[0,1]]),[131,132])
        out=fill_from_donor(np.zeros((2,2)),result,np.zeros((1,2)),np.full((1,3),.5),True)
        np.testing.assert_array_equal(out['provenance'],[0,0])
        np.testing.assert_array_equal(out['donor_index'],[-1,-1])

    def test_measured_nonplanar_surface_and_unsupported_geometry(self):
        grid=np.array([[x,y,5.+.15*x*x] for x in np.arange(-2,2.01,.1) for y in np.arange(-2,2.01,.1)])
        s=RoadSurface(np.eye(4),np.array([0.,0.,5.]),grid)
        z,support=s.height(np.array([[1.,0.],[10.,0.]]))
        self.assertGreater(z[0],5.1);self.assertTrue(support[0]);self.assertFalse(support[1])
        xyz,valid,_=s.intersect(np.zeros((1,3)),np.array([[.2,0,1.]])/np.sqrt(1.04))
        self.assertTrue(valid[0]);self.assertAlmostEqual(xyz[0,2],s.height(xyz[:,:2])[0][0],places=2)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    receipt=dict(status='PASSED_CPU_BEHAVIOR_CONTRACTS' if result.wasSuccessful() else 'FAILED_CPU_BEHAVIOR_CONTRACTS',
                 tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                 GPU_used=False,training_updates=0,actual_scene_binding_verified=False,quality_verified=False)
    Path(__file__).with_name('cpu_test_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
