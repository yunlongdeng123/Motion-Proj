"""CPU tests of metric geometry and actual GaussianModel optimizer resizing.

Pass GAUSSIAN_MODEL_PATH for the implementation under test. The actual class is
loaded from its AST, avoiding unrelated GPU-extension imports. Only its hard-
coded torch.zeros(device='cuda') allocations are redirected to CPU in one test;
append/prune/Adam operations themselves run unchanged from repository source.
"""
import ast
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from paper_sgcm import (SGCMConfig, SGCMController, SGCMGeometryError,
                        create_sgcm_controller, fit_ground_plane, transform_plane, transform_points,
                        validate_similarity)


def load_actual_gaussian_class():
    fallback = Path(__file__).with_name('gaussian_model_under_test.py')
    source = Path(os.environ.get('GAUSSIAN_MODEL_PATH',str(fallback)))
    tree = ast.parse(source.read_text(encoding='utf-8'))
    definition = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='GaussianModel')
    namespace = {'torch':torch,'nn':nn,'np':np,'BasicPointCloud':object}
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['GaussianModel']


def controller(scale=1.0, transform=None, config=None):
    transform = np.diag([scale,scale,scale,1.0]) if transform is None else transform
    plane = transform_plane(np.array([0.,0.,1.,0.]),transform)
    bounds = [np.array([[-1.,-1.,-1.],[1.,1.,2.]])]
    return SGCMController(config or SGCMConfig(),validate_similarity(transform),plane,
                          np.linalg.inv(transform),bounds,{})


class GeometryTests(unittest.TestCase):
    def test_exact_thresholds_and_outside_region(self):
        xyz=np.array([[0,0,.25],[0,0,.25001],[0,0,.5],[0,0,.50001],[2,0,1.]])
        masks=controller().classify_numpy(xyz,np.array([21.,21.,21.,21.,21.]))
        np.testing.assert_array_equal(masks['split'],[True,False,False,False,False])
        np.testing.assert_array_equal(masks['clone'],[False,True,True,False,False])
        np.testing.assert_array_equal(masks['prune'],[False,False,False,True,False])

    def test_metric_scale_rotation_translation_invariance(self):
        angle=.6; rotation=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
        world=np.array([[0,0,.1],[0,0,.4],[0,0,.8],[3,0,.8]])
        for scale in [.01,.1,2.5]:
            affine=np.eye(4);affine[:3,:3]=rotation*scale;affine[:3,3]=[4,-2,3]
            masks=controller(transform=affine).classify_numpy(transform_points(world,affine),np.full(4,30.))
            np.testing.assert_allclose(masks['distance_m'],world[:,2],atol=1e-12)
            np.testing.assert_array_equal(masks['prune'],[False,False,True,False])
            np.testing.assert_array_equal(masks['clone'],[False,True,False,False])
            np.testing.assert_array_equal(masks['split'],[True,False,False,False])

    def test_reject_anisotropic_transform(self):
        with self.assertRaises(SGCMGeometryError):
            validate_similarity(np.diag([.1,.2,.1,1.]))

    def test_ground_fit_with_dynamic_height_outliers(self):
        rng=np.random.RandomState(21)
        xy=rng.uniform(-4,4,(12000,2)); z=.05*xy[:,0]-.03*xy[:,1]-1.7+rng.normal(0,.02,len(xy))
        points=np.column_stack((xy,z)); object_points=points[:1200].copy();object_points[:,2]+=rng.uniform(.5,2.0,1200)
        points=np.concatenate((points,object_points))
        bounds=[np.array([[-1.,-1.,-1.],[1.,1.,1.]])]
        plane,stats=fit_ground_plane(points,bounds,SGCMConfig())
        predicted=-(plane[0]*xy[:,0]+plane[1]*xy[:,1]+plane[3])/plane[2]
        self.assertLess(np.sqrt(np.mean((predicted-(.05*xy[:,0]-.03*xy[:,1]-1.7))**2)),.05)
        self.assertGreater(stats['inlier_cells'],100)

    def test_insufficient_ground_fails_no_fallback(self):
        with self.assertRaises(SGCMGeometryError):
            fit_ground_plane(np.zeros((2,3)),[np.array([[-1,-1,-1],[1,1,1]])],SGCMConfig())

    def test_schedule_avoids_generic_interval(self):
        c=controller()
        self.assertFalse(c.due(500));self.assertTrue(c.due(550));self.assertFalse(c.due(600));self.assertFalse(c.due(15050))

    def test_schedule_overlap_rejected_before_training(self):
        with self.assertRaisesRegex(ValueError,'overlaps generic densification'):
            SGCMConfig.from_args(types.SimpleNamespace(sgcm_offset=0))
        with self.assertRaisesRegex(ValueError,'overlaps generic densification'):
            SGCMConfig.from_args(types.SimpleNamespace(sgcm_interval=150,sgcm_offset=50))
        SGCMConfig.from_args(types.SimpleNamespace())

    @unittest.skipUnless(os.environ.get('PAPER_SGCM_FIT_SOURCE'),'Real scene is opt-in via environment')
    def test_real_scene_ground_fit(self):
        c=create_sgcm_controller(os.environ['PAPER_SGCM_FIT_SOURCE'],types.SimpleNamespace())
        self.assertGreater(c.metadata['ground_fit']['inlier_cells'],30)
        self.assertGreater(c.model_scale,0)


class ActualOptimizerTopologyTests(unittest.TestCase):
    def make_model(self):
        model=load_actual_gaussian_class().__new__(load_actual_gaussian_class())
        n=5
        tensors={'xyz':torch.tensor([[0.,0,.75],[.1,0,.4],[.2,0,.1],[3,0,.75],[0,.2,.1]]),
                 'f_dc':torch.full((n,1,3),.2),'f_rest':torch.full((n,15,3),.3),
                 'opacity':torch.full((n,1),-.7),'scaling':torch.tensor([[.03,.04,.02]]).log().repeat(n,1),
                 'rotation':torch.tensor([[1.,0,0,0]]).repeat(n,1),'t':torch.arange(n).float()[:,None]*.1,
                 'scaling_t':torch.full((n,1),-.4),'velocity':torch.full((n,3),.2)}
        mapping={'xyz':'_xyz','f_dc':'_features_dc','f_rest':'_features_rest','opacity':'_opacity',
                 'scaling':'_scaling','rotation':'_rotation','t':'_t','scaling_t':'_scaling_t','velocity':'_velocity'}
        groups=[]
        for key,value in tensors.items():
            parameter=nn.Parameter(value.clone());setattr(model,mapping[key],parameter)
            groups.append({'name':key,'params':[parameter],'lr':.001})
        model.optimizer=torch.optim.Adam(groups,lr=.001)
        model.scaling_activation=torch.exp;model.scaling_inverse_activation=torch.log
        sum(parameter.square().sum() for group in groups for parameter in group['params']).backward()
        model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            model._xyz.copy_(tensors['xyz']);model._rotation.copy_(tensors['rotation']);model._scaling.copy_(tensors['scaling'])
        model.max_radii2D=torch.arange(n).float()+1
        model.xyz_gradient_accum=torch.arange(n).float()[:,None]+2
        model.t_gradient_accum=torch.arange(n).float()[:,None]+3
        model.denom=torch.arange(n).float()[:,None]+4
        return model

    def test_actual_clone_split_prune_adam_state_and_next_step(self):
        model=self.make_model()
        states={group['name']:model.optimizer.state[group['params'][0]]['exp_avg'].clone() for group in model.optimizer.param_groups}
        old_time=model._t.detach().clone(); old_mean=model._xyz[2].detach().clone()
        old_cov=torch.diag(model.get_scaling[2].square().detach())
        original_zeros=torch.zeros
        def cpu_zeros(*args,**kwargs):
            if kwargs.get('device')=='cuda': kwargs['device']='cpu'
            return original_zeros(*args,**kwargs)
        with patch.object(torch,'zeros',cpu_zeros):
            stats=controller().refine(model,torch.tensor([10.,10.,30.,10.,10.]),550)
        self.assertEqual((stats['pruned'],stats['cloned'],stats['split_parents']),(1,1,1))
        self.assertEqual(len(model.get_xyz),6)
        for group in model.optimizer.param_groups:
            parameter=group['params'][0];state=model.optimizer.state[parameter]
            self.assertEqual(len(parameter),6)
            torch.testing.assert_close(state['exp_avg'][:3],states[group['name']][[1,3,4]])
            torch.testing.assert_close(state['exp_avg'][3:],torch.zeros_like(state['exp_avg'][3:]))
            self.assertEqual(tuple(state['exp_avg_sq'].shape),tuple(parameter.shape))
        self.assertEqual(len(model.optimizer.state),9)
        torch.testing.assert_close(model.max_radii2D,torch.tensor([2.,4.,5.,0.,0.,0.]))
        torch.testing.assert_close(model._t.detach()[3:],old_time[[1,2,2]])
        means=model._xyz.detach()[4:]
        torch.testing.assert_close(means.mean(0),old_mean)
        covariance=sum(torch.diag(model.get_scaling[i].detach().square())+torch.outer(means[i-4]-old_mean,means[i-4]-old_mean) for i in [4,5])/2
        torch.testing.assert_close(covariance,old_cov)
        sum(group['params'][0].square().sum() for group in model.optimizer.param_groups).backward()
        model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
        self.assertTrue(all(torch.isfinite(group['params'][0]).all() for group in model.optimizer.param_groups))

    def test_resource_cap_and_stale_radii(self):
        model=self.make_model();c=controller(config=SGCMConfig(max_new_points=0))
        stats=c.refine(model,torch.tensor([10.,10.,30.,10.,10.]),550)
        self.assertEqual(len(model.get_xyz),4);self.assertEqual(stats['cloned'],0);self.assertEqual(stats['split_parents'],0)
        with self.assertRaises(RuntimeError):
            c.refine(model,torch.ones(5),650)


if __name__=='__main__':
    unittest.main(verbosity=2)
