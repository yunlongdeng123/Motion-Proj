import unittest,tempfile
from pathlib import Path
import torch
from shadow_rebuild.ownership import split_moments,classify_evidence,split_boundary,split_plane
from fixture import create
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel


class OwnershipMath(unittest.TestCase):
    def test_split_keeps_both_sides_and_preserves_moments(self):
        cut=torch.tensor([-1.5,-.4,0.,.8,1.5],dtype=torch.float64)
        mass,mean,var=split_moments(cut)
        torch.testing.assert_close(mass.sum(-1),torch.ones_like(cut))
        torch.testing.assert_close((mass*mean).sum(-1),torch.zeros_like(cut),atol=1e-12,rtol=0)
        torch.testing.assert_close((mass*(var+mean.square())).sum(-1),torch.ones_like(cut))
        self.assertTrue(bool((mass>0).all()))

    def test_invisible_unknown_and_body_are_not_road(self):
        e=torch.tensor([[0.,0.,0.,0.],[0.,0.,.6,1.],[4.,0.,2.,8.],[10.,0.,0.,10.],[0.,0.,8.,9.]])
        f=torch.tensor([[0,0,0],[0,0,2],[4,0,4],[5,0,0],[0,0,5]])
        c=classify_evidence(e,f,torch.tensor([5,5,5,5,5]),[5])
        self.assertEqual(c['road_candidate'].tolist(),[False,False,False,False,True])
        self.assertEqual(c['mixed'].tolist(),[False,False,True,False,False])
        self.assertTrue(bool(c['unknown'][0]))


class BoundaryTopology(unittest.TestCase):
    def test_oblique_plane_split_preserves_3D_moments_and_both_children(self):
        from shadow_rebuild.math3d import covariance
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'data';config=create(path);d=Dataset(path,synthetic=True)
            model=SceneModel(d.seed,config);field=model.field;opt=model.optimizer('geometry');n=len(field.xyz)
            row=torch.tensor([1]);cut=torch.tensor([.35]);direction=torch.tensor([[.6,.8,0.]])
            mu=field.xyz[row].detach().clone();sigma=covariance(field.quaternion[row],field.scales[row]).detach().clone()
            mass,_,_=split_moments(cut);split_plane(field,row,direction,cut,opt)
            rr=torch.tensor([1,n]);childmu=field.xyz[rr];childcov=covariance(field.quaternion[rr],field.scales[rr])
            mean=(mass[0,:,None]*childmu).sum(0);delta=childmu-mu
            second=(mass[0,:,None,None]*(childcov+delta[:,:,None]*delta[:,None,:])).sum(0)
            torch.testing.assert_close(mean,mu[0],atol=1e-6,rtol=1e-5);torch.testing.assert_close(second,sigma[0],atol=1e-6,rtol=1e-4)
            self.assertEqual(int(field.owner[1]),int(field.owner[n]))

    def test_split_preserves_identity_and_unaffected_adam(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'data';config=create(path);dataset=Dataset(path,synthetic=True)
            model=SceneModel(dataset.seed,config);field=model.field;optimizer=model.optimizer('geometry')
            loss=sum(x.square().sum() for x in model.parameters() if x.requires_grad);loss.backward();optimizer.step()
            n=len(field.xyz);before=field.xyz.detach().clone();owner=field.owner.clone()
            adam=optimizer.state[field.xyz]['exp_avg'].clone()
            split_boundary(field,torch.tensor([1]),torch.tensor([0]),torch.tensor([.2]),optimizer)
            self.assertEqual(len(field.xyz),n+1);torch.testing.assert_close(field.owner[:n],owner)
            self.assertEqual(int(field.owner[-1]),int(owner[1]))
            torch.testing.assert_close(field.xyz[0],before[0]);torch.testing.assert_close(optimizer.state[field.xyz]['exp_avg'][0],adam[0])
            self.assertEqual(float(optimizer.state[field.xyz]['exp_avg'][1].abs().sum()),0)
            self.assertEqual(float(optimizer.state[field.xyz]['exp_avg'][-1].abs().sum()),0)
            self.assertNotEqual(float(field.xyz[1,0].detach()),float(field.xyz[-1,0].detach()))

    def test_occluded_exterior_not_forced_to_background_but_visible_road_is_penalized(self):
        from types import SimpleNamespace
        from shadow_rebuild.losses import mask_terms
        masks=torch.zeros(1,2,1,dtype=torch.bool)
        f=SimpleNamespace(masks=masks,mask_known=torch.tensor([True]))
        raw=torch.ones(1,2,1,requires_grad=True)
        # Same asset Gaussian: fully occluded at first pixel, visible at second.
        trans=torch.tensor([[[0.],[1.]]]);visible=raw*trans
        outside=mask_terms(dict(raw_actor_alpha=raw,visible_labels=visible),f,True,True)[1]
        grad=torch.autograd.grad(outside,raw)[0]
        self.assertEqual(float(grad[0,0,0]),0);self.assertGreater(float(grad[0,1,0]),0)
