import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from world_material_field import gaussian_plan,WorldMaterialField


class FieldTests(unittest.TestCase):
    def make(self):
        T=np.eye(4);T[:2,:2]*=.02
        p=gaussian_plan(np.array([[.4,.4,0.],[.5,.4,0.]]),np.array([[0.,0.,1.]]*2),np.array([[.013,.013,.003]]*2),T)
        return WorldMaterialField(np.full((48,48,3),.3),np.ones((48,48)),np.ones((48,48)),p)

    def test_GH_moments_and_constant(self):
        m=self.make();w=m.weights.numpy();x=m.nodes.numpy()
        np.testing.assert_allclose(w.sum(),1,atol=1e-14)
        np.testing.assert_allclose((w[:,None]*x).sum(0),0,atol=1e-14)
        np.testing.assert_allclose(np.einsum('q,qi,qj->ij',w,x,x),np.eye(2),atol=1e-14)
        rgb,mass=m.query();np.testing.assert_allclose(rgb.detach().numpy(),.3,atol=1e-7)
        np.testing.assert_allclose(mass.numpy(),1,atol=1e-14)

    def test_real_parameter_gradient_and_stage_freeze(self):
        m=self.make();m.set_stage('A');m.query()[0].sum().backward()
        self.assertGreater(float(m.low.grad.abs().sum()),0);self.assertIsNone(m.high.grad)
        m.zero_grad();m.set_stage('B');m.high.data[20,20]=.1;m.query()[0].sum().backward()
        self.assertIsNone(m.low.grad);self.assertGreater(float(m.high.grad.abs().sum()),0)

    def test_field_reload_requeries_exact(self):
        m=self.make();m.low.data[3,3]=.1;m.high.data[20,20]=.2
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'field.pt';m.save(path,{'CPU_test':True});other,_=WorldMaterialField.load(path)
            torch.testing.assert_close(m.query()[0],other.query()[0],rtol=0,atol=0)


if __name__=='__main__':torch.set_num_threads(1);unittest.main()
