import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from world_texture_runtime import WorldTextureRuntime,compose_native_eligible,restore_common_A,FixedPixelReadLayout,FixedGaussianReadLayout

torch.set_num_threads(1)
class RuntimeTests(unittest.TestCase):
    def model(self,gate=1):
        shape=(24,24);return WorldTextureRuntime(np.full((*shape,3),.3),np.ones(shape),np.full(shape,gate),np.ones(shape),np.ones(shape),low_mask=np.ones((5,5)))

    def test_teacher_gate_opens_high_gradient_without_source_tag(self):
        uv=torch.tensor([[[11.,11.]]]);weights=torch.ones(1)
        old=self.model(0);new=self.model(1)
        old.set_stage('B');new.set_stage('B')
        old.sample(uv,weights)[0].sum().backward();new.sample(uv,weights)[0].sum().backward()
        self.assertEqual(float(old.high.grad.abs().sum()),0);self.assertGreater(float(new.high.grad.abs().sum()),0)
        self.assertEqual([x for x,_ in new.named_parameters()],['low','high'])

    def test_constant_sampling_and_empty_source(self):
        m=self.model();uv=torch.tensor([[[8.,8.],[9.,8.]],[[100.,100.],[101.,100.]]])
        rgb,mass=m.sample(uv,torch.tensor([.5,.5]));torch.testing.assert_close(rgb[0],torch.full((3,),.3))
        self.assertEqual(float(mass[1]),0);self.assertTrue(torch.isfinite(rgb).all())

    def test_deferred_preserves_readonly_and_fallback_without_roadalpha(self):
        baseline=torch.full((3,2,2),.4);E=torch.full((3,2,2),.05);W=torch.tensor([[.2,0.],[.1,.3]])
        X=torch.full((2,2,3),.8,requires_grad=True);valid=torch.tensor([[True,True],[False,True]])
        E[:,0,1]=0;out=compose_native_eligible(baseline,E,W,X,valid)
        torch.testing.assert_close(out[:,0,1],baseline[:,0,1]);torch.testing.assert_close(out[:,1,0],baseline[:,1,0])
        out.sum().backward();self.assertEqual(float(X.grad[0,1].abs().sum()),0);self.assertGreater(float(X.grad[0,0].abs().sum()),0)

    def test_common_A_fork_keeps_teacher_gate_and_Adam_snapshot(self):
        old=self.model(0);new=self.model(1)
        opt=torch.optim.Adam([dict(params=[old.low],lr=.01),dict(params=[old.high],lr=.003)])
        old.low.grad=torch.ones_like(old.low);opt.step()
        state=dict(stage='A',cursor=200,reader='pixel',identity='fixed_inputs',params=dict(low=old.low.detach().clone(),high=old.high.detach().clone()),
            optimizer=copy.deepcopy(opt.state_dict()),trace=[86,110]*100,RNG={'CPU_test':True})
        original=copy.deepcopy(state);other=torch.optim.Adam([dict(params=[new.low],lr=.01),dict(params=[new.high],lr=.003)])
        restore_common_A(state,new,other,reader='pixel',expected_identity='fixed_inputs')
        self.assertTrue(torch.all(new.H_gate==1));new.low.grad=torch.ones_like(new.low);other.step()
        for key,v in original['optimizer']['state'][0].items():self.assertTrue(torch.equal(v,state['optimizer']['state'][0][key]))
        with self.assertRaises(ValueError):restore_common_A(state,new,other,reader='average',expected_identity='fixed_inputs')

    def test_persistent_field_requeries_after_reload(self):
        m=self.model();m.high.data[11,11]=.5;uv=torch.tensor([[[11.,11.]]]);w=torch.ones(1)
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'field.pt';m.save(path,{'renderer':'pixel','GPU_validated':False});n,meta=WorldTextureRuntime.load(path)
            torch.testing.assert_close(m.sample(uv,w)[0],n.sample(uv,w)[0],rtol=0,atol=0);self.assertFalse(meta['GPU_validated'])

    def test_geometry_and_common_masks_cannot_be_optimized(self):
        m=self.model();uv=torch.tensor([[[11.,11.]]])
        with self.assertRaises(ValueError):m.sample(uv,torch.ones(1,requires_grad=True))
        with self.assertRaises(ValueError):m.sample(uv,torch.ones(1),torch.ones((1,1),requires_grad=True))
        with self.assertRaises(ValueError):WorldTextureRuntime(np.ones((24,24,3)),np.ones((24,24)),np.ones((24,24)),np.ones((24,24)),np.ones((24,24)),low_mask=np.ones((4,4)))

    def test_grouped_pixel_and_global_Gaussian_rows_scatter_by_identity(self):
        pixel=FixedPixelReadLayout([3,0,2],[1,1,0],2,2,row_id_kind='pixel_flat')
        v=torch.tensor([[.3]*3,[.1]*3,[.2]*3],requires_grad=True);field,valid=pixel(v)
        torch.testing.assert_close(field[0,0],v[1]);torch.testing.assert_close(field[1,1],v[0]);self.assertFalse(valid[1,0])
        field.sum().backward();torch.testing.assert_close(v.grad,torch.ones_like(v))
        gaussian=FixedGaussianReadLayout([31,5],[2,5,11,31],row_id_kind='global_world_row')
        old=torch.zeros((4,3));new=gaussian(old,torch.tensor([[.3]*3,[.1]*3]))
        torch.testing.assert_close(new[1],torch.full((3,),.1));torch.testing.assert_close(new[3],torch.full((3,),.3));self.assertEqual(float(old.sum()),0)
        with self.assertRaises(ValueError):FixedGaussianReadLayout([31],[2,5,11],row_id_kind='global_world_row')

if __name__=='__main__':unittest.main()
