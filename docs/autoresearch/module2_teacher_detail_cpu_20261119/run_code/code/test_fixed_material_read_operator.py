import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from world_texture_runtime import WorldTextureRuntime
from fixed_material_read_operator import ReadCompiler,FixedReadOperator

torch.set_num_threads(1)
class OperatorTests(unittest.TestCase):
    def test_forward_gradient_and_reload_equal_dense_quadrature(self):
        rng=np.random.default_rng(23);shape=(24,24)
        base=rng.uniform(.15,.65,(*shape,3));valid=np.ones(shape);valid[10:12,10:12]=0
        gates={'observed':rng.uniform(0,.3,shape),'teacher':rng.uniform(.2,1,shape)}
        uv=rng.uniform(7,16,(7,13,2));weights=rng.uniform(.1,1,13);weights/=weights.sum()
        compiler=ReadCompiler(base,valid,gates);compiler.add(uv[:3],weights);compiler.add(uv[3:],weights)
        op=compiler.finish(np.arange(7),{'kind':'CPU_same_quadrature_fixture'})
        for gate,g in gates.items():
            model=WorldTextureRuntime(base,valid,g,np.ones(shape),np.ones(shape),low_mask=np.ones((5,5)))
            model.low.data.copy_(torch.from_numpy(rng.normal(0,.2,model.low.shape)).float())
            model.high.data.copy_(torch.from_numpy(rng.normal(0,.2,model.high.shape)).float())
            dense=model.sample(torch.from_numpy(uv),torch.from_numpy(weights))[0];L,H=model.fields();fast=op.torch_query(L,H,gate)
            torch.testing.assert_close(dense,fast,rtol=0,atol=6e-8)
            dense.sum().backward();gl=model.low.grad.clone();gh=model.high.grad.clone();model.zero_grad()
            L,H=model.fields();op.torch_query(L,H,gate).sum().backward()
            torch.testing.assert_close(gl,model.low.grad,rtol=0,atol=2e-7);torch.testing.assert_close(gh,model.high.grad,rtol=0,atol=2e-7)
            with tempfile.TemporaryDirectory()as d:
                op.save(Path(d)/'read');other=FixedReadOperator.load(Path(d)/'read')
                np.testing.assert_array_equal(op.numpy_query(L.detach(),H.detach(),gate),other.numpy_query(L.detach(),H.detach(),gate))

    def test_gate_is_not_interpolated_premultiplied_H(self):
        b=np.full((4,4,3),.3);v=np.ones((4,4));g=np.zeros((4,4));g[:,2:]=1
        compiler=ReadCompiler(b,v,{'t':g},low_stride=1);compiler.add(np.array([[[1.5,1.]]]),np.ones(1));op=compiler.finish([0],{})
        H=np.zeros_like(b);H[:,2:]=.2;L=np.zeros_like(b)
        # 两个独立双线性值相乘：.5*.1=.05，非插值(g*H)=.1。
        np.testing.assert_allclose(op.numpy_query(L,H,'t'),.35,atol=1e-14)

if __name__=='__main__':unittest.main()
