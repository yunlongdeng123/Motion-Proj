import numpy as np
from scipy import sparse
import torch
import pytest
from shared_field_gs_read import SharedAverageGSRead

class Fields:
    def __init__(self):
        self.low=torch.tensor(np.arange(12).reshape(2,2,3)/100,dtype=torch.float32,requires_grad=True)
        self.high=torch.tensor(np.arange(27).reshape(3,3,3)/300,dtype=torch.float32,requires_grad=True)
    def fields(self):return self.low,self.high

def fixture():
    L=np.array([[.3,.2,.5,0],[0,.1,.3,.6]]);H=np.array([[.2,.2,0,0,.1,0,0,0,0],[0,0,0,.1,.2,.2,0,0,0]])
    return SharedAverageGSRead(np.ones((2,3))*.2,sparse.csr_matrix(L),sparse.csr_matrix(H),[30,10],[10,20,30],np.array([True,False,True]),np.ones((3,3),np.float32)*.11,low_shape=(2,2),high_shape=(3,3)),L,H

def test_global_ids_and_readonly_preserved():
    r,L,H=fixture();m=Fields();c=r.colors(m)
    expected=np.ones((3,3),np.float32)*.11
    expected[[2,0]]=np.clip(.2+L@m.low.detach().numpy().reshape(-1,3)+H@m.high.detach().numpy().reshape(-1,3),0,1)
    np.testing.assert_allclose(c.detach().numpy(),expected,atol=0,rtol=0)
    assert torch.equal(c[1],torch.tensor([.11,.11,.11]))

def test_sparse_backward_equals_dense_reference():
    r,L,H=fixture();m=Fields();c=r.colors(m);loss=(c**2).sum();g=torch.autograd.grad(loss,[m.low,m.high])
    m2=Fields();e=(.2+torch.tensor(L)@m2.low.reshape(-1,3).double()+torch.tensor(H)@m2.high.reshape(-1,3).double()).clamp(0,1).float()
    ref=torch.autograd.grad((e**2).sum(),[m2.low,m2.high])
    assert all(torch.equal(a,b)for a,b in zip(g,ref))

def test_scope_cannot_be_silently_changed():
    r,_,_=fixture()
    with pytest.raises(ValueError,match='scope'):
        SharedAverageGSRead(r.base,r.KL,r.KH,r.row_ids,r.active_world_rows,np.array([True,True,True]),r.parent_rgb,low_shape=(2,2),high_shape=(3,3))

def test_oracle_initial_ci_is_same_field_H0():
    r,L,H=fixture();m=Fields()
    with pytest.raises(ValueError):r.initial_ci_numpy(m)
    with torch.no_grad():m.high.zero_()
    assert np.array_equal(r.initial_ci_numpy(m),r.colors(m).detach().numpy())

def test_native_interface_is_explicit_not_pixel_composition():
    r,_,_=fixture();m=Fields()
    class Context:
        active_world_rows=np.array([10,20,30])
        def render_road(self,raw,colors):return {'raw':raw,'colors':colors,'CPU_stub_only':True}
    result=r.render(Context(),m,110)
    assert result['raw']==110 and result['colors'].shape==(3,3)and result['CPU_stub_only']
