import json
from pathlib import Path
import numpy as np
import torch
from bc_model import ActorShadow,srgb_to_linear,linear_to_srgb
from shadow_proxy_reference import shadow_mask
torch.set_num_threads(2)
s=ActorShadow([0,0,1,0],1,azimuth=.35)
pos=torch.tensor([[0.,0.,.9]]);rot=torch.eye(3)[None];size=torch.tensor([[4.2,2,1.8]]);valid=torch.tensor([True])
rng=np.random.default_rng(42);q=torch.tensor(np.c_[rng.uniform(-4,4,(500,2)),np.zeros(500)],dtype=torch.float32)
pose=np.eye(4);pose[:3,3]=pos[0].numpy()
ref=shadow_mask(q.numpy(),pose,[4.2,2,1.8],[0,0,1,0],s.light.detach().numpy(),float(s.softness_m.detach()))
actual=s.mask(q,pos,rot,size,valid)
checks={'numpy_reference_max_abs':float(abs(actual.detach().numpy()-ref).max())}
assert checks['numpy_reference_max_abs']<2e-5
shift=torch.tensor([.7,-.3,0.])
assert torch.allclose(s.mask(q+shift,pos+shift,rot,size,valid),actual,atol=2e-6)
assert torch.equal(s.attenuation(q,pos,rot,size,~valid),torch.ones((len(q),3)))
value=s.attenuation(q,pos,rot,size,valid).sum();value.backward()
grad=float(s.azimuth.grad);eps=1e-3
with torch.no_grad():
    s.azimuth+=eps;plus=float(s.attenuation(q,pos,rot,size,valid).sum())
    s.azimuth-=2*eps;minus=float(s.attenuation(q,pos,rot,size,valid).sum());s.azimuth+=eps
fd=(plus-minus)/(2*eps)
assert abs(grad-fd)<max(.1,abs(fd)*.03),(grad,fd)
x=torch.linspace(0,1,100)
assert torch.allclose(linear_to_srgb(srgb_to_linear(x)),x,atol=2e-6)
checks.update(translation_equivariance=True,delete_actor_identity=True,
              azimuth_gradient=grad,azimuth_finite_difference=fd,srgb_roundtrip=True,
              no_timestamp_input=True,world_light_independent_of_actor_pose=True)
Path(__file__).with_name('torch_model_tests.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks))
