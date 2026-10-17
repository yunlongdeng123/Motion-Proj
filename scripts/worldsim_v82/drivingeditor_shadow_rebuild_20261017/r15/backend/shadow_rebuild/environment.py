"""Original EnvLight cube shape, axes and nvdiffrast kernel; explicit device buffer."""
import torch
from torch import nn
class EnvLight(nn.Module):
    def __init__(self,resolution=1024):
        super().__init__();self.register_buffer('to_opengl',torch.tensor([[1.,0,0],[0,0,1],[0,-1,0]]));self.base=nn.Parameter(.5*torch.ones(6,resolution,resolution,3))
    def forward(self,l):
        import nvdiffrast.torch as dr
        l=(l.reshape(-1,3)@self.to_opengl.T).reshape(*l.shape).contiguous();prefix=l.shape[:-1]
        if len(prefix)!=3:l=l.reshape(1,1,-1,3)
        return dr.texture(self.base[None],l,filter_mode='linear',boundary_mode='cube').view(*prefix,3)
