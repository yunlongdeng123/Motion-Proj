"""A/C2 共享表面连续更新器；排序结构差异，输入信息不削弱。"""
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation
from .first_event_trace import trace,boundary_neighborhood
def features(surface,obs,device='cpu'):
    near=boundary_neighborhood(surface,obs);chain=trace(surface,obs);r=len(obs['positive_actor']);k=len(surface.center)
    actual=np.zeros((r,k));actual[chain['ray'],chain['patch']]=1
    o=np.broadcast_to(obs['origins_actor_m'][:,None,:],(r,k,3))/10
    d=np.broadcast_to(obs['directions_actor'][:,None,:],(r,k,3));y=np.broadcast_to(obs['observed_first_range_m'][:,None],(r,k))
    depth=near['plane_t']
    # 保留原始深度排序；单调可逆缩放避免把远处不同层截成同一个值。
    edge=np.concatenate([o,d,(y/10)[:,:,None],np.arcsinh(depth/10)[:,:,None],np.arcsinh(depth-y)[:,:,None],
        np.arcsinh(near['uv']),np.arcsinh(near['signed_boundary_distance'])[:,:,None],
        np.broadcast_to(obs['positive_actor'][:,None,None],(r,k,1)),actual[:,:,None],near['valid_plane'][:,:,None]],axis=-1)
    node=np.c_[surface.center/3,surface.rotation[:,:,2],np.log(surface.radius)]
    # 显式顺序只是一种结构先验；C2 同时接收完整深度与全部几何集合。
    order=np.argsort(np.where(near['valid_plane'],depth,np.inf),axis=1,kind='stable')[:,::-1].copy()
    return (torch.tensor(node,dtype=torch.float32,device=device),torch.tensor(edge,dtype=torch.float32,device=device),torch.tensor(order,dtype=torch.long,device=device))
class SurfaceDynamics(nn.Module):
    def __init__(self,width=32,ordered=True):
        super().__init__();self.ordered=ordered
        self.edge=nn.Sequential(nn.Linear(15,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        if ordered:self.chain=nn.GRU(width,width,batch_first=True)
        else:
            layers=[nn.Linear(width*2,width),nn.SiLU()]
            for _ in range(4):layers.extend([nn.Linear(width,width),nn.SiLU()])
            self.set_context=nn.Sequential(*layers)
        self.patch=nn.Sequential(nn.Linear(width*2+14,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.delta=nn.Linear(width,14)
    def forward(self,node,edge,order):
        x=self.edge(edge)
        if self.ordered:
            ordered=x.gather(1,order[:,:,None].expand_as(x));h,_=self.chain(ordered)
            x=torch.zeros_like(h).scatter(1,order[:,:,None].expand_as(h),h)
        else:
            context=self.set_context(torch.cat([x.mean(1),x.amax(1)],dim=-1));x=x+context[:,None]
        shared=torch.cat([x.mean(0),x.amax(0),node],dim=-1)
        return self.delta(self.patch(shared))
def apply_continuous(s,delta):
    result=s.copy();x=np.asarray(delta)
    result.center+=np.clip(x[:,:3],-.5,.5)
    result.rotation=Rotation.from_rotvec(np.clip(x[:,3:6],-.3,.3)).as_matrix()@result.rotation
    result.radius*=np.exp(np.clip(x[:,6:],-.7,.7));return result
