"""同见证、同候选的八步共享几何更新；A链结构/C2集合强控制。"""
import torch
from torch import nn

class WitnessDynamics(nn.Module):
    def __init__(self,width=32,ordered=True):
        super().__init__();self.width=width;self.ordered=ordered
        self.edge=nn.Sequential(nn.Linear(15,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        if ordered:self.chain=nn.GRU(width,width,batch_first=True)
        else:self.set_context=nn.Sequential(nn.Linear(width*2,width),nn.SiLU(),*[layer for _ in range(4) for layer in [nn.Linear(width,width),nn.SiLU()]])
        self.patch=nn.Sequential(nn.Linear(width*2+30,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.memory=nn.GRUCell(width,width)
        self.delta=nn.Linear(width,14);nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
        self.rank=nn.Sequential(nn.Linear(width*3+40,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))
    def forward(self,f,hidden=None):
        b,r,p,_=f['edge'].shape;active=f['active'];x=self.edge(f['edge'])*f['edge_valid'][...,None]
        if self.ordered:
            ordered=x.gather(2,f['order'][...,None].expand_as(x));h,_=self.chain(ordered.reshape(b*r,p,-1))
            x=torch.zeros_like(x).scatter(2,f['order'][...,None].expand_as(x),h.reshape_as(x))
        else:
            context=self.set_context(torch.cat([x.sum(2)/active.sum(-1)[:,None,None].clamp_min(1),x.amax(2)],-1));x=x+context[:,:,None]
        x=x*active[:,None,:,None]
        evidence=torch.log1p(f['evidence'].float()).flatten(-2)
        node=self.patch(torch.cat([x.mean(1),x.amax(1),f['node'],evidence],-1))
        if hidden is None:hidden=torch.zeros_like(node)
        hidden=self.memory(node.reshape(b*p,-1),hidden.reshape(b*p,-1)).reshape(b,p,-1)*active[...,None]
        delta=self.delta(hidden)*active[...,None]
        touch=f['all_delta'].abs().sum(-1)*active[:,None]
        local=torch.einsum('bep,bpw->bew',touch,hidden)/touch.sum(-1,keepdim=True).clamp_min(1e-6)
        glob=torch.cat([hidden.sum(1)/active.sum(-1,keepdim=True).clamp_min(1),hidden.amax(1)],-1)
        score=self.rank(torch.cat([local,glob[:,None].expand(-1,local.shape[1],-1),f['candidate']],-1)).squeeze(-1)
        return delta,score,hidden

def learning_loss(delta,score,f):
    active=f['active'];reg=((delta-f['target_delta']).square()*active[...,None]).sum()/active.sum().clamp_min(1)/14
    target=f['target_score'].clamp(-5,5);mask=f['valid']
    rank=torch.nn.functional.smooth_l1_loss(score[mask],target[mask])
    # 完全相等/近似相等事件按相同代价监督；不强迫网络记住任意argmin并列索引。
    best=target.masked_fill(~mask,float('inf')).amin(-1,keepdim=True)
    probabilities=torch.softmax(-(target-best)*5,dim=-1)*mask
    probabilities=probabilities/probabilities.sum(-1,keepdim=True).clamp_min(1e-9)
    ce=-(probabilities*torch.log_softmax(-score.masked_fill(~mask,1e4)*5,dim=-1)).sum(-1).mean()
    return 100*reg+rank+.1*ce,{'delta_mse':reg.detach(),'rank_loss':rank.detach()}
