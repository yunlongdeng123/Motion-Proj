"""只用 O/N/U/Q；复用r24已验证的四个hook，不加入身份/外观特征。"""
import torch
from torch import nn
from torch.nn import functional as F
from common import S
from state_adapter import StateResidualAdapter

def pack_onuq(state):
    o,n,u,q = (state[k].float() for k in ('O','N','U','Q'))
    if not all(v.shape == o.shape for v in (n,u,q)): raise ValueError('条件尺寸不同')
    if not all(torch.isfinite(v).all() for v in (o,n,u,q)): raise ValueError('非有限条件')
    if not all(((v>=0)&(v<=1)).all() for v in (o,n,u,q)): raise ValueError('条件超界')
    if not torch.allclose(o+n+u, torch.ones_like(o), atol=1e-5): raise ValueError('O/N/U必须互斥覆盖')
    if torch.any(q[u==1]!=0): raise ValueError('未知处不可有确定置信度')
    return torch.stack((o,n,u,q),1)

class SemanticAdapter(StateResidualAdapter):
    def __init__(self, width=32):
        super().__init__(width)
        self.features = nn.ModuleList([nn.Sequential(nn.Conv2d(4,width,3,padding=1),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=1),nn.SiLU()) for _ in self.channels])

    def set_condition(self, geometry, hole):
        if geometry.ndim!=4 or geometry.shape[1]!=4: raise ValueError('只接受ON UQ四通道')
        if hole.shape!=(geometry.shape[0],1,*geometry.shape[-2:]): raise ValueError('洞尺寸错误')
        if not torch.isfinite(geometry).all(): raise ValueError('非有限条件')
        self._geometry,self._hole=geometry.detach(),hole.detach();self._scales={}

    def residual(self,index,activation):
        if not self.enabled or self._geometry is None: return activation
        if activation.shape[:2]!=(self._geometry.shape[0],self.channels[index]):
            raise ValueError('使用顺序CFG；条件与帧数/通道必须一致')
        size=activation.shape[-2:]
        if size not in self._scales:
            # 类别和可信度取面积平均；不丢稀疏证据，不把一个点放大成整块已知。
            self._scales[size]=(F.adaptive_avg_pool2d(self._geometry.float(),size),
                               F.adaptive_max_pool2d(self._hole.float(),size))
        g,h=self._scales[size]
        delta=self.zero_heads[index](self.features[index](g))*h
        return activation+delta.to(activation.dtype)

def checks():
    torch.set_num_threads(1);torch.manual_seed(6201)
    a=SemanticAdapter(4);s={k:torch.zeros(2,8,8) for k in ('O','N','U','Q')};s['U'][:]=1
    s['O'][:,2:6,2:6]=1;s['U'][:,2:6,2:6]=0;s['Q'][:,2:6,2:6]=.5
    g=pack_onuq(s);h=s['O'][:,None];a.set_condition(g,h)
    x=torch.randn(2,320,8,8);assert torch.equal(x,a.residual(0,x))
    a.residual(0,x).square().mean().backward();assert a.zero_heads[0].weight.grad.abs().sum()>0
    with torch.no_grad():a.zero_heads[0].weight.fill_(.01)
    y=a.residual(0,x);assert (y-x).abs().sum()>0
    assert torch.equal(y.permute(0,2,3,1)[~h[:,0].bool()],x.permute(0,2,3,1)[~h[:,0].bool()])
    a.enabled=False;assert torch.equal(x,a.residual(0,x))
    s['N'][:]=1
    try:pack_onuq(s)
    except ValueError:pass
    else:raise AssertionError('冲突未被拒绝')
    a=SemanticAdapter();return {'zero_init_exact':True,'off_exact':True,'head_gradient':True,
        'conflict_rejected':True,'outside_hole_direct_residual_zero':True,
        'trainable_parameters':sum(p.numel() for p in a.parameters()),'channels':['O','N','U','Q']}

if __name__=='__main__':
    import json
    print(json.dumps(checks()))
