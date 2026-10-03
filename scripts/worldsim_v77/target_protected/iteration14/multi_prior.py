"""真正进入UNet的多先验支路：参考RGB交叉注意力、BEV编码、2D控制图。

原始DriveEditor参数不改；零初始化保证接入前后等价。CPU合同不替代真实GPU验收。
"""
import torch
from torch import nn
from torch.nn import functional as F


class MultiPriorBranch(nn.Module):
    channels = (320,640,1280,1280)
    blocks = (2,5,8,11)

    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.bev_encoder = nn.Sequential(nn.Conv2d(10,width,3,padding=1),nn.GroupNorm(4,width),
            nn.SiLU(),nn.Conv2d(width,width,3,padding=1),nn.SiLU())
        self.geometry_encoder = nn.Sequential(nn.Conv2d(12,width,3,padding=1),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=1),nn.SiLU())
        self.reference_encoder = nn.Sequential(nn.Conv2d(4,width,3,padding=1),nn.SiLU())
        self.reference_pose = nn.Sequential(nn.Linear(13,width),nn.SiLU(),nn.Linear(width,width))
        self.query_time = nn.Linear(1,width)
        self.parameters_encoder = nn.Linear(4,width)
        self.queries = nn.ModuleList(nn.Conv2d(c,width,1) for c in self.channels)
        self.attention = nn.ModuleList(nn.MultiheadAttention(width,4,batch_first=True) for _ in self.channels)
        self.geometry_heads = nn.ModuleList(nn.Conv2d(width,c,1) for c in self.channels)
        self.rgb_heads = nn.ModuleList(nn.Conv2d(width,c,1) for c in self.channels)
        for head in list(self.geometry_heads)+list(self.rgb_heads):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        self.enabled = True
        self.use_rgb = True
        self.use_geometry = True
        self._condition = None
        self._handles = []

    def set_condition(self, *, geometry, bev, bev_rays, reference_latents, reference_valid,
                      reference_pose, query_time, parameters, hole):
        """只接已擦除参考的冻结VAE latent；禁止在本支路偷偷读取原始RGB/GT。"""
        t = geometry.shape[0]
        if geometry.ndim!=4 or geometry.shape[1]!=12 or bev.shape[:2]!=(t,10):
            raise ValueError('几何/BEV通道不匹配')
        if hole.shape!=(t,1,*geometry.shape[-2:]): raise ValueError('独立编辑mask尺寸不匹配')
        if bev_rays.shape[:2]!=(t,4) or bev_rays.shape[-1]!=2: raise ValueError('BEV采样必须保留4个深度假设')
        r = reference_latents.shape[0]
        if reference_latents.ndim!=4 or reference_latents.shape[1]!=4 or reference_pose.shape!=(r,13):
            raise ValueError('参考VAE latent/pose形状错误')
        if reference_valid.shape!=(r,1,*reference_latents.shape[-2:]): raise ValueError('参考有效范围缺失')
        if query_time.shape!=(t,1) or parameters.shape!=(t,4): raise ValueError('时间/参数字段错误')
        items = dict(geometry=geometry,bev=bev,bev_rays=bev_rays,reference_latents=reference_latents,
            reference_valid=reference_valid,reference_pose=reference_pose,query_time=query_time,
            parameters=parameters,hole=hole)
        if not all(torch.isfinite(x).all() for x in items.values()): raise ValueError('非有限输入')
        self._condition = {k:v.detach() for k,v in items.items()}

    def residual(self, index, activation):
        if not self.enabled or self._condition is None: return activation
        c = self._condition
        t = c['geometry'].shape[0]
        if activation.shape[:2]!=(t,self.channels[index]):
            raise ValueError('使用顺序CFG，禁止把不同时间的条件广播到错误批次')
        # 注意力固定最多18x32个query；避免在全分辨率像素上制造昂贵全连接。
        size = (min(18,activation.shape[-2]),min(32,activation.shape[-1]))
        hole = F.adaptive_max_pool2d(c['hole'].float(),activation.shape[-2:])
        delta = torch.zeros_like(activation, dtype=torch.float32)
        params = self.parameters_encoder(c['parameters'].float())[:,:,None,None]
        if self.use_geometry:
            g = self.geometry_encoder(F.adaptive_avg_pool2d(c['geometry'].float(),size))
            b = self.bev_encoder(c['bev'].float())
            rays = c['bev_rays'].float()
            rays = F.interpolate(rays.permute(0,1,4,2,3).flatten(0,1),size=size,mode='bilinear',align_corners=False)
            rays = rays.reshape(t,4,2,*size).permute(0,1,3,4,2)
            # 查询像素射线上的多个距离，不能把未知像素强行压在一个假道路平面上。
            sampled = F.grid_sample(b,rays.reshape(t,4*size[0],size[1],2),align_corners=False)
            sampled = sampled.reshape(t,self.width,4,*size).mean(2)
            out = self.geometry_heads[index](g+sampled+params)
            delta = delta+F.interpolate(out,size=activation.shape[-2:],mode='bilinear',align_corners=False)
        if self.use_rgb:
            features = self.reference_encoder(c['reference_latents'].float())
            features = F.adaptive_avg_pool2d(features,(8,8)).flatten(2).transpose(1,2)
            features = features+self.reference_pose(c['reference_pose'].float())[:,None,:]
            valid = F.adaptive_avg_pool2d(c['reference_valid'].float(),(8,8)).flatten()>0.95
            tokens = features.reshape(1,-1,self.width).expand(t,-1,-1)
            # 一个恒零的空token使全遮挡参考仍有定义，不把灰块当真实车辆外观。
            tokens = torch.cat((tokens,torch.zeros(t,1,self.width,device=tokens.device)),1)
            invalid = torch.cat((~valid,torch.zeros(1,dtype=torch.bool,device=valid.device)))[None].expand(t,-1)
            query = self.queries[index](F.adaptive_avg_pool2d(activation.float(),size)).flatten(2).transpose(1,2)
            query = query+self.query_time(c['query_time'].float())[:,None,:]
            attn,_ = self.attention[index](query,tokens,tokens,key_padding_mask=invalid,need_weights=False)
            out = self.rgb_heads[index](attn.transpose(1,2).reshape(t,self.width,*size)+params)
            delta = delta+F.interpolate(out,size=activation.shape[-2:],mode='bilinear',align_corners=False)
        return activation+(delta*hole).to(activation.dtype)

    def attach(self, unet):
        if self._handles: raise RuntimeError('禁止重复挂载支路')
        for index, block in enumerate(self.blocks):
            if block>=len(unet.input_blocks): raise ValueError('官方UNet结构与预期不符')
            self._handles.append(unet.input_blocks[block].register_forward_hook(
                lambda module,args,output,i=index:self.residual(i,output)))

    def detach(self):
        for handle in self._handles: handle.remove()
        self._handles = []


def checks():
    torch.set_num_threads(1);torch.manual_seed(6201)
    branch = MultiPriorBranch(8)
    g = torch.zeros(2,12,16,24);g[:,2]=1
    h = torch.zeros(2,1,16,24);h[:,:,4:12,5:18]=1
    cond = dict(geometry=g,bev=torch.rand(2,10,24,24),bev_rays=torch.rand(2,4,8,12,2)*2-1,
        reference_latents=torch.rand(3,4,8,8),reference_valid=torch.ones(3,1,8,8),
        reference_pose=torch.rand(3,13),query_time=torch.tensor([[0.],[.9]]),parameters=torch.ones(2,4),hole=h)
    branch.set_condition(**cond)
    x = torch.randn(2,320,16,24)
    assert torch.equal(branch.residual(0,x),x)
    branch.residual(0,x).square().mean().backward()
    assert branch.geometry_heads[0].weight.grad.abs().sum()>0 and branch.rgb_heads[0].weight.grad.abs().sum()>0
    with torch.no_grad():
        branch.geometry_heads[0].weight.fill_(.01);branch.rgb_heads[0].weight.fill_(.01)
    branch.zero_grad();a=branch.residual(0,x);a.square().mean().backward()
    assert branch.bev_encoder[0].weight.grad.abs().sum()>0
    assert branch.reference_encoder[0].weight.grad.abs().sum()>0
    outside = ~h[:,0].bool()
    assert torch.equal(a.permute(0,2,3,1)[outside],x.permute(0,2,3,1)[outside])
    branch.set_condition(**(cond|{'reference_latents':cond['reference_latents']+1}))
    assert not torch.equal(a,branch.residual(0,x))
    branch.use_rgb=False
    geometry_only = branch.residual(0,x)
    branch.set_condition(**(cond|{'reference_latents':cond['reference_latents']*0}))
    assert torch.equal(geometry_only,branch.residual(0,x))
    branch.use_rgb=True;branch.use_geometry=False
    rgb_only = branch.residual(0,x)
    branch.set_condition(**(cond|{'reference_latents':cond['reference_latents']*0,'bev':cond['bev']*0}))
    assert torch.equal(rgb_only,branch.residual(0,x))
    branch.enabled=False;assert torch.equal(x,branch.residual(0,x))
    try: branch.set_condition(**(cond|{'reference_latents':torch.full_like(cond['reference_latents'],float('nan'))}))
    except ValueError: pass
    else: raise AssertionError('未拒绝非有限参考')
    return {'zero_init_exact':True,'off_exact':True,'RGB_and_BEV_have_gradient_after_head_opens':True,
        'reference_change_reaches_output':True,'ablation_routes_independent':True,
        'outside_mask_direct_residual_zero':True,'nonfinite_rejected':True,
        'tested_on':'CPU controlled activations; not official full-model forward/backward',
        'parameters':sum(p.numel() for p in MultiPriorBranch().parameters()),'new_backbones':'RGB latent encoder + lightweight BEV CNN'}


if __name__=='__main__':
    import json
    print(json.dumps(checks()))
