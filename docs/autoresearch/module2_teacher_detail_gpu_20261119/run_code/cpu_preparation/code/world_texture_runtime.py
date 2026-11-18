"""双读取方式共用的世界材质算子；设备通用，本轮仅CPU验收。

本模块不创建相机或native上下文，不自动训练。像素读取要消费真实原生eligible权重，
不能由整张road alpha替代。低/高频参数在世界网格共享，原始来源标签不参与隐式门控。
"""
from pathlib import Path
import copy,json
import numpy as np
import torch
import torch.nn.functional as F

def blur(field,sigma=3.):
    radius=int(np.ceil(3*sigma));x=torch.arange(-radius,radius+1,dtype=field.dtype,device=field.device)
    k=torch.exp(-.5*(x/sigma)**2);k=k/k.sum();C=field.shape[-1]
    a=field.permute(2,0,1)[None]
    a=F.conv2d(F.pad(a,(radius,radius,0,0),mode='replicate'),k[None,None,None,:].expand(C,1,1,-1),groups=C)
    a=F.conv2d(F.pad(a,(0,0,radius,radius),mode='replicate'),k[None,None,:,None].expand(C,1,-1,1),groups=C)
    return a[0].permute(1,2,0)

def sample_bilinear(field,uv):
    if uv.requires_grad:raise ValueError('本协议的几何/相机/UV均冻结')
    H,W,_=field.shape;shape=uv.shape[:-1]
    u=uv.to(device=field.device,dtype=torch.float64)
    grid=torch.stack((u[...,0]*2/(W-1)-1,u[...,1]*2/(H-1)-1),-1).reshape(1,1,-1,2)
    value=F.grid_sample(field.to(torch.float64).permute(2,0,1)[None],grid,align_corners=True,padding_mode='zeros')
    return value[0,:,0].T.reshape(*shape,field.shape[-1])

class WorldTextureRuntime(torch.nn.Module):
    def __init__(self,base,valid,H_gate,latent_mask,reg_support,*,low_mask=None,low_stride=6,low_bound=.30,high_bound=.34):
        super().__init__();base=np.asarray(base);H,W,C=base.shape
        if C!=3 or H<2 or W<2:raise ValueError('世界材质须为至少2×2的HWC RGB')
        for key,value in [('base',base),('valid',np.asarray(valid)[...,None]),('H_gate',np.asarray(H_gate)[...,None])]:
            self.register_buffer(key,torch.as_tensor(np.array(value,copy=True),dtype=torch.float64))
        for key,value in [('latent_mask',latent_mask),('reg_support',reg_support)]:
            self.register_buffer(key,torch.as_tensor(np.asarray(value)[...,None].copy(),dtype=torch.float32))
        self.low_stride=int(low_stride);self.low_bound=float(low_bound);self.high_bound=float(high_bound)
        self.low=torch.nn.Parameter(torch.zeros((int(np.ceil((H-1)/low_stride))+1,int(np.ceil((W-1)/low_stride))+1,3)))
        self.high=torch.nn.Parameter(torch.zeros((H,W,3)))
        if low_mask is None:raise ValueError('必须显式提供跨arms相同的common low mask')
        self.register_buffer('low_mask',torch.as_tensor(np.asarray(low_mask)[...,None].copy(),dtype=torch.float32))
        if self.low_mask.shape!=self.low[...,:1].shape:raise ValueError('common low mask shape错误')
        for name,buffer in self.named_buffers():
            if not bool(torch.isfinite(buffer).all())or bool(((buffer<0)|(buffer>1)).any()):raise ValueError('非法材质/门/支持：'+name)
        if self.valid.shape!=(H,W,1)or self.H_gate.shape!=(H,W,1)or self.latent_mask.shape!=(H,W,1)or self.reg_support.shape!=(H,W,1):raise ValueError('世界场shape不一致')

    def fields(self):
        L=self.low_bound*torch.tanh(self.low)*self.low_mask
        raw=torch.tanh(self.high)*self.latent_mask
        H=.5*self.high_bound*(raw-blur(raw,3.))
        return L,H

    def set_stage(self,stage):
        if stage not in ['A','B','C']:raise ValueError(stage)
        self.low.requires_grad_(stage in ['A','C']);self.high.requires_grad_(stage in ['B','C'])

    def sample(self,uv,weights,geometric_valid=None):
        """同一算子读Gaussian核或pixel核；调用者提供且冻结真实采样计划。"""
        u=torch.as_tensor(uv,device=self.base.device,dtype=torch.float64)
        w=torch.as_tensor(weights,device=u.device,dtype=torch.float64)
        if w.requires_grad:raise ValueError('读取核权重必须冻结')
        if w.ndim==1:w=w.expand(u.shape[:-1])
        if geometric_valid is not None:
            geometric=torch.as_tensor(geometric_valid,device=u.device,dtype=torch.float64)
            if geometric.requires_grad:raise ValueError('几何有效性必须冻结')
            w=w*geometric
        L,H=self.fields();v=sample_bilinear(self.valid,u)
        b=sample_bilinear(self.base*self.valid,u)
        l=sample_bilinear(L,u/self.low_stride);h=sample_bilinear(H,u);g=sample_bilinear(self.H_gate,u)
        mass=(w[...,None]*v).sum(-2)
        color=(w[...,None]*(b+v*(l+g*h))).sum(-2)/mass.clamp_min(1e-12)
        return color.clamp(0,1).to(torch.float32),mass[...,0]

    def regularizers(self):
        L,H=self.fields();R=self.reg_support;den=R.sum().clamp_min(1)
        effective=self.H_gate.to(H.dtype)*H
        mx=self.low_mask[:,1:,0]*self.low_mask[:,:-1,0];my=self.low_mask[1:,:,0]*self.low_mask[:-1,:,0]
        low=((L[:,1:]-L[:,:-1]).square()*mx[...,None]).sum()+((L[1:]-L[:-1]).square()*my[...,None]).sum()
        return dict(low_adjacent_delta_squared=low/(3*(mx.sum()+my.sum()).clamp_min(1)),
            effective_high_energy=(effective.square()*R).sum()/(3*den),
            effective_high_lowfrequency_leakage=(blur(effective).square()*R).sum()/(3*den),
            effective_high_weighted_DC=((effective*R).sum((0,1))/den).square().mean())

    def save(self,path,metadata):
        torch.save(dict(schema='teacher_world_texture_runtime.v1',state=self.state_dict(),metadata=metadata,
            settings=dict(low_stride=self.low_stride,low_bound=self.low_bound,high_bound=self.high_bound)),Path(path))

    @classmethod
    def load(cls,path,device='cpu'):
        saved=torch.load(path,map_location='cpu',weights_only=False);s=saved['state']
        model=cls(s['base'].numpy(),s['valid'][...,0].numpy(),s['H_gate'][...,0].numpy(),
            s['latent_mask'][...,0].numpy(),s['reg_support'][...,0].numpy(),low_mask=s['low_mask'][...,0].numpy(),**saved['settings'])
        model.load_state_dict(s);return model.to(device),saved['metadata']

def compose_native_eligible(baseline,eligible_RGB,eligible_weight,world_RGB,valid):
    """低层代数；调用方须先认证basis的reader/camera/scope/K，不能据此宣称native通过。"""
    if any(x.requires_grad for x in (baseline,eligible_RGB,eligible_weight)):raise ValueError('原生basis必须冻结')
    replacement=eligible_weight[None]*world_RGB.movedim(-1,0)
    delta=torch.where(valid[None],replacement-eligible_RGB,torch.zeros_like(eligible_RGB))
    return baseline+delta

class FixedPixelReadLayout(torch.nn.Module):
    """采样按(nx,ny)分组后必须按pixel ID散射，不能reshape造成错位/条纹。"""
    def __init__(self,row_ids,mass,height,width,*,row_id_kind):
        super().__init__()
        if row_id_kind!='pixel_flat':raise ValueError('像素读取必须显式pixel_flat身份')
        ids=np.asarray(row_ids,np.int64);m=np.asarray(mass)
        if ids.ndim!=1 or len(ids)!=len(m)or len(np.unique(ids))!=len(ids)or np.any(ids<0)or np.any(ids>=height*width):raise ValueError('像素row身份错误')
        self.height=int(height);self.width=int(width);self.register_buffer('pixel_rows',torch.from_numpy(ids.copy()))
        valid=np.zeros(height*width,bool);valid[ids]=m>1e-12
        self.register_buffer('material_valid_grid',torch.from_numpy(valid.reshape(height,width)))

    def forward(self,values):
        if values.shape!=(len(self.pixel_rows),3):raise ValueError('像素读取颜色数不同')
        field=values.new_zeros((self.height*self.width,3)).index_copy(0,self.pixel_rows,values)
        return field.reshape(self.height,self.width,3),self.material_valid_grid

class FixedGaussianReadLayout(torch.nn.Module):
    """平均读取row_ids是完整拓扑global world ID，不能直接当active数组下标。"""
    def __init__(self,row_ids,active_world_rows,*,row_id_kind):
        super().__init__()
        if row_id_kind!='global_world_row':raise ValueError('高斯读取必须显式global_world_row身份')
        active=np.asarray(active_world_rows,np.int64);ids=np.asarray(row_ids,np.int64)
        if active.ndim!=1 or np.any(np.diff(active)<=0)or len(np.unique(ids))!=len(ids):raise ValueError('active/world身份顺序错误')
        pos=np.searchsorted(active,ids)
        if np.any(pos>=len(active))or not np.array_equal(active[pos],ids):raise ValueError('有world ID不在active集合中')
        self.active_count=len(active);self.register_buffer('active_positions',torch.from_numpy(pos.copy()))

    def forward(self,readonly_colors,values):
        if readonly_colors.shape!=(self.active_count,3)or values.shape!=(len(self.active_positions),3):raise ValueError('高斯颜色数不同')
        return readonly_colors.clone().index_copy(0,self.active_positions,values)

def restore_common_A(common,model,optimizer,*,reader,expected_identity,high_seed=None):
    """只恢复参数/Adam，不加载arm的门控buffer；防止新teacher门被旧snapshot覆盖。"""
    if common['stage']!='A'or common['cursor']!=200 or common['reader']!=reader or common['identity']!=expected_identity:raise ValueError('不是同reader/同输入的A200')
    if bool((common['params']['high']!=0).any()):raise ValueError('共用A前缀必须保持H=0')
    for name in ['low','high']:
        p=getattr(model,name);x=common['params'][name]
        if p.shape!=x.shape:raise ValueError('common参数shape不同')
        with torch.no_grad():p.copy_(x.to(p.device))
    optimizer.load_state_dict(copy.deepcopy(common['optimizer']))
    if model.high in optimizer.state and optimizer.state[model.high]:raise ValueError('A阶段不应有H Adam moments')
    if high_seed is not None:
        seed=torch.as_tensor(high_seed,device=model.high.device,dtype=model.high.dtype)
        if seed.shape!=model.high.shape or not bool(torch.isfinite(seed).all()):raise ValueError('HF seed shape/finite不同')
        with torch.no_grad():model.high.copy_(seed)
    return dict(cursor=200,trace=common['trace'].copy(),RNG=copy.deepcopy(common['RNG']),
        reader=reader,arm_buffers_preserved=True,T2_H_seed_only=high_seed is not None)
