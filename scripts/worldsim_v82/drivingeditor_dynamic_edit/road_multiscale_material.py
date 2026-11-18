"""固定Σ的共享世界材质：GH15[A0 + ΔL + mH·ΔH]；无每Gaussian可训练RGB。"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_plan(world_xyz,normal_world,sigma_m,world_from_uv,order=15):
    """Σ先投影到道路切平面，再映到既有世界XY atlas坐标；法向厚度不入颜色核。"""
    if order not in (7,11,15,21,41):raise ValueError('正式候选GH15；其它阶数只作CPU积分参照')
    xyz=np.asarray(world_xyz,np.float64);normal=np.asarray(normal_world,np.float64)
    normal=normal/np.linalg.norm(normal,axis=1,keepdims=True);sigma=np.asarray(sigma_m,np.float64)
    if not np.allclose(sigma[:,0],sigma[:,1],atol=1e-12,rtol=0):raise ValueError('当前道路要求两个切向sigma相等；不得忽略非轴对称Σ')
    A=np.asarray(world_from_uv,np.float64);J=np.linalg.inv(A[:2,:2])
    center=(xyz[:,:2]-A[:2,3])@J.T
    covariance=sigma[:,0,None,None]**2*(np.eye(2)[None]-normal[:,:2,None]*normal[:,None,:2])
    covariance=np.einsum('ab,nbc,dc->nad',J,covariance,J)
    chol=np.linalg.cholesky(covariance+np.eye(2)*1e-14)
    x,w=np.polynomial.hermite.hermgauss(order)
    xx,yy=np.meshgrid(x,x,indexing='xy');wx,wy=np.meshgrid(w,w,indexing='xy')
    return dict(center_uv=center,chol_uv=chol,standard_nodes=np.stack([xx.ravel(),yy.ravel()],1)*np.sqrt(2),
                weights=(wx*wy).ravel()/np.pi,world_from_uv=A,order=order)


def sample_field(field,uv):
    """float64 UV与bilinear weights；field HWC，返回 N×Q×C。"""
    H,W,C=field.shape
    grid=torch.stack((uv[...,0]*2/(W-1)-1,uv[...,1]*2/(H-1)-1),-1)[None]
    value=F.grid_sample(field.to(torch.float64).permute(2,0,1)[None],grid,
        mode='bilinear',padding_mode='zeros',align_corners=True)
    return value[0].permute(1,2,0)


def blur(field,sigma_cells):
    radius=int(np.ceil(3*sigma_cells));x=torch.arange(-radius,radius+1,device=field.device,dtype=field.dtype)
    k=torch.exp(-.5*(x/sigma_cells)**2);k=k/k.sum();C=field.shape[-1]
    a=field.permute(2,0,1)[None]
    a=F.conv2d(F.pad(a,(radius,radius,0,0),mode='replicate'),k[None,None,None,:].expand(C,1,1,-1),groups=C)
    a=F.conv2d(F.pad(a,(0,0,radius,radius),mode='replicate'),k[None,None,:,None].expand(C,1,-1,1),groups=C)
    return a[0].permute(1,2,0)


class WorldMaterialField(torch.nn.Module):
    def __init__(self,A0,valid,mH,plan,*,low_stride=6,low_bound=.30,high_bound=.04,highpass_sigma_cells=3.):
        super().__init__()
        for key,value in [('A0',A0),('valid',np.asarray(valid)[...,None]),('mH',np.asarray(mH)[...,None]),
                          ('center_uv',plan['center_uv']),('chol_uv',plan['chol_uv']),
                          ('nodes',plan['standard_nodes']),('weights',plan['weights'])]:
            self.register_buffer(key,torch.as_tensor(np.array(value,copy=True),dtype=torch.float64))
        H,W,_=np.asarray(A0).shape
        self.low=torch.nn.Parameter(torch.zeros((int(np.ceil((H-1)/low_stride))+1,int(np.ceil((W-1)/low_stride))+1,3)))
        self.high=torch.nn.Parameter(torch.zeros((H,W,3)))
        self.low_stride=low_stride;self.low_bound=low_bound;self.high_bound=high_bound
        self.highpass_sigma_cells=highpass_sigma_cells
        self.register_buffer('high_parameter_mask',(self.mH>0).to(torch.float32))
        self.register_buffer('high_output_mask',(self.mH>0).to(torch.float32))
        self.register_buffer('low_parameter_mask',torch.ones_like(self.low[...,:1]))
        if 'sample_uv' in plan:
            self.register_buffer('sample_uv',torch.as_tensor(np.array(plan['sample_uv'],copy=True)))
            self.register_buffer('geometric_sample_valid',torch.as_tensor(np.array(plan.get('geometric_sample_valid',np.ones(plan['sample_uv'].shape[:2],bool)),copy=True),dtype=torch.bool))

    def set_stage(self,stage):
        if stage not in ('A','B','C','L'):raise ValueError(stage)
        self.low.requires_grad_(stage in ('A','C','L'));self.high.requires_grad_(stage in ('B','C'))

    def fields(self):
        low=self.low_bound*torch.tanh(self.low)*self.low_parameter_mask
        raw=torch.tanh(self.high)*self.high_parameter_mask
        high=.5*self.high_bound*(raw-blur(raw,self.highpass_sigma_cells))
        return low,high

    def query(self,*,chunk=2048,center_only=False,indices=None):
        low,high=self.fields();result=[];mass=[]
        if indices is None:indices=torch.arange(len(self.center_uv),device=self.center_uv.device)
        for begin in range(0,len(indices),chunk):
            row=indices[begin:begin+chunk];center=self.center_uv[row]
            if center_only:
                uv=center[:,None,:];weight=self.weights.new_ones(1)
            else:
                uv=self.sample_uv[row].to(torch.float64) if hasattr(self,'sample_uv') else center[:,None,:]+torch.einsum('nij,qj->nqi',self.chol_uv[row],self.nodes)
                weight=self.weights
            valid=sample_field(self.valid,uv)
            base=sample_field(self.A0*self.valid,uv)
            if not center_only and hasattr(self,'geometric_sample_valid'):
                geometric=self.geometric_sample_valid[row,...,None]
                valid=valid*geometric;base=base*geometric
            L=sample_field(low,uv/self.low_stride)
            H=sample_field(high,uv)
            gate=sample_field(self.mH,uv)
            denominator=(weight[None,:,None]*valid).sum(1)
            # 缺材质部分不把占位RGB纳入；保留有效核质量供资格/回退审计。
            color=(weight[None,:,None]*(base+valid*(L+gate*H))).sum(1)/denominator.clamp_min(1e-12)
            result.append(color.clamp(0,1).to(torch.float32));mass.append(denominator[:,0])
        return torch.cat(result),torch.cat(mass)

    def effective_high_statistics(self,support=None):
        _,h=self.fields();effective=self.mH.to(h.dtype)*h
        weight=self.high_output_mask if support is None else support
        denominator=weight.sum().clamp_min(1)
        return dict(effective_high_DC=(effective*weight).sum((0,1))/denominator,
            effective_high_energy=(effective.square()*weight).sum()/(3*denominator),
            effective_high_lowfrequency_energy=(blur(effective,self.highpass_sigma_cells).square()*weight).sum()/(3*denominator))

    def save(self,path,metadata):
        """权威共享field与核；派生ci不作为可训练参数。"""
        torch.save(dict(schema='world_multiscale_material.v1',state=self.state_dict(),metadata=metadata,
            settings=dict(low_stride=self.low_stride,low_bound=self.low_bound,high_bound=self.high_bound,
                          highpass_sigma_cells=self.highpass_sigma_cells)),Path(path))

    @classmethod
    def load(cls,path,*,device='cpu'):
        saved=torch.load(path,map_location='cpu',weights_only=False);s=saved['state']
        plan=dict(center_uv=s['center_uv'].numpy(),chol_uv=s['chol_uv'].numpy(),standard_nodes=s['nodes'].numpy(),weights=s['weights'].numpy())
        if 'sample_uv' in s:plan['sample_uv']=s['sample_uv'].numpy()
        if 'geometric_sample_valid' in s:plan['geometric_sample_valid']=s['geometric_sample_valid'].numpy()
        model=cls(s['A0'].numpy(),s['valid'][...,0].numpy(),s['mH'][...,0].numpy(),plan,**saved['settings'])
        model.load_state_dict(s);return model.to(device),saved['metadata']
