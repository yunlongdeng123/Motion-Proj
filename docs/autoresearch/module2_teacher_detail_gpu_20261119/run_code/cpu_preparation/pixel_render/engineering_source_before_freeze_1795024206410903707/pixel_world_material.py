"""CPU参考：世界材质按像素足迹读取，仅替换显式eligible原生贡献。

不是CUDA raster，不将测试GS权重称作native证据，不使用每高斯完整Sigma做像素过滤。
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as nnf


def cpu_tensor(value,dtype=torch.float64):
    if torch.is_tensor(value)and value.device.type!='cpu':raise ValueError('本轮仅CPU tensor')
    if isinstance(value,np.ndarray)and not value.flags.writeable:value=value.copy()
    return torch.as_tensor(value,dtype=dtype,device='cpu')


def bilinear(field,uv):
    field=cpu_tensor(field);uv=cpu_tensor(uv)
    if uv.requires_grad:raise ValueError('固定相机/geometry的UV不能是可训练参数')
    shape=uv.shape[:-1];h,w,c=field.shape
    grid=torch.stack((2*uv[...,0]/(w-1)-1,2*uv[...,1]/(h-1)-1),-1).reshape(1,1,-1,2)
    value=nnf.grid_sample(field.permute(2,0,1)[None],grid,mode='bilinear',padding_mode='zeros',align_corners=True)
    return value[0,:,0].T.reshape(*shape,c)


def blur(field,sigma=3.):
    r=int(np.ceil(3*sigma));x=torch.arange(-r,r+1,dtype=field.dtype)
    k=torch.exp(-.5*(x/sigma)**2);k=k/k.sum();c=field.shape[-1]
    a=field.permute(2,0,1)[None]
    a=nnf.conv2d(nnf.pad(a,(r,r,0,0),mode='replicate'),k[None,None,None,:].expand(c,1,1,-1),groups=c)
    a=nnf.conv2d(nnf.pad(a,(0,0,r,r),mode='replicate'),k[None,None,:,None].expand(c,1,-1,1),groups=c)
    return a[0].permute(1,2,0)


class SharedWorldTexture(torch.nn.Module):
    """与P相同低/高频表达；许可与最终共同base有效性由显式ledger给出。

    prior source/known/unknown只作来源，不隐式复用旧valid或旧高频latent门。
    """
    def __init__(self,base_srgb,field_valid,H_permission,*,latent_mask=None,reg_support=None,low_stride=6,low_bound=.30,high_bound=.34):
        super().__init__();base=cpu_tensor(base_srgb).clone();h,w,_=base.shape
        valid=cpu_tensor(field_valid).clone();gate=cpu_tensor(H_permission).clone()
        if valid.shape!=(h,w)or gate.shape!=(h,w):raise ValueError('世界场有效性/许可shape不同')
        if not bool(torch.isfinite(base).all())or not bool(torch.isfinite(gate).all())or bool(((gate<0)|(gate>1)).any()):raise ValueError('材质/许可非finite或越界')
        mask=cpu_tensor(np.ones((h,w))if latent_mask is None else latent_mask).clone()
        reg=cpu_tensor(mask if reg_support is None else reg_support).clone()
        for key,value in [('base',base),('valid',valid[...,None]),('H_permission',gate[...,None]),('latent_mask',mask[...,None]),('reg_support',reg[...,None])]:self.register_buffer(key,value)
        self.low_stride=int(low_stride);self.low_bound=float(low_bound);self.high_bound=float(high_bound)
        self.low=torch.nn.Parameter(torch.zeros((int(np.ceil((h-1)/low_stride))+1,int(np.ceil((w-1)/low_stride))+1,3),dtype=torch.float64))
        self.high=torch.nn.Parameter(torch.zeros_like(base))
        low_mask=torch.zeros_like(self.low[...,:1]);yy,xx=np.nonzero(mask.numpy()>0)
        for dy,dx in[(0,0),(0,1),(1,0),(1,1)]:
            y=np.minimum(yy//low_stride+dy,len(low_mask)-1);x=np.minimum(xx//low_stride+dx,low_mask.shape[1]-1)
            low_mask[y,x]=1
        self.register_buffer('low_mask',low_mask)

    @classmethod
    def from_permission_ledger(cls,path,mode='observed_plus_teacher_H',*,bound_recipe=None):
        recipe_path=Path(path).parents[2]/'protocol/bound_recipe.json'if bound_recipe is None else bound_recipe
        if isinstance(recipe_path,(str,Path)):
            if not Path(recipe_path).is_file():raise ValueError('Q入口缺中央锁定bound_recipe，不回退P历史.04')
            recipe=json.loads(Path(recipe_path).read_text())
        else:recipe=dict(recipe_path)
        if not {'low_bound','high_bound'}<=recipe.keys():raise ValueError('bound_recipe必须显式low_bound/high_bound')
        with np.load(path,allow_pickle=False)as z:
            key={'observed_H':'observed_H_prior','observed_plus_teacher_H':'mH_union'}[mode]
            return cls(z['common_base_srgb'],z['common_material_valid'],z[key],latent_mask=z['common_latent_mask'],reg_support=z['common_reg_support'],
                low_bound=recipe['low_bound'],high_bound=recipe['high_bound'])

    def fields(self):
        low=self.low_bound*torch.tanh(self.low)*self.low_mask
        raw=torch.tanh(self.high)*self.latent_mask
        high=.5*self.high_bound*(raw-blur(raw,3.))
        return low,high

    def sample(self,uv_samples,weights=None,geometric_valid=None):
        """任意固定UV采样；可给Gaussian均值积分或真正pixel足迹，调用方必须标明。

        高低频与许可在积分内；只使用共同final material-valid，不读原prior标签。
        """
        uv=cpu_tensor(uv_samples);shape=uv.shape[:-2];q=uv.shape[-2]
        weight=cpu_tensor(np.full(q,1/q)if weights is None else weights)
        if weight.ndim==1:weight=weight.expand(*shape,q)
        if geometric_valid is not None:weight=weight*cpu_tensor(geometric_valid)
        low,high=self.fields();v=bilinear(self.valid,uv)
        b=bilinear(self.base*self.valid,uv)
        L=bilinear(low,uv/self.low_stride);H=bilinear(high,uv);m=bilinear(self.H_permission,uv)
        denominator=(weight[...,None]*v).sum(-2)
        rgb=(weight[...,None]*(b+v*(L+m*H))).sum(-2)/denominator.clamp_min(1e-12)
        return rgb.clamp(0,1),denominator[...,0]

    def save_asset(self,path,*,chart,scope_world_rows,geometry_reference,source_ledger,permission_mode):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        torch.save(dict(schema='pixel_world_material_cpu_asset.v1',state=self.state_dict(),
            settings=dict(low_stride=self.low_stride,low_bound=self.low_bound,high_bound=self.high_bound)),path/'world_texture.pt')
        np.savez_compressed(path/'chart_scope.npz',**chart,eligible_world_rows=np.asarray(scope_world_rows,np.int64))
        data=dict(schema='pixel_world_material_renderer.v1',renderer_mode='eligible_native_weights_pixel_world',
            material_authority='world_texture.pt',geometry_reference=str(geometry_reference),source_ledger=str(source_ledger),
            chart_scope='chart_scope.npz',permission_mode=permission_mode,
            camera_cache_is_asset=False,new_view_or_edit_requires_fresh_native_basis=True,
            new_view_requires_true_G_s_ray_map=True,pixel_filter='pixel UV footprint adaptive box supersample; no full Gaussian Sigma',
            CPU_prototype_only=True,native_validated=False,formal_allowed=False)
        (path/'renderer.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

    @classmethod
    def load_asset(cls,path):
        path=Path(path);meta=json.loads((path/'renderer.json').read_text())
        if meta['renderer_mode']!='eligible_native_weights_pixel_world':raise ValueError('不能把pixel资产当baked平均RGB资产')
        s=torch.load(path/'world_texture.pt',map_location='cpu',weights_only=False);v=s['state']
        model=cls(v['base'],v['valid'][...,0],v['H_permission'][...,0],latent_mask=v['latent_mask'][...,0],reg_support=v['reg_support'][...,0],**s['settings'])
        model.load_state_dict(v);return model,meta


def pixel_box_samples(center_uv,jacobian_uv,*,max_axis_samples=16,samples_per_texel=2):
    """局部pixel UV微分→自适应box采样；无Gaussian尺度输入，超预算显式不合格。

    全图不预生成大矩阵，调用者按pixel chunk调用。不同n分组可由未来GPU后端实现。
    """
    uv=np.asarray(center_uv,np.float64);J=np.asarray(jacobian_uv,np.float64)
    if uv.shape[:-1]!=J.shape[:-2]or J.shape[-2:]!=(2,2):raise ValueError('pixel UV/J shape不符')
    need=np.maximum(1,np.ceil(samples_per_texel*np.max(np.linalg.norm(J,axis=-2),axis=-1)).astype(int))
    n=int(min(max_axis_samples,max(1,int(need.max()))))
    v=(np.arange(n)+.5)/n-.5;xx,yy=np.meshgrid(v,v,indexing='xy');offset=np.stack((xx.ravel(),yy.ravel()),-1)
    samples=uv[...,None,:]+np.einsum('...ij,qj->...qi',J,offset)
    return samples,np.full(n*n,1/(n*n)),dict(axis_samples=n,required_axis_max=int(need.max()),
        sampling_qualified=bool(np.all(need<=max_axis_samples)),filter='pixel box, local canonical-triangle UV differential',
        Gaussian_Sigma_applied=False)


@dataclass
class ContributionBasis:
    baseline_rgb:torch.Tensor
    eligible_rgb:torch.Tensor
    eligible_weight:torch.Tensor
    alpha:torch.Tensor
    protected:torch.Tensor
    mode:str
    evidence:str='CPU_reference_not_native'
    binding:dict=None

    def validate(self):
        for value in(self.baseline_rgb,self.eligible_rgb,self.eligible_weight,self.alpha,self.protected):
            if value.device.type!='cpu' or value.requires_grad:raise ValueError('basis/geometry必须是冻结CPU数据')
        if self.mode not in('road_only','joint'):raise ValueError('必须区分road W与joint alphaT')
        if bool((self.eligible_weight<0).any())or bool((self.eligible_weight>self.alpha+1e-12).any()):raise ValueError('eligible贡献不合法')
        if self.mode=='joint':
            if bool((self.eligible_weight[self.protected]>0).any()):raise ValueError('joint K上W_e非0：必须修world row scope，不能屏幕copy K')
            if bool((self.eligible_rgb[:,self.protected]!=0).any()):raise ValueError('joint K上eligibleRGB非0')


def shade_pixel_material(basis,world_rgb,material_valid,*,map_binding=None):
    """冻结原生权重下的deferred颜色shader；没有整road alpha贴图，也不copy联合K图。

    未知/无真surface交点只回退eligible原ci贡献；只读和非道路贡献始终保留。
    """
    basis.validate();rgb=cpu_tensor(world_rgb)
    if basis.binding is not None and basis.binding!=map_binding:raise ValueError('新camera/edit/geometry必须重取basis，禁止复用旧五帧W')
    if rgb.shape==(*basis.alpha.shape,3):rgb=rgb.movedim(-1,0)
    if rgb.shape!=basis.baseline_rgb.shape:raise ValueError('像素材质RGB shape不符')
    valid=cpu_tensor(material_valid,dtype=torch.bool)
    replacement=basis.eligible_weight[None]*rgb
    delta=torch.where(valid[None],replacement-basis.eligible_rgb,torch.zeros_like(basis.eligible_rgb))
    return basis.baseline_rgb+delta


def cpu_front_to_back(alphas,colors,eligible,*,mode='joint',protected=None,environment=None):
    """小型CPU代数参考，不模拟native tile/early-stop二进制细节。"""
    a=cpu_tensor(alphas);c=cpu_tensor(colors);e=cpu_tensor(eligible,dtype=torch.bool)
    if a.ndim!=3 or c.shape!=(len(a),3)or e.shape!=(len(a),):raise ValueError('CPU片元shape不同')
    if bool(((a<0)|(a>=1)).any()):raise ValueError('片元alpha须在[0,1)')
    before=torch.cat((torch.ones_like(a[:1]),torch.cumprod(1-a,0)[:-1]),0);w=a*before
    baseline=torch.einsum('nhw,nc->chw',w,c);selected=torch.einsum('nhw,nc->chw',w*e[:,None,None],c)
    total=1-torch.prod(1-a,0)
    if environment is not None:baseline=baseline+(1-total)[None]*cpu_tensor(environment)
    K=torch.zeros_like(total,dtype=torch.bool)if protected is None else cpu_tensor(protected,dtype=torch.bool)
    return ContributionBasis(baseline,selected,w[e].sum(0),total,K,mode),w


@dataclass(frozen=True)
class PinholeCamera:
    K:np.ndarray
    world_from_camera:np.ndarray
    width:int
    height:int
    center_offset:float=.5

    def rays(self,pixels=None):
        if pixels is None:
            x,y=np.meshgrid(np.arange(self.width),np.arange(self.height));pixels=np.stack((x,y),-1)
        p=np.asarray(pixels,np.float64)+self.center_offset
        d=np.concatenate(((p-self.K[:2,2])/np.diag(self.K)[:2],np.ones((*p.shape[:-1],1))),-1)
        return self.world_from_camera[:3,3].copy(),d@self.world_from_camera[:3,:3].T

    def plane_map(self,normal=(0,0,1),offset=0,pixels=None):
        origin,d=self.rays(pixels);n=np.asarray(normal,np.float64);den=d@n
        t=(offset-origin@n)/np.where(np.abs(den)>1e-12,den,1)
        valid=(np.abs(den)>1e-12)&(t>0);X=origin+t[...,None]*d
        R=self.world_from_camera[:3,:3];cols=[]
        for axis in(0,1):
            dr=R[:,axis]/self.K[axis,axis];dt=-t*(n@dr)/np.where(np.abs(den)>1e-12,den,1)
            cols.append((t[...,None]*dr+d*dt[...,None])[...,:2])
        return X,valid,np.stack(cols,-1)


def look_at_camera(position,target,width=96,height=64,focal=90):
    p=np.asarray(position,float);f=np.asarray(target,float)-p;f=f/np.linalg.norm(f)
    r=np.cross(f,[0,0,1]);r=r/np.linalg.norm(r);down=np.cross(f,r)
    pose=np.eye(4);pose[:3,:3]=np.stack((r,down,f),1);pose[:3,3]=p
    K=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1.]])
    return PinholeCamera(K,pose,width,height)
