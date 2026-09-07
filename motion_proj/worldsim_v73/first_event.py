"""固定射线管内几何首表面的截断NLL代理；无opacity，缺失支持仍需coverage。"""
import torch
from .surface_visibility import BeamTubeFreeSpaceLoss


class FirstSurfaceEventLoss(BeamTubeFreeSpaceLoss):
    def __init__(self,width_m=.03,resolution=32,ray_chunk=32,sigma_m=.2,cap=28.):
        super().__init__(width_m=width_m,resolution=resolution,ray_chunk=ray_chunk)
        self.sigma_m=sigma_m
        self.cap=cap

    def forward(self,vertices,faces,origins,directions,observed_first_range_m):
        count=len(origins)
        if not count or not len(faces):
            return vertices.sum()*0+(self.cap if count else 0), {
                'supervised_rays':count,'no_support_rays':count,'capped_rays':count,
                'mean_geometric_return_mass':0.}
        faces=faces.to(dtype=torch.int32).contiguous()
        terms=[]; masses=[]; missing=[]; capped=[]
        for start in range(0,count,self.ray_chunk):
            origin=origins[start:start+self.ray_chunk].float()
            direction=directions[start:start+self.ray_chunk].float()
            observed=observed_first_range_m[start:start+self.ray_chunk].float()
            reference=torch.zeros_like(direction); reference[:,2]=1
            vertical=direction[:,2].abs()>.9
            reference[vertical,2]=0; reference[vertical,1]=1
            right=torch.nn.functional.normalize(torch.cross(direction,reference,dim=-1),dim=-1)
            up=torch.cross(right,direction,dim=-1)
            relative=vertices.float()[None]-origin[:,None]
            x=(relative*right[:,None]).sum(-1)/(3*self.width_m)
            y=(relative*up[:,None]).sum(-1)/(3*self.width_m)
            distance=(relative*direction[:,None]).sum(-1)
            # 深度裁剪只用当前几何；不让target裁掉早面或选择最接近target的后面。
            far=distance.detach().amax(-1).clamp_min(0)+1.
            z=2*distance/far[:,None]-1
            clip=torch.stack([x,y,z,torch.ones_like(z)],-1).contiguous()
            raster,_=self.dr.rasterize(self.context,clip,faces,resolution=[self.resolution,self.resolution])
            valid=raster[...,3]>0
            depth,_=self.dr.interpolate(distance[...,None].contiguous(),raster,faces)
            log_value=-.5*((depth[...,0]-observed[:,None,None])/self.sigma_m).square()
            shift=log_value.detach().masked_fill(~valid,-torch.inf).amax((-2,-1))
            shift=torch.where(torch.isfinite(shift),shift,torch.zeros_like(shift))
            # 先去除无效像素的指数，避免巨大exp乘0；平移不改变有支持的对数和梯度。
            exponent=torch.where(valid,log_value-shift[:,None,None],torch.zeros_like(log_value))
            value=exponent.exp()*valid.float()
            aa=self.dr.antialias(value[...,None].contiguous(),raster,clip,faces)[...,0]
            score=(aa*self.pixel_weights).sum((-2,-1))
            log_score=(score.clamp_min(torch.finfo(score.dtype).tiny).log()+shift).clamp_max(0)
            # 数值clamp之后统一截断所有严重错配；不把这个floor改成几何透射率。
            nll=(-log_score).clamp_max(self.cap)
            terms.append(nll)
            with torch.no_grad():
                occupied=self.dr.antialias(valid.float()[...,None].contiguous(),raster,clip,faces)[...,0]
                mass=(occupied*self.pixel_weights).sum((-2,-1))
                masses.append(mass); missing.append(mass<=0); capped.append(nll.detach()>=self.cap)
        return torch.cat(terms).mean(), {
            'supervised_rays':count,'no_support_rays':torch.cat(missing).sum().item(),
            'capped_rays':torch.cat(capped).sum().item(),
            'mean_geometric_return_mass':torch.cat(masses).mean().item()}
