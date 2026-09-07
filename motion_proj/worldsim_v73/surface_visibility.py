"""同一三角表面在已观测有限射线管内的几何覆盖代理，无可学习opacity。"""
import torch
from torch import nn


class BeamTubeFreeSpaceLoss(nn.Module):
    def __init__(self,width_m=.03,resolution=32,ray_chunk=32,tolerance_m=.20,penalty='coverage'):
        super().__init__()
        import nvdiffrast.torch as dr
        self.dr=dr
        self.context=dr.RasterizeCudaContext()
        self.width_m=width_m
        self.resolution=resolution
        self.ray_chunk=ray_chunk
        self.tolerance_m=tolerance_m
        self.penalty=penalty
        axis=(torch.arange(resolution,device='cuda')+.5)/resolution*2-1
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        # 宽度是固定训练代理；有限3σ外不产生约束，不声称为传感器实测光束。
        weights=torch.exp(-4.5*(xx.square()+yy.square()))
        self.register_buffer('pixel_weights',weights/weights.sum())

    def forward(self,vertices,faces,origins,directions,observed_first_range_m):
        if not len(origins): return vertices.sum()*0
        faces=faces.to(dtype=torch.int32).contiguous()
        terms=[]
        for start in range(0,len(origins),self.ray_chunk):
            origin=origins[start:start+self.ray_chunk].float()
            direction=directions[start:start+self.ray_chunk].float()
            end=(observed_first_range_m[start:start+self.ray_chunk]-self.tolerance_m).clamp_min(1e-3)
            reference=torch.zeros_like(direction); reference[:,2]=1
            vertical=direction[:,2].abs()>.9
            reference[vertical,2]=0; reference[vertical,1]=1
            right=torch.nn.functional.normalize(torch.cross(direction,reference,dim=-1),dim=-1)
            up=torch.cross(right,direction,dim=-1)
            relative=vertices.float()[None]-origin[:,None]
            x=(relative*right[:,None]).sum(-1)/(3*self.width_m)
            y=(relative*up[:,None]).sum(-1)/(3*self.width_m)
            distance=(relative*direction[:,None]).sum(-1)
            z=2*distance/end[:,None]-1
            # 相机只读；clip的z范围恰为原始首回波前的已观测free，后方不参加。
            clip=torch.stack([x,y,z,torch.ones_like(z)],-1).contiguous()
            raster,_=self.dr.rasterize(self.context,clip,faces,resolution=[self.resolution,self.resolution])
            occupied=(raster[...,3:]>0).float()
            if self.penalty=='range':
                # 原生rast的z/w不传位置梯度；显式插值米制沿束距离获得支持内梯度。
                depth,_=self.dr.interpolate(distance[...,None].contiguous(),raster,faces)
                value=occupied*(end[:,None,None,None]-depth).clamp_min(0)
            else:
                value=occupied
            coverage=self.dr.antialias(value.contiguous(),raster,clip,faces)
            terms.append((coverage[...,0]*self.pixel_weights).sum((-2,-1)))
        # 固定footprint积分再按真实束平均；range模式单位为米，coverage为比例。
        # 重复表面不增加独立概率机会，两者均是有限宽度代理，不冒充字面中心束。
        return torch.cat(terms).mean()
