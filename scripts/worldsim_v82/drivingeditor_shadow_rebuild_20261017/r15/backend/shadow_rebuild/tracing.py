"""直接积分 Actor 3D 高斯密度，太阳射线和相机共享几何与 opacity。

不是 IRGS 的 OptiX/2DGS 完整传输实现：这里实现各向异性 3D 密度的解析
半射线积分与有限太阳盘。没有学习五盒体、独立投影纹理或深度修正网络。
opacity 转体密度采用最薄轴穿心光学厚度匹配；这是显式近似，须 GPU 实验验证。
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .math3d import disk_directions, normalize, logit


def gaussian_ray_integral(origins, directions, means, covariances, amplitude, lower=0.0, upper=None):
    """返回 [R,G] 光学厚度；方向必须单位化，距离与尺度都为米。

    rho(x)=amplitude*exp(-.5*(x-mu)^T inv(Sigma) (x-mu))。
    使用 erfc 避免正尾部 erf 相减消失；无射线离散采样误差。
    """
    return _precision_integral(origins,directions,means,torch.linalg.inv(covariances),amplitude,lower,upper)


def _precision_integral(origins,directions,means,precision,amplitude,lower=0.,upper=None):
    difference = origins[:, None]-means[None]
    a = torch.einsum("ri,gij,rj->rg", directions, precision, directions).clamp_min(1e-12)
    b = torch.einsum("ri,gij,rgj->rg", directions, precision, difference)
    c = torch.einsum("rgi,gij,rgj->rg", difference, precision, difference)
    transverse = (c-b.square()/a).clamp_min(0)
    lo = torch.sqrt(a/2)*(torch.as_tensor(lower, device=a.device, dtype=a.dtype)+b/a)
    if upper is None:
        cdf = .5*torch.erfc(lo)
    else:
        hi = torch.sqrt(a/2)*(torch.as_tensor(upper, device=a.device, dtype=a.dtype)+b/a)
        # 两个正尾用 erfc；两个负尾用反射，跨零直接相减。
        cdf = .5*torch.where(lo >= 0, torch.erfc(lo)-torch.erfc(hi),
                            torch.where(hi <= 0, torch.erfc(-hi)-torch.erfc(-lo), torch.erf(hi)-torch.erf(lo)))
    return amplitude[None]*torch.exp(-.5*transverse)*torch.sqrt(2*math.pi/a)*cdf.clamp_min(0)


def _ray_aabb(origins, directions, low, high):
    # detached broad phase 只剔除聚合 6-sigma AABB；完整解析积分保留细阶段梯度。
    parallel = directions.abs() < 1e-10
    safe = torch.where(parallel, torch.ones_like(directions), directions)
    t1, t2 = (low-origins)/safe, (high-origins)/safe
    near = torch.minimum(t1, t2).masked_fill(parallel, -torch.inf).amax(-1).clamp_min(0)
    far = torch.maximum(t1, t2).masked_fill(parallel, torch.inf).amin(-1)
    invalid = (parallel & ((origins < low) | (origins > high))).any(-1)
    return (far >= near) & ~invalid


def dense_geometry_transmittance(origins, directions, geometry, ray_chunk=256, gaussian_chunk=512, cull=True):
    mask = geometry["owner"] >= 0
    if not bool(mask.any()):
        return origins[:, 0]*0+1
    means, cov = geometry["xyz"][mask], geometry["cov"][mask]
    opacity = geometry["opacity"][mask, 0].clamp(1e-6, 1-1e-6)
    sigma_min = torch.linalg.eigvalsh(cov).amin(-1).clamp_min(1e-10).sqrt()
    amplitude = -torch.log1p(-opacity)/(math.sqrt(2*math.pi)*sigma_min)
    values = []
    for r0 in range(0, len(origins), ray_chunk):
        o, d = origins[r0:r0+ray_chunk], directions[r0:r0+ray_chunk]
        tau = o[:, 0]*0
        for g0 in range(0, len(means), gaussian_chunk):
            m, s, amp = means[g0:g0+gaussian_chunk], cov[g0:g0+gaussian_chunk], amplitude[g0:g0+gaussian_chunk]
            with torch.no_grad():
                radius = 6*torch.diagonal(s.detach(), dim1=-2, dim2=-1).sqrt()
                hit = _ray_aabb(o.detach(), d.detach(), (m.detach()-radius).amin(0), (m.detach()+radius).amax(0)) if cull else torch.ones(len(o), dtype=torch.bool, device=o.device)
            ids = hit.nonzero().flatten()
            if not len(ids):
                continue
            def integrate(oo, dd, mm, ss, aa):
                return gaussian_ray_integral(oo, dd, mm, ss, aa).sum(-1)
            args = (o[ids], d[ids], m, s, amp)
            # 大场景只保留分块输入，反传重算积分中间项，避免 R*G*太阳采样数常驻显存。
            if torch.is_grad_enabled() and len(ids)*len(m) > 16384 and any(a.requires_grad for a in args):
                contribution = checkpoint(integrate, *args, use_reentrant=False)
            else:
                contribution = integrate(*args)
            tau = tau.index_add(0, ids, contribution)
        values.append(torch.exp(-tau.clamp_max(80)))
    return torch.cat(values) if values else origins.new_empty(0)


def geometry_transmittance(origins,directions,geometry,ray_chunk=256,gaussian_chunk=512,cull=True):
    """先按真实Actor粗筛射线，再沿其高斯块积分；同一帧复用精度矩阵。"""
    if not cull:
        return dense_geometry_transmittance(origins,directions,geometry,ray_chunk,gaussian_chunk,False)
    groups=getattr(geometry,'shadow_groups',None)
    if groups is None:
        groups=[]
        owner=geometry['owner'];selected=owner>=0
        if selected.any():
            means=geometry['xyz'][selected];cov=geometry['cov'][selected];owners=owner[selected]
            precision=torch.linalg.inv(cov)
            sigma=geometry['minimum_scale'][selected] if 'minimum_scale' in geometry else torch.linalg.eigvalsh(cov).amin(-1).clamp_min(1e-10).sqrt()
            opacity=geometry['opacity'][selected,0].clamp(1e-6,1-1e-6)
            amplitude=-torch.log1p(-opacity)/(math.sqrt(2*math.pi)*sigma)
            radius=6*torch.diagonal(cov.detach(),dim1=-2,dim2=-1).sqrt()
            for actor in owners.unique().tolist():
                ids=(owners==actor).nonzero().flatten()
                lo=means[ids].detach()-radius[ids];hi=means[ids].detach()+radius[ids]
                groups.append((means[ids],precision[ids],amplitude[ids],lo,hi))
        if hasattr(geometry,'__dict__'):geometry.shadow_groups=groups
    tau=origins[:,0]*0
    for means,precision,amplitude,lo,hi in groups:
        hit=_ray_aabb(origins.detach(),directions.detach(),lo.amin(0),hi.amax(0))
        actor_rays=hit.nonzero().flatten()
        for r0 in range(0,len(actor_rays),ray_chunk):
            rays=actor_rays[r0:r0+ray_chunk];o=origins[rays];d=directions[rays]
            subtotal=o[:,0]*0
            for g0 in range(0,len(means),gaussian_chunk):
                mask=_ray_aabb(o.detach(),d.detach(),lo[g0:g0+gaussian_chunk].amin(0),hi[g0:g0+gaussian_chunk].amax(0))
                active=mask.nonzero().flatten()
                if not len(active):continue
                args=(o[active],d[active],means[g0:g0+gaussian_chunk],precision[g0:g0+gaussian_chunk],amplitude[g0:g0+gaussian_chunk])
                def integrate(oo,dd,mm,pp,aa):return _precision_integral(oo,dd,mm,pp,aa).sum(-1)
                if torch.is_grad_enabled() and len(active)*len(args[2])>16384 and any(a.requires_grad for a in args):
                    contribution=checkpoint(integrate,*args,use_reentrant=False)
                else:contribution=integrate(*args)
                subtotal=subtotal.index_add(0,active,contribution)
            tau=tau.index_add(0,rays,subtotal)
    return torch.exp(-tau.clamp_max(80))


class SunAmbient(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.azimuth = nn.Parameter(torch.tensor(math.radians(config.sun_azimuth_degrees)))
        elevation = (config.sun_elevation_degrees-5)/80
        self.elevation_logit = nn.Parameter(logit(torch.tensor(elevation)))
        self.ambient_logit = nn.Parameter(logit(torch.full((3,), config.ambient_fraction)))
        self.radius_logit = nn.Parameter(logit(torch.tensor(min(.99, config.sun_angular_radius_degrees/5))))
        self.strength_logit = nn.Parameter(logit(torch.tensor(config.shadow_strength)))

    def direction(self):
        e = torch.deg2rad(5+80*self.elevation_logit.sigmoid())
        a = self.azimuth
        return torch.stack((e.cos()*a.cos(), e.cos()*a.sin(), e.sin()))

    def visibility(self, position, normal, geometry, enabled=True):
        if not enabled or not self.config.shadows:
            return position[:, 0]*0+1
        radius = torch.deg2rad(5*self.radius_logit.sigmoid())
        directions = disk_directions(self.direction(), radius, self.config.sun_samples)
        origins = position+normal*self.config.ray_bias_m
        samples = [geometry_transmittance(origins, d.expand_as(origins), geometry,
                    self.config.ray_chunk, self.config.gaussian_chunk) for d in directions]
        return torch.stack(samples).mean(0)

    def unshadowed_and_rho(self, position, normal, albedo, geometry, receiver, enabled=True):
        """C=C0*(1-rho)，rho只由当前编辑后Actor几何、接收面和光照决定。

        rho是RGB衰减比，保留环境光；不存在时间embedding或独立黑纹理。
        主相机/太阳射线共享实际xyz、协方差与opacity，因此pose编辑直接影响rho。
        """
        # 总 irradiance 的尺度固定，防止全局 albedo/light 任意互相缩放。
        ambient = self.ambient_logit.sigmoid()
        cosine = (normal*self.direction()).sum(-1).clamp_min(0)
        active = (receiver.detach() > .001) & (normal.detach().norm(dim=-1) > .5)
        ids = active.nonzero().flatten()
        visibility = position[:, 0]*0+1
        if len(ids):
            traced = self.visibility(position[ids], normal[ids], geometry, enabled=enabled)
            visibility = visibility.index_copy(0, ids, traced)
        strength = self.strength_logit.sigmoid()
        irradiance0 = ambient+(1-ambient)*cosine[:,None]
        removed=(1-ambient)*cosine[:,None]*strength*(1-visibility[:,None])*receiver[:,None]
        rho=(removed/irradiance0.clamp_min(1e-8)).clamp(0,1)
        return albedo*irradiance0,rho,visibility

    def shade(self, position, normal, albedo, geometry, receiver, enabled=True):
        c0,rho,visibility=self.unshadowed_and_rho(position,normal,albedo,geometry,receiver,enabled)
        return c0*(1-rho),visibility
