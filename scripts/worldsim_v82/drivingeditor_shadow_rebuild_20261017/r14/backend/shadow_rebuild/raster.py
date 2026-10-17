"""主视角共享 alpha*T 的 G-buffer；CPU EWA 参考与既有 CUDA ABI 适配器。"""
from typing import NamedTuple
import math
import torch
from .math3d import normalize


class Camera:
    def __init__(self, intrinsic, world_from_camera, height, width, device="cpu"):
        self.K = torch.as_tensor(intrinsic, dtype=torch.float32, device=device)
        self.pose = torch.as_tensor(world_from_camera, dtype=torch.float32, device=device)
        if self.K.shape != (3, 3) or self.pose.shape != (4, 4) or not bool(torch.isfinite(self.K).all()) or not bool(torch.isfinite(self.pose).all()):
            raise ValueError("相机矩阵尺寸/数值无效")
        eye = torch.eye(3, device=self.pose.device)
        if not bool(torch.allclose(self.pose[:3, :3].T@self.pose[:3, :3], eye, atol=2e-4)) or not bool(torch.isclose(torch.linalg.det(self.pose[:3, :3]), self.pose.new_tensor(1.), atol=2e-4)):
            raise ValueError("相机必须是米坐标 proper rigid pose")
        if not bool(torch.allclose(self.pose[3], self.pose.new_tensor([0, 0, 0, 1]), atol=1e-5)) or not bool(torch.allclose(self.K[2], self.K.new_tensor([0, 0, 1]), atol=1e-6)):
            raise ValueError("相机齐次行无效")
        if not bool(self.K[0, 0] > 0) or not bool(self.K[1, 1] > 0) or abs(float(self.K[0, 1])) > 1e-6 or abs(float(self.K[1, 0])) > 1e-6:
            raise ValueError("主光栅器需要无skew的正焦距针孔相机")
        self.view = torch.linalg.inv(self.pose)
        self.height, self.width = int(height), int(width)
        self.center = self.pose[:3, 3]

    def project(self, xyz):
        camera = xyz@self.view[:3, :3].T+self.view[:3, 3]
        uvw = camera@self.K.T
        return uvw[:, :2]/uvw[:, 2:3].clamp_min(1e-6), camera[:, 2]


def reference_raster(camera, xyz, cov, opacity, channels):
    """小数据 CPU 验证用。严格限制 N*H*W，禁止误作全场景 GPU fallback。"""
    h, w = camera.height, camera.width
    if len(xyz)*h*w > 4_000_000:
        raise ValueError("CPU 参考光栅器容量限制；正式场景必须通过 CUDA ABI 验收")
    if len(xyz) == 0:
        return channels.new_zeros(h, w, channels.shape[-1])
    uv, z = camera.project(xyz)
    cam = xyz@camera.view[:3, :3].T+camera.view[:3, 3]
    fx, fy = camera.K[0, 0], camera.K[1, 1]
    zero = z*0
    j = torch.stack((fx/z.clamp_min(1e-5), zero, -fx*cam[:, 0]/z.square().clamp_min(1e-5),
                     zero, fy/z.clamp_min(1e-5), -fy*cam[:, 1]/z.square().clamp_min(1e-5)), -1).reshape(-1, 2, 3)
    cv = camera.view[:3, :3]@cov@camera.view[:3, :3].T
    projected = j@cv@j.transpose(-1, -2)+torch.eye(2, device=xyz.device)*.3
    inv = torch.linalg.inv(projected)
    y, x = torch.meshgrid(torch.arange(h, device=xyz.device), torch.arange(w, device=xyz.device), indexing="ij")
    delta = torch.stack((x, y), -1)[None]-uv[:, None, None]
    exponent = torch.einsum("nhwi,nij,nhwj->nhw", delta, inv, delta)
    alpha = (opacity[:, 0, None, None]*torch.exp(-.5*exponent)).clamp_max(.99)
    alpha = alpha*(z > .01)[:, None, None]
    order = torch.argsort(z.detach())
    alpha, channels = alpha[order], channels[order]
    trans = torch.cumprod(torch.cat((torch.ones_like(alpha[:1]), 1-alpha+1e-10), 0), 0)[:-1]
    return torch.einsum("nhw,nc->hwc", alpha*trans, channels)


class _Settings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


def cuda_raster(camera, xyz, cov, opacity, channels, capacity=10):
    """仅支持已审计的 DrivingEditor 四输出 feature rasterizer，不安装/替换内核。"""
    from diff_gaussian_rasterization import GaussianRasterizer
    if len(xyz) == 0:
        return channels.new_zeros(camera.height, camera.width, channels.shape[-1])
    k, h, w = camera.K, camera.height, camera.width
    # 非中心主点投影也保留；与本代码 +Z / row-vector CUDA 矩阵约定配对。
    near, far = .01, 1000.0
    p = torch.zeros(4, 4, device=xyz.device)
    p[0, 0], p[1, 1] = 2*k[0, 0]/w, 2*k[1, 1]/h
    p[0, 2], p[1, 2] = 2*(k[0, 2]+.5)/w-1, 2*(k[1, 2]+.5)/h-1
    p[2, 2], p[2, 3], p[3, 2] = far/(far-near), -far*near/(far-near), 1
    settings = _Settings(h, w, w/(2*float(k[0, 0])), h/(2*float(k[1, 1])), xyz.new_zeros(3),
                         1.0, camera.view.T.contiguous(), (p@camera.view).T.contiguous(), 0, camera.center, False, False)
    rast = GaussianRasterizer(raster_settings=settings)
    packed = cov[:, (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)].contiguous()
    screen = torch.zeros_like(xyz, requires_grad=True)
    output = []
    for c0 in range(0, channels.shape[-1], capacity):
        payload = channels[:, c0:c0+capacity].contiguous()
        result = rast(means3D=xyz.contiguous(), means2D=screen, shs=None,
                      colors_precomp=xyz.new_zeros(len(xyz), 3), opacities=opacity.contiguous(),
                      scales=None, rotations=None, cov3D_precomp=packed, features=payload,
                      mask=torch.ones(len(xyz), dtype=torch.bool, device=xyz.device))
        if not isinstance(result, (tuple, list)) or len(result) != 4 or result[2].shape != (payload.shape[-1], h, w):
            raise RuntimeError("CUDA feature ABI 未通过；禁止静默切换到不同合成语义")
        output.append(result[2].permute(1, 2, 0))
    return torch.cat(output, -1)


def render_buffers(camera, geometry, actor_count, backend="reference", max_features=10, isolated=True,competition=False):
    z = (geometry["xyz"]@camera.view[:3, :3].T+camera.view[:3, 3])[:, 2:3]
    ids = torch.arange(actor_count, device=z.device)
    labels = (geometry["owner"][:, None] == ids).to(z.dtype)
    road = geometry["road"][:, None].to(z.dtype)
    channels = torch.cat((torch.ones_like(z), z, geometry["xyz"], geometry["normal"], geometry["albedo"],
                          geometry["appearance"], geometry["view_residual"], road, labels,
                          geometry["xyz"]*road, geometry["normal"]*road,
                          geometry["albedo"]*road, geometry["appearance"]*road,
                          geometry["appearance"]*(geometry["owner"] >= 0)[:, None]), -1)

    def raster(g, value):
        if backend == "reference":
            return reference_raster(camera, g["xyz"], g["cov"], g["opacity"], value)
        if backend != "cuda":
            raise ValueError("未知主光栅器")
        return cuda_raster(camera, g["xyz"], g["cov"], g["opacity"], value, max_features)

    raw = raster(geometry, channels)
    alpha = raw[..., 0]
    denom = alpha.clamp_min(1e-6)[..., None]
    package = dict(alpha=alpha, depth=raw[..., 1]/denom[..., 0], position=raw[..., 2:5]/denom,
                   normal=normalize(raw[..., 5:8]/denom), albedo=raw[..., 8:11]/denom,
                   appearance=raw[..., 11:14], view_residual=raw[..., 14:17]/denom,
                   road=raw[..., 17]/denom[..., 0], visible_labels=raw[..., 18:18+actor_count])
    base = 18+actor_count
    road_denom = raw[..., 17:18].clamp_min(1e-6)
    package.update(road_alpha=raw[..., 17], road_position=raw[..., base:base+3]/road_denom,
                   road_normal=normalize(raw[..., base+3:base+6]/road_denom),
                   road_albedo=raw[..., base+6:base+9]/road_denom,
                   road_appearance=raw[..., base+9:base+12], dynamic_visible_rgb=raw[..., base+12:base+15])
    if isolated:
        individual = []
        for actor in range(actor_count):
            mask = geometry["owner"] == actor
            sub = {key: value[mask] for key, value in geometry.items()}
            individual.append(raster(sub, channels[mask, :1])[..., 0])
        package["raw_actor_alpha"] = torch.stack(individual, -1)
    if competition:
        for name,mask in [('static',geometry['owner']<0),('dynamic',geometry['owner']>=0)]:
            sub={key:value[mask] for key,value in geometry.items()}
            moment=raster(sub,torch.cat((torch.ones_like(z),z),-1)[mask])
            package[name+'_isolated_alpha']=moment[...,0]
            package[name+'_isolated_depth']=moment[...,1]/moment[...,0].clamp_min(1e-6)
    return package
