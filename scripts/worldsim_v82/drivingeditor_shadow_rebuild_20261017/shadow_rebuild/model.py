import math
import torch
from torch import nn
import torch.nn.functional as F
from .field import GaussianField
from .tracing import SunAmbient
from .raster import render_buffers
from .math3d import normalize


class SceneModel(nn.Module):
    def __init__(self, seed, config):
        super().__init__()
        self.config = config
        self.field = GaussianField(seed, config)
        self.light = SunAmbient(config)
        self.sky_logit = nn.Parameter(torch.zeros(1, 3, 32, 64))

    def sky(self, camera):
        y, x = torch.meshgrid(torch.arange(camera.height, device=camera.K.device),
                              torch.arange(camera.width, device=camera.K.device), indexing="ij")
        pixel = torch.stack((x, y, torch.ones_like(x)), -1).to(camera.K.dtype)
        ray = normalize((pixel@torch.linalg.inv(camera.K).T)@camera.pose[:3, :3].T)
        longitude = torch.atan2(ray[..., 1], ray[..., 0])/math.pi
        latitude = 2*torch.asin(ray[..., 2].clamp(-1, 1))/math.pi
        grid = torch.stack((longitude, latitude), -1)[None]
        return F.grid_sample(self.sky_logit.sigmoid(), grid, align_corners=False, padding_mode="border")[0].permute(1, 2, 0)

    def buffers(self, frame, actor_count, backend, edits=None, isolated=True):
        geometry = self.field.evaluate(frame.poses, frame.valid, frame.camera.center, edits)
        package = render_buffers(frame.camera, geometry, actor_count, backend, self.config.max_features, isolated)
        sky = self.sky(frame.camera)
        package["canonical"] = package["appearance"]+(1-package["alpha"])[..., None]*sky
        package["sky"] = sky
        return geometry, package

    def shade_pixels(self, package, geometry, indices, shadows=True):
        h, w = package["alpha"].shape
        flatten = lambda name: package[name].reshape(h*w, *package[name].shape[2:])[indices]
        # 接收面的属性同样使用全场景 alpha*T 的 road 条件通道，不混入车身深度/法线。
        position, normal, albedo = flatten("road_position"), flatten("road_normal"), flatten("road_albedo")
        alpha, road_alpha = flatten("alpha"), flatten("road_alpha")
        receiver = (road_alpha.detach() > 1e-5).to(road_alpha.dtype)
        radiance, visibility = self.light.shade(position, normal, albedo, geometry, receiver, enabled=shadows)
        # 本轮分解道路材质和车辆投射阴影；Actor自阴影/反射仍由新训练的canonical
        # radiance解释，避免强制简化Lambert模型毁掉车漆/灯光。Actor外观始终可训练。
        nonroad = flatten("appearance")-flatten("road_appearance")
        rgb = road_alpha[:, None]*radiance+nonroad+(1-alpha[:, None])*flatten("sky")
        return rgb, visibility

    def full_image(self, package, geometry, chunk=2048, shadows=True):
        size = package["alpha"].numel()
        pixels, shadows_out = [], []
        for first in range(0, size, chunk):
            ids = torch.arange(first, min(first+chunk, size), device=package["alpha"].device)
            rgb, vis = self.shade_pixels(package, geometry, ids, shadows)
            pixels.append(rgb)
            shadows_out.append(vis)
        shape = package["alpha"].shape
        return torch.cat(pixels).reshape(*shape, 3), torch.cat(shadows_out).reshape(shape)

    def set_stage(self, stage):
        self.field.set_stage(stage)
        for parameter in self.light.parameters():
            parameter.requires_grad_(stage != "geometry")

    def optimizer(self, stage):
        self.set_stage(stage)
        c = self.config
        rates = {"xyz": c.position_lr, "log_scales": c.shape_lr, "quaternion": c.shape_lr,
                 "opacity_logit": c.opacity_lr}
        groups = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            lr = c.light_lr if name.startswith("light.") else rates.get(name.split(".")[-1], c.appearance_lr)
            if stage == "joint":
                lr *= c.joint_lr_scale
            groups.append(dict(params=[parameter], lr=lr, name=name))
        return torch.optim.Adam(groups, eps=1e-8)
