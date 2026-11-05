"""车体坐标中的持久高斯资产。中心与支撑始终受表面锚点约束。"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn


def measured_surface(asset):
    xyz = np.asarray(asset['xyz_local_m'], dtype=np.float32)
    rgb = np.asarray(asset['rgb'], dtype=np.float32)
    known = np.asarray(asset['rgb_valid'], dtype=bool)
    box = np.asarray(asset['box_lwh_m'], dtype=np.float32)
    if len(xyz) < 4 or not known.any():
        raise ValueError('需要至少四个实测锚点及独立 RGB 证据')
    if rgb[known].max() > 1 or rgb[known].min() < 0:
        raise ValueError('对象 RGB 必须为 [0,1] sRGB')
    tree = cKDTree(xyz)
    distance, neighbours = tree.query(xyz, k=min(len(xyz), 12), workers=1)
    points = xyz[neighbours]
    centred = points - points.mean(1, keepdims=True)
    covariance = centred.transpose(0, 2, 1) @ centred
    normals = np.linalg.eigh(covariance)[1][:, :, 0]
    sigma = np.clip(distance[:, min(3, distance.shape[1] - 1)] * .66, .018, .095)
    nearby = cKDTree(xyz[known]).query(xyz[~known], workers=1)[1]
    initial = rgb.copy()
    initial[~known] = rgb[known][nearby]
    return xyz, normals, sigma, initial, box


def extend_colour(xyz, colour, targets):
    distance, rows = cKDTree(xyz).query(targets, k=min(len(xyz), 24), workers=1)
    weight = np.exp(-np.median(distance, axis=1) / .65)[:, None]
    return weight * np.median(colour[rows], axis=1) + (1 - weight) * np.median(colour, axis=0)


class ObjectLayer(nn.Module):
    def __init__(self, xyz, normals, sigma, rgb, box, *, generated=False):
        super().__init__()
        def buf(name, value):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        buf('anchor', xyz)
        buf('normal', normals)
        buf('sigma', np.asarray(sigma).reshape(-1))
        buf('initial_rgb', np.clip(rgb, .005, .995))
        buf('half_box', np.asarray(box) * .5)
        self.generated = generated
        if generated:
            return
        self.offset = nn.Parameter(torch.zeros_like(self.anchor))
        self.log_size = nn.Parameter(torch.zeros_like(self.sigma))
        self.coverage_logit = nn.Parameter(torch.full_like(self.sigma, 1.75))
        self.colour_logit = nn.Parameter(torch.logit(self.initial_rgb))

    @classmethod
    def from_measurements(cls, asset):
        return cls(*measured_surface(asset))

    @classmethod
    def from_fixed_pvd(cls, prior, observed):
        xyz = prior['canonical_xyz']
        sigma = prior['sigma_tangent_m'].reshape(-1)
        idx = np.asarray(prior['prior_origin'], dtype=bool) & ~np.asarray(prior['observed_anchor'], dtype=bool)
        # 新建的实测面优先。PVD 几何、法线和采样尺度保持原值。
        observed_xyz = observed.anchor.detach().cpu().numpy()
        gap = cKDTree(observed_xyz).query(xyz, workers=1)[0]
        idx &= gap > np.maximum(.045, sigma)
        if not idx.any():
            raise ValueError('固定 PVD 在实测面去重后没有剩余提案')
        # 用车体坐标中的局部 RGB 中位数延续外观。颜色不来自历史渲染。
        out = cls(xyz[idx], prior['canonical_normals'][idx], sigma[idx],
                  extend_colour(observed_xyz, observed.initial_rgb.detach().cpu().numpy(), xyz[idx]),
                  observed.half_box.detach().cpu().numpy() * 2, generated=True)
        out.register_buffer('pvd_source_rows', torch.as_tensor(np.flatnonzero(idx)))
        return out

    def update_inferred_colour(self, observed):
        xyz = observed.anchor.detach().cpu().numpy()
        inferred_xyz = self.anchor.detach().cpu().numpy()
        colour = observed.colour_logit.detach().sigmoid().cpu().numpy()
        value = extend_colour(xyz, colour, inferred_xyz)
        self.initial_rgb.copy_(self.initial_rgb.new_tensor(value))

    def geometry(self):
        if self.generated:
            return self.anchor, self.sigma, torch.full_like(self.sigma, .995), self.initial_rgb
        bound = torch.minimum(self.sigma * .45, self.sigma.new_full((), .035))[:, None]
        move = self.offset.tanh()
        move = bound * move / move.norm(dim=1, keepdim=True).clamp_min(1)
        # 保留实测锚点本身的微小盒边误差，同时避免进一步向外移动。
        limit = torch.maximum(self.half_box + .04, self.anchor.abs())
        xyz = torch.maximum(-limit, torch.minimum(self.anchor + move, limit))
        sigma = self.sigma * (1 + .25 * self.log_size.tanh())
        return xyz, sigma, .995 * self.coverage_logit.sigmoid(), self.colour_logit.sigmoid()

    def raster_rows(self, matrix, camera, pipe, channels, precomputed):
        xyz, sigma, opacity, rgb = self.geometry()
        transform = matrix[:3, :3]
        model_scale = transform.norm(dim=0).mean()
        means = xyz @ transform.T + matrix[:3, 3]
        normal = torch.nn.functional.normalize(self.normal @ transform.T, dim=1)
        q = torch.stack((1 + normal[:, 2], -normal[:, 1], normal[:, 0], torch.zeros_like(sigma)), 1)
        q = torch.where((q.norm(dim=1) < 1e-5)[:, None], q.new_tensor([0., 1., 0., 0.]), q)
        q = torch.nn.functional.normalize(q, dim=1)
        thickness = torch.full_like(sigma, .003) if self.generated else (sigma * .16).clamp(.003, .012)
        scales = torch.stack((sigma, sigma, thickness), 1) * model_scale
        cov = None
        if pipe.compute_cov3D_python:
            from utils.general_utils import build_scaling_rotation, strip_symmetric
            basis = build_scaling_rotation(scales, q)
            cov = strip_symmetric(basis @ basis.transpose(1, 2))
        depth = means @ camera.world_view_transform[:3, 2:3] + camera.world_view_transform[3, 2]
        dc = ((rgb - .5) / .28209479177387814)[:, None]
        shs = None if precomputed else torch.cat((dc, dc.new_zeros((len(dc), channels - 1, 3))), 1)
        return dict(means3D=means, means2D=torch.zeros_like(means), shs=shs,
                    colors_precomp=rgb if precomputed else None,
                    features=torch.cat((depth, torch.ones_like(depth)), 1), opacities=opacity[:, None],
                    scales=scales if cov is None else None, rotations=q if cov is None else None,
                    cov3D_precomp=cov, mask=torch.ones_like(sigma, dtype=torch.bool))

    def anchor_loss(self):
        return self.offset.tanh().square().mean() + self.log_size.tanh().square().mean()
