"""世界坐标道路外观场。参考曝光纹理和成像增益分开，几何只读。"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


def srgb_to_linear(value):
    return torch.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(value):
    value = value.clamp(0, 1)
    return torch.where(value <= 0.0031308, value * 12.92, 1.055 * value.pow(1 / 2.4) - 0.055)


class WorldRoadAppearanceField(torch.nn.Module):
    """颜色定义在固定表面参数坐标，单位由 world_from_uv 显式给出。

    world_from_uv @ [u, v, 0, 1] 对应整数像素中心。纹理表示参考曝光下的
    线性 RGB 外观。单目自然光 RGB 无法唯一给出反射率、粗糙度或金属度。
    view_gain 仅是明确记录的成像增益，不能当作已恢复的光照。
    """

    def __init__(self, texture_srgb, world_from_uv, support, provenance=None, world_xyz=None):
        super().__init__()
        texture = torch.as_tensor(texture_srgb, dtype=torch.float32)
        matrix = torch.as_tensor(world_from_uv, dtype=torch.float64)
        support = torch.as_tensor(support, dtype=torch.float32)
        if texture.ndim != 3 or texture.shape[-1] != 3 or min(texture.shape[:2]) < 2:
            raise ValueError("texture 必须是 H×W×3 且两轴至少2像素")
        if support.shape != texture.shape[:2]:
            raise ValueError("support 与纹理尺寸不匹配")
        if (not torch.isfinite(texture).all() or texture.min() < 0 or texture.max() > 1
                or not torch.isfinite(support).all() or support.min() < 0 or support.max() > 1):
            raise ValueError("纹理及支持度须为有限 [0,1] 数")
        if matrix.shape != (4, 4) or not torch.isfinite(matrix).all():
            raise ValueError("world_from_uv 必须是有限4×4矩阵")
        if not torch.allclose(matrix[3], matrix.new_tensor([0, 0, 0, 1])):
            raise ValueError("world_from_uv 必须是仿射变换")
        inverse = torch.linalg.inv(matrix)
        self.height, self.width = texture.shape[:2]
        self.register_buffer("world_from_uv", matrix)
        self.register_buffer("uv_from_world", inverse)
        self.register_buffer("reference_linear", srgb_to_linear(texture).permute(2, 0, 1)[None])
        self.register_buffer("support", support[None, None])
        yy, xx = torch.meshgrid(torch.arange(self.height, dtype=torch.float64),
                                torch.arange(self.width, dtype=torch.float64), indexing="ij")
        coords = torch.stack((xx, yy, torch.zeros_like(xx)), -1)
        if world_xyz is not None:
            xyz = torch.as_tensor(world_xyz, dtype=torch.float64)
            if xyz.shape != (self.height, self.width, 3) or not torch.isfinite(xyz).all():
                raise ValueError("world_xyz 必须是固定表面的 H×W×3 米制坐标")
            surface_coords = xyz @ inverse[:3, :3].T + inverse[:3, 3]
            # 输入世界坐标常以float32保存，千米级平移会产生数千分之一像素舍入。
            if not torch.allclose(surface_coords[:, :, :2], coords[:, :, :2], atol=.01, rtol=0):
                raise ValueError("world_xyz与像素中心变换不一致，请显式核对corner/centre约定")
            coords = surface_coords
        else:
            xyz = coords @ matrix[:3, :3].T + matrix[:3, 3]
        self.register_buffer("surface_world_xyz", xyz)
        self.register_buffer("surface_height", coords[:, :, 2].to(torch.float32)[None, None])
        if provenance is None:
            provenance = torch.where(support > 0, 3, 0).to(torch.uint8)
        provenance = torch.as_tensor(provenance, dtype=torch.uint8)
        if provenance.shape != support.shape or provenance.max() > 3:
            raise ValueError("provenance 必须是HxW的0未知/1观测/2迁移/3生成")
        self.register_buffer("provenance", provenance)

    def query(self, world_xyz, view_gain=(1.0, 1.0, 1.0)):
        xyz = torch.as_tensor(world_xyz, device=self.world_from_uv.device, dtype=torch.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not torch.isfinite(xyz).all():
            raise ValueError("query world_xyz 必须是有限N×3米制坐标")
        uvh = xyz @ self.uv_from_world[:3, :3].T + self.uv_from_world[:3, 3]
        inside = ((uvh[:, :2] >= -.01) & (uvh[:, :2] <= uvh.new_tensor([self.width - 1, self.height - 1]) + .01)).all(1)
        xy = uvh[:, :2] / uvh.new_tensor([self.width - 1, self.height - 1]) * 2 - 1
        grid = xy.to(torch.float32)[None, None]
        height = functional.grid_sample(self.surface_height, grid, align_corners=True).reshape(-1)
        inside &= (uvh[:, 2] - height).abs() <= .001
        colour = functional.grid_sample(self.reference_linear, grid, align_corners=True).squeeze(0).squeeze(1).T
        confidence = functional.grid_sample(self.support, grid, align_corners=True).reshape(-1) * inside
        gain = colour.new_tensor(view_gain)
        if gain.shape != (3,) or not torch.isfinite(gain).all() or (gain <= 0).any():
            raise ValueError("view_gain 须为3个有限正数")
        return dict(rgb_linear=colour * gain, confidence=confidence, uv=uvh[:, :2], inside=inside)

    def gaussian_rows(self):
        """相同固定表面点只从场取色，高斯不保存独立可训练的颜色。"""
        world = self.surface_world_xyz.reshape(-1, 3)
        sampled = self.query(world)
        return dict(world_xyz=world, rgb_linear=sampled["rgb_linear"],
                    rgb_srgb=linear_to_srgb(sampled["rgb_linear"]),
                    confidence=sampled["confidence"], active=sampled["confidence"] > 0.5)

    @classmethod
    def from_files(cls, texture, support, world_from_uv, provenance=None, world_xyz=None):
        rgb = np.asarray(Image.open(Path(texture)).convert("RGB"), dtype=np.float32) / 255
        mask = np.asarray(Image.open(Path(support)).convert("L"), dtype=np.float32) / 255
        return cls(rgb, world_from_uv, mask, provenance, world_xyz)


def check_coordinate_bridge(device="cpu"):
    # 大世界坐标、旋转地面、非方形纹理和半像素插值覆盖实际接线风险。
    image = torch.rand(7, 11, 3, generator=torch.Generator().manual_seed(42))
    matrix = np.array([[0, -.04, 0, 2320], [.04, 0, 0, 1365], [0, 0, 1, -20], [0, 0, 0, 1]], float)
    field = WorldRoadAppearanceField(image, matrix, torch.ones(7, 11)).to(device)
    rows = field.gaussian_rows()
    error = (rows["rgb_srgb"].cpu() - image.reshape(-1, 3)).abs().max().item()
    assert error < 2e-5, error
    assert rows["active"].all()
    centre = torch.as_tensor([2320 - .04 * 2.5, 1365 + .04 * 4.5, -20], dtype=torch.float64, device=device)
    result = field.query(centre[None])
    expected = srgb_to_linear(image[2:4, 4:6]).mean((0, 1)).to(device)
    assert torch.allclose(result["rgb_linear"][0], expected, atol=1e-5)
    off_surface = centre.clone(); off_surface[2] += .01
    assert field.query(off_surface[None])["confidence"].item() == 0
    assert len(list(field.parameters())) == 0
    return {"device": device, "max_rgb_roundtrip_error": error, "fixed_geometry": True,
            "linear_interpolation": True, "off_surface_rejected": True,
            "material_semantics": "reference-exposure appearance, PBR maps unidentifiable"}


if __name__ == '__main__':
    import argparse
    import json
    parser=argparse.ArgumentParser(description='固定世界外观场坐标与插值回归')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    print(json.dumps(check_coordinate_bridge(parser.parse_args().device),ensure_ascii=False,indent=2))
