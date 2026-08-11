#!/usr/bin/env python3
"""WorldSim V4 AD-GS 最小 CUDA 冒烟测试，不依赖未使用的语义预处理环境。"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch


ADGS_ROOT = os.environ.get("ADGS_ROOT", "/root/autodl-tmp/third_party/AD-GS")
sys.path.insert(0, ADGS_ROOT)


def main() -> None:
    import cv2
    import open3d
    import plyfile
    import pytorch3d
    from pytorch3d.ops import knn_points
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from simple_knn._C import distCUDA2
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2

    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability(0) == (8, 6)
    points = torch.rand(100_000, 3, device="cuda")
    distances = distCUDA2(points)
    assert torch.isfinite(distances).all()
    knn = knn_points(points[:1024][None], points[1024:3072][None], K=8)
    assert knn.idx.shape == (1, 1024, 8)
    assert torch.isfinite(knn.dists).all()

    height, width = 120, 160
    fov = 1.0
    world_view = torch.tensor(
        getWorld2View2(np.eye(3, dtype=np.float32), np.array([0.0, 0.0, 4.0], dtype=np.float32))
    ).cuda().transpose(0, 1)
    projection = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fov).cuda().transpose(0, 1)
    settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=np.tan(fov * 0.5),
        tanfovy=np.tan(fov * 0.5),
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=world_view,
        projmatrix=(world_view.unsqueeze(0).bmm(projection.unsqueeze(0))).squeeze(0),
        sh_degree=0,
        campos=world_view.inverse()[3, :3],
        prefiltered=False,
        inv_depth=False,
        debug=False,
    )
    count, semantic_dims = 1024, 8
    means3d = (torch.randn(count, 3, device="cuda") * 0.25).requires_grad_(True)
    means2d = torch.zeros_like(means3d, requires_grad=True)
    shs = torch.randn(count, 1, 3, device="cuda", requires_grad=True)
    opacities = torch.rand(count, 1, device="cuda", requires_grad=True)
    scales = (torch.rand(count, 3, device="cuda") * 0.02 + 0.01).requires_grad_(True)
    rotations = torch.zeros(count, 4, device="cuda")
    rotations[:, 0] = 1.0
    rotations.requires_grad_(True)
    semantic = torch.rand(count, semantic_dims, device="cuda", requires_grad=True)
    flow_points = (means3d.detach() + 0.01).requires_grad_(True)
    color, radii, depth, opacity, flow, sem = GaussianRasterizer(raster_settings=settings)(
        means3D=means3d,
        means2D=means2d,
        shs=shs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        flow_points=flow_points,
        semantic=semantic,
    )
    assert int((radii > 0).sum()) > 0
    loss = color.mean() + depth.mean() + opacity.mean() + flow.mean() + sem.mean()
    loss.backward()
    assert means3d.grad is not None and torch.isfinite(means3d.grad).all()
    print(
        {
            "status": "passed",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "opencv": cv2.__version__,
            "open3d": open3d.__version__,
            "plyfile": plyfile.__file__,
            "pytorch3d": pytorch3d.__version__,
            "visible_gaussians": int((radii > 0).sum()),
            "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        }
    )


if __name__ == "__main__":
    main()
