"""AD-GS 环境冒烟测试：import / 单 Gaussian forward+backward / LPIPS / PyTorch3D。

对应 DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md 8.1 第 5 步。
必须在 adgs 环境下运行，且 AD-GS 仓库根目录需在 sys.path 中。
"""
import os
import sys

import numpy as np
import torch

ADGS_ROOT = os.environ.get("ADGS_ROOT", "/root/autodl-tmp/third_party/AD-GS")
sys.path.insert(0, ADGS_ROOT)


def section(name):
    print(f"\n===== {name} =====", flush=True)


def main():
    section("versions")
    import torchaudio
    import torchvision
    print("python           ", sys.version.split()[0])
    print("torch            ", torch.__version__, "| cuda", torch.version.cuda)
    print("torchvision      ", torchvision.__version__)
    print("torchaudio       ", torchaudio.__version__)
    print("cudnn            ", torch.backends.cudnn.version())
    print("arch_list        ", torch.cuda.get_arch_list())
    print("device           ", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("OMP_NUM_THREADS  ", os.environ.get("OMP_NUM_THREADS"))
    print("TORCH_CUDA_ARCH  ", os.environ.get("TORCH_CUDA_ARCH_LIST"))

    section("core imports")
    import cv2
    import mmcv
    import open3d
    from nuscenes.nuscenes import NuScenes  # noqa: F401  只验证可导入
    print("numpy", np.__version__, "| opencv", cv2.__version__,
          "| open3d", open3d.__version__, "| mmcv", mmcv.__version__)
    print("nuscenes-devkit NuScenes 类导入 ok")

    section("simple-knn CUDA kernel")
    from simple_knn._C import distCUDA2
    pts = torch.rand(100000, 3, device="cuda")
    dist = distCUDA2(pts)
    assert torch.isfinite(dist).all(), "simple-knn 产生了非有限值"
    print("distCUDA2 ok | N=100000 | mean", float(dist.mean()))

    section("gaussian rasterizer forward + backward")
    from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                             GaussianRasterizer)
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2

    H, W = 240, 320
    fovx = fovy = 1.0
    znear, zfar = 0.01, 100.0

    R = np.eye(3, dtype=np.float32)
    T = np.array([0.0, 0.0, 4.0], dtype=np.float32)
    world_view = torch.tensor(getWorld2View2(R, T)).cuda().transpose(0, 1)
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).cuda().transpose(0, 1)
    full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    campos = world_view.inverse()[3, :3]

    N, K = 4096, 8
    means3D = torch.randn(N, 3, device="cuda") * 0.3
    means3D.requires_grad_(True)
    means2D = torch.zeros_like(means3D, requires_grad=True)
    means2D.retain_grad()
    shs = torch.randn(N, 1, 3, device="cuda", requires_grad=True)
    opacities = torch.rand(N, 1, device="cuda", requires_grad=True)
    scales = (torch.rand(N, 3, device="cuda") * 0.02 + 0.01).requires_grad_(True)
    rotations = torch.zeros(N, 4, device="cuda")
    rotations[:, 0] = 1.0
    rotations.requires_grad_(True)
    semantic = torch.rand(N, K, device="cuda", requires_grad=True)
    flow_points = (means3D.detach() + 0.01).requires_grad_(True)

    settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=np.tan(fovx * 0.5), tanfovy=np.tan(fovy * 0.5),
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=world_view, projmatrix=full_proj,
        sh_degree=0, campos=campos,
        prefiltered=False, inv_depth=False, debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    color, radii, depth, opac, flow, sem = rasterizer(
        means3D=means3D, means2D=means2D, shs=shs, colors_precomp=None,
        opacities=opacities, scales=scales, rotations=rotations,
        flow_points=flow_points, semantic=semantic,
    )
    visible = int((radii > 0).sum())
    print(f"forward ok | color{tuple(color.shape)} depth{tuple(depth.shape)} "
          f"opacity{tuple(opac.shape)} flow{tuple(flow.shape)} semantic{tuple(sem.shape)}")
    print(f"visible gaussians {visible}/{N} | color range "
          f"[{float(color.min()):.4f}, {float(color.max()):.4f}]")
    assert visible > 0, "没有任何 Gaussian 被光栅化，相机设置或 kernel 有问题"
    assert torch.isfinite(color).all(), "渲染结果含 NaN/Inf"

    loss = color.mean() + depth.mean() + opac.mean() + flow.mean() + sem.mean()
    loss.backward()
    for name, tensor in [("means3D", means3D), ("shs", shs), ("opacities", opacities),
                         ("scales", scales), ("rotations", rotations), ("semantic", semantic)]:
        assert tensor.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(tensor.grad).all(), f"{name} 梯度含 NaN/Inf"
    print("backward ok | grad norms:",
          {n: round(float(t.grad.norm()), 4)
           for n, t in [("means3D", means3D), ("shs", shs), ("opacities", opacities),
                        ("scales", scales), ("semantic", semantic)]})

    section("LPIPS")
    import lpips
    lpips_fn = lpips.LPIPS(net="vgg").cuda()
    a = torch.rand(1, 3, 128, 128, device="cuda") * 2 - 1
    b = a + torch.randn_like(a) * 0.05
    value = lpips_fn(a, b)
    print("lpips(vgg) ok |", float(value))
    assert torch.isfinite(value).all()

    section("PyTorch3D")
    import pytorch3d
    from pytorch3d.ops import knn_points
    from pytorch3d.structures import Meshes
    from pytorch3d.transforms import quaternion_to_matrix
    print("pytorch3d", pytorch3d.__version__)
    p1 = torch.rand(1, 2048, 3, device="cuda")
    p2 = torch.rand(1, 4096, 3, device="cuda")
    knn = knn_points(p1, p2, K=8)
    print("knn_points CUDA ok | dists", tuple(knn.dists.shape))
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    print("quaternion_to_matrix ok |", quaternion_to_matrix(quat).shape)
    verts = torch.rand(1, 100, 3, device="cuda")
    faces = torch.randint(0, 100, (1, 200, 3), device="cuda")
    mesh = Meshes(verts=verts, faces=faces)
    from pytorch3d.ops import sample_points_from_meshes
    sampled = sample_points_from_meshes(mesh, 512)
    print("sample_points_from_meshes CUDA ok |", tuple(sampled.shape))

    section("peak memory")
    print("max_allocated MB", round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1))
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
