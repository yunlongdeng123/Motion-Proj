import torch

from motion_proj.auditor.providers import LidarCalibratedDepthProvider
from motion_proj.auditor.depth_anything import relative_inverse_to_depth


class _SignedRelativeDepth:
    def depth(self, frames):
        k, _, h, w = frames.shape
        values = torch.linspace(-2.0, 3.0, h * w).reshape(h, w)
        return values.unsqueeze(0).repeat(k, 1, 1)


def test_relative_inverse_depth_is_positive_and_reverses_order():
    inverse_depth = torch.linspace(-2.0, 3.0, 64).reshape(8, 8)
    depth = relative_inverse_to_depth(inverse_depth, default_depth=20.0)
    assert torch.isfinite(depth).all()
    assert float(depth.min()) > 0
    assert depth[0, 0] > depth[-1, -1]


def test_lidar_calibration_clamps_nonphysical_depth():
    provider = LidarCalibratedDepthProvider(
        device="cpu", enable=False, min_points=2, min_depth=0.5, max_depth=80.0,
    )
    provider.model = _SignedRelativeDepth()
    frames = torch.zeros(2, 3, 8, 8)
    lidar = torch.zeros(2, 8, 8)
    lidar[:, 5, 5] = 10.0
    lidar[:, 6, 6] = 20.0
    depth = provider.estimate(frames, {"lidar_depth": lidar})
    assert torch.isfinite(depth).all()
    assert float(depth.min()) >= 0.5
    assert float(depth.max()) <= 80.0
    assert all(item["lidar_points"] == 2 for item in provider.last_diagnostics)
    assert all(item["scale"] is not None for item in provider.last_diagnostics)
