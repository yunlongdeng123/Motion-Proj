import torch

from motion_proj.utils import geometry as G


def test_identity_ego_flow_is_zero():
    K = torch.tensor([[500.0, 0, 224.0], [0, 500.0, 128.0], [0, 0, 1.0]])
    cam2ego = torch.eye(4)
    e0 = torch.eye(4)
    e1 = torch.eye(4)
    depth = torch.full((16, 28), 10.0)
    flow = G.ego_induced_flow(depth, K, cam2ego, e0, e1)
    assert flow.abs().max() < 1e-3


def test_se3_inverse():
    R = G.quaternion_to_matrix(torch.tensor([0.92, 0.0, 0.38, 0.0]))
    T = G.make_transform(R, torch.tensor([1.0, 2.0, 3.0]))
    err = (G.se3_inverse(T) @ T - torch.eye(4)).abs().max()
    assert err < 1e-5


def test_project_backproject_roundtrip():
    K = torch.tensor([[500.0, 0, 14.0], [0, 500.0, 8.0], [0, 0, 1.0]])
    depth = torch.full((16, 28), 7.0)
    uv = G.pixel_grid(16, 28)
    X = G.backproject(uv, depth, K)
    uv2 = G.project_points(X, K)
    assert (uv2 - uv).abs().max() < 1e-3


def test_nonzero_flow_under_translation():
    K = torch.tensor([[500.0, 0, 14.0], [0, 500.0, 8.0], [0, 0, 1.0]])
    cam2ego = torch.eye(4)
    e0 = torch.eye(4)
    e1 = torch.eye(4)
    e1[2, 3] = 1.0
    depth = torch.full((16, 28), 10.0)
    flow = G.ego_induced_flow(depth, K, cam2ego, e0, e1)
    assert flow.abs().max() > 1e-2
