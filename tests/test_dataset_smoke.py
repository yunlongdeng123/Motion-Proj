import os

import numpy as np
import pytest
import torch

from motion_proj.auditor.boxes_nuscenes import build_tracks
from motion_proj.auditor.state import MotionState, Track
from motion_proj.projector import DynamicsProjector
from motion_proj.projector.energies import e_obj
from motion_proj.projector.smoothing import smooth_tracks

DATAROOT = "/root/autodl-tmp/data/nuscenes"


def test_build_tracks_links_instances():
    bpf = [
        [{"instance_token": "a", "category": "car", "xyxy": np.array([10, 10, 30, 30], np.float32), "center_depth": 12.0}],
        [
            {"instance_token": "a", "category": "car", "xyxy": np.array([12, 11, 33, 31], np.float32), "center_depth": 12.5},
            {"instance_token": "b", "category": "car", "xyxy": np.array([40, 40, 60, 60], np.float32), "center_depth": 8.0},
        ],
    ]
    tracks = build_tracks(bpf, 2)
    assert len(tracks) == 2
    a = next(t for t in tracks if t.instance_token == "a")
    assert a.present.tolist() == [True, True]


def test_smoothing_reduces_object_energy():
    K = 4
    xyxy = torch.tensor(
        [[20, 20, 40, 40], [24, 19, 45, 41], [19, 21, 39, 39], [28, 22, 50, 44]], dtype=torch.float32
    )
    tr = Track("inst0", "car", xyxy, torch.full((K,), 12.0), torch.ones(K, dtype=torch.bool))
    before = e_obj([tr])
    after = e_obj(smooth_tracks([tr], lam=5.0))
    assert after < before


def test_projector_runs_on_synthetic_state():
    K, H, W = 4, 64, 96
    frames = torch.rand(K, 3, H, W) * 2 - 1
    depth = torch.full((K, H, W), 15.0)
    e2g = torch.eye(4).unsqueeze(0).repeat(K, 1, 1)
    for t in range(K):
        e2g[t, 2, 3] = 0.5 * t
    intr = torch.tensor([[120.0, 0, 48.0], [0, 120.0, 32.0], [0, 0, 1.0]])
    tr = Track(
        "inst0", "car",
        torch.tensor([[20, 20, 40, 40], [24, 19, 45, 41], [19, 21, 39, 39], [28, 22, 50, 44]], dtype=torch.float32),
        torch.full((K,), 12.0), torch.ones(K, dtype=torch.bool),
    )
    state = MotionState(
        u_static=torch.zeros(K - 1, H, W, 2), u_ego=torch.zeros(K - 1, H, W, 2),
        static_mask=torch.ones(K - 1, H, W), flow_conf=torch.ones(K - 1, H, W),
        depth=depth, tracks=[tr],
        meta={"intrinsics": intr, "cam2ego": torch.eye(4), "ego2global": e2g, "hw": (H, W), "sample_id": "t0"},
    )
    res = DynamicsProjector().project(frames, state)
    assert res.x_dagger.shape == (K, 3, H, W)
    assert res.mask.shape == (K, 1, H, W)
    assert res.metadata["energies"]["obj_after"] <= res.metadata["energies"]["obj_before"] + 1e-6


@pytest.mark.skipif(not os.path.isdir(os.path.join(DATAROOT, "v1.0-mini")), reason="nuScenes mini not extracted")
def test_nuscenes_dataset_item():
    from omegaconf import OmegaConf

    from motion_proj.data import NuScenesFutureVideoDataset

    cfg = OmegaConf.create(
        {
            "version": "v1.0-mini", "dataroot": DATAROOT, "cameras": ["CAM_FRONT"],
            "num_frames": 4, "frame_stride": 1, "height": 128, "width": 224,
            "min_box_visibility": 2, "use_sweeps": False,
        }
    )
    ds = NuScenesFutureVideoDataset(cfg)
    assert len(ds) > 0
    item = ds[0]
    assert item["frames"].shape == (4, 3, 128, 224)
    assert item["ego2global"].shape == (4, 4, 4)
