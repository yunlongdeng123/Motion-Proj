import numpy as np
import torch

from motion_proj.data.nuscenes_dataset import NuScenesFutureVideoDataset


class _Box:
    center = np.array([0.0, 0.0, 10.0], dtype=np.float32)
    wlh = np.ones(3, dtype=np.float32)

    def __init__(self, token):
        self.token = token

    def corners(self):
        return np.array(
            [[-1, -1, 1, 1, -1, -1, 1, 1], [-1, 1, -1, 1, -1, 1, -1, 1], [9] * 8],
            dtype=np.float32,
        )


class _NuScenes:
    def get(self, table, token):
        if table == "sample_annotation":
            return {
                "instance_token": token,
                "category_name": "vehicle.car",
                "visibility_token": "1" if token == "low" else "2",
                "attribute_tokens": [],
            }
        raise KeyError((table, token))

    def box_velocity(self, token):
        return np.zeros(3, dtype=np.float32)


def test_min_box_visibility_is_applied_before_track_schema():
    dataset = NuScenesFutureVideoDataset.__new__(NuScenesFutureVideoDataset)
    dataset.nusc = _NuScenes()
    dataset.min_vis = 2
    dataset.W, dataset.H = 100, 80
    intrinsics = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    rows = dataset._boxes_to_2d([_Box("low"), _Box("kept")], intrinsics, 1.0, 1.0, 100, 80)
    assert [row["instance_token"] for row in rows] == ["kept"]
    assert rows[0]["visibility"] == 2

