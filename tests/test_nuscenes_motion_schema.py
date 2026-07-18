import numpy as np
import torch

from motion_proj.data.nuscenes_dataset import NuScenesFutureVideoDataset


class _Box:
    token = "ann-1"
    center = np.array([0.0, 0.0, 10.0], dtype=np.float32)
    wlh = np.array([2.0, 4.0, 1.5], dtype=np.float32)

    def corners(self):
        return np.array(
            [
                [-1, -1, -1, -1, 1, 1, 1, 1],
                [-1, -1, 1, 1, -1, -1, 1, 1],
                [9, 11, 9, 11, 9, 11, 9, 11],
            ],
            dtype=np.float32,
        )


class _NuScenes:
    def get(self, table, token):
        if table == "sample_annotation":
            return {
                "instance_token": "instance-1",
                "category_name": "vehicle.car",
                "visibility_token": "4",
                "attribute_tokens": ["attr-moving"],
            }
        if table == "attribute":
            return {"name": "vehicle.moving"}
        raise KeyError((table, token))

    def box_velocity(self, token):
        assert token == "ann-1"
        return np.array([3.0, 0.5, 0.0], dtype=np.float32)


def test_additive_motion_schema_preserves_legacy_fields():
    dataset = NuScenesFutureVideoDataset.__new__(NuScenesFutureVideoDataset)
    dataset.nusc = _NuScenes()
    dataset.min_vis = 2
    dataset.W, dataset.H = 100, 80
    intrinsics = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    row = dataset._boxes_to_2d([_Box()], intrinsics, 1.0, 1.0, 100, 80)[0]

    assert {
        "instance_token", "category", "xyxy", "center_depth", "size3d", "visibility",
    } <= row.keys()
    assert {
        "annotation_token", "attributes", "center_cam", "corners_cam", "velocity_global",
    } <= row.keys()
    assert row["annotation_token"] == "ann-1"
    assert row["attributes"] == ["vehicle.moving"]
    assert row["center_cam"].shape == (3,)
    assert row["corners_cam"].shape == (8, 3)
    assert row["velocity_global"].shape == (3,)
    assert row["xyxy"][0] <= 50.0 <= row["xyxy"][2]
    assert row["xyxy"][1] <= 40.0 <= row["xyxy"][3]

