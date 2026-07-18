import torch

from motion_proj.data.real_motion_targets import build_actor_residual_targets


def _box(token, frame, visibility=4):
    return {
        "annotation_token": f"ann-{token}-{frame}",
        "instance_token": token,
        "category": "vehicle.car",
        "attributes": ["vehicle.moving"],
        "visibility": visibility,
        "xyxy": [40.0, 30.0, 60.0, 50.0],
        "center_cam": [0.1 * frame, 0.0, 10.0],
        "velocity_global": [0.2, 0.0, 0.0],
    }


def test_only_same_instance_in_adjacent_visible_frames_forms_target():
    identity = torch.eye(4)
    sample = {
        "frames": torch.zeros((3, 3, 80, 100)),
        "boxes": [
            [_box("a", 0), _box("gap", 0), _box("low", 0)],
            [_box("a", 1), _box("low", 1, visibility=1)],
            [_box("a", 2), _box("gap", 2)],
        ],
        "timestamps": torch.tensor([1_000_000, 1_500_000, 2_000_000]),
        "intrinsics": torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]),
        "cam2ego": identity,
        "ego2global": identity.unsqueeze(0).repeat(3, 1, 1),
    }
    rows = build_actor_residual_targets(sample, min_visibility=2)
    assert [(row["instance_token"], row["frame_index"]) for row in rows] == [("a", 0), ("a", 1)]
    assert rows[0]["annotation_token_t"] == "ann-a-0"
    assert rows[1]["annotation_token_tp1"] == "ann-a-2"
