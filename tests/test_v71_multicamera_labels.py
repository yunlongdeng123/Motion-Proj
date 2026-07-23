import numpy as np

from motion_proj.resim.label_regeneration import raw_projected_box


def test_same_3d_actor_keeps_identity_across_camera_projection():
    corners = np.array([
        [-1, -1, 5], [1, -1, 5], [1, 1, 5], [-1, 1, 5],
        [-1, -1, 7], [1, -1, 7], [1, 1, 7], [-1, 1, 7],
    ])
    k = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]])
    for camera_id in ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"):
        value = raw_projected_box(corners, k, (100, 80))
        assert value["status"] == "projected", camera_id
