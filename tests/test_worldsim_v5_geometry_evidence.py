from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v5.geometry_evidence import (
    gaussian_geometry,
    normal_proxies,
    view_angle_cosine,
)


def test_normal_proxy_uses_smallest_axis_and_faces_reference_camera() -> None:
    covariance = np.asarray([np.diag([0.01, 1.0, 2.0])])
    centers = np.asarray([[0.0, 0.0, 0.0]])
    normal = normal_proxies(covariance, centers, np.asarray([2.0, 0.0, 0.0]))
    assert np.allclose(normal, [[1.0, 0.0, 0.0]])
    assert np.allclose(
        view_angle_cosine(
            centers=centers, normals=normal, camera_center=np.asarray([2.0, 0.0, 0.0])
        ),
        [1.0],
    )


def test_gaussian_geometry_is_finite_and_symmetric() -> None:
    result = gaussian_geometry(
        means=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        quaternions_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        reference_camera_center=np.asarray([0.0, 0.0, 5.0]),
    )
    assert np.allclose(result["covariance"], np.swapaxes(result["covariance"], 1, 2))
    assert result["normal_available"].tolist() == [1]
