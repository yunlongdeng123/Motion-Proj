import numpy as np

from motion_proj.worldsim_v61.occupancy import (
    UNKNOWN,
    Transform,
    VoxelGridSpec,
    voxelize_oriented_box,
)


def test_named_transform_roundtrip_and_composition() -> None:
    angle = np.deg2rad(30.0)
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    matrix[:3, 3] = [3.0, -2.0, 0.5]
    transform = Transform(dst="world", src="actor", matrix=matrix)
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, -2.0, 0.25]])
    np.testing.assert_allclose(transform.inverse().apply(transform.apply(points)), points, atol=1e-12)
    identity = transform.then(transform.inverse())
    np.testing.assert_allclose(identity.matrix, np.eye(4), atol=1e-12)


def test_oriented_voxelization_rejects_corner_aabb_inflation() -> None:
    spec = VoxelGridSpec(
        frame="grid", origin_m=(-5.0, -5.0, -2.0), voxel_size_m=0.2, shape=(50, 50, 20)
    )
    angle = np.deg2rad(45.0)
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    transform = Transform(dst="grid", src="box", matrix=matrix)
    indices, corner_aabb_count = voxelize_oriented_box(
        spec, transform, np.asarray([4.0, 1.6, 1.4])
    )
    assert 0 < indices.shape[0] < corner_aabb_count


def test_actor_removal_contract_is_unknown_not_free() -> None:
    spec = VoxelGridSpec(
        frame="grid", origin_m=(-2.0, -2.0, -1.0), voxel_size_m=0.2, shape=(20, 20, 10)
    )
    transform = Transform(dst="grid", src="box", matrix=np.eye(4))
    indices, _ = voxelize_oriented_box(spec, transform, np.asarray([2.0, 1.0, 1.0]))
    semantics = np.ones(spec.shape, dtype=np.uint8)
    semantics[indices[:, 0], indices[:, 1], indices[:, 2]] = UNKNOWN
    assert np.all(semantics[indices[:, 0], indices[:, 1], indices[:, 2]] == UNKNOWN)
