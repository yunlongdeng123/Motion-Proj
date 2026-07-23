import numpy as np

from motion_proj.resim.observation_evidence import (
    DYNAMIC_OCCUPIED,
    RAY_FREE,
    UNKNOWN,
    ObservationEvidenceFrame,
)
from motion_proj.resim.safety_geometry import (
    GridSpec,
    OrientedBox,
    voxelize_aabb_baseline,
    voxelize_oriented_box,
)


def test_oriented_voxelization_is_not_rotated_corner_aabb():
    grid = GridSpec((-4, -4, -1), (4, 4, 1), 0.25)
    box = OrientedBox((0, 0, 0), (4, 1, 1), np.pi / 4)
    oriented = voxelize_oriented_box(box, grid)
    coarse = voxelize_aabb_baseline(box, grid)
    assert oriented.sum() > 0
    assert oriented.sum() < coarse.sum() * 0.6
    assert np.all(oriented <= coarse)


def test_legacy_dynamic_volume_restores_unknown_not_free():
    grid = GridSpec((0, 0, 0), (2, 2, 2), 1)
    semantics = np.zeros(grid.shape, dtype=np.uint8)
    semantics[0, 0, 0] = DYNAMIC_OCCUPIED
    semantics[1, 1, 1] = RAY_FREE
    instance = np.zeros(grid.shape, dtype=np.int32)
    instance[0, 0, 0] = 8
    frame = ObservationEvidenceFrame.from_legacy_o0(
        semantics=semantics,
        mask_lidar=np.ones(grid.shape, dtype=np.uint8),
        instance_id=instance,
        grid=grid,
    )
    removed = frame.without_actor(8)
    composite, ids = removed.composite()
    assert composite[0, 0, 0] == UNKNOWN
    assert composite[1, 1, 1] == RAY_FREE
    assert ids.sum() == 0
