import numpy as np

from motion_proj.resim.observation_evidence import ObservationEvidenceFrame
from motion_proj.resim.safety_geometry import GridSpec, OrientedBox


def test_actor_edit_is_reversible_and_does_not_contaminate_other_instance():
    grid = GridSpec((-3, -3, -1), (3, 3, 1), 0.5)
    shape = grid.shape
    legacy_semantics = np.zeros(shape, dtype=np.uint8)
    legacy_ids = np.zeros(shape, dtype=np.int32)
    legacy_ids[1:3, 1:3, :] = 7
    legacy_ids[-3:-1, -3:-1, :] = 9
    legacy_semantics[legacy_ids > 0] = 3
    frame = ObservationEvidenceFrame.from_legacy_o0(
        semantics=legacy_semantics,
        mask_lidar=np.zeros(shape, dtype=np.uint8),
        instance_id=legacy_ids,
        grid=grid,
    )
    other_before = frame.dynamic_instance_layers[9].copy()
    edited = frame.without_actor(7).with_actor_box(
        7, OrientedBox((0, 0, 0), (2, 1, 1), 0.4, 7)
    )
    assert np.array_equal(other_before, edited.dynamic_instance_layers[9])
    restored = edited.without_actor(7)
    assert 7 not in restored.dynamic_instance_layers
    assert np.array_equal(restored.base_state, frame.base_state)
    assert restored.content_hash() == frame.without_actor(7).content_hash()
