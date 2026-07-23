from motion_proj.resim.safety_geometry import OrientedBox, obb_intersects, swept_obb_collision


def test_crossing_between_frames_is_detected():
    a0 = OrientedBox((-2, 0, 0), (1, 1, 1), 0)
    a1 = OrientedBox((2, 0, 0), (1, 1, 1), 0)
    b0 = OrientedBox((0, -2, 0), (1, 1, 1), 0)
    b1 = OrientedBox((0, 2, 0), (1, 1, 1), 0)
    assert not obb_intersects(a0, b0)
    assert not obb_intersects(a1, b1)
    result = swept_obb_collision(a0, a1, b0, b1)
    assert result["collision"]
    assert 0.2 < result["first_collision_fraction"] < 0.75


def test_vertical_separation_is_respected():
    low = OrientedBox((0, 0, 0), (2, 2, 1), 0)
    high = OrientedBox((0, 0, 3), (2, 2, 1), 0)
    assert not obb_intersects(low, high)
