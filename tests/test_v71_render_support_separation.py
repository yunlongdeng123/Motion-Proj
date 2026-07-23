import pytest

from motion_proj.resim.render_support import RenderSupportFrame, safety_input_from_render_support


def test_gaussian_support_cannot_be_used_as_safety_geometry():
    support = RenderSupportFrame("a" * 64, 10, ("cam0:t0",), ("cam1:t0",), 0.2, 0.1)
    assert len(support.content_hash()) == 64
    with pytest.raises(TypeError):
        safety_input_from_render_support(support)
