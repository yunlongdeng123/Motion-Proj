import numpy as np

from motion_proj.resim.label_regeneration import visible_box_from_mask


def test_visibility_distinguishes_invisible_visible_and_truncated():
    invisible = visible_box_from_mask(np.zeros((5, 6), bool))
    assert invisible["status"] == "invisible" and invisible["xyxy"] is None
    middle = np.zeros((5, 6), bool)
    middle[1:3, 2:4] = True
    assert visible_box_from_mask(middle)["status"] == "visible"
    edge = np.zeros((5, 6), bool)
    edge[0:2, 2:4] = True
    assert visible_box_from_mask(edge)["status"] == "truncated"
