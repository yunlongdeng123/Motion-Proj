import numpy as np

from motion_proj.resim.label_regeneration import (
    LIMITED_SEMANTIC_VALUES,
    limited_semantic_mask,
)


def test_limited_semantics_never_infer_unmodelled_classes():
    background = np.array([[0.0, 0.8], [0.9, 0.9]])
    instance = np.array([[0, 0], [7, 0]])
    ignore = np.array([[False, False], [False, True]])
    value = limited_semantic_mask(background, instance, alpha_threshold=0.5, ignore_mask=ignore)
    assert set(np.unique(value)).issubset(LIMITED_SEMANTIC_VALUES)
    assert value.tolist() == [[0, 1], [2, 255]]
