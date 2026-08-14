from __future__ import annotations

import numpy as np
import pytest

from motion_proj.worldsim_v5.ownership_renderer import compose_ownership_opacity


def test_ownership_probability_multiplies_immutable_base_alpha() -> None:
    base = np.asarray([[0.2], [0.5], [1.0]], dtype=np.float32)
    ownership = np.asarray([0.0, 0.4, 1.0], dtype=np.float32)
    base_before = base.copy()
    ownership_before = ownership.copy()
    result = compose_ownership_opacity(base, ownership)
    assert np.allclose(result, [0.0, 0.2, 1.0])
    assert np.array_equal(base, base_before)
    assert np.array_equal(ownership, ownership_before)


@pytest.mark.parametrize(
    ("base", "ownership", "message"),
    [
        (np.asarray([1.1]), np.asarray([0.5]), "base_opacity"),
        (np.asarray([0.5]), np.asarray([np.nan]), "probability"),
        (np.asarray([0.5, 0.5]), np.asarray([0.5]), "一一对齐"),
    ],
)
def test_ownership_opacity_rejects_invalid_payload(
    base: np.ndarray, ownership: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compose_ownership_opacity(base, ownership)
