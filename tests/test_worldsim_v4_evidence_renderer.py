from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v4.evidence_renderer import probability_to_logit


def test_probability_logit_roundtrip() -> None:
    probability = np.asarray([0.0, 0.1, 0.5, 0.9, 1.0])
    logits = probability_to_logit(probability)
    restored = 1.0 / (1.0 + np.exp(-logits))
    np.testing.assert_allclose(restored[1:-1], probability[1:-1], atol=1e-6)
    assert np.isfinite(logits).all()
