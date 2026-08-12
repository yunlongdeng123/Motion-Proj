from __future__ import annotations

import numpy as np
import pytest

from motion_proj.worldsim_v4.beta_fusion import (
    BetaEvidence,
    prior_from_probability,
    product_weight,
    update_beta_evidence,
    update_from_counts,
)


def test_beta_update_matches_preregistered_equations() -> None:
    prior = prior_from_probability(np.asarray([0.25, 0.75]), strength=4.0)
    fused = update_beta_evidence(
        prior,
        evidence=np.asarray([1.0, 0.0]),
        weight=np.asarray([2.0, 3.0]),
    )
    np.testing.assert_allclose(fused.alpha, [3.0, 3.0])
    np.testing.assert_allclose(fused.beta, [3.0, 4.0])
    np.testing.assert_allclose(fused.posterior, [0.5, 3.0 / 7.0])
    np.testing.assert_allclose(
        fused.uncertainty,
        fused.alpha * fused.beta / ((fused.alpha + fused.beta) ** 2 * (fused.alpha + fused.beta + 1.0)),
    )


def test_count_fusion_multiplies_mask_visibility_depth_and_lidar() -> None:
    prior = BetaEvidence(np.ones(2), np.ones(2))
    factors = {
        "mask": np.asarray([0.5, 1.0]),
        "visibility": np.asarray([1.0, 0.5]),
        "depth": 0.8,
        "lidar": np.asarray([0.5, 1.0]),
    }
    np.testing.assert_allclose(product_weight(factors, shape=(2,)), [0.2, 0.4])
    fused = update_from_counts(
        prior,
        positive_count=np.asarray([3, 2]),
        negative_count=np.asarray([1, 4]),
        factors=factors,
    )
    np.testing.assert_allclose(fused.alpha, [1.6, 1.8])
    np.testing.assert_allclose(fused.beta, [1.2, 2.6])


def test_invalid_beta_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        BetaEvidence(np.asarray([0.0]), np.asarray([1.0]))
    with pytest.raises(ValueError):
        product_weight({"depth": np.asarray([1.1])}, shape=(1,))
