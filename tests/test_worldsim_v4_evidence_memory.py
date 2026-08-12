from __future__ import annotations

import numpy as np
import pytest

from motion_proj.worldsim_v4.beta_fusion import BetaEvidence
from motion_proj.worldsim_v4.evidence_temporal import TemporalEvidenceMemory, temporal_update


def test_temporal_memory_uses_frozen_beta_parameter_ema() -> None:
    previous = BetaEvidence(np.asarray([9.0]), np.asarray([1.0]))
    observed = BetaEvidence(np.asarray([1.0]), np.asarray([9.0]))
    updated = temporal_update(previous, observed, retention=0.75)
    np.testing.assert_allclose(updated.alpha, [7.0])
    np.testing.assert_allclose(updated.beta, [3.0])


def test_temporal_memory_rejects_identity_order_drift() -> None:
    memory = TemporalEvidenceMemory(np.asarray([10, 20]), retention=0.5)
    observed = BetaEvidence(np.ones(2), np.ones(2))
    memory.update(np.asarray([10, 20]), observed)
    with pytest.raises(ValueError, match="identity/order"):
        memory.update(np.asarray([20, 10]), observed)
