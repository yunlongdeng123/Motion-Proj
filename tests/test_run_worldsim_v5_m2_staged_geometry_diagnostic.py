from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.run_worldsim_v5_m2_staged_geometry_diagnostic import (
    M2StagedGeometryError,
    load_config,
    mechanism_conclusion,
)


def _gates() -> dict:
    return {
        "minimum_evaluable_views": 4,
        "gaussianization_primary": {
            "minimum_view_count": 4,
            "minimum_post_minus_pre_mae_m": 0.1,
        },
        "candidate_builder_primary": {
            "minimum_view_count": 4,
            "minimum_raw_mae_m": 0.5,
        },
        "unlock_g3_if": {
            "minimum_g0_raw_failure_views": 3,
            "g0_raw_failure_mae_m": 0.5,
        },
    }


def _row(raw: float, delta: float, status: str = "done") -> dict:
    return {
        "status": status,
        "staged_metrics": {
            "raw_geometry_error": {"mae_m": raw},
            "gaussianization_delta_mae_m": delta,
        },
    }


def test_mechanism_conclusion_prioritizes_gaussianization() -> None:
    rows = [_row(0.7, 0.2) for _ in range(4)] + [_row(0.1, 0.0) for _ in range(2)]
    result = mechanism_conclusion(rows, _gates())
    assert result["conclusion"] == "gaussianization_is_primary_mechanism_on_model_proxy"
    assert result["g3_unlocked_for_next_development_run"] is True


def test_mechanism_conclusion_abstains_on_short_denominator() -> None:
    rows = [_row(0.8, 0.3), _row(0.8, 0.3), {"status": "abstain"}]
    result = mechanism_conclusion(rows, _gates())
    assert result["conclusion"] == "insufficient_evaluable_views_keep_g3_locked"
    assert result["g3_unlocked_for_next_development_run"] is False


def test_four_evaluable_and_two_unavailable_preserve_frozen_denominator() -> None:
    rows = [_row(0.1, 0.0) for _ in range(4)] + [
        {"status": "abstain", "reason": "ABSTAIN_SAM_MASK_UNAVAILABLE"},
        {"status": "abstain", "reason": "ABSTAIN_SAM_MASK_UNAVAILABLE"},
    ]
    result = mechanism_conclusion(rows, _gates())
    assert result["evaluable_view_count"] == 4
    assert result["conclusion"] == "g0_and_gaussianization_not_primary_on_model_proxy"


def test_config_freezes_development_only_contract(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
schema_version: worldsim_v5_m2_geometry_first_development_v1
task_id: WS-V5-M2-GEOMETRY-FIRST-REPAIR-01
status: running
phase: staged_geometry_contract_smoke
scope:
  validation_quality_read: false
  heldout_quality_read: false
  test_quality_read: false
  parameter_search_performed: false
  router_refit_performed: false
view_protocol: {expected_view_count: 1, frames: [2], cameras: [0]}
surface: {active_models: [G0_ROBUST_PLANE]}
reference: {independent_geometry_claim_allowed: false}
""",
        encoding="utf-8",
    )
    assert load_config(path)["status"] == "running"
    payload = path.read_text().replace("test_quality_read: false", "test_quality_read: true")
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(M2StagedGeometryError, match="restriction"):
        load_config(path)
