from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v51.evidence.abstention import (
    build_semantic_unknown_state,
    finalize_selective_semantic_metrics,
    posterior_entropy,
    selective_semantic_statistics,
)


def test_unknown_requires_entropy_and_sparse_or_disagreeing_evidence() -> None:
    result = build_semantic_unknown_state(
        conditional_actor_probability=np.asarray([0.5, 0.5, 0.01, 0.9]),
        effective_observation_count=np.asarray([0.1, 0.3, 0.1, 0.3]),
        cross_view_disagreement=np.asarray([0.0, 0.0, 0.1, 0.1]),
        effective_count_maximum=0.2,
        entropy_minimum=0.5,
        disagreement_minimum=0.05,
    )

    assert result["unknown_probability"].tolist() == [1.0, 0.0, 0.0, 0.0]
    total = (
        result["posterior_actor"]
        + result["posterior_background"]
        + result["unknown_probability"]
    )
    assert np.array_equal(total, np.ones(4, dtype=np.float32))
    assert posterior_entropy(np.asarray([0.5]))[0] == pytest.approx(1.0)


def test_selective_metrics_preserve_abstained_denominator() -> None:
    statistics = selective_semantic_statistics(
        probability=np.asarray([0.9, 0.8, 0.6, 0.1]),
        target=np.asarray([1, 0, 0, 0]),
        unknown_probability=np.asarray([0.0, 1.0, 1.0, 0.0]),
        probability_threshold=0.5,
        abstain_threshold=0.5,
    )
    metrics = finalize_selective_semantic_metrics(statistics)

    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["error_at_coverage"] == pytest.approx(0.0)
    assert metrics["selective_semantic_risk"] == pytest.approx(0.01)
    assert metrics["unknown_precision"] == pytest.approx(1.0)
    assert metrics["unknown_recall_on_errors"] == pytest.approx(1.0)
    assert metrics["accepted_subset_error"] == pytest.approx(0.1)
    assert metrics["abstained_subset_error"] == pytest.approx(0.7)
    assert metrics["denominators"]["total_pixel_count"] == 4


def test_empty_abstained_subset_is_explicit_not_silently_dropped() -> None:
    metrics = finalize_selective_semantic_metrics(
        selective_semantic_statistics(
            probability=np.asarray([0.1, 0.9]),
            target=np.asarray([0, 1]),
            unknown_probability=np.zeros(2),
            probability_threshold=0.5,
            abstain_threshold=0.5,
        )
    )
    assert metrics["coverage"] == 1.0
    assert metrics["unknown_precision"] is None
    assert metrics["abstained_subset_error"] is None


def test_unknown_config_is_frozen_and_quality_blind() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/worldsim_v51/m1_unary_unknown_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["status"] == "frozen_before_quality_read"
    assert config["comparator"] == "A1"
    assert config["unknown"]["calibration_quality_read"] is False
    assert config["unknown"]["rule_expression"] == (
        "high_entropy AND (low_effective_count OR high_cross_view_disagreement)"
    )
    assert config["unknown"]["image_abstain_threshold"] == 0.5
    assert config["restrictions"]["parameter_search"] is False
