from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v51.evidence.effective_count import (
    audit_fractional_concentration_cap,
    kish_effective_count,
)


def test_kish_count_cannot_cap_unit_interval_fractional_mass() -> None:
    weights = [
        np.asarray([1.0, 1.0, 1.0]),
        np.asarray([0.5, 0.5, 0.5]),
        np.asarray([0.1, 0.4, 0.9]),
    ]
    weight_sum = np.asarray([value.sum() for value in weights])
    square_sum = np.asarray([np.square(value).sum() for value in weights])
    audit = audit_fractional_concentration_cap(
        weight_sum, square_sum, epsilon=0.0
    )

    assert np.all(
        audit["kish_effective_count_without_epsilon"]
        >= audit["fractional_concentration"]
    )
    assert np.array_equal(
        audit["capped_concentration"], audit["fractional_concentration"]
    )
    assert audit["replacement_amplification"][1] == pytest.approx(1.5)


def test_kish_formula_is_permutation_invariant_not_correlation_aware() -> None:
    first = np.asarray([0.2, 0.5, 0.8])
    second = first[[2, 0, 1]]
    assert kish_effective_count(
        np.asarray([first.sum()]),
        np.asarray([np.square(first).sum()]),
        epsilon=1e-12,
    ) == pytest.approx(
        kish_effective_count(
            np.asarray([second.sum()]),
            np.asarray([np.square(second).sum()]),
            epsilon=1e-12,
        )
    )


def test_a3_audit_config_forbids_quality_read() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/worldsim_v51/m1_effective_count_audit_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["parent"] == "A2"
    assert config["effective_count"]["correlation_observable_present"] is False
    assert config["restrictions"]["evaluation_artifact_read"] is False
    assert config["restrictions"]["gpu_renderer"] is False
