from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from motion_proj.worldsim_v3.a2_formal import (
    QUALITY_DIRECTIONS,
    compare_vectors,
    quality_vector,
    select_matched_checkpoint,
    validate_a2_d1_formal_contract,
)


PROJECT = Path(__file__).resolve().parents[1]


def test_frozen_formal_contract_is_valid() -> None:
    contract = yaml.safe_load(
        (PROJECT / "configs/worldsim_v3/a2_d1_formal_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    validate_a2_d1_formal_contract(contract)
    drifted = deepcopy(contract)
    drifted["matched_gaussian_budget"]["maximum_relative_gap"] = 0.05
    with pytest.raises(ValueError, match="relative-gap"):
        validate_a2_d1_formal_contract(drifted)


def test_matched_checkpoint_uses_closest_then_earliest() -> None:
    result = select_matched_checkpoint(
        100,
        [
            {"step": 15_000, "rigid_gaussians": 102, "checkpoint": "later"},
            {"step": 10_000, "rigid_gaussians": 98, "checkpoint": "earlier"},
            {"step": 5_000, "rigid_gaussians": 80, "checkpoint": "far"},
        ],
        0.02,
    )
    assert result["status"] == "done"
    assert result["selected"]["step"] == 10_000
    assert result["selected"]["relative_gap"] == pytest.approx(0.02)


def test_matched_checkpoint_abstains_without_mutating_budget() -> None:
    result = select_matched_checkpoint(
        100,
        [{"step": 5_000, "rigid_gaussians": 90, "checkpoint": "candidate"}],
        0.02,
    )
    assert result["status"] == "ABSTAIN_BUDGET_NOT_MATCHED"
    assert result["read_only"] is True


def _region(value: float) -> dict[str, float | str]:
    return {
        "status": "done",
        "psnr": 20.0 + value,
        "ssim": 0.7 + value / 100,
        "masked_lpips_alex_tight_crop_256px": 0.2 - value / 100,
    }


def _evaluation(value: float) -> dict:
    return {
        "heldout_metrics": {
            "image_metrics/test/psnr": 25.0 + value,
            "image_metrics/test/ssim": 0.8 + value / 100,
            "image_metrics/test/lpips": 0.2 - value / 100,
        },
        "actor_metrics": {
            "roles": {
                role: {
                    "status": "done",
                    "actor_region": _region(value),
                    "boundary_band": _region(value),
                }
                for role in ("high-support", "boundary-support")
            },
            "non_target": {
                "status": "done",
                "quality": {
                    "status": "done",
                    "psnr": 30.0 + value,
                    "ssim": 0.9 + value / 100,
                    "masked_lpips_alex_tight_crop_256px": 0.1 - value / 100,
                    "mean_absolute_error": 0.05 - value / 1000,
                },
            },
        },
    }


def test_quality_vector_and_exact_pareto_dominance() -> None:
    d0 = quality_vector(_evaluation(0.0))
    d1 = quality_vector(_evaluation(1.0))
    assert set(d0) == set(QUALITY_DIRECTIONS)
    result = compare_vectors(d0, d1, QUALITY_DIRECTIONS)
    assert result["verdict"] == "d1_strictly_dominates_d0"
    assert result["d0_better_axis_count"] == 0
    assert result["no_numeric_tolerance"] is True


def test_exact_pareto_reports_tradeoff() -> None:
    d0 = {"quality": 1.0, "cost": 1.0}
    d1 = {"quality": 2.0, "cost": 2.0}
    result = compare_vectors(d0, d1, {"quality": "max", "cost": "min"})
    assert result["verdict"] == "tradeoff_non_dominated"
