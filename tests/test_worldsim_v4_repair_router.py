from __future__ import annotations

import pytest

from motion_proj.worldsim_v4.repair_candidates import (
    GaussianAssetBinding,
    RepairCandidate,
)
from motion_proj.worldsim_v4.repair_compiler import (
    RepairRequest,
    compile_repair_delta,
)
from motion_proj.worldsim_v4.repair_risk import (
    RepairRiskWeights,
    score_repair_candidate,
)
from motion_proj.worldsim_v4.repair_router import (
    RiskRepairRouter,
    route_hard_priority,
)
from motion_proj.worldsim_v4.selective_metrics import (
    SelectiveSample,
    selective_group_metrics,
    selective_risk_curve,
)


def asset(name: str) -> GaussianAssetBinding:
    return GaussianAssetBinding(
        path=f"/assets/{name}.npz",
        sha256="a" * 64,
        bytes=128,
        gaussian_count=4,
    )


def candidate(
    identifier: str,
    method: str,
    *,
    photo: float,
    geometry: float,
    temporal: float,
    uncertainty: float,
    cost: float = 0.0,
) -> RepairCandidate:
    return RepairCandidate(
        candidate_id=identifier,
        method=method,
        gaussians=asset(identifier),
        photo_risk=photo,
        geometry_risk=geometry,
        temporal_risk=temporal,
        uncertainty=uncertainty,
        compute_cost=cost,
        provenance=method.lower(),
    )


def weights() -> RepairRiskWeights:
    return RepairRiskWeights(1.0, 2.0, 3.0, 4.0, 0.5)


def test_candidate_rejects_unbounded_risk() -> None:
    with pytest.raises(ValueError, match="photo_risk"):
        candidate(
            "bad",
            "DONOR",
            photo=1.1,
            geometry=0.0,
            temporal=0.0,
            uncertainty=0.0,
        )


def test_risk_score_persists_every_weighted_term() -> None:
    row = candidate(
        "donor",
        "DONOR",
        photo=0.1,
        geometry=0.2,
        temporal=0.3,
        uncertainty=0.4,
        cost=0.5,
    )
    score = score_repair_candidate(row, weights())
    assert score.photo == pytest.approx(0.1)
    assert score.geometry == pytest.approx(0.4)
    assert score.temporal == pytest.approx(0.9)
    assert score.uncertainty == pytest.approx(1.6)
    assert score.compute_cost == pytest.approx(0.25)
    assert score.total == pytest.approx(3.25)


def test_risk_router_selects_minimum_total_not_hard_priority() -> None:
    observed = candidate(
        "observed",
        "OBSERVED",
        photo=0.8,
        geometry=0.8,
        temporal=0.8,
        uncertainty=0.8,
    )
    donor = candidate(
        "donor",
        "DONOR",
        photo=0.05,
        geometry=0.05,
        temporal=0.05,
        uncertainty=0.05,
    )
    decision = RiskRepairRouter(weights=weights(), threshold=1.0).route(
        [observed, donor]
    )
    assert decision.action == "DONOR"
    assert decision.candidate_id == "donor"
    assert decision.accepted


def test_risk_router_abstains_above_threshold() -> None:
    decision = RiskRepairRouter(weights=weights(), threshold=0.1).route(
        [
            candidate(
                "generated",
                "GENERATED",
                photo=0.2,
                geometry=0.2,
                temporal=0.2,
                uncertainty=0.2,
            )
        ]
    )
    assert decision.action == "ABSTAIN"
    assert decision.abstain_reason == "ABSTAIN_RISK_THRESHOLD"
    assert decision.selected_score is not None


def test_risk_router_tie_break_is_explicit_and_deterministic() -> None:
    rows = [
        candidate(
            "z-donor",
            "DONOR",
            photo=0.1,
            geometry=0.1,
            temporal=0.1,
            uncertainty=0.1,
        ),
        candidate(
            "a-observed",
            "OBSERVED",
            photo=0.1,
            geometry=0.1,
            temporal=0.1,
            uncertainty=0.1,
        ),
    ]
    decision = RiskRepairRouter(weights=weights(), threshold=2.0).route(rows)
    assert decision.candidate_id == "a-observed"


def test_hard_priority_is_a_separate_ablation() -> None:
    rows = [
        candidate(
            "donor",
            "DONOR",
            photo=0.0,
            geometry=0.0,
            temporal=0.0,
            uncertainty=0.0,
        ),
        candidate(
            "observed",
            "OBSERVED",
            photo=1.0,
            geometry=1.0,
            temporal=1.0,
            uncertainty=1.0,
        ),
    ]
    decision = route_hard_priority(rows)
    assert decision.action == "OBSERVED"
    assert decision.policy == "hard_priority"


def request() -> RepairRequest:
    return RepairRequest(
        request_id="scene-0994-remove-high",
        scene="scene-0994",
        hole_id="high-support",
        frames=(10, 11),
        camera_ids=(0, 1),
        erase_gaussian_ids=(3, 4),
        base_checkpoint_sha256="b" * 64,
    )


def test_compiler_abstain_is_atomic_noop() -> None:
    row = candidate(
        "unsafe",
        "GENERATED",
        photo=1.0,
        geometry=1.0,
        temporal=1.0,
        uncertainty=1.0,
    )
    delta = compile_repair_delta(
        request=request(),
        candidates=[row],
        router=RiskRepairRouter(weights=weights(), threshold=0.1),
    )
    assert delta.applied_erase_gaussian_ids == ()
    assert delta.insert_asset is None
    assert not delta.base_mutated
    assert delta.to_dict()["rollback_checkpoint_sha256"] == "b" * 64


def test_compiler_accepts_content_addressed_delta_without_base_mutation() -> None:
    row = candidate(
        "safe",
        "DONOR",
        photo=0.01,
        geometry=0.01,
        temporal=0.01,
        uncertainty=0.01,
    )
    delta = compile_repair_delta(
        request=request(),
        candidates=[row],
        router=RiskRepairRouter(weights=weights(), threshold=1.0),
    )
    assert delta.applied_erase_gaussian_ids == (3, 4)
    assert delta.insert_asset == row.gaussians
    assert delta.decision.action == "DONOR"
    assert len(delta.sha256) == 64


def test_selective_group_gate_requires_abstain_error_separation() -> None:
    metrics = selective_group_metrics(
        [
            SelectiveSample("a", 0.1, 0.2, True),
            SelectiveSample("b", 0.2, 0.4, True),
            SelectiveSample("c", 0.9, 1.2, False),
        ]
    )
    assert metrics["coverage"] == pytest.approx(2.0 / 3.0)
    assert metrics["accepted_mean_error"] == pytest.approx(0.3)
    assert metrics["abstain_mean_error"] == pytest.approx(1.2)
    assert metrics["meaningful_abstention_gate"] is True


def test_selective_curve_retains_lowest_uncertainty() -> None:
    curve = selective_risk_curve(
        [
            SelectiveSample("high", 0.9, 2.0, False),
            SelectiveSample("low", 0.1, 0.1, True),
            SelectiveSample("mid", 0.5, 0.6, True),
        ],
        requested_coverages=[1.0 / 3.0, 2.0 / 3.0, 1.0],
    )
    assert [row["retained_count"] for row in curve] == [1, 2, 3]
    assert curve[0]["mean_error"] == pytest.approx(0.1)
    assert curve[-1]["mean_error"] == pytest.approx(0.9)
