import json
from pathlib import Path

import numpy as np

from motion_proj.worldsim_v7.actor_reliability import ActorResidualDistribution
from motion_proj.worldsim_v7.boundary_cost_density import (
    LogCostMixtureDensity,
    boundary_state_cost,
)
from motion_proj.worldsim_v7.physical_compiler import (
    ActorState,
    HazardPreservingPhysicalCompiler,
    PhysicalEvidence,
    SurfaceAction,
)
from motion_proj.worldsim_v7.runtime_surface import (
    ReliabilitySurface,
    apply_binary_isotonic_map,
    fit_binary_isotonic_map,
)
from motion_proj.worldsim_v7.sceneir_adapter import SE3
from motion_proj.worldsim_v7.validity_hazard import (
    HazardFeatures,
    ValidityFeatures,
    ValidityHazardFactorizer,
)
from scripts.prepare_worldsim_v7_av2 import validate_cohort


def _actor() -> ActorState:
    return ActorState("actor-7", ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (4.2, 1.8, 1.6))


def test_physical_compiler_exposes_four_actions_without_actor_mutation():
    actor = _actor()
    compiler = HazardPreservingPhysicalCompiler()
    evidence = [
        PhysicalEvidence(actor, "keep", 4, 3, 2, True, 0.0, 0.02),
        PhysicalEvidence(actor, "project", 4, 3, 2, True, 0.08, 0.12),
        PhysicalEvidence(actor, "complete", 4, 3, 3, True, 0.0, 0.02, hole_radius_m=0.3),
        PhysicalEvidence(actor, "unknown", 0, 0, 0, False, 0.0, 0.0),
    ]
    decisions = compiler.compile_many(evidence)
    assert {decision.action for decision in decisions} == set(SurfaceAction)
    assert all(decision.actor == actor for decision in decisions)
    assert not decisions[-1].collision_query_enabled


def test_validity_and_hazard_inputs_are_conditionally_separated():
    factorizer = ValidityHazardFactorizer()
    clean = ValidityFeatures(0.0, 0.02, 0.01, 0.0, 1.0, 0.99)
    artifact = ValidityFeatures(0.7, 0.8, 0.5, 0.4, 0.1, 0.2)
    safe = HazardFeatures(8.0, 8.0, 0.0, 0.0, 0.0)
    hazardous = HazardFeatures(0.7, 0.4, 12.0, 0.9, 0.8)
    assert factorizer.score(clean, safe).artifact_probability == factorizer.score(clean, hazardous).artifact_probability
    assert factorizer.score(clean, hazardous).hazard_probability == factorizer.score(artifact, hazardous).hazard_probability
    assert factorizer.score(artifact, safe).artifact_probability > factorizer.score(clean, safe).artifact_probability
    assert factorizer.score(clean, hazardous).hazard_probability > factorizer.score(clean, safe).hazard_probability


def test_actor_projection_boundary_cost_density_and_surface_are_monotone():
    residual = ActorResidualDistribution(
        actor_id="actor-7",
        horizons_s=np.asarray([1.0, 2.0]),
        mean_xy_m=np.asarray([[0.1, 0.0], [0.2, 0.0]]),
        covariance_xy_m2=np.asarray([np.eye(2) * 0.01, np.eye(2) * 0.04]),
    )
    projected_mean, projected_scale = residual.project_to_boundary(
        np.asarray([[1.0, 0.0], [1.0, 0.0]])
    )
    assert np.allclose(projected_mean, [0.1, 0.2])
    assert np.allclose(projected_scale, [0.1, 0.2])

    samples = residual.sample(16, seed=3)
    costs = boundary_state_cost(
        samples,
        np.asarray([[1.0, 0.0], [1.0, 0.0]]),
        np.asarray([1.0, 0.5]),
    )
    assert costs.shape == (16,)
    assert np.all(costs >= 0.0)

    densities = [
        LogCostMixtureDensity(np.asarray([1.0]), np.asarray([-1.0]), np.asarray([0.4])),
        LogCostMixtureDensity(np.asarray([1.0]), np.asarray([0.0]), np.asarray([0.5])),
    ]
    surface = ReliabilitySurface.from_densities(
        np.asarray([1.0, 2.0]), np.asarray([0.1, 0.5, 1.0]), densities
    )
    assert np.all(np.diff(surface.probabilities, axis=1) >= 0.0)
    assert np.all(np.diff(surface.probabilities, axis=0) <= 0.0)
    assert surface.query(1.5, 0.8) <= surface.query(1.5, 1.0)


def test_isotonic_map_and_se3_inverse():
    thresholds, values = fit_binary_isotonic_map(
        np.asarray([0.1, 0.2, 0.3, 0.4]), np.asarray([0.0, 1.0, 0.0, 1.0])
    )
    calibrated = apply_binary_isotonic_map(np.asarray([0.1, 0.2, 0.3, 0.4]), thresholds, values)
    assert np.all(np.diff(calibrated) >= 0.0)

    tied_thresholds, tied_values = fit_binary_isotonic_map(
        np.asarray([0.1, 0.1, 0.2]), np.asarray([0.0, 1.0, 1.0])
    )
    assert np.allclose(tied_thresholds, [0.1, 0.2])
    assert np.allclose(tied_values, [0.5, 1.0])

    transform = SE3.from_quaternion_translation(
        np.asarray([1.0, 0.0, 0.0, 0.0]), np.asarray([1.0, 2.0, 3.0])
    )
    points = np.asarray([[2.0, 4.0, 6.0]])
    assert np.allclose(transform.inverse().transform_points(transform.transform_points(points)), points)


def test_frozen_av2_cohort_enforces_zero_shot_contract():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "worldsim_v7"
        / "av2_zero_shot_cohort_v1.json"
    )
    cohort = json.loads(config_path.read_text(encoding="utf-8"))
    assert validate_cohort(cohort) is cohort
    assert [row["index"] for row in cohort["logs"]] == list(range(0, 150, 5))
