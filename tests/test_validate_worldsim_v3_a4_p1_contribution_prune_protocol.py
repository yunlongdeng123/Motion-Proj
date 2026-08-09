from __future__ import annotations

from copy import deepcopy

from omegaconf import OmegaConf
import pytest

from scripts.validate_worldsim_v3_a4_p1_contribution_prune_protocol import (
    DEFAULT_PROTOCOL,
    validate_schema,
)


def load_protocol() -> dict:
    return OmegaConf.to_container(OmegaConf.load(DEFAULT_PROTOCOL), resolve=True)


def test_frozen_a4_p1_schema_is_valid() -> None:
    validate_schema(load_protocol())


def test_source_checkpoint_mutation_cannot_be_authorized() -> None:
    protocol = deepcopy(load_protocol())
    protocol["authorization"]["source_checkpoint_mutation_authorized"] = True
    with pytest.raises(RuntimeError, match="authorization"):
        validate_schema(protocol)


def test_p2_p3_p4_remain_unauthorized() -> None:
    protocol = deepcopy(load_protocol())
    protocol["authorization"]["p2_fp16_authorized"] = True
    with pytest.raises(RuntimeError, match="authorization"):
        validate_schema(protocol)


def test_heldout_frames_cannot_enter_ranking_partition() -> None:
    protocol = deepcopy(load_protocol())
    protocol["contribution_contract"]["training_discovery_frames"][0] = 10
    with pytest.raises(RuntimeError, match="training discovery frames"):
        validate_schema(protocol)


def test_heldout_cannot_influence_ranking() -> None:
    protocol = deepcopy(load_protocol())
    protocol["contribution_contract"]["heldout_may_influence_ranking"] = True
    with pytest.raises(RuntimeError, match="heldout leakage"):
        validate_schema(protocol)


def test_candidate_grid_cannot_be_result_selected() -> None:
    protocol = deepcopy(load_protocol())
    protocol["candidate_contract"]["arms"][2]["prune_fraction"] = 0.15
    with pytest.raises(RuntimeError, match="candidate arms"):
        validate_schema(protocol)


def test_actor_quality_threshold_cannot_be_relaxed() -> None:
    protocol = deepcopy(load_protocol())
    protocol["quality_contract"]["actor_endpoints"]["metrics"][0][
        "maximum_regression"
    ] = 0.50
    with pytest.raises(RuntimeError, match="actor quality thresholds"):
        validate_schema(protocol)


def test_frozen_candidate_masks_cannot_be_regenerated() -> None:
    protocol = deepcopy(load_protocol())
    protocol["quality_contract"]["candidate_mask_regeneration_forbidden"] = False
    with pytest.raises(RuntimeError, match="mask regeneration"):
        validate_schema(protocol)


def test_selection_fallback_remains_exact_alias() -> None:
    protocol = deepcopy(load_protocol())
    protocol["selection_contract"]["no_eligible_candidate"]["selected_asset"] = (
        "best_observed_candidate"
    )
    with pytest.raises(RuntimeError, match="fallback"):
        validate_schema(protocol)


def test_resource_ceiling_cannot_be_relaxed() -> None:
    protocol = deepcopy(load_protocol())
    protocol["resource_ceilings"]["peak_torch_allocated_mib"] = 24_000
    with pytest.raises(RuntimeError, match="resource ceilings"):
        validate_schema(protocol)


def test_completed_stage_overwrite_is_forbidden() -> None:
    protocol = deepcopy(load_protocol())
    protocol["recovery_contract"]["completed_stage_policy"] = "overwrite"
    with pytest.raises(RuntimeError, match="overwrite policy"):
        validate_schema(protocol)
