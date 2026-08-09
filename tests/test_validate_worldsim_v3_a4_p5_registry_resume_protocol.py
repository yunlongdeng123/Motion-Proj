from __future__ import annotations

from copy import deepcopy

from omegaconf import OmegaConf
import pytest

from scripts.validate_worldsim_v3_a4_p5_registry_resume_protocol import (
    DEFAULT_PROTOCOL,
    validate_schema,
)


def load_protocol() -> dict:
    return OmegaConf.to_container(OmegaConf.load(DEFAULT_PROTOCOL), resolve=True)


def test_frozen_a4_p5_schema_is_valid() -> None:
    validate_schema(load_protocol())


def test_checkpoint_copy_cannot_be_authorized() -> None:
    protocol = deepcopy(load_protocol())
    protocol["authorization"]["checkpoint_copy_authorized"] = True
    with pytest.raises(RuntimeError, match="forbidden authorization"):
        validate_schema(protocol)


def test_p1_p2_p3_remain_unauthorized() -> None:
    protocol = deepcopy(load_protocol())
    protocol["authorization"]["p2_fp16_authorized"] = True
    with pytest.raises(RuntimeError, match="forbidden authorization"):
        validate_schema(protocol)


def test_registry_counts_are_frozen() -> None:
    protocol = deepcopy(load_protocol())
    protocol["registry_contract"]["actor_assets"]["available_actor_count"] = 24
    with pytest.raises(RuntimeError, match="actor assets"):
        validate_schema(protocol)


def test_compact_actor_fields_are_frozen() -> None:
    protocol = deepcopy(load_protocol())
    protocol["registry_contract"]["actor_assets"]["required_compact_fields"].pop()
    with pytest.raises(RuntimeError, match="actor compact fields"):
        validate_schema(protocol)


def test_resource_ceiling_cannot_be_relaxed() -> None:
    protocol = deepcopy(load_protocol())
    protocol["resource_ceilings"]["wall_time_seconds"] = 300
    with pytest.raises(RuntimeError, match="resource ceilings"):
        validate_schema(protocol)


def test_completed_stage_overwrite_is_forbidden() -> None:
    protocol = deepcopy(load_protocol())
    protocol["recovery_contract"]["completed_stage_policy"] = "overwrite"
    with pytest.raises(RuntimeError, match="overwrite policy"):
        validate_schema(protocol)
