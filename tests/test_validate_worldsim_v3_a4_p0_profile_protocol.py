from __future__ import annotations

from copy import deepcopy

from omegaconf import OmegaConf
import pytest

from scripts.validate_worldsim_v3_a4_p0_profile_protocol import (
    DEFAULT_PROTOCOL,
    validate_schema,
)


def load_protocol() -> dict:
    return OmegaConf.to_container(OmegaConf.load(DEFAULT_PROTOCOL), resolve=True)


def test_frozen_a4_p0_schema_is_valid() -> None:
    validate_schema(load_protocol())


def test_rejected_r1_cannot_be_profile_input() -> None:
    protocol = deepcopy(load_protocol())
    protocol["selected_asset"]["variant"] = "r1-reactivate"
    with pytest.raises(RuntimeError, match="selected variant"):
        validate_schema(protocol)


def test_gpu_ceiling_cannot_be_relaxed() -> None:
    protocol = deepcopy(load_protocol())
    protocol["resource_ceilings"]["peak_torch_allocated_mib"] = 20000
    with pytest.raises(RuntimeError, match="resource ceilings"):
        validate_schema(protocol)


def test_completed_stage_overwrite_is_forbidden() -> None:
    protocol = deepcopy(load_protocol())
    protocol["recovery_contract"]["completed_stage_policy"] = "overwrite"
    with pytest.raises(RuntimeError, match="overwrite policy"):
        validate_schema(protocol)
