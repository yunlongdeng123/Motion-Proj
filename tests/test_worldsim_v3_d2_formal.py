from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from motion_proj.worldsim_v3.d2_formal import (
    validate_a2_d2_formal_contract,
)


PROJECT = Path(__file__).resolve().parents[1]


def contract() -> dict:
    return yaml.safe_load(
        (PROJECT / "configs/worldsim_v3/a2_d2_formal_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_d2_formal_contract_is_valid() -> None:
    validate_a2_d2_formal_contract(contract())


def test_d2_formal_rejects_retraining_d1_alias() -> None:
    drifted = contract()
    drifted["d1_reference_alias"]["mode"] = "retrain"
    with pytest.raises(ValueError, match="immutable exact alias"):
        validate_a2_d2_formal_contract(drifted)


def test_d2_formal_rejects_matched_target_drift() -> None:
    drifted = deepcopy(contract())
    drifted["matched_gaussian_budget"]["target_count"] += 1
    with pytest.raises(ValueError, match="matched D1 target"):
        validate_a2_d2_formal_contract(drifted)


def test_d2_formal_keeps_d3_conditional() -> None:
    drifted = deepcopy(contract())
    drifted["formal_gate"]["d3_not_automatically_unlocked"] = False
    with pytest.raises(ValueError, match="D3 must remain conditional"):
        validate_a2_d2_formal_contract(drifted)
