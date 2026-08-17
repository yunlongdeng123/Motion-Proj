from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v51.protocol import (
    DEVELOPMENT_ROLE_ORDER,
    ProtocolError,
    load_yaml,
    validate_development_roles,
    validate_scope,
)


def test_scope_and_development_roles_bind_v5_cohort() -> None:
    scope = load_yaml(ROOT / "configs/worldsim_v51/p0_m1_scope_v1.yaml")
    roles = load_yaml(ROOT / "configs/worldsim_v51/development_roles_v1.yaml")

    scope_report = validate_scope(ROOT, scope)
    roles_report = validate_development_roles(ROOT, roles)

    assert scope_report["cohort_sha256"] == (
        "553373159023218b44615be27aeeb5533a6c585be276e06425235fe09b6b48b1"
    )
    assert tuple(
        roles_report["historical_diagnostic"]
        + roles_report["screening"]
        + roles_report["development_confirmation"]
    ) == DEVELOPMENT_ROLE_ORDER
    assert roles_report["validation_quality_read"] is False
    assert roles_report["test_quality_read"] is False


def test_development_role_reselection_fails_closed() -> None:
    roles = load_yaml(ROOT / "configs/worldsim_v51/development_roles_v1.yaml")
    roles["roles"]["screening"]["scenes"] = ["scene-0359", "scene-0998"]

    with pytest.raises(ProtocolError, match="场景身份或顺序漂移"):
        validate_development_roles(ROOT, roles)


def test_scope_requires_m2_m3_pending() -> None:
    scope = load_yaml(ROOT / "configs/worldsim_v51/p0_m1_scope_v1.yaml")
    scope["first_round_authorization"]["m2_status"] = "running"

    with pytest.raises(ProtocolError, match="M2/M3"):
        validate_scope(ROOT, scope)
