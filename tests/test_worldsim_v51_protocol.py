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
    sha256_file,
    validate_development_roles,
    validate_scope,
    validate_stage_b_authorization,
)
from scripts.fetch_worldsim_v51_dinov2_asset import (
    validate_config as validate_dino_download,
)
from scripts.fetch_worldsim_v51_dinov2_asset_parallel import (
    _ranges as parallel_ranges,
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


def test_stage_b_authorization_preserves_u2_b3_and_quality_locks() -> None:
    authorization = load_yaml(
        ROOT / "configs/worldsim_v51/stage_b_authorization_v1.yaml"
    )

    report = validate_stage_b_authorization(ROOT, authorization)

    assert report["authorized_fallback"] == "U2_B3"
    assert report["expected_image_count"] == 240
    assert report["expected_image_bytes"] == 39747172
    assert report["validation_quality_read"] is False
    assert report["test_quality_read"] is False
    assert report["m2_status"] == "pending"
    assert report["m3_status"] == "pending"


def test_stage_b_route_reordering_fails_closed() -> None:
    authorization = load_yaml(
        ROOT / "configs/worldsim_v51/stage_b_authorization_v1.yaml"
    )
    routes = authorization["m1_route_policy"]["route_order"]
    routes[0], routes[1] = routes[1], routes[0]

    with pytest.raises(ProtocolError, match="route 顺序"):
        validate_stage_b_authorization(ROOT, authorization)


def test_stage_b_dinov2_download_binds_input_freeze_and_keeps_quality_locked() -> None:
    config, freeze = validate_dino_download(
        ROOT / "configs/worldsim_v51/stage_b_dinov2_download_v1.yaml"
    )

    assert freeze["status"] == "done"
    assert freeze["frozen_denominators"]["image_count"] == 240
    assert config["asset"]["content_length_bytes"] == 4546140349
    assert config["asset"]["etag_is_sha256"] is False
    assert config["locks"]["quality_read"] is False
    assert config["locks"]["validation_quality_read"] is False
    assert config["locks"]["test_quality_read"] is False
    assert config["locks"]["m2_status"] == "pending"
    assert config["locks"]["m3_status"] == "pending"


def test_stage_b_parallel_download_and_terminal_asset_freeze_are_exact() -> None:
    config = load_yaml(
        ROOT / "configs/worldsim_v51/stage_b_dinov2_download_parallel_v1.yaml"
    )
    freeze = load_yaml(ROOT / config["input_freeze"]["path"])
    asset_freeze = load_yaml(
        ROOT / "configs/worldsim_v51/stage_b_dinov2_asset_freeze_v1.yaml"
    )
    start = config["frozen_prefix"]["bytes"]
    stop = config["asset"]["content_length_bytes"]
    ranges = parallel_ranges(
        start, stop, config["parallel_download"]["segment_count"]
    )

    assert freeze["status"] == "done"
    assert len(ranges) == 14
    assert ranges[0][0] == start
    assert ranges[-1][1] == stop - 1
    assert sum(last - first + 1 for first, last in ranges) == stop - start
    assert config["integrity"]["multipart_etag"]["expected_part_count"] == 542
    assert config["locks"]["quality_read"] is False
    assert config["locks"]["m2_status"] == "pending"
    assert config["locks"]["m3_status"] == "pending"
    assert asset_freeze["status"] == "done"
    assert asset_freeze["asset"]["bytes"] == stop
    assert asset_freeze["asset"]["multipart_part_count"] == 542
    assert asset_freeze["asset"]["temporary_prefix_and_segments_removed"] is True
    run = Path(asset_freeze["canonical_run"]["path"])
    for relative, expected in asset_freeze["canonical_run"]["hashes"].items():
        assert (run / relative).is_file()
        assert sha256_file(run / relative) == expected
