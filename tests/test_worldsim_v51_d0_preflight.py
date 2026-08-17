from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v51.protocol import sha256_file
from scripts.audit_worldsim_v51_d0_preflight import THRESHOLDS, validate_config


CONFIG = ROOT / "configs/worldsim_v51/stage_d_progressive_preflight_v1.yaml"
FREEZE = ROOT / "configs/worldsim_v51/stage_d_progressive_preflight_freeze_v1.yaml"


def test_d0_preregistration_freezes_faithful_mechanism_and_route() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    method = config["faithful_mechanism"]
    assert tuple(method["progressive_thresholds"]) == THRESHOLDS
    assert method["maximum_logical_distance"] == 2
    assert method["logical_distance_decay"] == 0.5
    assert method["pair_similarity"] == "cosine"
    assert method["seed_source"] == "frozen_u2_b3_posterior"
    assert method["node"] == "raw_gaussian"
    assert method["node_change"] is False
    assert method["parameter_search"] is False
    assert config["route"]["next_route_on_rejection"] == "SUPER_PRIMITIVE_OR_ANCHOR"
    assert config["failure_ledger_refs"] == [
        "V51-F31",
        "V51-F32",
        "V51-F33",
        "V51-F34",
        "V51-F35",
    ]


def test_d0_preregistration_keeps_all_future_quality_locked() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["preflight"]["quality_read"] is False
    assert config["preflight"]["inspect_quality_payloads"] is False
    locks = config["locks"]
    assert locks["screening_quality_read"] is False
    assert locks["confirmation_quality_read"] is False
    assert locks["validation_quality_read"] is False
    assert locks["test_quality_read"] is False
    assert locks["m2_status"] == "pending"
    assert locks["m3_status"] == "pending"


def test_d0_h_gate_matches_normative_candidate_promotion_rule() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    gate = config["h_gate_after_operator_freeze"]
    assert gate["minimum_positive_boundary_f1_scenes"] == 2
    assert gate["minimum_scene_balanced_boundary_f1_delta_exclusive"] == 0.0
    assert gate["minimum_scene_balanced_iou_delta"] == 0.0
    assert gate["maximum_scene_balanced_false_negative_semantic_mass_delta"] == 0.02
    assert gate["parameter_search_after_read"] is False


def test_d0_preflight_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/audit_worldsim_v51_d0_preflight.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in result.stdout
    assert "--output" in result.stdout


def test_d0_preflight_freeze_binds_byte_exact_no_quality_run() -> None:
    freeze = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))
    assert freeze["status"] == "done"
    assert freeze["canonical_run"]["byte_exact_replay"] is True
    report_spec = freeze["canonical_run"]["report"]
    report = Path(freeze["canonical_run"]["path"]) / report_spec["path"]
    assert report.is_file()
    assert report.stat().st_size == report_spec["bytes"]
    assert sha256_file(report) == report_spec["sha256"]
    assert freeze["locks"]["quality_read"] is False
    assert freeze["locks"]["screening_quality_read"] is False
    assert freeze["locks"]["confirmation_quality_read"] is False
    assert freeze["locks"]["validation_quality_read"] is False
    assert freeze["locks"]["test_quality_read"] is False
    assert freeze["locks"]["m2_status"] == "pending"
    assert freeze["locks"]["m3_status"] == "pending"
