from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = (
    PROJECT_ROOT
    / "docs"
    / "run_manifests"
    / "worldsim-v5.2.1-human-review-attribution-v1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_human_attribution_manifest_is_exact_and_split_safe() -> None:
    manifest = _json(PACKAGE_DIR / "manifest.json")
    contract = _json(PACKAGE_DIR / "backtest_contract.json")
    rows = _jsonl(PACKAGE_DIR / "cases.jsonl")

    assert manifest["status"] == "done"
    assert manifest["counts"] == {
        "attribution_unresolved": 1,
        "base_failure": 9,
        "m123_confirmation": 3,
        "m123_discovery": 5,
        "m123_eligible": 8,
        "total": 18,
    }
    for name, metadata in manifest["outputs"].items():
        assert _sha256(PACKAGE_DIR / metadata["path"]) == metadata["sha256"], name

    assert len(rows) == 18
    assert [row["review_order"] for row in rows] == list(range(1, 19))
    assert len({row["case_id"] for row in rows}) == 18
    assert Counter(row["research_gate"] for row in rows) == {
        "BASE_FAILURE": 9,
        "M123_ELIGIBLE": 8,
        "ATTRIBUTION_UNRESOLVED": 1,
    }

    eligible = {row["case_id"]: row for row in rows if row["eligible_for_primary_eval"]}
    assert set(eligible) == set(contract["design_case_ids"] + contract["confirmation_case_ids"])
    assert {row["base"] for row in eligible.values()} == {"streetgs"}
    assert all(row["dataset"] == "nuscenes" for row in rows)
    assert all(row["dataset_split"] == "trainval" for row in rows)
    assert all(row["source"]["target_path"].startswith("/root/autodl-tmp/data/") for row in rows)
    assert all(len(row["source"]["target_sha256"]) == 64 for row in rows)

    assert all(eligible[case_id]["use_role"] == "design" for case_id in contract["design_case_ids"])
    assert all(
        eligible[case_id]["use_role"] == "confirmation_only"
        for case_id in contract["confirmation_case_ids"]
    )
    forbidden = set(contract["base_sentinel_case_ids"] + contract["diagnostic_only_case_ids"])
    for module in contract["module_cohorts"].values():
        assert forbidden.isdisjoint(module["discovery"])
        assert forbidden.isdisjoint(module["confirmation"])

    assert contract["locks"] == {
        "base_failure_used_for_m123_primary_eval": False,
        "confirmation_used_for_design": False,
        "fresh_validation_test_kitti_read": False,
        "manual_visual_diagnosis_is_causal_proof": False,
        "threshold_refit_on_confirmation": False,
    }


def test_autoresearch_config_binds_the_frozen_case_ids() -> None:
    config_text = (
        PROJECT_ROOT / "configs" / "worldsim_v52" / "m123_autoresearch_v1.yaml"
    ).read_text(encoding="utf-8")
    contract = _json(PACKAGE_DIR / "backtest_contract.json")

    assert "unattended_overnight: true" in config_text
    assert "user_confirmation_required: false" in config_text
    assert "confirmation_used_for_design: false" in config_text
    for case_id in contract["design_case_ids"] + contract["confirmation_case_ids"]:
        assert config_text.count(case_id) == 1
