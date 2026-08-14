from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STATUSES = {"pending", "running", "blocked", "done", "rejected"}


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_v5_p0_scope_authorizes_only_first_phase() -> None:
    scope = _yaml(ROOT / "configs/worldsim_v5/p0_scope_v1.yaml")
    assert scope["schema_version"] == "worldsim_v5_p0_scope_v1"
    assert scope["status"] == "running"
    assert set(scope["task_registry"].values()) <= ALLOWED_STATUSES
    assert scope["task_registry"]["WS-V5-M1B-REVERSIBLE-SEMANTIC-SPLIT-01"] == "pending"
    assert scope["task_registry"]["WS-V5-D1-KITTI-ADAPTER-01"] == "blocked"
    assert scope["authorization"]["allowed_now"] == [
        "WS-V5-P0-SCOPE-FREEZE-01",
        "WS-V5-M1-D0-BAYES-FORENSICS-01",
        "WS-V5-M2-D0-GEOMETRY-FORENSICS-01",
        "result_blind_dataset_and_adapter_audit",
    ]
    assert scope["closeout_gate"]["training_started"] is False
    assert scope["closeout_gate"]["fresh_quality_read"] is False
    for output in scope["outputs"].values():
        assert (ROOT / output).is_file()


def test_fresh_cohort_contract_excludes_every_v4_scene() -> None:
    fresh = _yaml(ROOT / "configs/worldsim_v5/nuscenes_fresh_cohort_v1.yaml")
    v4 = _yaml(ROOT / "configs/worldsim_v4/nuscenes_cohort_v1.yaml")
    old_roles = v4["freeze"]["scene_roles"]
    old_scenes = set(
        old_roles["development"] + old_roles["validation"] + old_roles["test"]
    )
    excluded = fresh["v4_exclusion"]["scenes"]
    assert len(excluded) == len(set(excluded)) == 30
    assert set(excluded) == old_scenes
    assert fresh["protocol"]["scene_counts"] == {
        "development": 8,
        "validation": 8,
        "test": 20,
    }
    assert fresh["protocol"]["total_scene_count"] == 36
    assert fresh["freeze"]["selection_status"] == "pending"
    assert fresh["freeze"]["scene_roles"] == {
        "development": [],
        "validation": [],
        "test": [],
    }
    assert fresh["restrictions"]["fresh_test_quality_read"] is False
    assert fresh["restrictions"]["v4_scene_confirmatory_reuse"] is False


def test_v5_navigation_and_forensic_boundaries_are_explicit() -> None:
    plan = (ROOT / "docs/WORLDSIM_V5_STRUCTDELTA_PLAN.md").read_text(encoding="utf-8")
    m1 = (ROOT / "docs/WS_V5_M1_FAILURE_FORENSICS.md").read_text(encoding="utf-8")
    m2 = (ROOT / "docs/WS_V5_M2_GEOMETRY_FORENSICS.md").read_text(encoding="utf-8")
    assert "WS-V5-P0-SCOPE-FREEZE-01=running" in plan
    assert "fresh quality 未读" in plan
    assert "per-view observation" in m1
    assert "blocked_evidence_missing" in m1
    assert "192/214" in m2
    assert "accepted-only repair 并未" in m2
    assert "m1_forensics_v1.yaml" in m1
    assert "m2_forensics_v1.yaml" in m2


def test_p0_forensic_machine_contracts_are_registered() -> None:
    scope = _yaml(ROOT / "configs/worldsim_v5/p0_scope_v1.yaml")
    m1 = _yaml(ROOT / "configs/worldsim_v5/m1_forensics_v1.yaml")
    m2 = _yaml(ROOT / "configs/worldsim_v5/m2_forensics_v1.yaml")
    assert m1["scope"] == "v4_historical_diagnostic_only"
    assert m2["scope"] == "v4_historical_diagnostic_only"
    assert m1["restrictions"]["parameter_search_performed"] is False
    assert m2["restrictions"]["router_refit_performed"] is False
    assert m1["output_contract"]["state_audit"] == "artifacts/state_audit.json"
    assert m2["output_contract"]["geometry_audit"] == "artifacts/geometry_audit.json"
    assert "m1_formal_forensic_run_done" in scope["closeout_gate"]["requires"]
    assert "m2_formal_forensic_run_done" in scope["closeout_gate"]["requires"]
