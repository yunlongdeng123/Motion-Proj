from __future__ import annotations

import copy
import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml

from motion_proj.worldsim_v4.p0_contract import (
    P0ContractError,
    SCHEMA_VERSION,
    TASK_ID,
    audit_config,
    run_audit,
)


def _config() -> dict:
    literature = [
        {
            "name": f"source-{index}",
            "primary_url": f"https://example.com/source-{index}",
            "role": "paper_only",
            "execution_state": "paper_only",
        }
        for index in range(12)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": "done",
        "project": {
            "head_at_start": "a" * 40,
            "v33_closeout_commit": "b" * 40,
            "plan_sha256": "c" * 64,
            "plan_path": "docs/WORLDSIM_V4_EVIDELTA_GS_PLAN.md",
            "branch": "research/worldsim-v4-evidelta",
            "v33_canonical_read_only": True,
        },
        "method_schema": {
            "base_asset": {},
            "evidence_state": {},
            "multi_view_update": {},
            "temporal_memory": {},
            "calibration": {"fit_split": "development"},
            "authenticity": {},
            "repair_risk": {},
            "temporal_transform": {},
            "reversible_delta": {"base_immutable": True, "rollback_render_sha_exact": True},
        },
        "datasets": {
            "nuscenes": {
                "scene_counts": {"development": 6, "validation": 6, "test": 18},
                "scene_disjoint": True,
                "selection_uses_model_results": False,
                "test_read_count": 1,
            },
            "kitti": {
                "root": "/missing/kitti",
                "root_present_at_p0": False,
                "download_allowed": False,
                "method_threshold_source": "frozen_nuscenes",
                "layout_state": "blocked_local_dataset_missing",
            },
        },
        "baselines": {
            "v33_frozen": {"tier": "A", "same_split": True},
            "streetgs": {"tier": "A", "same_split": True},
            "ad_gs": {"tier": "A", "same_split": True},
        },
        "metrics": {
            "image": {
                "primary": ["psnr", "ssim", "lpips_alex"],
                "regions": ["global", "static", "actor", "boundary", "edit_roi"],
            },
            "engineering": [
                "wall",
                "peak_nvidia_vram",
                "peak_cgroup_ram",
                "asset_bytes",
                "cold_load",
                "fps",
                "pipeline_success_rate",
                "valid_edit_yield",
                "retry_amplification",
                "resume_efficiency",
            ],
            "statistics": {"unit": "scene", "include_failed_blocked_abstain": True},
        },
        "task_registry": {TASK_ID: "done", "WS-V4-D0-NUSCENES-COHORT-01": "pending"},
        "literature": literature,
        "gates": {
            "paper_claim_frozen": True,
            "math_schema_frozen": True,
            "baseline_matrix_frozen": True,
            "metrics_schema_frozen": True,
            "dataset_protocol_frozen": True,
            "kitti_layout_audited": True,
            "no_training": True,
            "no_model_inference": True,
            "no_weight_download": True,
            "d0_authorized": True,
            "d1_authorized": False,
            "b0_authorized": False,
            "m1_authorized": False,
            "m2_authorized": False,
            "m3_authorized": False,
            "test_authorized": False,
        },
    }


def test_p0_contract_accepts_frozen_paper_first_schema() -> None:
    result = audit_config(_config())
    assert result["nuscenes_scene_count"] == 30
    assert result["literature_count"] == 12
    assert result["gates"]["d0_authorized"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["datasets"]["nuscenes"]["scene_counts"].update(test=17), "6/6/18"),
        (lambda c: c["datasets"]["kitti"].update(download_allowed=True), "禁止下载"),
        (lambda c: c["metrics"]["image"].update(primary=["psnr", "ssim"]), "图像主指标"),
        (lambda c: c["metrics"]["statistics"].update(unit="pixel"), "scene"),
        (lambda c: c["gates"].update(m1_authorized=True), "m1_authorized"),
    ],
)
def test_p0_contract_fails_closed(mutate, message) -> None:
    config = copy.deepcopy(_config())
    mutate(config)
    with pytest.raises(P0ContractError, match=message):
        audit_config(config)


def test_run_audit_records_immutable_source_snapshots(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(
        ["git", "-C", str(project), "switch", "-q", "-c", "research/worldsim-v4-evidelta"],
        check=True,
    )
    subprocess.run(["git", "-C", str(project), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(project), "config", "user.name", "Test"], check=True)
    files = {
        "motion_proj/worldsim_v4/p0_contract.py": "contract\n",
        "scripts/audit_worldsim_v4_start.py": "audit\n",
        "tests/test_worldsim_v4_p0_contract.py": "test\n",
        "configs/worldsim_v4/p0_scope_v1.yaml": "config\n",
        "docs/WORLDSIM_V4_EVIDELTA_GS_PLAN.md": "plan\n",
    }
    for relpath, content in files.items():
        path = project / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "add", "."], check=True)
    subprocess.run(["git", "-C", str(project), "commit", "-q", "-m", "init"], check=True)
    head = subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip()
    config = _config()
    config["project"]["head_at_start"] = head
    config["project"]["v33_closeout_commit"] = head
    config["project"]["plan_sha256"] = hashlib.sha256(b"plan\n").hexdigest()
    config["project"]["branch"] = subprocess.check_output(
        ["git", "-C", str(project), "branch", "--show-current"], text=True
    ).strip()
    config_path = project / "configs/worldsim_v4/p0_scope_v1.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run_dir = tmp_path / "run"
    summary = run_audit(config_path, run_dir, project_root=project)
    assert summary["repository"]["plan_sha256_exact"] is True
    assert len(summary["source_snapshots"]) == 5
    assert (run_dir / "manifest.json").is_file()
    with pytest.raises(P0ContractError, match="禁止复用"):
        run_audit(config_path, run_dir, project_root=project)
