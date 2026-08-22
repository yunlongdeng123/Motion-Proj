"""P7R 只恢复 source-proven 形式合同，不重复模型 forward。"""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/worldsim_v61/p7r_irwm_contract_recovery_v1.yaml"
RUNNER_PATH = REPO_ROOT / "scripts/run_worldsim_v61_p7r_irwm_contract_recovery.py"


def test_recovery_scope_is_exact_and_analysis_only() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["task_id"] == "WS-V61-P7R-IRWM-CONTRACT-RECOVERY-01"
    assert config["hypothesis_id"] == "WS-V61-H-P7-IRWM-002"
    assert config["contract"]["required_h001_false_checks"] == [
        "environment_versions_exact", "model_state_exact"
    ]
    assert config["contract"]["allowed_missing_keys"] == [
        "pts_bbox_head.transformer.reference_points.weight",
        "pts_bbox_head.transformer.reference_points.bias",
    ]
    assert config["contract"]["expected_environment"]["detectron2"] == "0.6+cu111"
    assert config["contract"]["required_source_statement"] == "del self.transformer.reference_points"


def test_runner_cannot_repeat_gpu_or_model_inference() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "CUDA_VISIBLE_DEVICES" not in source
    assert "run_worldsim_v61_p7_irwm_worker" not in source
    assert '"new_gpu_forward_started": False' in source
    assert '"model_loaded": False' in source


def test_recovery_binds_complete_h001_authority() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert set(config["sources"]["h001_files"]) == {
        "P7_GATE.json", "SUMMARY.json", "WORKER_PLAN.json", "WORKER_REPORT.json",
        "CURRENT_PREDICTED_OCCUPANCY.npz", "RESOURCE_AUDIT.json", "MANIFEST.json", "TERMINAL.json",
    }
    assert all(len(value) == 64 for value in config["sources"]["h001_files"].values())
    assert config["sources"]["official_deleted_parameter_source"]["sha256"] == (
        "cb3079214f0c664ae81784b70e8d76003246ca4f6d5dc126db507f28c4a2e4fd"
    )
