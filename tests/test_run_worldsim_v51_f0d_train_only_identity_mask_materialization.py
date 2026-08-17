from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_f0d_train_only_identity_mask_materialization import (
    _record_chain,
    _scene_command,
    _validate_config,
)
from scripts.audit_worldsim_v51_f0d_train_only_identity_mask_materialization import audit


CONFIG = ROOT / "configs/worldsim_v51/stage_f_f0d_train_only_identity_mask_materialization_v1.yaml"


def test_f0d_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_worldsim_v51_f0d_train_only_identity_mask_materialization.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout


def test_f0d_auditor_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_worldsim_v51_f0d_train_only_identity_mask_materialization.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output" in result.stdout


def test_f0d_config_validates_exact_45_view_scene_local_contract() -> None:
    config, records = _validate_config(CONFIG)
    assert len(records) == 45
    assert [row["scene"] for row in config["scene_contracts"]] == [
        "scene-0471",
        "scene-1087",
        "scene-0379",
    ]
    assert [row["record_count"] for row in config["scene_contracts"]] == [15, 15, 15]
    assert len({Path(row["path"]).name for row in records}) == 15
    assert len({(row["scene"], Path(row["path"]).name) for row in records}) == 45
    assert config["locks"]["full_materialization_execution"] is True
    assert config["decision"]["identity_training_authorized"] is False


def test_f0d_scene_cli_keeps_grid32_and_upstream_batch64() -> None:
    config, _ = _validate_config(CONFIG)
    command = _scene_command(config, Path("/input"), Path("/output"))
    assert command[command.index("--SAM_NUM_POINTS_PER_SIDE") + 1] == "32"
    assert command[command.index("--SAM_NUM_POINTS_PER_BATCH") + 1] == "64"


def test_f0d_output_record_chain_is_order_and_content_sensitive() -> None:
    rows = [
        {"scene": "scene-a", "frame": 0, "camera": 0, "mask_sha256": "a"},
        {"scene": "scene-a", "frame": 0, "camera": 1, "mask_sha256": "b"},
    ]
    first = _record_chain(rows)
    assert first == _record_chain(rows)
    assert first != _record_chain(list(reversed(rows)))
    changed = [dict(row) for row in rows]
    changed[1]["mask_sha256"] = "c"
    assert first != _record_chain(changed)


def test_f0d_r035_independent_audit_replays_blocked_boundary() -> None:
    result = audit(
        CONFIG,
        Path(
            "/root/autodl-tmp/runs/worldsim_v51/"
            "WS-V51-M1-F-IDENTITY-EMBEDDING-01/"
            "20260818T120000Z__m1-stage-f-f0d-train-materialization-s20260814-r035"
        ),
    )
    assert result["status"] == "pass"
    assert result["audited_run_status"] == "blocked"
    assert result["materialization"]["canonical_mask_count"] == 15
    assert result["materialization"]["complete"] is False
    assert result["failed_scene"]["failure_class"] == (
        "CUDA_CUBLAS_STATUS_INTERNAL_ERROR"
    )
    assert result["failed_scene"]["explicit_pytorch_oom"] is False
    assert result["quality_read"] is False
