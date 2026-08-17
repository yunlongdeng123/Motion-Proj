import hashlib
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_f0a_environment_one_view_smoke import (
    _one_view_command,
    _prepare_torch_hub_assets,
    _validate_config,
    parse_last_json_line,
)
from scripts.audit_worldsim_v51_f0a_environment_one_view_smoke import audit


def test_f0a_environment_smoke_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_worldsim_v51_f0a_environment_one_view_smoke.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout


def test_f0a_environment_smoke_auditor_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_worldsim_v51_f0a_environment_one_view_smoke.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output" in result.stdout


def test_f0a_environment_smoke_r032_audit_passes() -> None:
    result = audit(
        ROOT / "configs/worldsim_v51/stage_f_f0a_environment_one_view_smoke_v6.yaml",
        Path(
            "/root/autodl-tmp/runs/worldsim_v51/"
            "WS-V51-M1-F-IDENTITY-EMBEDDING-01/"
            "20260818T090000Z__m1-stage-f-f0a-environment-one-view-s20260814-r032"
        ),
    )
    assert result["status"] == "pass"
    assert result["mask_histogram"] == {"0": 1440000}
    assert result["quality_read"] is False


def test_f0a_environment_smoke_freeze_is_terminal_and_locked() -> None:
    freeze = yaml.safe_load(
        (
            ROOT
            / "configs/worldsim_v51/stage_f_f0a_environment_one_view_smoke_freeze_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert freeze["status"] == "done"
    assert freeze["independent_audit"]["status"] == "pass"
    assert freeze["one_view_output"]["interpretation"]["quality_claim"] is False
    assert freeze["governance"]["materialization_authorized"] is False


def test_f0a_environment_smoke_formal_config_validates() -> None:
    config = _validate_config(
        ROOT / "configs/worldsim_v51/stage_f_f0a_environment_one_view_smoke_v6.yaml"
    )
    assert config["one_view"]["interpretation"]["association_capability_claim"] is False
    assert config["one_view"]["upstream_defaults"] == {
        "SAM_NUM_POINTS_PER_SIDE": 64,
        "SAM_NUM_POINTS_PER_BATCH": 64,
    }
    assert config["runtime_environment"]["PYTORCH_CUDA_ALLOC_CONF"] == (
        "max_split_size_mb:128"
    )
    assert config["one_view"]["arguments"]["SAM_NUM_POINTS_PER_BATCH"] == 32
    assert config["one_view"]["arguments"]["SAM_NUM_POINTS_PER_SIDE"] == 32
    assert config["decision"]["materialization_authorized"] is False


def test_f0a_grid_recovery_uses_preregistered_official_cli_arguments() -> None:
    config = _validate_config(
        ROOT / "configs/worldsim_v51/stage_f_f0a_environment_one_view_smoke_v6.yaml"
    )
    command = _one_view_command(
        config,
        Path("/isolated/bin/python"),
        Path("/input"),
        Path("/output"),
    )
    index = command.index("--SAM_NUM_POINTS_PER_BATCH")
    assert command[index + 1] == "32"
    index = command.index("--SAM_NUM_POINTS_PER_SIDE")
    assert command[index + 1] == "32"


def test_parse_last_json_line_preserves_solver_banner() -> None:
    payload = parse_last_json_line(
        'Restricted license - for non-production use only\n{"status": 2, "solution": 1.0}\n',
        "Gurobi",
    )
    assert payload == {
        "status": 2,
        "solution": 1.0,
        "stdout_prefix": ["Restricted license - for non-production use only"],
    }


def test_prepare_torch_hub_assets_copies_then_reuses_exact_asset(tmp_path: Path) -> None:
    source = tmp_path / "source.pth"
    source.write_bytes(b"frozen torchvision weight")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    config = {
        "torch_home": str(tmp_path / "torch-home"),
        "partial_suffix": ".partial",
        "assets": {
            "resnet": {
                "source_cache_path": str(source),
                "filename": "resnet.pth",
                "url": "https://example.invalid/resnet.pth",
                "bytes": source.stat().st_size,
                "sha256": digest,
            }
        },
    }
    first = _prepare_torch_hub_assets(config)
    second = _prepare_torch_hub_assets(config)
    assert first["assets"][0]["acquisition"] == "copied_then_atomic_publish"
    assert second["assets"][0]["acquisition"] == "reused_exact_canonical_asset"
    assert first["assets"][0]["sha256"] == digest
    assert not (tmp_path / "torch-home/hub/checkpoints/resnet.pth.partial").exists()
