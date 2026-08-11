from __future__ import annotations

from pathlib import Path

import pytest

from scripts.restore_worldsim_v4_adgs_env import AdgsEnvironmentError, build_commands, load_config


def valid_config() -> str:
    return """\
task_id: WS-V4-B0-MATCHED-BASELINES-01
status: running
policy:
  restore_mode: clone_frozen_local_environment_then_build_vendored_extensions
  network_access: false
  test_quality_read: false
paths:
  project_root: /project
  target_environment: /env/adgs
  adgs_root: /third/AD-GS
  plyfile_wheel: /weights/plyfile.whl
runtime_contract:
  torch_cuda: '11.8'
  torch_cuda_arch_list: '8.6'
extensions:
  - {name: simple_knn, source: submodules/simple-knn, import: simple_knn._C}
  - {name: diff_gaussian_rasterization, source: submodules/rasterizer, import: diff_gaussian_rasterization}
"""


def test_load_config_freezes_offline_cuda_contract(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(valid_config(), encoding="utf-8")
    config = load_config(path)
    assert config["policy"]["network_access"] is False
    assert config["runtime_contract"]["torch_cuda_arch_list"] == "8.6"


def test_load_config_rejects_network_or_extension_drift(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(valid_config().replace("network_access: false", "network_access: true"), encoding="utf-8")
    with pytest.raises(AdgsEnvironmentError, match="离线恢复策略"):
        load_config(path)


def test_build_commands_are_offline_and_use_target_python(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(valid_config(), encoding="utf-8")
    commands = build_commands(load_config(path))
    assert [name for name, _, _ in commands] == [
        "install_plyfile",
        "build_simple_knn",
        "build_diff_gaussian_rasterization",
        "cuda_smoke",
    ]
    for _, command, _ in commands:
        assert command[0].replace("\\", "/") == "/env/adgs/bin/python"
    for _, command, _ in commands[:-1]:
        assert "--no-index" in command
