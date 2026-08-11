from __future__ import annotations

from pathlib import Path

import pytest

from scripts.restore_worldsim_v4_adgs_env import (
    AdgsEnvironmentError,
    build_commands,
    load_config,
    materialize_extension_sources,
)


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
runtime_wheels:
  - {name: roma, import: roma, path: /weights/roma.whl, bytes: 10, sha256: abc}
external_extensions:
  - {name: pytorch3d, import: pytorch3d._C, source_root: /third/pytorch3d, commit: abc123}
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
        "install_runtime_roma",
        "build_simple_knn",
        "build_diff_gaussian_rasterization",
        "build_external_pytorch3d",
        "cuda_smoke",
    ]
    for _, command, _ in commands:
        assert command[0].replace("\\", "/") == "/env/adgs/bin/python"
    for _, command, _ in commands[:-1]:
        assert "--no-index" in command


def test_extension_builds_use_run_local_source_copies(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    adgs = tmp_path / "third" / "AD-GS"
    (adgs / "submodules/simple-knn").mkdir(parents=True)
    (adgs / "submodules/rasterizer").mkdir(parents=True)
    (adgs / "submodules/simple-knn/setup.py").write_text("# simple\n", encoding="utf-8")
    (adgs / "submodules/rasterizer/setup.py").write_text("# raster\n", encoding="utf-8")
    pytorch3d = tmp_path / "third" / "pytorch3d"
    pytorch3d.mkdir(parents=True)
    (pytorch3d / "setup.py").write_text("# pytorch3d\n", encoding="utf-8")
    path.write_text(
        valid_config()
        .replace("/third/AD-GS", str(adgs))
        .replace("/third/pytorch3d", str(pytorch3d)),
        encoding="utf-8",
    )
    config = load_config(path)
    build_root = materialize_extension_sources(config, tmp_path / "run/build_source")
    commands = build_commands(config, build_root)
    assert commands[2][2] == build_root / "submodules/simple-knn"
    assert commands[3][2] == build_root / "submodules/rasterizer"
    assert commands[4][2] == build_root / "external/pytorch3d"
    assert (adgs / "submodules/simple-knn/build").exists() is False
    assert (build_root / "submodules/simple-knn/setup.py").read_text(encoding="utf-8") == "# simple\n"
    assert (build_root / "external/pytorch3d/setup.py").read_text(encoding="utf-8") == "# pytorch3d\n"


def test_environment_and_training_configs_share_compatibility_contract() -> None:
    environment = load_config(Path("configs/worldsim_v4/adgs_environment_v1.yaml"))
    training = pytest.importorskip("yaml").safe_load(
        Path("configs/worldsim_v4/adgs_training_v1.yaml").read_text(encoding="utf-8")
    )
    source = environment["source_contract"]
    implementation = training["implementation"]
    assert source["compatibility_patch_sha256"] == implementation["compatibility_patch_sha256"]
    assert source["expected_modified_files"] == implementation["expected_modified_files"]
    assert "utils/flow_utils.py" in source["expected_modified_files"]
    assert [row["import"] for row in environment["runtime_wheels"]] == ["roma"]
    assert [row["import"] for row in environment["external_extensions"]] == ["pytorch3d._C"]
