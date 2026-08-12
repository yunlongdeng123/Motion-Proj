from __future__ import annotations

from pathlib import Path

from scripts.build_worldsim_v4_v33_actor_registry import build_command, resolve_scene_row


def test_resolve_scene_row_requires_exact_scene_and_no_test_read() -> None:
    inputs = {
        "schema_version": "worldsim_v4_v33_replay_inputs_v1",
        "partition_contract": "sample_index_mod_5",
        "test_quality_read": False,
        "scenes": [{"scene": "scene-0242"}],
    }

    assert resolve_scene_row(inputs, "scene-0242") == {"scene": "scene-0242"}


def test_build_command_uses_d0_high_actor_and_exact_base(tmp_path: Path) -> None:
    row = {
        "scene": "scene-0242",
        "base_checkpoint": {"path": "/runs/base.pth"},
        "actors": {
            "high_support": {"instance_token": "high-token"},
            "boundary_support": {"instance_token": "boundary-token"},
        },
    }
    config = {
        "runtimes": {
            "drivestudio_python": "/env/bin/python",
            "drivestudio_checkout": "/third_party/drivestudio",
        },
        "inputs": {"raw_metadata": "/data/v1.0-trainval"},
    }

    command = build_command(
        project_root=tmp_path,
        replay_config=config,
        scene_row=row,
        output=tmp_path / "registry.json",
    )

    assert command[0] == "/env/bin/python"
    assert command[command.index("--selected-token") + 1] == "high-token"
    assert command[command.index("--scene-name") + 1] == "scene-0242"
    assert command[command.index("--checkpoint") + 1] == "/runs/base.pth"
