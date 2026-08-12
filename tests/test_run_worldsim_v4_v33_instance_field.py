from __future__ import annotations

from pathlib import Path

from scripts.run_worldsim_v4_v33_instance_field import (
    RUN_ROOT,
    build_commands,
    run_root_for,
)


def test_build_commands_freezes_development_for_all_eval_steps() -> None:
    project = Path("/project")
    run_dir = RUN_ROOT / "instance-scene0255-r1"
    config_path = run_dir / "instance.yaml"
    config = {
        "runtimes": {
            "drivestudio_python": "/env/ds/bin/python",
            "sam_python": "/env/sam/bin/python",
        },
        "provenance": {"evaluation_partition": "development"},
    }

    commands = build_commands(
        config_path=config_path,
        run_dir=run_dir,
        project_root=project,
        config=config,
    )

    assert len(commands) == 5
    for command in commands[:3]:
        assert command[command.index("--partition") + 1] == "development"
    assert commands[3][commands[3].index("--eval-partition") + 1] == "development"
    assert commands[4][commands[4].index("--evaluation-partition") + 1] == "development"
    assert commands[3][commands[3].index("--phase") + 1] == "formal"
    assert "smoke" not in commands[3]


def test_validation_task_uses_its_own_run_namespace() -> None:
    config = {"task_id": "WS-V4-M1-EVIDENCE-FIELD-01"}
    assert run_root_for(config) == Path(
        "/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01"
    )
