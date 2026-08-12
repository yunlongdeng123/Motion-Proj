from __future__ import annotations

from pathlib import Path

from scripts.run_worldsim_v4_v33_semantic_lift import (
    RUN_ROOT,
    build_commands,
    run_root_for,
)


def test_build_commands_runs_train_only_semantics_and_v4_finalizer() -> None:
    project = Path("/project")
    run_dir = RUN_ROOT / "semantic-scene0255-r1"
    config_path = run_dir / "semantic.yaml"
    config = {
        "runtimes": {
            "drivestudio_python": "/env/ds/bin/python",
            "sam_python": "/env/sam/bin/python",
        }
    }

    commands = build_commands(
        config_path=config_path,
        run_dir=run_dir,
        project_root=project,
        config=config,
    )

    assert len(commands) == 5
    assert commands[1][1].endswith("prepare_worldsim_v32_s1_prompts.py")
    assert commands[2][0] == "/env/sam/bin/python"
    assert commands[3][1].endswith("lift_worldsim_v32_semantics.py")
    assert commands[4][commands[4].index("--run-root") + 1] == str(RUN_ROOT)
    assert all("heldout" not in " ".join(command) for command in commands)


def test_validation_task_uses_its_own_run_namespace() -> None:
    config = {"task_id": "WS-V4-M1-EVIDENCE-FIELD-01"}
    assert run_root_for(config) == Path(
        "/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01"
    )
