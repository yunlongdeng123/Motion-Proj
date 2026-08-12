from pathlib import Path

from scripts.run_worldsim_v4_v33_spatial_delta import (
    RUN_ROOT,
    build_commands,
    run_root_for,
)


def test_spatial_runner_builds_package_then_development_eval() -> None:
    run = RUN_ROOT / "spatial-scene0255-r1"
    commands = build_commands(
        config_path=run / "spatial.yaml",
        run_dir=run,
        project_root=Path("/project"),
        config={"runtimes": {"drivestudio_python": "/env/bin/python"}},
    )
    assert commands[0][1].endswith("build_worldsim_v33_s4_spatial_delta.py")
    assert commands[1][1].endswith("evaluate_worldsim_v33_s4_spatial_delta.py")
    assert commands[1][commands[1].index("--package-manifest-sha256") + 1] == "__RESOLVE_AFTER_PACKAGE__"


def test_spatial_runner_derives_validation_run_root_from_task_id() -> None:
    assert run_root_for({"task_id": "WS-V4-M1-EVIDENCE-FIELD-01"}) == Path(
        "/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01"
    )
