from pathlib import Path

from scripts.run_worldsim_v4_v33_spatial_delta import RUN_ROOT, build_commands


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
