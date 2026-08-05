from pathlib import Path

from scripts.run_worldsim_v3_a0_smoke import (
    PATCHED_DRIVESTUDIO,
    PROCESSED_ROOT,
    build_train_command,
)


def test_build_train_command_uses_patched_tree_and_scene0255() -> None:
    command = build_train_command(Path("/tmp/a0-smoke"), 3)

    assert command[1] == str(PATCHED_DRIVESTUDIO / "tools/train.py")
    assert "data.scene_idx=204" in command
    assert f"data.data_root={PROCESSED_ROOT}" in command
    assert "trainer.optim.num_iters=3" in command
    assert "logging.saveckpt_freq=3" in command
    assert "render.render_full=false" in command
    assert "render.render_test=false" in command
