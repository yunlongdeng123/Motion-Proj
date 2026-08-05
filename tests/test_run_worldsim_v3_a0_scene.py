from pathlib import Path

from scripts.run_worldsim_v3_a0_scene import (
    BOUNDARY_TOKEN,
    HIGH_TOKEN,
    NUM_ITERS,
    PATCHED_DRIVESTUDIO,
    build_eval_command,
    build_train_command,
    compact_actor,
)


def test_formal_commands_freeze_native_scene0255_contract() -> None:
    command, checkpoint = build_train_command(Path("/tmp/a0-formal"))

    assert command[1] == str(PATCHED_DRIVESTUDIO / "tools/train.py")
    assert "data.scene_idx=204" in command
    assert f"trainer.optim.num_iters={NUM_ITERS}" in command
    assert "data.pixel_source.test_image_stride=10" in command
    assert "render.render_test=false" in command
    assert checkpoint.name == "checkpoint_final.pth"

    eval_command = build_eval_command(checkpoint)
    assert eval_command[1] == str(PATCHED_DRIVESTUDIO / "tools/eval.py")
    assert str(checkpoint) in eval_command
    assert "render.render_test=true" in eval_command
    assert "render.render_full=false" in eval_command


def test_compact_actor_keeps_ablation_fields() -> None:
    row = {
        "instance_token": HIGH_TOKEN,
        "class_name": "vehicle.trailer",
        "availability": "available",
        "rigid_model_index": 7,
        "checkpoint_tensor_slice": {"gaussian_count": 123},
    }

    assert compact_actor(row) == {
        "instance_token": HIGH_TOKEN,
        "class_name": "vehicle.trailer",
        "availability": "available",
        "rigid_model_index": 7,
        "gaussian_count": 123,
    }
    assert compact_actor(None) is None
    assert BOUNDARY_TOKEN != HIGH_TOKEN
