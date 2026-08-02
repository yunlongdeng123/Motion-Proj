from pathlib import Path

import pytest

from scripts import run_dr_v2_m1_dggt as runner


def test_parse_metrics_requires_exact_finite_values(tmp_path: Path):
    stdout = tmp_path / "stdout.log"
    stdout.write_text(
        "PSNR: 21.5\nSSIM: 0.81\nLPIPS: 0.24\n"
        "Avg Inference Time (s): 1.25\n"
    )
    result = runner.parse_metrics({"stage": "smoke", "stdout": str(stdout)})
    assert result == {
        "PSNR": 21.5,
        "SSIM": 0.81,
        "LPIPS(ALEX)": 0.24,
        "inference_time_seconds": 1.25,
    }

    stdout.write_text("PSNR: 21.5\n")
    with pytest.raises(RuntimeError, match="SSIM"):
        runner.parse_metrics({"stage": "smoke", "stdout": str(stdout)})


@pytest.mark.parametrize("views,depth_count", [(1, 4), (3, 12)])
def test_validate_output_checks_nonempty_protocol_artifacts(
    tmp_path: Path, views: int, depth_count: int
):
    scene = tmp_path / "001"
    scene.mkdir()
    for index in range(4):
        (scene / f"view_{index:04d}.png").write_bytes(b"png")
    for index in range(depth_count):
        (scene / f"view_{index:04d}.npy").write_bytes(b"npy")
    (scene / "rendered_video.mp4").write_bytes(b"video")
    (scene / "comparison.mp4").write_bytes(b"video")

    result = runner.validate_output(tmp_path, views)
    assert result["image_count"] == 4
    assert result["depth_count"] == depth_count
    assert result["video_count"] == 2


def test_validate_output_fails_closed_on_empty_file(tmp_path: Path):
    scene = tmp_path / "001"
    scene.mkdir()
    for index in range(4):
        (scene / f"view_{index:04d}.png").write_bytes(b"png")
        (scene / f"view_{index:04d}.npy").write_bytes(b"npy")
    (scene / "rendered_video.mp4").write_bytes(b"")
    (scene / "comparison.mp4").write_bytes(b"video")
    with pytest.raises(RuntimeError, match="缺失或为空"):
        runner.validate_output(tmp_path, 1)


def test_resource_stop_reason_keeps_oom_and_memory_semantics_separate():
    sample = {
        "memory_max_bytes": 100,
        "memory_current_bytes": 91,
        "memory_events": {"oom": 0, "oom_kill": 0},
        "disk_free_bytes": 100 * 1024**3,
    }
    reason, count = runner.stop_reason(sample, sample["memory_events"], 0)
    assert reason is None
    reason, count = runner.stop_reason(sample, sample["memory_events"], count)
    assert reason == "cgroup memory 连续两个采样达到 90%"

    sample["memory_current_bytes"] = 1
    sample["memory_events"] = {"oom": 1, "oom_kill": 0}
    reason, _ = runner.stop_reason(sample, {"oom": 0, "oom_kill": 0}, 0)
    assert reason == "memory.events oom 增加"


def test_inference_command_freezes_mode_and_disables_diffusion(tmp_path: Path):
    command = runner.inference_command(tmp_path, "003", 3, tmp_path / "out")
    assert command[0] == runner.PYTHON
    assert command[command.index("--scene_names") + 1] == "003"
    assert command[command.index("--input_views") + 1] == "3"
    assert command[command.index("--sequence_length") + 1] == "4"
    assert command[command.index("--mode") + 1] == "2"
    assert "-diffusion" not in command


def test_base_env_keeps_cuda_abi_on_dggt_environment():
    env = runner.base_env()
    assert env["CUDA_HOME"] == str(runner.ENV)
    assert env["PATH"].split(":", 1)[0] == str(runner.ENV / "bin")
    assert env["TORCH_CUDA_ARCH_LIST"] == "8.6"
    assert env["OMP_NUM_THREADS"] == "8"
    assert "/usr/local/cuda/bin" not in env["PATH"].split(":")[:2]


def test_constraints_pin_torch_compatible_transformers_stack():
    constraints = (
        Path(__file__).parents[1] / "configs/env/dggt_v2_constraints.txt"
    ).read_text(encoding="utf-8")
    assert "transformers==4.48.3" in constraints
    assert "tokenizers==0.21.0" in constraints
    assert "diffusers==0.32.2" in constraints
    assert "flow-vis==0.1" in constraints
