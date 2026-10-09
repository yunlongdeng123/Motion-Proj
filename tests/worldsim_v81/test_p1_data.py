"""YouTube-VOS P1 官方采样分布与固定掩码。"""

from pathlib import Path

from PIL import Image
import pytest
import torch

from motion_proj.worldsim_v81.p1_data import YouTubeVOSP1Dataset


def make_video(root: Path, name: str, frames: int, offset: int = 0) -> None:
    folder = root / "JPEGImages" / name
    folder.mkdir(parents=True)
    for frame in range(frames):
        Image.new("RGB", (12, 8), (offset + frame, 0, 0)).save(folder / f"{frame:05d}.jpg")


def test_random_video_and_contiguous_start_with_official_fixed_mask(tmp_path):
    make_video(tmp_path, "a", 26)
    make_video(tmp_path, "b", 27, offset=100)
    make_video(tmp_path, "short", 25)
    dataset = YouTubeVOSP1Dataset(tmp_path, seed=123, samples=100)
    profile = dataset.profile()
    assert profile["videos"] == 2 and profile["rejected_short"] == 1
    assert profile["available_windows"] == 5 and len(dataset) == 100
    choices = [dataset.choice(step) for step in range(100)]
    assert choices == [dataset.choice(step) for step in range(100)]
    assert {index for index, _ in choices} == {0, 1}
    assert {start for _, start in choices} == {0, 1, 2}
    batch = dataset[0]
    assert batch["target_rgb"].shape == (25, 3, 256, 256)
    assert batch["hole_mask"].shape == (25, 1, 256, 256)
    assert torch.all(batch["hole_mask"][..., :84] == 1)
    assert torch.all(batch["hole_mask"][..., 84:172] == 0)
    assert torch.all(batch["hole_mask"][..., 172:] == 1)
    assert torch.all(batch["visible_rgb"] * batch["hole_mask"] == 0)
    # 文件名排序对应时间连续片段，起点由当前 step 决定。
    first_value = (batch["target_rgb"][0, 0, 128, 128].item() + 1) * 127.5
    baseline = 0 if batch["video_id"] == "a" else 100
    assert abs(first_value - (baseline + batch["clip_start"])) < 3


def test_p1_requires_train_jpegimages_with_more_than_25_frames(tmp_path):
    make_video(tmp_path, "short", 25)
    with pytest.raises(ValueError, match="超过25"):
        YouTubeVOSP1Dataset(tmp_path)
    with pytest.raises(FileNotFoundError, match="JPEGImages"):
        YouTubeVOSP1Dataset(tmp_path / "wrong")
