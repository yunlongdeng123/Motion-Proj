import json

import numpy as np
from PIL import Image
import torch

from motion_proj.worldsim_v81.data import discover_videos, OutpaintingDataset


def test_actual_rgb_loading_and_hidden_condition(tmp_path):
    directory = tmp_path / "JPEGImages" / "sample"
    directory.mkdir(parents=True)
    for frame in range(3):
        image = np.full((8, 12, 3), 100 + frame, dtype=np.uint8)
        Image.fromarray(image).save(directory / f"{frame:05d}.png")
    accepted, rejected = discover_videos(directory.parent, "train", frames=3)
    assert len(accepted) == 1 and not rejected
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(accepted[0]) + "\n", encoding="utf-8")
    sample = OutpaintingDataset(manifest, frames=3, size=(8, 8))[0]
    assert sample["target_rgb"].shape == (3, 3, 8, 8)
    mask = sample["hole_mask"].expand_as(sample["target_rgb"]).bool()
    assert torch.all(sample["visible_rgb"][mask] == 0)
    assert torch.equal(sample["target_rgb"][~mask], sample["visible_rgb"][~mask])


def test_short_video_rejected(tmp_path):
    directory = tmp_path / "short"
    directory.mkdir()
    Image.new("RGB", (8, 8)).save(directory / "00000.jpg")
    accepted, rejected = discover_videos(tmp_path, "train", frames=25)
    assert not accepted and rejected[0]["reason"] == "too_few_frames"


def test_published_border_boundaries():
    from motion_proj.worldsim_v81.masks import make_border_outpaint_mask
    mask = make_border_outpaint_mask(256, 256)[0, 0, 0]
    assert torch.all(mask[:84] == 1)
    assert torch.all(mask[84:172] == 0)
    assert torch.all(mask[172:] == 1)
