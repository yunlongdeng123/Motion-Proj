"""P1 可见条件、参考帧与官方 pipeline 导入的 CPU 契约。"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from motion_proj.worldsim_v81 import infer_p1
from motion_proj.worldsim_v81.infer_p1 import (
    _official_pipeline_class, build_pairs_for_all_frames, compute_structure_term,
    effective_amp, load_sequence, select_reference_frame_indices,
)
from motion_proj.worldsim_v81.model_bridge import DEFAULT_EXTERNAL


def test_public_structure_term_and_reference_pair_tree():
    gradient = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4
    inverse = 255 - gradient
    assert compute_structure_term(gradient, gradient) == pytest.approx(1.0)
    assert compute_structure_term(gradient, inverse) < 0
    frames = [Image.fromarray(gradient if i % 2 == 0 else inverse) for i in range(25)]
    refs = select_reference_frame_indices(frames, 4)
    assert refs[0] == 0 and refs[-1] == 24
    chain, to_frame = build_pairs_for_all_frames(25, refs)
    assert len(chain) + len(to_frame) == 24
    assert chain == [(refs[j], refs[j - 1]) for j in range(len(refs) - 1, 0, -1)]
    assert {target for _, target in chain + to_frame} == set(range(24))


def test_sequence_uses_first_25_frames_and_visible_center_only(tmp_path):
    roots = [tmp_path / name for name in ("a", "b")]
    for variant, root in enumerate(roots):
        folder = root / "sequence"
        folder.mkdir(parents=True)
        for index in range(26):
            array = np.full((256, 256, 3), 20 if index % 2 else 220, dtype=np.uint8)
            array[:, :32] = 255 * variant
            array[:, 224:] = 255 * variant
            Image.fromarray(array).save(folder / f"{index:05d}.png")
    a = load_sequence(roots[0], "sequence", 0.125)
    b = load_sequence(roots[1], "sequence", 0.125)
    assert len(a[0]) == len(a[1]) == len(a[2]) == 25
    assert a[0][0].name == "00000.png" and a[0][-1].name == "00024.png"
    assert a[3] == (32, 224)
    assert a[4] == b[4] and a[5] == b[5]
    assert np.array(a[1][0])[:, :32].mean() != np.array(b[1][0])[:, :32].mean()
    assert np.array(a[2][0])[:, :32].sum() == 0


def test_short_sequence_fails_instead_of_implicit_padding(tmp_path):
    folder = tmp_path / "short"
    folder.mkdir()
    Image.new("RGB", (256, 256)).save(folder / "00000.jpg")
    with pytest.raises(ValueError, match="25"):
        load_sequence(tmp_path, "short", 0.33)


def test_explicit_full_video_uses_every_frame_and_global_reference_indices(tmp_path):
    folder = tmp_path / "complete"
    folder.mkdir()
    for index in range(27):
        Image.new("RGB", (256, 256), (index, 20, 30)).save(folder / f"{index:05d}.png")
    paths, target, visible, _, refs, pairs = load_sequence(
        tmp_path, "complete", 0.125, full_video=True)
    assert len(paths) == len(target) == len(visible) == 27
    assert paths[-1].name == "00026.png"
    assert refs[0] == 0 and refs[-1] == 26
    assert len(pairs) == 26 and {target for _, target in pairs} == set(range(26))


def test_full_video_vae_encodes_without_extra_full_rgb_tensor(monkeypatch):
    calls = []

    def fake_encode(_vae, pixels, *, scaled, sample, chunk):
        calls.append((pixels.shape[1], scaled, sample, chunk))
        return pixels[:, :, :1]

    monkeypatch.setattr(infer_p1, "encode_video", fake_encode)
    visible = torch.zeros(1, 9, 3, 4, 4)
    latent = infer_p1._encode_condition_full_video(
        None, visible, torch.Generator().manual_seed(7), chunk=4)
    assert latent.shape == (1, 9, 1, 4, 4)
    assert calls == [(4, False, False, 4), (4, False, False, 4), (1, False, False, 4)]
    assert latent.device == visible.device


def test_full_video_cli_is_explicit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["infer_p1", "--data-root", "/tmp/data",
                     "--sequence-id", "s", "--checkpoint", "/tmp/checkpoint.pt",
                     "--output-dir", "/tmp/output", "--side-ratio", "0.125",
                     "--mode", "paper-feedforward", "--full-video"])
    assert infer_p1.parse_args().full_video is True


def test_fixed_public_pipeline_imports_without_weights():
    pipeline = _official_pipeline_class(DEFAULT_EXTERNAL)
    assert pipeline.__name__ == "StableVideoDiffusionPipeline"


def test_literal_public_uses_source_fp16_inside_bf16_requested_run():
    assert effective_amp("literal-public", "bf16") == "fp16"
    assert effective_amp("literal-public", "fp16") == "fp16"
    assert effective_amp("paper-feedforward", "bf16") == "bf16"
