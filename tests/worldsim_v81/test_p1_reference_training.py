"""P1 参考链训练的 CPU 数据流与恢复协议契约。"""

import sys
import importlib.util
from pathlib import Path

import pytest
import torch

from motion_proj.worldsim_v81.train_p1 import (
    paired_flow_loss,
    parse_args,
    raft_flows_for_pairs,
    reference_pairs_from_visible,
    validate_resume_protocol,
)


def _visible_clip():
    mask = torch.zeros(1, 25, 1, 256, 256)
    mask[..., :84] = 1
    mask[..., 172:] = 1
    visible = torch.zeros(1, 25, 3, 256, 256)
    yy, xx = torch.meshgrid(torch.arange(256), torch.arange(88), indexing="ij")
    checker = (((xx // 8 + yy // 8) % 2) * 2 - 1).float()
    for index in range(25):
        visible[0, index, :, :, 84:172] = checker * (-1 if (index // 4) % 2 else 1)
    return visible, mask


def test_reference_pairs_use_only_visible_center_and_form_unique_parent_tree():
    visible, mask = _visible_clip()
    pairs = reference_pairs_from_visible(visible, mask)
    assert len(pairs) == 24
    assert len({target for _, target in pairs}) == 24
    assert (24, 20) in pairs and (4, 0) in pairs
    assert {target for _, target in pairs} == set(range(24))
    # 洞区数据无论如何变化，都不能改变参考选择。
    altered = visible.clone()
    altered[..., :84] = 1
    altered[..., 172:] = -1
    assert reference_pairs_from_visible(altered, mask) == pairs


def test_reference_pairs_reject_time_varying_mask():
    visible, mask = _visible_clip()
    mask[:, 1, :, :, 0] = 0
    with pytest.raises(ValueError, match="静态 mask"):
        reference_pairs_from_visible(visible, mask)


class RecordingFlowLoss:
    l1_criterion = torch.nn.L1Loss()

    def __init__(self):
        self.calls = []

    def ternary_loss(self, completed, ground_truth, mask, current, shift):
        self.calls.append((current[:, 0, 0, 0].tolist(), shift[:, 0, 0, 0].tolist()))
        return (completed - ground_truth).abs().mean()


def test_paired_flow_loss_uses_actual_source_and_target_frames_in_both_directions():
    pairs = [(24, 20), (4, 0)]
    frames = torch.arange(25, dtype=torch.float32).view(1, 25, 1, 1, 1).expand(1, 25, 3, 4, 4)
    masks = torch.zeros(1, 25, 1, 4, 4)
    masks[..., 0] = 1
    truth = (torch.zeros(1, 2, 2, 4, 4), torch.zeros(1, 2, 2, 4, 4))
    predicted = tuple(torch.ones_like(flow, requires_grad=True) for flow in truth)
    recorder = RecordingFlowLoss()
    l1, warp = paired_flow_loss(recorder, predicted, truth, masks, frames, pairs)
    assert recorder.calls == [([24.0, 4.0], [20.0, 0.0]),
                              ([20.0, 0.0], [24.0, 4.0])]
    assert l1 > 0 and warp > 0
    (l1 + warp).backward()
    assert all(flow.grad is not None and torch.isfinite(flow.grad).all() for flow in predicted)


def test_raft_pair_chunks_keep_reference_order():
    class FakeRaft:
        def __init__(self):
            self.seen = []

        def forward_pairs(self, frames, pairs, *, iters, bidirectional):
            assert iters == 20 and bidirectional
            self.seen.extend(pairs)
            values = torch.tensor([s * 25 + t for s, t in pairs], dtype=torch.float32)
            forward = values.view(1, -1, 1, 1, 1).expand(1, -1, 2, 4, 4)
            return forward, -forward

    raft = FakeRaft()
    pairs = [(24, 20), (4, 0), (8, 5)]
    forward, backward = raft_flows_for_pairs(raft, torch.empty(1, 25, 3, 4, 4),
                                               pairs, iters=20, pair_chunk=2)
    assert raft.seen == pairs
    assert forward[0, :, 0, 0, 0].tolist() == [620.0, 100.0, 205.0]
    assert torch.equal(backward, -forward)


def test_resume_protocol_rejects_old_all_frames_checkpoint_in_reference_run():
    legacy = {"format": "worldsim_v81_seen_to_scene_p1_youtube_vos"}
    assert validate_resume_protocol(legacy, "literal-allframes") == "literal-allframes"
    with pytest.raises(ValueError, match="禁止跨传播协议恢复"):
        validate_resume_protocol(legacy, "reference-m4")
    assert validate_resume_protocol({"propagation_protocol": "reference-m4"},
                                    "reference-m4") == "reference-m4"
    with pytest.raises(ValueError, match="禁止跨传播协议恢复"):
        validate_resume_protocol({"propagation_protocol": "reference-m4"},
                                 "paper-bidirectional-m4")
    assert validate_resume_protocol({"propagation_protocol": "paper-bidirectional-m4"},
                                    "paper-bidirectional-m4") == "paper-bidirectional-m4"


def test_reference_protocol_is_default_and_literal_mode_is_explicit(monkeypatch, tmp_path):
    command = ["train_p1", "--output-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", command)
    assert parse_args().propagation_protocol == "reference-m4"
    monkeypatch.setattr(sys, "argv", command + ["--propagation-protocol", "literal-allframes"])
    assert parse_args().propagation_protocol == "literal-allframes"


def test_chunked_official_ternary_loss_preserves_values_and_flow_gradients():
    from motion_proj.worldsim_v81.model_bridge import DEFAULT_EXTERNAL
    source = Path(DEFAULT_EXTERNAL) / "utils/loss.py"
    if not source.exists():
        pytest.skip("固定官方FlowLoss未安装；远端CPU验证此项")
    spec = importlib.util.spec_from_file_location("p1_fixed_official_loss", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loss = module.FlowLoss()
    torch.manual_seed(2026)
    pairs = [(0, 1), (0, 2), (2, 4), (4, 6), (3, 5)]
    frames = torch.rand(1, 7, 3, 9, 10)
    mask = torch.zeros(1, 7, 1, 9, 10)
    for i in range(7):
        mask[:, i, :, :, :i + 1] = 1
    truth = tuple(torch.randn(1, 5, 2, 9, 10) * .1 for _ in range(2))
    full = tuple((x + torch.randn_like(x) * .03).requires_grad_() for x in truth)
    chunks = tuple(x.detach().clone().requires_grad_() for x in full)
    full_parts = paired_flow_loss(loss, full, truth, mask, frames, pairs)
    chunk_parts = paired_flow_loss(loss, chunks, truth, mask, frames, pairs, warp_chunk=2)
    for actual, expected in zip(chunk_parts, full_parts):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    sum(full_parts).backward()
    sum(chunk_parts).backward()
    for actual, expected in zip(chunks, full):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-5, atol=2e-6)
