"""固定输入探针的 CPU 接线测试；不加载模型权重或运行 GPU。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from motion_proj.worldsim_v81 import p1_fixed_capacity_probe as probe
from motion_proj.worldsim_v81.reference_propagation import build_reference_plan
from motion_proj.worldsim_v81.train_p1 import FORMAT


def _source_state(root: Path, step: int = 500) -> dict:
    profile = {"rgb_root": str(root), "videos": 10, "available_windows": 100,
               "seed": 123, "frames": 25, "size": 256, "mask_ratio_each_side": .33}
    return {"format": FORMAT, "step": step,
            "propagation_protocol": "paper-bidirectional-m4",
            "optimizer_name": "Adam",
            "optimizer": {"param_groups": [{"lr": 1e-5, "weight_decay": 0}]},
            "args": {"data_root": str(root), "lr": 1e-5, "amp": "bf16"},
            "data_profile": profile}


def test_source_contract_requires_formal_checkpoint_and_fixed_clip(tmp_path):
    state = _source_state(tmp_path)
    batch = {"video_id": [probe.VIDEO_ID], "clip_start": torch.tensor([probe.CLIP_START])}
    assert probe.validate_source(state, state["data_profile"], batch,
                                 data_root=tmp_path, updates=64) == (500, "bf16")
    source100 = {**state, "step": 100}
    assert probe.validate_source(source100, state["data_profile"], batch,
                                 data_root=tmp_path, updates=1) == (100, "bf16")
    for bad in ({**state, "format": probe.PROBE_FORMAT}, {**state, "step": 1000},
                {**state, "propagation_protocol": "reference-m4"}):
        with pytest.raises(ValueError):
            probe.validate_source(bad, state["data_profile"], batch,
                                  data_root=tmp_path, updates=16)
    with pytest.raises(ValueError, match="1..64"):
        probe.validate_source(state, state["data_profile"], batch,
                              data_root=tmp_path, updates=65)
    changed = {**state["data_profile"], "seed": 999}
    with pytest.raises(ValueError, match="数据池"):
        probe.validate_source(state, changed, batch, data_root=tmp_path, updates=16)
    with pytest.raises(ValueError, match="固定片段"):
        probe.validate_source(state, state["data_profile"],
                              {**batch, "clip_start": torch.tensor([3])},
                              data_root=tmp_path, updates=16)


class _FCNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(.1))
        self.calls = 0

    def forward_bidirect_flow(self, flow, mask):
        self.calls += 1
        return (flow[0] + self.bias, flow[1] + self.bias), None

    def combine_flow(self, flow, predicted, mask):
        return predicted


class _UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))
        self.inputs = []

    def forward(self, x, timestep, clip, *, added_time_ids):
        self.inputs.append((x.detach().clone(), timestep.detach().clone(),
                            clip.detach().clone(), added_time_ids.detach().clone()))
        return SimpleNamespace(sample=x[:, :, :4] * self.weight + x[:, :, 4:] * .4)


class _FlowLoss:
    @staticmethod
    def l1_criterion(predicted, target):
        return (predicted - target).abs().mean()

    @staticmethod
    def ternary_loss(completed, truth, mask, current, shift):
        return ((completed - truth).square() * mask).mean()


def test_fixed_inputs_stay_exact_while_fcnet_and_propagator_recompute(monkeypatch):
    fcnet, propagator, unet = _FCNet(), torch.nn.Linear(1, 1, bias=False), _UNet()
    with torch.no_grad():
        propagator.weight.fill_(.3)
    components = SimpleNamespace(fcnet=fcnet, propagator=propagator, unet=unet,
                                 flow_loss=_FlowLoss(), vae=torch.nn.Identity(),
                                 image_encoder=torch.nn.Identity(), raft=torch.nn.Identity())
    condition_calls = []

    def propagate(module, latent, forward, backward, mask, plan):
        condition_calls.append((latent.detach().clone(), forward.detach().clone(),
                                module.weight.detach().clone()))
        condition = latent + module.weight.view(1, 1, 1, 1, 1) * forward.mean()
        return None, None, condition

    monkeypatch.setattr(probe, "propagate_reference_latents", propagate)
    mask = torch.ones(1, 2, 1, 2, 2)
    mask[..., 0] = 0
    target_latent = torch.full((1, 2, 4, 2, 2), .35)
    flow = torch.zeros(1, 1, 2, 2, 2)
    cache = {"plan": build_reference_plan(2, [0, 1]), "mask": mask,
             "flow": (flow, flow.clone()), "target_latent": target_latent,
             "cond_latent": torch.full_like(target_latent, .12),
             "noise": torch.full_like(target_latent, .23),
             "clip": torch.full((1, 1, 2), .71),
             "time_ids": torch.tensor([[7., 127., .049787]]),
             "sigma": 2.0137527, "cond_sigma": .049787}
    snapshot = {key: value.clone() for key, value in cache.items()
                if isinstance(value, torch.Tensor)}
    batch = {"target_rgb": torch.zeros(1, 2, 3, 2, 2)}
    optimizer = torch.optim.SGD([*fcnet.parameters(), *propagator.parameters(),
                                 *unet.parameters()], lr=.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    first = probe.fixed_train_step(components, batch, cache, optimizer, scaler, amp="bf16")
    second = probe.fixed_train_step(components, batch, cache, optimizer, scaler, amp="bf16")
    assert len(condition_calls) == 2 and fcnet.calls >= 2
    assert torch.equal(condition_calls[0][0], condition_calls[1][0])
    assert not torch.equal(condition_calls[0][1], condition_calls[1][1])
    assert not torch.equal(condition_calls[0][2], condition_calls[1][2])
    for index in (1, 2, 3):
        assert torch.equal(unet.inputs[0][index], unet.inputs[1][index])
    assert torch.equal(unet.inputs[0][0][:, :, :4], unet.inputs[1][0][:, :, :4])
    assert not torch.equal(unet.inputs[0][0][:, :, 4:], unet.inputs[1][0][:, :, 4:])
    for key, value in snapshot.items():
        assert torch.equal(cache[key], value)
    for row in (first, second):
        assert row["sigma"] == pytest.approx(cache["sigma"])
        assert row["frozen_grad_tensors"] == 0
        assert row["optimizer_updated"] is True
        assert all(group["status"] == "ok" for group in row["gradients"].values())


def test_terminal_state_carries_distinct_format_and_restart_material(tmp_path):
    components = SimpleNamespace(unet=torch.nn.Linear(2, 2),
                                 fcnet=torch.nn.Linear(2, 2),
                                 propagator=torch.nn.Linear(2, 2))
    parameters = [p for module in (components.unet, components.propagator,
                                   components.fcnet) for p in module.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "probe_state_final.pt"
    probe.save_terminal_state(path, components, optimizer, scheduler, scaler,
                              source_checkpoint=tmp_path / "step500.pt", source_step=500,
                              updates=64, update_attempts=64, seed=2026,
                              input_frames=["frame0.jpg"],
                              profile={"seed": 123})
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["format"] == probe.PROBE_FORMAT != FORMAT
    assert (state["source_step"], state["probe_updates"],
            state["probe_update_attempts"], state["formal_training_updates"]) == (500, 64, 64, 0)
    assert {"models", "optimizer", "scheduler", "scaler", "torch_rng",
            "cuda_rng", "numpy_rng", "python_rng"} <= set(state)
    assert not path.with_suffix(".pt.tmp").exists()
