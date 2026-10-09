"""固定片段容量入口的纯 CPU 数据角色与更新契约。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from motion_proj.worldsim_v81.p1_capacity_probe import (
    PROBE_FORMAT,
    build_teacher_condition,
    parse_args,
    query_images_from_visible,
    save_probe_state,
    validate_probe_contract,
)
from motion_proj.worldsim_v81.reference_propagation import build_reference_plan
from motion_proj.worldsim_v81.train_p1 import FORMAT


def _state(root: Path) -> dict:
    return {"format": FORMAT, "step": 100,
            "propagation_protocol": "paper-bidirectional-m4",
            "optimizer_name": "Adam",
            "optimizer": {"param_groups": [{"lr": 1e-5, "weight_decay": 0}]},
            "args": {"data_root": str(root), "lr": 1e-5, "amp": "bf16"},
            "data_profile": {"videos": 10, "available_windows": 100}}


def test_capacity_contract_rejects_wrong_source_budget_and_optimizer(tmp_path):
    state = _state(tmp_path)
    profile = {"videos": 10, "available_windows": 100}
    batch = {"video_id": ["0fc958cde2"], "clip_start": torch.tensor([2])}
    assert validate_probe_contract(state, profile, batch, data_root=tmp_path, updates=32) == "bf16"
    with pytest.raises(ValueError, match="1..64"):
        validate_probe_contract(state, profile, batch, data_root=tmp_path, updates=65)
    with pytest.raises(ValueError, match="第100步"):
        validate_probe_contract({**state, "step": 1000}, profile, batch,
                                data_root=tmp_path, updates=32)
    with pytest.raises(ValueError, match="Adam"):
        validate_probe_contract({**state, "optimizer_name": "AdamW"}, profile, batch,
                                data_root=tmp_path, updates=32)
    with pytest.raises(ValueError, match="固定训练片段"):
        validate_probe_contract(state, profile, {**batch, "clip_start": torch.tensor([3])},
                                data_root=tmp_path, updates=32)


def test_query_pil_condition_cannot_read_hidden_gt():
    mask = torch.ones(1, 25, 1, 8, 8)
    mask[..., 2:6] = 0
    visible = torch.zeros(1, 25, 3, 8, 8)
    visible[..., 2:6] = .2
    first = {"target_rgb": torch.zeros_like(visible), "visible_rgb": visible,
             "hole_mask": mask}
    second = {**first, "target_rgb": torch.ones_like(visible)}
    target1, query1, edges1 = query_images_from_visible(first)
    target2, query2, edges2 = query_images_from_visible(second)
    assert edges1 == edges2 == (2, 6)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(query1, query2))
    assert target1[0].tobytes() != target2[0].tobytes()
    assert query1[0].getpixel((0, 0)) == (0, 0, 0)


class _FlowComplete:
    def __init__(self):
        self.shift = 0.0

    def forward_bidirect_flow(self, flows, masks):
        forward = torch.zeros_like(flows[0])
        forward[:, :, 0] = self.shift
        return (forward, -forward), None

    def combine_flow(self, flows, predicted, masks):
        return predicted


class _CapturePropagator:
    use_fb_check = False

    def backward_warp_cached(self, image, flow):
        batch, _, height, width = image.shape
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).to(image).unsqueeze(0).expand(batch, -1, -1, -1)
        grid = grid + flow.permute(0, 2, 3, 1)
        grid[..., 0] = 2 * grid[..., 0] / (width - 1) - 1
        grid[..., 1] = 2 * grid[..., 1] / (height - 1) - 1
        return F.grid_sample(image, grid, align_corners=True)

    def _post_fuse(self, original, masks, future, past, *unused):
        return past, future, future


def test_teacher_rebuilds_condition_after_trainable_flow_update():
    plan = build_reference_plan(2, [0, 1])
    mask = torch.ones(1, 2, 1, 11, 11)
    mask[..., 4:6] = 0
    latent = torch.zeros(1, 2, 4, 11, 11)
    latent[0, 1, 0, 5, 4] = 1
    flow = torch.zeros(1, 1, 2, 11, 11)
    cache = {"plan": plan, "mask": mask, "flow": (flow, flow.clone()),
             "cond_latent": latent}
    fcnet = _FlowComplete()
    components = SimpleNamespace(fcnet=fcnet, propagator=_CapturePropagator())
    before = build_teacher_condition(components, cache, torch.device("cpu"))
    fcnet.shift = 1.0
    after = build_teacher_condition(components, cache, torch.device("cpu"))
    assert not torch.equal(before, after)
    assert float(before[0, 0, 0, 5, 3]) == pytest.approx(0)
    assert float(after[0, 0, 0, 5, 3]) == pytest.approx(1)


def test_cli_keeps_experiment_explicit(tmp_path):
    args = parse_args(["--checkpoint", str(tmp_path / "step100.pt"),
                       "--output-dir", str(tmp_path / "capacity"), "--updates", "32"])
    assert args.updates == 32 and args.query_steps == 25


def test_probe_state_is_recoverable_but_not_a_formal_train_checkpoint(tmp_path):
    components = SimpleNamespace(unet=torch.nn.Linear(2, 2),
                                 fcnet=torch.nn.Linear(2, 2),
                                 propagator=torch.nn.Linear(2, 2))
    parameters = [p for module in (components.unet, components.fcnet,
                                   components.propagator) for p in module.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=1e-5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    generator = torch.Generator().manual_seed(2026)
    path = tmp_path / "probe_state_final.pt"
    save_probe_state(path, components, optimizer, scheduler, scaler, generator,
                     source_checkpoint=tmp_path / "p1-checkpoint-000100.pt",
                     updates=32, seed=2026,
                     data_profile={"videos": 10, "available_windows": 100},
                     input_frames=["frame0.jpg"], amp="bf16")
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["format"] == PROBE_FORMAT != FORMAT
    assert state["source_step"] == 100 and state["probe_updates"] == 32
    assert set(state["models"]) == {"unet_temporal", "fcnet", "propagator"}
    assert state["diagnostic_profile"]["input_frames"] == ["frame0.jpg"]
    assert {"optimizer", "scheduler", "scaler", "torch_rng", "cuda_rng",
            "numpy_rng", "python_rng", "cfg_rng"} <= set(state)
    assert not path.with_suffix(".pt.tmp").exists()
