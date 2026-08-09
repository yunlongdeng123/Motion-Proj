#!/usr/bin/env python
"""在真实 DriveStudio RigidNodes 路径验证 A2 ancestry 与 module-off 等价性。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Union

import torch
from omegaconf import OmegaConf
from torch.nn import Parameter

from models.nodes.rigid import RigidNodes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def control(enabled: bool) -> Any:
    return OmegaConf.create(
        {
            "sh_degree": 0,
            "warmup_steps": 0,
            "reset_alpha_interval": 10,
            "refine_interval": 1,
            "sh_degree_interval": 1000,
            "n_split_samples": 2,
            "reset_alpha_value": 0.01,
            "densify_grad_thresh": 0.5,
            "densify_size_thresh": 0.5,
            "cull_alpha_thresh": 0.005,
            "cull_scale_thresh": 10.0,
            "cull_screen_size": 10.0,
            "split_screen_size": 10.0,
            "stop_screen_size_at": 100,
            "stop_split_at": 100,
            "cull_out_of_bound": False,
            "a2_ancestry": {"enabled": enabled},
        }
    )


def instance_inputs() -> dict[str, dict[str, Any]]:
    pose = torch.eye(4).repeat(3, 1, 1)
    frame_info = torch.ones(3, dtype=torch.bool)
    return {
        "actor-a": {
            "pts": torch.tensor(
                [
                    [-1.0, 0.0, 0.0],
                    [-0.5, 0.1, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.5, -0.1, 0.0],
                ]
            ),
            "colors": torch.full((4, 3), 0.25),
            "poses": pose.clone(),
            "size": torch.tensor([8.0, 4.0, 3.0]),
            "frame_info": frame_info.clone(),
            "num_pts": 4,
        },
        "actor-b": {
            "pts": torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [0.2, 1.5, 0.0],
                    [-0.2, 2.0, 0.0],
                    [0.0, 2.5, 0.0],
                ]
            ),
            "colors": torch.full((4, 3), 0.75),
            "poses": pose.clone(),
            "size": torch.tensor([8.0, 4.0, 3.0]),
            "frame_info": frame_info.clone(),
            "num_pts": 4,
        },
    }


def create_model(enabled: bool) -> RigidNodes:
    torch.manual_seed(17)
    model = RigidNodes(
        class_name="RigidNodes",
        ctrl=control(enabled),
        reg=OmegaConf.create({}),
        scene_scale=1.0,
        scene_origin=torch.zeros(3),
        num_train_images=1,
        device=torch.device("cpu"),
    )
    model.create_from_pcd(instance_inputs())
    scales = torch.full_like(model._scales, 0.1)
    scales[0] = 1.0
    model._scales = Parameter(torch.log(scales))
    opacities = torch.full_like(model._opacities, 0.1)
    opacities[-1] = 0.001
    model._opacities = Parameter(torch.logit(opacities))
    return model


def run_refinement(enabled: bool) -> tuple[RigidNodes, dict[str, TensorLike]]:
    model = create_model(enabled)
    groups = [
        {"params": params, "name": name}
        for name, params in model.get_gaussian_param_groups().items()
    ]
    optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
    zero_loss = sum(parameter.sum() * 0.0 for group in groups for parameter in group["params"])
    zero_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model.step = 5
    model.xys_grad_norm = torch.zeros(model.num_points)
    model.xys_grad_norm[0] = 1.0
    model.xys_grad_norm[1] = 1.0
    model.vis_counts = torch.ones(model.num_points)
    model.max_2Dsize = torch.zeros(model.num_points)
    torch.manual_seed(23)
    model.refinement_after(5, optimizer)
    state = model.state_dict()
    return model, state


TensorLike = Union[torch.Tensor, dict[str, Any]]


def native_tensor_state(state: dict[str, TensorLike]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state.items()
        if key != "worldsim_a2_ancestry" and isinstance(value, torch.Tensor)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    off_model, off_state = run_refinement(False)
    on_model, on_state = run_refinement(True)
    off_native = native_tensor_state(off_state)
    on_native = native_tensor_state(on_state)
    keys_equal = set(off_native) == set(on_native)
    mismatched = sorted(
        key
        for key in set(off_native) & set(on_native)
        if not torch.equal(off_native[key], on_native[key])
    )
    off_has_ancestry = "worldsim_a2_ancestry" in off_state
    on_has_ancestry = "worldsim_a2_ancestry" in on_state
    ledger = on_model._a2_ancestry
    ledger.validate(expected_actor_ids=on_model.point_ids[..., 0])

    restored = RigidNodes(
        class_name="RigidNodes",
        ctrl=control(True),
        reg=OmegaConf.create({}),
        scene_scale=1.0,
        scene_origin=torch.zeros(3),
        num_train_images=1,
        device=torch.device("cpu"),
    )
    restored.load_state_dict(on_state)
    restored._a2_ancestry.validate(
        expected_actor_ids=restored.point_ids[..., 0]
    )

    result = {
        "status": "done",
        "task_id": "WS-V3-A2-ACTOR-DENSIFY-01",
        "component": "A2-I0 module-off equivalence smoke",
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "off_native_checkpoint_keys_equal_on": keys_equal,
        "native_tensor_bitwise_equal": keys_equal and not mismatched,
        "mismatched_native_tensors": mismatched,
        "off_has_ancestry_checkpoint_key": off_has_ancestry,
        "on_has_ancestry_checkpoint_key": on_has_ancestry,
        "off_gaussian_count": off_model.num_points,
        "on_gaussian_count": on_model.num_points,
        "on_ancestry": ledger.summary(),
        "roundtrip_ancestry": restored._a2_ancestry.summary(),
    }
    if not keys_equal or mismatched or off_has_ancestry or not on_has_ancestry:
        result["status"] = "blocked"
        result["reason"] = "MODULE_OFF_EQUIVALENCE_FAILED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "done":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
