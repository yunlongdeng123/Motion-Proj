#!/usr/bin/env python
"""在真实 DriveStudio RigidNodes 路径验证 A2-D1 actor quota。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import OmegaConf
from torch.nn import Parameter

from models.nodes.rigid import RigidNodes
from motion_proj.worldsim_v3.actor_quota import validate_a2_d1_contract


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def control(contract: dict[str, Any]) -> Any:
    rigid = contract["actor_densification"]["rigid_nodes"]
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
            "split_screen_size": 0.05,
            "stop_screen_size_at": 100,
            "stop_split_at": 100,
            "cull_out_of_bound": False,
            "a2_ancestry": {"enabled": True},
            "a2_actor_quota": {
                "enabled": True,
                "densify_grad_threshold": rigid[
                    "densify_grad_threshold"
                ],
                **rigid["quota"],
            },
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


def create_model(contract: dict[str, Any]) -> RigidNodes:
    torch.manual_seed(17)
    model = RigidNodes(
        class_name="RigidNodes",
        ctrl=control(contract),
        reg=OmegaConf.create({}),
        scene_scale=1.0,
        scene_origin=torch.zeros(3),
        num_train_images=1,
        device=torch.device("cpu"),
    )
    model.create_from_pcd(instance_inputs())
    scales = torch.full_like(model._scales, 0.1)
    scales[1] = 1.0
    model._scales = Parameter(torch.log(scales))
    model._opacities = Parameter(
        torch.logit(torch.full_like(model._opacities, 0.1))
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_a2_d1_contract(contract)
    model = create_model(contract)
    initial_summary = model.get_a2_actor_quota_summary()
    groups = [
        {"params": params, "name": name}
        for name, params in model.get_gaussian_param_groups().items()
    ]
    optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
    zero_loss = sum(
        parameter.sum() * 0.0
        for group in groups
        for parameter in group["params"]
    )
    zero_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model.step = 5
    model.xys_grad_norm = torch.tensor(
        [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.0002, 0.0]
    )
    model.vis_counts = torch.ones(model.num_points)
    model.max_2Dsize = torch.zeros(model.num_points)
    model.max_2Dsize[2] = 0.1
    torch.manual_seed(23)
    model.refinement_after(5, optimizer)

    state = model.state_dict()
    final_summary = model.get_a2_actor_quota_summary()
    ancestry_summary = model._a2_ancestry.summary()
    restored = RigidNodes(
        class_name="RigidNodes",
        ctrl=control(contract),
        reg=OmegaConf.create({}),
        scene_scale=1.0,
        scene_origin=torch.zeros(3),
        num_train_images=1,
        device=torch.device("cpu"),
    )
    restored.load_state_dict(copy.deepcopy(state))
    roundtrip_summary = restored.get_a2_actor_quota_summary()

    expected_counts = [10, 6]
    actual_counts = [
        row["current"] for row in final_summary["actors"]
    ]
    maximum_respected = all(
        row["current"] <= row["maximum"]
        for row in final_summary["actors"]
    )
    result = {
        "status": "done",
        "task_id": contract["task_id"],
        "component": "A2-D1 actor quota synthetic integration smoke",
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "initial_quota": initial_summary,
        "final_quota": final_summary,
        "roundtrip_quota": roundtrip_summary,
        "ancestry": ancestry_summary,
        "quota_checkpoint_key_present": (
            "worldsim_a2_actor_quota" in state
        ),
        "ancestry_checkpoint_key_present": (
            "worldsim_a2_ancestry" in state
        ),
        "maximum_respected": maximum_respected,
        "expected_actor_counts": expected_counts,
        "actual_actor_counts": actual_counts,
    }
    valid = (
        result["quota_checkpoint_key_present"]
        and result["ancestry_checkpoint_key_present"]
        and maximum_respected
        and actual_counts == expected_counts
        and final_summary == roundtrip_summary
        and final_summary["counters"]["accepted_children"] == 8
        and final_summary["counters"][
            "rejected_by_maximum_parents"
        ]
        == 1
        and ancestry_summary["live_gaussians"] == 16
    )
    if not valid:
        result["status"] = "blocked"
        result["reason"] = "A2_D1_SYNTHETIC_QUOTA_CONTRACT_FAILED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "done":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
