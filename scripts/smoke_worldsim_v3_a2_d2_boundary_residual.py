#!/usr/bin/env python
"""在真实 DriveStudio RigidNodes 路径验证 A2-D2 边界/残差干预。"""

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
from motion_proj.worldsim_v3.actor_quota import D1_RANKING, D2_RANKING
from motion_proj.worldsim_v3.boundary_residual import (
    BoundaryResidualPolicy,
    validate_a2_d2_contract,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def control(
    protocol: dict[str, Any],
    *,
    d2_enabled: bool,
    include_d2_config: bool = True,
) -> Any:
    inherited = protocol["paired_intervention"]["d1_inherited_exactly"]
    payload: dict[str, Any] = {
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
        "a2_ancestry": {"enabled": True},
        "a2_actor_quota": {
            "enabled": True,
            "densify_grad_threshold": inherited[
                "rigid_densify_grad_threshold"
            ],
            "minimum_initial_multiplier": inherited[
                "minimum_initial_multiplier"
            ],
            "minimum_absolute_floor": inherited[
                "minimum_absolute_floor"
            ],
            "maximum_initial_multiplier": inherited[
                "maximum_initial_multiplier"
            ],
            "maximum_absolute_cap": inherited["maximum_absolute_cap"],
            "ranking": D2_RANKING if d2_enabled else D1_RANKING,
            "below_threshold_policy": inherited[
                "below_threshold_policy"
            ],
            "budget_policy": "gradient_ranked_prefix",
        },
    }
    if include_d2_config:
        policy = BoundaryResidualPolicy.from_contract(protocol)
        payload["a2_boundary_residual"] = {
            "enabled": d2_enabled,
            "boundary_radius_pixels": policy.boundary_radius_pixels,
            "mask_binarization_threshold": (
                policy.mask_binarization_threshold
            ),
            "scale_cap_threshold_multiplier": (
                policy.scale_cap_threshold_multiplier
            ),
            "ranking": policy.ranking,
        }
    return OmegaConf.create(payload)


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


def create_model(
    protocol: dict[str, Any],
    *,
    d2_enabled: bool,
    include_d2_config: bool = True,
) -> RigidNodes:
    torch.manual_seed(17)
    model = RigidNodes(
        class_name="RigidNodes",
        ctrl=control(
            protocol,
            d2_enabled=d2_enabled,
            include_d2_config=include_d2_config,
        ),
        reg=OmegaConf.create({}),
        scene_scale=1.0,
        scene_origin=torch.zeros(3),
        num_train_images=1,
        device=torch.device("cpu"),
    )
    model.create_from_pcd(instance_inputs())
    model._scales = Parameter(torch.log(torch.ones_like(model._scales)))
    model._opacities = Parameter(
        torch.logit(torch.full_like(model._opacities, 0.1))
    )
    return model


def make_optimizer(model: RigidNodes) -> torch.optim.Adam:
    groups = [
        {"params": params, "name": name}
        for name, params in model.get_gaussian_param_groups().items()
    ]
    optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
    loss = sum(
        parameter.sum()
        for group in groups
        for parameter in group["params"]
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def prepare_refinement(model: RigidNodes) -> None:
    model.step = 5
    model.xys_grad_norm = torch.tensor(
        [0.1, 0.9, 0.8, 1.0, 0.2, 0.7, 0.6, 0.95]
    )
    model.vis_counts = torch.ones(model.num_points)
    model.max_2Dsize = torch.zeros(model.num_points)


def tensor_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bytes = left.detach().cpu().contiguous().numpy().tobytes()
    right_bytes = right.detach().cpu().contiguous().numpy().tobytes()
    return left_bytes == right_bytes


def nested_bitwise_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and tensor_bitwise_equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(nested_bitwise_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                nested_bitwise_equal(a, b) for a, b in zip(left, right)
            )
        )
    return left == right


def module_off_equivalence(protocol: dict[str, Any]) -> dict[str, Any]:
    reference = create_model(
        protocol, d2_enabled=False, include_d2_config=False
    )
    disabled = create_model(
        protocol, d2_enabled=False, include_d2_config=True
    )
    reference_optimizer = make_optimizer(reference)
    disabled_optimizer = make_optimizer(disabled)
    prepare_refinement(reference)
    prepare_refinement(disabled)

    torch.manual_seed(23)
    reference.refinement_after(5, reference_optimizer)
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(23)
    disabled.refinement_after(5, disabled_optimizer)
    disabled_rng = torch.get_rng_state().clone()

    reference_state = reference.state_dict()
    disabled_state = disabled.state_dict()
    return {
        "checkpoint_keys_equal": (
            reference_state.keys() == disabled_state.keys()
        ),
        "native_state_bitwise_equal": nested_bitwise_equal(
            reference_state, disabled_state
        ),
        "rng_state_equal": tensor_bitwise_equal(
            reference_rng, disabled_rng
        ),
        "d2_checkpoint_key_absent": (
            "worldsim_a2_boundary_residual" not in disabled_state
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    validate_a2_d2_contract(protocol)
    model = create_model(protocol, d2_enabled=True)
    optimizer = make_optimizer(model)
    model.filter_mask = torch.ones(model.num_points, dtype=torch.bool)
    observed_indices = torch.tensor([0, 1, 2, 4, 5, 6])
    model.record_a2_d2_projected_observations(
        filtered_indices=observed_indices,
        boundary=torch.tensor([1.0, 0.5, 0.25, 0.9, 0.4, 0.1]),
        photometric_residual=torch.tensor(
            [0.1, 0.9, 0.8, 0.2, 0.7, 0.6]
        ),
    )
    prepare_refinement(model)
    torch.manual_seed(23)
    model.refinement_after(5, optimizer)

    state = model.state_dict()
    d2_summary = model.get_a2_boundary_residual_summary()
    quota_summary = model.get_a2_actor_quota_summary()
    ancestry_summary = model._a2_ancestry.summary()
    restored = RigidNodes(
        class_name="RigidNodes",
        ctrl=control(protocol, d2_enabled=True),
        reg=OmegaConf.create({}),
        scene_scale=1.0,
        scene_origin=torch.zeros(3),
        num_train_images=1,
        device=torch.device("cpu"),
    )
    restored.load_state_dict(copy.deepcopy(state))
    roundtrip_d2 = restored.get_a2_boundary_residual_summary()
    roundtrip_quota = restored.get_a2_actor_quota_summary()

    scaling_group = next(
        group
        for group in optimizer.param_groups
        if group["name"] == model.class_prefix + "scaling"
    )
    scaling_parameter = scaling_group["params"][0]
    scaling_state = optimizer.state[scaling_parameter]
    cap_rows = torch.tensor([0, 1, 2, 4, 5, 6])
    moment_rows_zero = all(
        bool(torch.count_nonzero(scaling_state[key][cap_rows]) == 0)
        for key in ("exp_avg", "exp_avg_sq")
    )
    expected_actor_counts = [10, 10]
    actual_actor_counts = [
        row["current"] for row in quota_summary["actors"]
    ]
    maximum_respected = all(
        row["current"] <= row["maximum"]
        for row in quota_summary["actors"]
    )
    module_off = module_off_equivalence(protocol)

    result = {
        "status": "done",
        "task_id": protocol["task_id"],
        "component": "A2-D2 boundary/residual synthetic integration smoke",
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "boundary_residual": d2_summary,
        "roundtrip_boundary_residual": roundtrip_d2,
        "quota": quota_summary,
        "roundtrip_quota": roundtrip_quota,
        "ancestry": ancestry_summary,
        "d2_checkpoint_key_present": (
            "worldsim_a2_boundary_residual" in state
        ),
        "quota_checkpoint_key_present": (
            "worldsim_a2_actor_quota" in state
        ),
        "ancestry_checkpoint_key_present": (
            "worldsim_a2_ancestry" in state
        ),
        "expected_actor_counts": expected_actor_counts,
        "actual_actor_counts": actual_actor_counts,
        "maximum_respected": maximum_respected,
        "scale_optimizer_moment_rows_zero": moment_rows_zero,
        "module_off_equivalence": module_off,
    }
    counters = d2_summary["counters"]
    quota_counters = quota_summary["counters"]
    valid = (
        result["d2_checkpoint_key_present"]
        and result["quota_checkpoint_key_present"]
        and result["ancestry_checkpoint_key_present"]
        and d2_summary == roundtrip_d2
        and quota_summary == roundtrip_quota
        and counters["boundary_observations"] == 6
        and counters["photometric_residual_observations"] == 6
        and counters["refinement_events"] == 1
        and counters["capped_gaussians"] == 6
        and quota_counters["events"] == 1
        and quota_counters["accepted_split_parents"] == 6
        and quota_counters["accepted_children"] == 12
        and quota_counters["rejected_by_maximum_parents"] == 2
        and actual_actor_counts == expected_actor_counts
        and maximum_respected
        and moment_rows_zero
        and all(module_off.values())
    )
    if not valid:
        result["status"] = "blocked"
        result["reason"] = "A2_D2_SYNTHETIC_INTEGRATION_CONTRACT_FAILED"
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
