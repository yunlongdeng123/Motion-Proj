#!/usr/bin/env python
"""物化 WorldSim V3 A2-D2 配对 smoke 的不可变训练配置。"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from motion_proj.worldsim_v3.actor_quota import D1_RANKING, D2_RANKING
from motion_proj.worldsim_v3.boundary_residual import (
    BoundaryResidualPolicy,
    validate_a2_d2_contract,
)
from scripts.materialize_worldsim_v3_a2_d1_config import (
    materialize_config as materialize_d1_config,
)


VARIANTS = ("d1-actor-quota", "d2-boundary-residual")


def materialize_config(
    source: DictConfig,
    variant: str,
    num_iters: int,
    protocol: dict,
) -> DictConfig:
    validate_a2_d2_contract(protocol)
    if variant not in VARIANTS:
        raise ValueError(f"unknown A2-D2 variant: {variant}")
    smoke = protocol["paired_smoke"]
    if num_iters != int(smoke["num_iters"]):
        raise ValueError("A2-D2 smoke iteration budget drift")

    inherited = protocol["paired_intervention"]["d1_inherited_exactly"]
    quota = OmegaConf.create(
        {
            "densify_grad_threshold": inherited[
                "rigid_densify_grad_threshold"
            ],
            "minimum_initial_multiplier": inherited[
                "minimum_initial_multiplier"
            ],
            "minimum_absolute_floor": inherited["minimum_absolute_floor"],
            "maximum_initial_multiplier": inherited[
                "maximum_initial_multiplier"
            ],
            "maximum_absolute_cap": inherited["maximum_absolute_cap"],
            "ranking": D2_RANKING if variant == VARIANTS[1] else D1_RANKING,
            "below_threshold_policy": inherited["below_threshold_policy"],
            "budget_policy": "gradient_ranked_prefix",
        }
    )
    config = materialize_d1_config(
        source,
        "d1-actor-quota",
        num_iters,
        quota,
        stage="paired-smoke",
    )
    policy = BoundaryResidualPolicy.from_contract(protocol)
    config.model.RigidNodes.ctrl.a2_boundary_residual = {
        "enabled": variant == VARIANTS[1],
        "boundary_radius_pixels": policy.boundary_radius_pixels,
        "mask_binarization_threshold": policy.mask_binarization_threshold,
        "scale_cap_threshold_multiplier": (
            policy.scale_cap_threshold_multiplier
        ),
        "ranking": policy.ranking,
    }
    config.worldsim_v3.stage = "D2 paired engineering smoke"
    config.worldsim_v3.variant = variant
    config.worldsim_v3.boundary_residual_checkpoint_key = (
        "worldsim_a2_boundary_residual"
    )
    config.worldsim_v3.d1_inherited_exactly = True
    return config


def normalized_pair_payload(config: DictConfig) -> dict:
    payload = OmegaConf.to_container(config, resolve=True)
    payload["worldsim_v3"]["variant"] = "<paired-variant>"
    payload["model"]["RigidNodes"]["ctrl"]["a2_actor_quota"][
        "ranking"
    ] = "<paired-ranking>"
    payload["model"]["RigidNodes"]["ctrl"]["a2_boundary_residual"][
        "enabled"
    ] = "<paired-enabled>"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = OmegaConf.to_container(
        OmegaConf.load(args.protocol), resolve=True
    )
    config = materialize_config(
        OmegaConf.load(args.source_config),
        args.variant,
        args.num_iters,
        protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=args.output)
    print(OmegaConf.to_yaml(config.worldsim_v3))


if __name__ == "__main__":
    main()
