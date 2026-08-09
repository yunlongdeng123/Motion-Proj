#!/usr/bin/env python
"""物化 WorldSim V3 A2-D1 配对 smoke 的不可变训练配置。"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


VARIANTS = ("d0-native", "d1-actor-quota")


def materialize_config(
    source: DictConfig,
    variant: str,
    num_iters: int,
    actor_quota: DictConfig,
) -> DictConfig:
    if variant not in VARIANTS:
        raise ValueError(f"unknown A2-D1 variant: {variant}")
    if num_iters < 1000:
        raise ValueError("A2-D1 smoke must run at least 1000 iterations")
    config = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
    config.trainer.type = (
        "motion_proj.worldsim_v3.trainer.WorldSimV3Trainer"
    )
    config.trainer.optim.num_iters = int(num_iters)
    config.logging.saveckpt_freq = int(num_iters)
    config.render.render_full = False
    config.render.render_test = False
    config.render.render_novel = None
    del config.model["Affine"]
    del config.model["CamPose"]

    native_background_threshold = float(
        config.trainer.gaussian_ctrl_general_cfg.densify_grad_thresh
    )
    if native_background_threshold != 0.0005:
        raise ValueError("native background gradient threshold drift")
    config.trainer.gaussian_ctrl_general_cfg.a2_ancestry = {
        "enabled": True
    }
    quota_payload = OmegaConf.to_container(actor_quota, resolve=True)
    config.model.RigidNodes.ctrl.a2_actor_quota = {
        "enabled": variant == "d1-actor-quota",
        **quota_payload,
    }
    config.worldsim_v3 = {
        "task_id": "WS-V3-A2-ACTOR-DENSIFY-01",
        "stage": "D1 paired engineering smoke",
        "variant": variant,
        "seed": 0,
        "scene": "scene-0230",
        "background_densification": "native_unchanged",
        "background_densify_grad_threshold": (
            native_background_threshold
        ),
        "actor_quota_checkpoint_key": "worldsim_a2_actor_quota",
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--num-iters", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = OmegaConf.load(args.contract)
    rigid = contract.actor_densification.rigid_nodes
    actor_quota = OmegaConf.create(
        {
            "densify_grad_threshold": rigid.densify_grad_threshold,
            **OmegaConf.to_container(rigid.quota, resolve=True),
        }
    )
    config = materialize_config(
        OmegaConf.load(args.source_config),
        args.variant,
        args.num_iters,
        actor_quota,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=args.output)
    print(OmegaConf.to_yaml(config.worldsim_v3))


if __name__ == "__main__":
    main()
