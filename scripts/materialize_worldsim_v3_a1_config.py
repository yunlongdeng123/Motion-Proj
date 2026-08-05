#!/usr/bin/env python
"""Materialize immutable DriveStudio configs for WorldSim V3 A1 variants."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


VARIANTS = (
    "c0-off",
    "c1-native",
    "c2-factorized-isp",
    "c3-bounded-pose",
)


def materialize_config(source: DictConfig, variant: str, num_iters: int) -> DictConfig:
    if variant not in VARIANTS:
        raise ValueError(f"unknown A1 variant: {variant}")
    if num_iters <= 0:
        raise ValueError("num_iters must be positive")
    config = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
    config.trainer.type = "motion_proj.worldsim_v3.trainer.WorldSimV3Trainer"
    config.trainer.optim.num_iters = int(num_iters)
    config.logging.saveckpt_freq = int(num_iters)
    config.render.render_full = False
    config.render.render_test = False
    config.render.render_novel = None

    if variant == "c0-off":
        del config.model["Affine"]
        del config.model["CamPose"]
    elif variant != "c1-native":
        affine_optim = OmegaConf.to_container(config.model.Affine.optim, resolve=True)
        config.model.Affine = {
            "type": "motion_proj.worldsim_v3.calibration.FactorizedAffineTransform",
            "params": {
                "num_cameras": 3,
                "camera_embedding_dim": 4,
                "time_embedding_dim": 8,
                "num_time_frequencies": 2,
                "base_mlp_layer_width": 64,
            },
            "optim": affine_optim,
        }
        if variant == "c3-bounded-pose":
            pose_optim = OmegaConf.to_container(
                config.model.CamPose.optim, resolve=True
            )
            config.model.CamPose = {
                "type": "motion_proj.worldsim_v3.calibration.BoundedCameraOptModule",
                "params": {
                    "num_cameras": 3,
                    "max_translation_m": 0.15,
                    "max_rotation_deg": 2.0,
                    "translation_prior_weight": 1.0e-4,
                    "rotation_prior_weight": 1.0e-4,
                    "temporal_smoothness_weight": 1.0e-3,
                },
                "optim": pose_optim,
            }
    config.worldsim_v3 = {
        "task_id": "WS-V3-A1-CALIBRATION-01",
        "variant": variant,
        "rolling_shutter": "not_supported",
        "rolling_shutter_reason": (
            "nuScenes raw and processed contracts have no readout direction/time or row timing"
        ),
        "exposure_metadata": "unavailable",
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--num-iters", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = materialize_config(
        OmegaConf.load(args.source_config), args.variant, args.num_iters
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=args.output)
    print(OmegaConf.to_yaml(config.worldsim_v3))


if __name__ == "__main__":
    main()
