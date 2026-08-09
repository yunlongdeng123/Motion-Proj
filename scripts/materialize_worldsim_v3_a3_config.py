#!/usr/bin/env python
"""Materialize A3 R0 alias metadata or the engineering-only R1 config."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from motion_proj.worldsim_v3.local_refinement import (
    AUDIT_VERSION,
    TASK_ID,
    validate_a3_protocol,
    validate_a3_sidecar_manifest,
    sha256_file,
)


VARIANTS = ("r0-no-refine-exact-alias", "r1-reactivate")
SELECTED_CHECKPOINT_STEP = 30_000
PAIRED_LOSS_KEYS = {"rgb", "ssim", "mask", "depth"}


def _validate_source(
    source: DictConfig,
    protocol: dict[str, Any],
    *,
    source_config_sha256: str,
    checkpoint_sha256: str,
) -> None:
    dependencies = protocol["depends_on"]
    if source_config_sha256 != dependencies["selected_checkpoint_config_sha256"]:
        raise ValueError("A3 source checkpoint config SHA drift")
    if checkpoint_sha256 != dependencies["selected_checkpoint_sha256"]:
        raise ValueError("A3 selected checkpoint SHA drift")
    if int(source.seed) != 0:
        raise ValueError("A3 source seed drift")
    if int(source.data.scene_idx) != 179:
        raise ValueError("A3 source scene drift")
    if list(source.data.pixel_source.cameras) != [0, 1, 2]:
        raise ValueError("A3 source cameras drift")
    if int(source.data.pixel_source.test_image_stride) != 10:
        raise ValueError("A3 held-out split drift")
    if int(source.trainer.optim.num_iters) != SELECTED_CHECKPOINT_STEP:
        raise ValueError("A3 selected checkpoint step drift")
    if source.worldsim_v3.variant != "d2-boundary-residual":
        raise ValueError("A3 must start from the selected D2 asset")


def r0_alias_manifest(
    source: DictConfig,
    protocol: dict[str, Any],
    *,
    protocol_sha256: str,
    source_config_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Describe R0 without creating a config or a checkpoint."""

    validate_a3_protocol(protocol)
    _validate_source(
        source,
        protocol,
        source_config_sha256=source_config_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "audit_version": AUDIT_VERSION,
        "variant": VARIANTS[0],
        "mode": "immutable_exact_alias_no_optimizer_no_new_checkpoint",
        "formal_training_authorized": False,
        "protocol_sha256": protocol_sha256,
        "source_config_sha256": source_config_sha256,
        "checkpoint": protocol["depends_on"]["selected_checkpoint"],
        "checkpoint_sha256": checkpoint_sha256,
        "optimizer_steps": 0,
        "new_checkpoint_keys": False,
    }


def materialize_r1_config(
    source: DictConfig,
    protocol: dict[str, Any],
    sidecar_manifest: dict[str, Any],
    *,
    protocol_sha256: str,
    source_config_sha256: str,
    checkpoint_sha256: str,
    sidecar_manifest_path: str,
    optimizer_steps: int,
    stage: str = "paired-engineering-smoke",
) -> DictConfig:
    """Create an R1 config while keeping all numeric choices non-formal."""

    validate_a3_protocol(protocol)
    if stage != "paired-engineering-smoke":
        raise ValueError("A3 formal materialization is not authorized")
    if not isinstance(optimizer_steps, int) or optimizer_steps <= 0:
        raise ValueError("A3 engineering optimizer steps must be positive")
    _validate_source(
        source,
        protocol,
        source_config_sha256=source_config_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    validate_a3_sidecar_manifest(
        sidecar_manifest,
        protocol_sha256=protocol_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )

    config = OmegaConf.create(deepcopy(OmegaConf.to_container(source, resolve=True)))
    config.trainer.optim.num_iters = SELECTED_CHECKPOINT_STEP + optimizer_steps
    # R1 may only learn from externally authorized paired pixels.  Native
    # Gaussian and trainer regularizers have no S-A/S-B provenance and could
    # otherwise move an authorized opacity/scale row through a loss side door.
    for model_config in config.model.values():
        if isinstance(model_config, DictConfig) and "reg" in model_config:
            model_config.reg = {}
    for loss_name in list(config.trainer.losses):
        if loss_name not in PAIRED_LOSS_KEYS:
            del config.trainer.losses[loss_name]
    config.trainer.a3_local_refinement = {
        "enabled": True,
        "variant": VARIANTS[1],
        "engineering_only": True,
        "formal_training_authorized": False,
        "start_after_checkpoint_step": True,
        "disable_native_gaussian_postprocess": True,
        "require_external_paired_loss_masks": True,
        "native_regularizers_disabled": True,
        "sidecar_manifest": sidecar_manifest_path,
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    config.logging.saveckpt_freq = config.trainer.optim.num_iters
    config.worldsim_v3 = {
        "task_id": TASK_ID,
        "audit_version": AUDIT_VERSION,
        "stage": "A3 R1 paired engineering smoke only",
        "variant": VARIANTS[1],
        "seed": 0,
        "scene": "scene-0230",
        "source_variant": "d2-boundary-residual",
        "source_config_sha256": source_config_sha256,
        "source_checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": protocol_sha256,
        "sidecar_arrays_sha256": sidecar_manifest["arrays"]["sha256"],
        "optimizer_steps": optimizer_steps,
        "native_regularizers_disabled": True,
        "formal": False,
        "numeric_protocol_frozen": False,
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--sidecar-manifest", type=Path)
    parser.add_argument("--optimizer-steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    source = OmegaConf.load(args.source_config)
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    protocol_sha256 = sha256_file(args.protocol)
    source_config_sha256 = sha256_file(args.source_config)
    checkpoint_sha256 = sha256_file(args.source_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.variant == VARIANTS[0]:
        payload = r0_alias_manifest(
            source,
            protocol,
            protocol_sha256=protocol_sha256,
            source_config_sha256=source_config_sha256,
            checkpoint_sha256=checkpoint_sha256,
        )
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    if args.sidecar_manifest is None or args.optimizer_steps is None:
        parser.error("R1 requires --sidecar-manifest and --optimizer-steps")
    sidecar = json.loads(args.sidecar_manifest.read_text(encoding="utf-8"))
    config = materialize_r1_config(
        source,
        protocol,
        sidecar,
        protocol_sha256=protocol_sha256,
        source_config_sha256=source_config_sha256,
        checkpoint_sha256=checkpoint_sha256,
        sidecar_manifest_path=str(args.sidecar_manifest.resolve()),
        optimizer_steps=args.optimizer_steps,
    )
    OmegaConf.save(config=config, f=args.output)


if __name__ == "__main__":
    main()
