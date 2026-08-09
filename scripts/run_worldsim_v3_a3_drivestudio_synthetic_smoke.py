#!/usr/bin/env python
"""Verify A3 DriveStudio module-off exactness and fail-closed loss injection."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import torch


def _load_trainer(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load DriveStudio trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BasicTrainer


def _fake_trainer(*, a3_guard: object | None) -> SimpleNamespace:
    return SimpleNamespace(
        models={},
        gaussian_classes={},
        _a2_d2_diagnostic_maps=None,
        sky_opacity_loss_fn=None,
        depth_loss_fn=None,
        losses_dict=OmegaConf.create({"rgb": {"w": 0.8}, "ssim": {"w": 0.2}}),
        ssim=lambda left, right: (left - right).square().mean(),
        step=7,
        device=torch.device("cpu"),
        a3_local_refinement_guard=a3_guard,
    )


def run_smoke(original_root: Path, patched_root: Path) -> dict[str, object]:
    sys.path.insert(0, str(patched_root))
    original = _load_trainer(
        "worldsim_v3_a2_base_exact",
        original_root / "models/trainers/base.py",
    )
    patched = _load_trainer(
        "worldsim_v3_a3_base_patched",
        patched_root / "models/trainers/base.py",
    )
    generator = torch.Generator().manual_seed(0)
    image_infos = {
        "pixels": torch.rand(4, 5, 3, generator=generator),
        "sky_masks": torch.zeros(4, 5),
    }
    outputs = {
        "rgb": torch.rand(4, 5, 3, generator=generator),
        "opacity": torch.rand(4, 5, 1, generator=generator),
        "depth": torch.rand(4, 5, generator=generator),
    }
    original_losses = original.compute_losses(
        _fake_trainer(a3_guard=None), outputs, image_infos, {}
    )
    patched_losses = patched.compute_losses(
        _fake_trainer(a3_guard=None), outputs, image_infos, {}
    )
    module_off_exact = (
        original_losses.keys() == patched_losses.keys()
        and all(
            torch.equal(original_losses[name], patched_losses[name])
            for name in original_losses
        )
    )

    missing_masks_rejected = False
    try:
        patched.compute_losses(
            _fake_trainer(a3_guard=object()), outputs, image_infos, {}
        )
    except RuntimeError as error:
        missing_masks_rejected = "provenance is not authorized" in str(error)

    authorized = dict(image_infos)
    authorized.update(
        {
            "a3_paired_loss_authorized": torch.tensor(True),
            "a3_rgb_loss_mask": torch.tensor(
                [
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=torch.bool,
            ),
            "a3_geometry_loss_mask": torch.tensor(
                [
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=torch.bool,
            ),
        }
    )
    authorized_losses = patched.compute_losses(
        _fake_trainer(a3_guard=object()), outputs, authorized, {}
    )
    finite_authorized_losses = all(
        bool(torch.isfinite(value).all()) for value in authorized_losses.values()
    )
    passed = module_off_exact and missing_masks_rejected and finite_authorized_losses
    if not passed:
        raise RuntimeError("A3 DriveStudio synthetic integration smoke failed")
    return {
        "schema_version": 1,
        "task_id": "WS-V3-A3-LOCAL-REFINE-01",
        "audit_version": "A3-DRIVESTUDIO-SYNTHETIC-SMOKE-v1",
        "evidence_tier": "synthetic_contract_only_not_quality_evidence",
        "pass": passed,
        "module_off_loss_exact": module_off_exact,
        "module_off_loss_keys": list(original_losses),
        "missing_paired_masks_rejected": missing_masks_rejected,
        "authorized_masked_losses_finite": finite_authorized_losses,
        "formal_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/third_party/"
            "drivestudio-worldsim-v3-a2-d2-r8"
        ),
    )
    parser.add_argument(
        "--patched-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/third_party/"
            "drivestudio-worldsim-v3-a3-r1-r1"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = run_smoke(args.original_root, args.patched_root)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
