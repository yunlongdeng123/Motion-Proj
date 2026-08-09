#!/usr/bin/env python
"""Materialize the conservative heldout-safe A3 S-B/T0 engineering sidecar."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.dynamic_editing_v2.drivestudio_registry import require_token
from motion_proj.dynamic_editing_v2.pilot_metrics import (
    counterfactual_effect_mask,
)
from motion_proj.resim.drivestudio_adapter import gsplat_first_hit_from_info
from motion_proj.resim.label_regeneration import (
    lidar_measured_depth,
    render_expected_depth,
)
from motion_proj.worldsim_v3.local_refinement import (
    HELDOUT_FRAMES,
    SIDECAR_AUDIT_VERSION,
    TASK_ID,
    affected_pixel_mask,
    measured_background_support_mask,
    merge_s_b_row_observations,
    projected_background_rows,
    sha256_file,
    validate_a3_protocol,
)
from scripts.run_dr_v2_m4_pilot import move_actor_local_y


CAMERA_NAMES = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
EDITS = ("lateral", "delete")
EXPECTED_PROTOCOL_SHA256 = (
    "03fbf632645326692bbcf18ab18a08b5440c7733c709f925945c78018bb272d0"
)


def atomic_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def to_device(values: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in values.items()
    }


def uint8_rgb(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    if not np.isfinite(array).all():
        raise RuntimeError("A3 render contains non-finite RGB")
    return np.round(np.clip(array, 0, 1) * 255).astype(np.uint8)


def render_variant(
    *,
    trainer,
    dataset,
    checkpoint: Path,
    frame: int,
    camera: int,
    model_index: int,
    variant: str,
    device: torch.device,
    first_hit_alpha_threshold: float | None = None,
) -> dict[str, Any]:
    trainer.resume_from_checkpoint(str(checkpoint), load_only_model=True)
    trainer.set_eval()
    rigid = trainer.models["RigidNodes"]
    if variant == "lateral":
        move_actor_local_y(rigid, model_index, 1.0)
    elif variant == "delete":
        rigid.remove_instances([model_index])
    elif variant != "original":
        raise ValueError(f"unknown A3 render variant: {variant}")

    image_index = frame * dataset.pixel_source.num_cams + camera
    image_infos, camera_infos = dataset.full_image_set.get_image(
        image_index, camera_downscale=1.0
    )
    measured = image_infos["lidar_depth_map"].detach().float().cpu().numpy()
    image_infos = to_device(image_infos, device)
    camera_infos = to_device(camera_infos, device)
    with torch.inference_mode():
        outputs = trainer(image_infos, camera_infos)
    result = {
        "rgb": uint8_rgb(outputs["rgb"]),
        "depth_lidar_measured": measured,
        "image_index": image_index,
    }
    if first_hit_alpha_threshold is not None:
        expected = render_expected_depth(
            outputs["depth"].detach().float().cpu().numpy().squeeze(),
            outputs["opacity"].detach().float().cpu().numpy().squeeze(),
        )
        first_hit, first_hit_valid = gsplat_first_hit_from_info(
            trainer.info,
            alpha_threshold=first_hit_alpha_threshold,
        )
        result.update(
            {
                "depth_render_expected": expected,
                "depth_surface_first_hit": first_hit,
                "depth_surface_first_hit_valid": first_hit_valid,
                "means2d": trainer.info["means2d"].detach(),
                "radii": trainer.info["radii"].detach(),
            }
        )
    return result


def select_evidence_view(
    *,
    trainer,
    dataset,
    checkpoint: Path,
    model_index: int,
    valid_frames: list[int],
    device: torch.device,
    minimum_support_pixels: int,
    maximum_candidate_views: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    candidates = [
        (frame, camera)
        for frame in valid_frames
        if frame not in HELDOUT_FRAMES
        for camera in range(dataset.pixel_source.num_cams)
    ]
    for frame, camera in candidates[:maximum_candidate_views]:
        original = render_variant(
            trainer=trainer,
            dataset=dataset,
            checkpoint=checkpoint,
            frame=frame,
            camera=camera,
            model_index=model_index,
            variant="original",
            device=device,
        )
        deleted = render_variant(
            trainer=trainer,
            dataset=dataset,
            checkpoint=checkpoint,
            frame=frame,
            camera=camera,
            model_index=model_index,
            variant="delete",
            device=device,
        )
        lateral = render_variant(
            trainer=trainer,
            dataset=dataset,
            checkpoint=checkpoint,
            frame=frame,
            camera=camera,
            model_index=model_index,
            variant="lateral",
            device=device,
        )
        source = counterfactual_effect_mask(
            original["rgb"], deleted["rgb"], threshold_uint8=2, dilation_radius=2
        )
        edited = counterfactual_effect_mask(
            lateral["rgb"], deleted["rgb"], threshold_uint8=2, dilation_radius=2
        )
        support_counts = {}
        for edit in EDITS:
            edit_footprint = edited if edit == "lateral" else np.zeros_like(source)
            affected = affected_pixel_mask(
                source,
                edit_footprint,
                dilation_radius=3,
            )
            support = measured_background_support_mask(
                affected_mask=affected,
                source_actor_footprint=source,
                edited_actor_footprint=edit_footprint,
                depth_lidar_measured=original["depth_lidar_measured"],
            )
            support_counts[edit] = int(support.sum())
        attempt = {
            "frame": frame,
            "camera": camera,
            "camera_name": CAMERA_NAMES[camera],
            "source_footprint_pixels": int(source.sum()),
            "edited_footprint_pixels": int(edited.sum()),
            "s_b_t0_pixels": support_counts,
        }
        attempts.append(attempt)
        if (
            source.any()
            and edited.any()
            and all(
                support_counts[edit] >= minimum_support_pixels
                for edit in EDITS
            )
        ):
            return {
                **attempt,
                "source_footprint": source,
                "edited_footprint": edited,
                "original_rgb": original["rgb"],
                "delete_rgb": deleted["rgb"],
                "lateral_rgb": lateral["rgb"],
                "depth_lidar_measured": original["depth_lidar_measured"],
            }, attempts
    raise RuntimeError(
        "A3 could not find a heldout-safe view with S-B/T0 support: "
        f"attempted={len(attempts)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--drivestudio-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/third_party/"
            "drivestudio-worldsim-v3-a3-r1-r1"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-support-pixels", type=int, default=1)
    parser.add_argument("--maximum-candidate-views", type=int, default=60)
    parser.add_argument("--first-hit-alpha-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.minimum_support_pixels <= 0 or args.maximum_candidate_views <= 0:
        raise ValueError("A3 sidecar search limits must be positive")

    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_a3_protocol(protocol)
    protocol_sha256 = sha256_file(args.protocol)
    if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("A3 protocol file SHA drift")
    checkpoint_sha256 = sha256_file(args.checkpoint)
    config_sha256 = sha256_file(args.source_config)
    registry_sha256 = sha256_file(args.registry)
    dependencies = protocol["depends_on"]
    checks = {
        "protocol": protocol_sha256,
        "checkpoint": checkpoint_sha256,
        "config": config_sha256,
        "registry": registry_sha256,
    }
    expected = {
        "protocol": protocol_sha256,
        "checkpoint": dependencies["selected_checkpoint_sha256"],
        "config": dependencies["selected_checkpoint_config_sha256"],
        "registry": dependencies["selected_actor_registry_sha256"],
    }
    if checks != expected:
        raise RuntimeError(f"A3 immutable input drift: {checks!r}")

    args.output_dir.mkdir(parents=True)
    units_dir = args.output_dir / "units"
    qa_dir = args.output_dir / "qa"
    units_dir.mkdir()
    qa_dir.mkdir()
    sys.path.insert(0, str(args.drivestudio_root))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    config = OmegaConf.load(args.source_config)
    device = torch.device(args.device)
    dataset = DrivingDataset(data_cfg=config.data)
    trainer = import_str(config.trainer.type)(
        **config.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=config.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(str(args.checkpoint), load_only_model=True)
    trainer.set_eval()
    background_point_count = int(trainer.models["Background"]._means.shape[0])
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    row_observations: list[tuple[torch.Tensor, torch.Tensor]] = []
    unit_records: list[dict[str, Any]] = []
    selection_records: dict[str, Any] = {}

    for role in protocol["paired_design"]["actor_roles"]:
        actor_spec = protocol["paired_design"]["actors"][role]
        actor = require_token(registry, actor_spec["instance_token"])
        model_index = int(actor["rigid_model_index"])
        if model_index != int(actor_spec["rigid_model_index"]):
            raise RuntimeError(f"A3 actor registry drift: {role}")
        trainer.resume_from_checkpoint(str(args.checkpoint), load_only_model=True)
        valid_frames = (
            torch.where(
                trainer.models["RigidNodes"].instances_fv[:, model_index].bool()
            )[0]
            .cpu()
            .tolist()
        )
        selected, attempts = select_evidence_view(
            trainer=trainer,
            dataset=dataset,
            checkpoint=args.checkpoint,
            model_index=model_index,
            valid_frames=valid_frames,
            device=device,
            minimum_support_pixels=args.minimum_support_pixels,
            maximum_candidate_views=args.maximum_candidate_views,
        )
        selection_records[role] = {
            "selection_rule": "first_lexicographic_nonheldout_view_with_both_edit_footprints_and_T0_support",
            "selected": {
                key: value
                for key, value in selected.items()
                if not isinstance(value, np.ndarray)
            },
            "attempts": attempts,
        }
        frame, camera = int(selected["frame"]), int(selected["camera"])
        if frame in HELDOUT_FRAMES:
            raise RuntimeError("A3 held-out frame leaked into sidecar")
        for edit in EDITS:
            variant = edit
            rendered = render_variant(
                trainer=trainer,
                dataset=dataset,
                checkpoint=args.checkpoint,
                frame=frame,
                camera=camera,
                model_index=model_index,
                variant=variant,
                device=device,
                first_hit_alpha_threshold=args.first_hit_alpha_threshold,
            )
            source = selected["source_footprint"]
            edited = (
                selected["edited_footprint"]
                if edit == "lateral"
                else np.zeros_like(source)
            )
            affected = affected_pixel_mask(source, edited, dilation_radius=3)
            measured = lidar_measured_depth(rendered["depth_lidar_measured"])
            geometry = measured_background_support_mask(
                affected_mask=affected,
                source_actor_footprint=source,
                edited_actor_footprint=edited,
                depth_lidar_measured=measured.value,
            )
            affected_rows = projected_background_rows(
                means2d=rendered["means2d"],
                radii=rendered["radii"],
                pixel_mask=affected,
                background_point_count=background_point_count,
            ).cpu()
            supported_rows = projected_background_rows(
                means2d=rendered["means2d"],
                radii=rendered["radii"],
                pixel_mask=geometry,
                background_point_count=background_point_count,
            ).cpu()
            row_observations.append((affected_rows, supported_rows))

            unit_name = f"{role}__{edit}__frame_{frame:03d}_camera_{camera}"
            unit_path = units_dir / f"{unit_name}.npz"
            np.savez_compressed(
                unit_path,
                source_actor_footprint=source.astype(np.bool_),
                edited_actor_footprint=edited.astype(np.bool_),
                affected_pixel_mask=affected.astype(np.bool_),
                rgb_loss_mask=np.zeros_like(affected, dtype=np.bool_),
                geometry_loss_mask=geometry.astype(np.bool_),
                depth_render_expected=rendered["depth_render_expected"].value,
                depth_render_expected_valid=rendered["depth_render_expected"].valid,
                depth_surface_first_hit=rendered["depth_surface_first_hit"],
                depth_surface_first_hit_valid=rendered[
                    "depth_surface_first_hit_valid"
                ],
                depth_lidar_measured=measured.value,
                depth_lidar_measured_valid=measured.valid,
            )
            qa_prefix = qa_dir / unit_name
            imageio.imwrite(str(qa_prefix) + "__original.png", selected["original_rgb"])
            imageio.imwrite(str(qa_prefix) + "__edited.png", rendered["rgb"])
            imageio.imwrite(
                str(qa_prefix) + "__affected.png", affected.astype(np.uint8) * 255
            )
            imageio.imwrite(
                str(qa_prefix) + "__s_b_t0.png", geometry.astype(np.uint8) * 255
            )
            unit_records.append(
                {
                    "role": role,
                    "instance_token": actor_spec["instance_token"],
                    "rigid_model_index": model_index,
                    "edit": edit,
                    "frame": frame,
                    "camera": camera,
                    "camera_name": CAMERA_NAMES[camera],
                    "image_index": rendered["image_index"],
                    "heldout": False,
                    "path": str(unit_path.relative_to(args.output_dir)),
                    "sha256": sha256_file(unit_path),
                    "source_footprint_pixels": int(source.sum()),
                    "edited_footprint_pixels": int(edited.sum()),
                    "affected_pixels": int(affected.sum()),
                    "s_a_rgb_pixels": 0,
                    "s_b_t0_geometry_pixels": int(geometry.sum()),
                    "affected_background_rows": int(affected_rows.sum()),
                    "s_b_background_rows": int(supported_rows.sum()),
                }
            )

    affected_rows, strata_codes = merge_s_b_row_observations(
        row_observations,
        background_point_count=background_point_count,
    )
    arrays_path = args.output_dir / "rows.npz"
    np.savez_compressed(
        arrays_path,
        affected_background_rows=affected_rows.numpy().astype(np.bool_),
        support_strata_codes=strata_codes.numpy().astype(np.uint8),
    )
    atomic_json(args.output_dir / "selection.json", selection_records)
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "audit_version": SIDECAR_AUDIT_VERSION,
        "variant": "r1-reactivate",
        "formal_training_authorized": False,
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_config_sha256": config_sha256,
        "actor_registry_sha256": registry_sha256,
        "background_point_count": background_point_count,
        "arrays": {
            "path": arrays_path.name,
            "sha256": sha256_file(arrays_path),
            "affected_rows_key": "affected_background_rows",
            "support_strata_key": "support_strata_codes",
        },
        "evidence": {
            "support_provenance_complete": True,
            "support_mode": "conservative_S_B_T0_only",
            "s_a_status": "ABSTAIN_NOT_MATERIALIZED_IN_FIRST_PAIRED_GATE",
            "heldout_frames": HELDOUT_FRAMES,
            "heldout_excluded_from_support": True,
            "typed_depth_truth_tiers": {
                "depth_render_expected": "diagnostic",
                "depth_surface_first_hit": "T1",
                "depth_lidar_measured": "T0",
            },
            "first_hit_alpha_threshold": args.first_hit_alpha_threshold,
            "first_hit_alpha_status": "engineering_probe_not_frozen",
            "units": unit_records,
        },
        "counts": {
            "affected_background_rows": int(affected_rows.sum()),
            "s_a_background_rows": 0,
            "s_b_background_rows": int(((strata_codes == 1) & affected_rows).sum()),
            "s_c_affected_background_rows": int(
                ((strata_codes == 2) & affected_rows).sum()
            ),
        },
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
