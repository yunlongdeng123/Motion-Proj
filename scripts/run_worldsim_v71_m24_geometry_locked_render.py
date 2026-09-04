"""Render a frozen geometry-locked appearance carrier in StreetGS."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn import Parameter
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.dynamic_editing_v2.pilot_metrics import counterfactual_effect_mask


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device)
    return value


def _uint8_rgb(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise RuntimeError("render returned invalid RGB")
    return np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _load_runtime(config: Mapping[str, Any], device: torch.device) -> tuple[Any, Any]:
    checkout = Path(config["drivestudio_checkout"])
    sys.path.insert(0, str(checkout))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    source = OmegaConf.load(config["streetgs_config"])
    source.data.preload_device = "cpu"
    dataset = DrivingDataset(data_cfg=source.data)
    trainer = import_str(source.trainer.type)(
        **source.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=source.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    _restore_checkpoint(trainer, Path(config["streetgs_checkpoint"]), device)
    return dataset, trainer


def _restore_checkpoint(trainer: Any, checkpoint: Path, device: torch.device) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "RigidNodes" in state.get("models", {}):
        state["models"]["RigidNodes"] = _to_device(
            state["models"]["RigidNodes"], device
        )
    trainer.load_state_dict(state, load_only_model=True, strict=True)
    trainer.set_eval()
    del state


def _load_actor_carrier(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    sidecar = Path(config["m23_run"]) / "GEOMETRY_LOCKED_APPEARANCE_SIDECAR.npz"
    with np.load(sidecar, allow_pickle=False) as arrays:
        actor_tokens = arrays["actor_tokens"].astype(str).tolist()
        actor_position = actor_tokens.index(str(config["actor_token"]))
        offsets = arrays["offsets"]
        start, end = int(offsets[actor_position]), int(offsets[actor_position + 1])
        return {
            "centers": arrays["centers"][start:end].copy(),
            "scales": arrays["scales"][start:end].copy(),
            "features_dc": arrays["features_dc"][start:end].copy(),
            "features_rest": arrays["features_rest"][start:end].copy(),
            "opacity_logits": arrays["opacity_logits"][start:end].copy(),
        }


def _replace_actor(
    rigid: Any, actor_index: int, carrier: Mapping[str, np.ndarray]
) -> tuple[int, int]:
    actor_index = int(actor_index)
    keep = rigid.point_ids[..., 0] != actor_index
    removed = int((~keep).sum())
    if removed == 0:
        raise RuntimeError("target Actor has no original StreetGS Gaussians")
    device, dtype = rigid._means.device, rigid._means.dtype
    centers = torch.as_tensor(carrier["centers"], dtype=dtype, device=device)
    if "scales_xyz" in carrier:
        scales = torch.as_tensor(carrier["scales_xyz"], dtype=dtype, device=device)
    else:
        scalar_scales = torch.as_tensor(
            carrier["scales"], dtype=dtype, device=device
        )
        scales = scalar_scales[:, None].expand(-1, 3)
    if "quaternions" in carrier:
        quaternions = torch.as_tensor(
            carrier["quaternions"], dtype=dtype, device=device
        )
    else:
        quaternions = torch.zeros((len(centers), 4), dtype=dtype, device=device)
        quaternions[:, 0] = 1.0
    features_dc = torch.as_tensor(
        carrier["features_dc"], dtype=rigid._features_dc.dtype, device=device
    )
    features_rest = torch.as_tensor(
        carrier["features_rest"], dtype=rigid._features_rest.dtype, device=device
    )
    opacities = torch.as_tensor(
        carrier["opacity_logits"], dtype=rigid._opacities.dtype, device=device
    )
    point_ids = torch.full(
        (len(centers), 1),
        actor_index,
        dtype=rigid.point_ids.dtype,
        device=rigid.point_ids.device,
    )
    rigid._means = Parameter(torch.cat([rigid._means[keep], centers], dim=0))
    rigid._scales = Parameter(
        torch.cat([rigid._scales[keep], torch.log(scales)], dim=0)
    )
    rigid._quats = Parameter(torch.cat([rigid._quats[keep], quaternions], dim=0))
    rigid._features_dc = Parameter(
        torch.cat([rigid._features_dc[keep], features_dc], dim=0)
    )
    rigid._features_rest = Parameter(
        torch.cat([rigid._features_rest[keep], features_rest], dim=0)
    )
    rigid._opacities = Parameter(
        torch.cat([rigid._opacities[keep], opacities], dim=0)
    )
    rigid.point_ids = torch.cat([rigid.point_ids[keep], point_ids], dim=0)
    return removed, len(centers)


def _hide_actor(rigid: Any, actor_index: int) -> int:
    mask = rigid.point_ids[..., 0] == int(actor_index)
    count = int(mask.sum())
    if count == 0:
        raise RuntimeError("target Actor has no Gaussians to hide")
    with torch.no_grad():
        rigid._opacities[mask] = -100.0
    return count


def _render_variant(
    trainer: Any,
    dataset: Any,
    config: Mapping[str, Any],
    device: torch.device,
    variant: str,
    carrier: Mapping[str, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, int]]:
    _restore_checkpoint(trainer, Path(config["streetgs_checkpoint"]), device)
    rigid = trainer.models["RigidNodes"]
    actor_index = int(config["rigid_model_index"])
    counts = {"removed": 0, "inserted": 0, "hidden": 0}
    if variant == "geometry_locked":
        counts["removed"], counts["inserted"] = _replace_actor(
            rigid, actor_index, carrier
        )
    elif variant == "actor_hidden":
        counts["hidden"] = _hide_actor(rigid, actor_index)
    elif variant != "original":
        raise ValueError(f"unknown render variant: {variant}")

    images: dict[int, np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    for camera in config["cameras"]:
        camera = int(camera)
        image_index = int(config["frame"]) * dataset.pixel_source.num_cams + camera
        image_infos, camera_infos = dataset.full_image_set.get_image(
            image_index, camera_downscale=1.0
        )
        targets[camera] = _uint8_rgb(image_infos["pixels"])
        image_infos = _to_device(image_infos, device)
        camera_infos = _to_device(camera_infos, device)
        with torch.inference_mode():
            rendered = trainer(image_infos, camera_infos)["rgb"]
        images[camera] = _uint8_rgb(rendered)
        if hasattr(trainer, "info"):
            trainer.info = {}
    return images, targets, counts


def _mse_psnr(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> tuple[float | None, float | None, int, int]:
    valid = np.asarray(mask, dtype=np.bool_)
    if prediction.shape != target.shape or prediction.shape[:2] != valid.shape:
        raise ValueError("RGB and mask shapes do not align")
    count = int(valid.sum())
    if count == 0:
        return None, None, 0, 0
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    squared_sum = float(np.square(difference[valid]).sum())
    elements = count * 3
    mse = squared_sum / elements
    psnr = 20.0 * math.log10(255.0) - 10.0 * math.log10(mse) if mse > 0 else None
    return mse, psnr, count, elements


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "status.json", {"status": "running", "phase": "loading_runtime"}
    )
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M24 requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        carrier = _load_actor_carrier(config)
        dataset, trainer = _load_runtime(config, device)
        variants: dict[str, dict[int, np.ndarray]] = {}
        ground_truth: dict[int, np.ndarray] = {}
        variant_counts: dict[str, dict[str, int]] = {}
        for variant in ("original", "geometry_locked", "actor_hidden"):
            images, targets, counts = _render_variant(
                trainer, dataset, config, device, variant, carrier
            )
            variants[variant] = images
            ground_truth = targets
            variant_counts[variant] = counts

        rows: list[dict[str, Any]] = []
        total_original_squared = 0.0
        total_carrier_squared = 0.0
        total_elements = 0
        for camera in config["cameras"]:
            camera = int(camera)
            original = variants["original"][camera]
            changed = variants["geometry_locked"][camera]
            hidden = variants["actor_hidden"][camera]
            target = ground_truth[camera]
            footprint = counterfactual_effect_mask(
                original,
                hidden,
                threshold_uint8=int(config["evaluation"]["footprint_threshold_uint8"]),
                dilation_radius=int(config["evaluation"]["footprint_dilation_radius"]),
            )
            original_mse, original_psnr, pixels, elements = _mse_psnr(
                original, target, footprint
            )
            carrier_mse, carrier_psnr, _, _ = _mse_psnr(
                changed, target, footprint
            )
            full_mask = np.ones(original.shape[:2], dtype=np.bool_)
            _, original_full_psnr, _, _ = _mse_psnr(original, target, full_mask)
            _, carrier_full_psnr, _, _ = _mse_psnr(changed, target, full_mask)
            if original_mse is not None and carrier_mse is not None:
                total_original_squared += original_mse * elements
                total_carrier_squared += carrier_mse * elements
                total_elements += elements
            row = {
                "scene_name": config["scene_name"],
                "actor_token": config["actor_token"],
                "frame": int(config["frame"]),
                "camera": camera,
                "footprint_pixels": pixels,
                "original_actor_psnr_db": original_psnr,
                "geometry_locked_actor_psnr_db": carrier_psnr,
                "actor_psnr_delta_db": (
                    carrier_psnr - original_psnr
                    if carrier_psnr is not None and original_psnr is not None
                    else None
                ),
                "original_full_psnr_db": original_full_psnr,
                "geometry_locked_full_psnr_db": carrier_full_psnr,
                "full_psnr_delta_db": (
                    carrier_full_psnr - original_full_psnr
                    if carrier_full_psnr is not None and original_full_psnr is not None
                    else None
                ),
                "carrier_changed_pixels": int(
                    np.any(
                        np.abs(changed.astype(np.int16) - original.astype(np.int16))
                        > int(config["evaluation"]["footprint_threshold_uint8"]),
                        axis=2,
                    ).sum()
                ),
            }
            rows.append(row)
            imageio.imwrite(run_dir / f"camera_{camera}_target.png", target)
            imageio.imwrite(run_dir / f"camera_{camera}_original.png", original)
            imageio.imwrite(run_dir / f"camera_{camera}_geometry_locked.png", changed)
            imageio.imwrite(run_dir / f"camera_{camera}_actor_hidden.png", hidden)
            imageio.imwrite(
                run_dir / f"camera_{camera}_actor_footprint.png",
                footprint.astype(np.uint8) * 255,
            )

        visible_camera_count = sum(int(row["footprint_pixels"]) > 0 for row in rows)
        aggregate_original_psnr = (
            20.0 * math.log10(255.0)
            - 10.0 * math.log10(total_original_squared / total_elements)
            if total_elements and total_original_squared > 0
            else None
        )
        aggregate_carrier_psnr = (
            20.0 * math.log10(255.0)
            - 10.0 * math.log10(total_carrier_squared / total_elements)
            if total_elements and total_carrier_squared > 0
            else None
        )
        decisions = {
            "all_fixed_views_rendered": all(
                len(variants[name]) == len(config["cameras"])
                for name in ("original", "geometry_locked", "actor_hidden")
            ),
            "actor_visible_in_at_least_one_fixed_camera": visible_camera_count > 0,
        }
        passed = all(decisions.values())
        _write_jsonl(run_dir / "RENDER_ROWS.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.m24_geometry_locked_render.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": (
                "geometry_locked_carrier_rasterized"
                if passed
                else "geometry_locked_render_interface_rejected"
            ),
            "scene_name": config["scene_name"],
            "actor_token": config["actor_token"],
            "rigid_model_index": int(config["rigid_model_index"]),
            "hazardous": bool(config["hazardous"]),
            "frame": int(config["frame"]),
            "frame_selection": dict(config.get("frame_selection", {})),
            "cameras": [int(value) for value in config["cameras"]],
            "visible_camera_count": visible_camera_count,
            "physical_carrier_gaussian_count": int(len(carrier["centers"])),
            "variant_counts": variant_counts,
            "aggregate_actor_footprint_pixels": sum(
                int(row["footprint_pixels"]) for row in rows
            ),
            "aggregate_original_actor_psnr_db": aggregate_original_psnr,
            "aggregate_geometry_locked_actor_psnr_db": aggregate_carrier_psnr,
            "aggregate_actor_psnr_delta_db": (
                aggregate_carrier_psnr - aggregate_original_psnr
                if aggregate_carrier_psnr is not None
                and aggregate_original_psnr is not None
                else None
            ),
            "carrier_changed_pixels": sum(
                int(row["carrier_changed_pixels"]) for row in rows
            ),
            "rows": rows,
            "decisions": decisions,
            "geometry_source": "frozen_m8_m23_sidecar",
            "appearance_source": "frozen_same_actor_streetgs_sh_opacity",
            "geometry_gradient": False,
            "quality_selection": False,
            "checkpoint_written": False,
            "training": False,
            "physical_metric_read": False,
            "external_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device)
                / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "fixed_real_render",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": "fixed_real_render",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
