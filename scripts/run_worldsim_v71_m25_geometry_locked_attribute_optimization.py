"""Train only visual attributes on a frozen GT-supervised physical carrier."""

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
import torch
from torch.nn import functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_worldsim_v71_m24_geometry_locked_render as m24


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _view_key(view: Mapping[str, Any]) -> str:
    return f"f{int(view['frame']):03d}_c{int(view['camera'])}"


def _as_hwc(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == 3:
        return value.permute(1, 2, 0)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise RuntimeError(f"expected RGB image, got {tuple(value.shape)}")
    return value


def _load_views(dataset: Any, views: list[Mapping[str, Any]]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for view in views:
        frame, camera = int(view["frame"]), int(view["camera"])
        image_index = frame * dataset.pixel_source.num_cams + camera
        image_infos, camera_infos = dataset.full_image_set.get_image(
            image_index, camera_downscale=1.0
        )
        records[_view_key(view)] = {
            "frame": frame,
            "camera": camera,
            "image_infos": image_infos,
            "camera_infos": camera_infos,
            "target": _as_hwc(image_infos["pixels"]).detach().cpu(),
        }
    return records


def _render(
    trainer: Any,
    record: Mapping[str, Any],
    device: torch.device,
    require_grad: bool,
) -> torch.Tensor:
    image_infos = m24._to_device(record["image_infos"], device)
    camera_infos = m24._to_device(record["camera_infos"], device)
    if require_grad:
        rendered = trainer(image_infos, camera_infos)["rgb"]
    else:
        with torch.inference_mode():
            rendered = trainer(image_infos, camera_infos)["rgb"]
    rendered = _as_hwc(rendered)
    if hasattr(trainer, "info"):
        trainer.info = {}
    return rendered


def _render_records(
    trainer: Any,
    records: Mapping[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: _render(trainer, record, device, require_grad=False).detach().cpu()
        for key, record in records.items()
    }


def _make_footprints(
    original: Mapping[str, torch.Tensor],
    hidden: Mapping[str, torch.Tensor],
    threshold_uint8: int,
    dilation_radius: int,
) -> dict[str, np.ndarray]:
    footprints: dict[str, np.ndarray] = {}
    for key in original:
        footprints[key] = m24.counterfactual_effect_mask(
            m24._uint8_rgb(original[key]),
            m24._uint8_rgb(hidden[key]),
            threshold_uint8=threshold_uint8,
            dilation_radius=dilation_radius,
        )
    return footprints


def _masked_metric(
    prediction: torch.Tensor,
    target: torch.Tensor,
    footprint: np.ndarray,
) -> tuple[float | None, float | None, int, int]:
    return m24._mse_psnr(
        m24._uint8_rgb(prediction), m24._uint8_rgb(target), footprint
    )


def _pooled_psnr(
    images: Mapping[str, torch.Tensor],
    records: Mapping[str, Any],
    footprints: Mapping[str, np.ndarray],
) -> float | None:
    squared_sum, elements = 0.0, 0
    for key, prediction in images.items():
        mse, _, _, count = _masked_metric(
            prediction, records[key]["target"], footprints[key]
        )
        if mse is not None:
            squared_sum += mse * count
            elements += count
    if elements == 0:
        return None
    mse = squared_sum / elements
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse) if mse else None


def _metric_rows(
    split: str,
    views: list[Mapping[str, Any]],
    records: Mapping[str, Any],
    footprints: Mapping[str, np.ndarray],
    original: Mapping[str, torch.Tensor],
    initial: Mapping[str, torch.Tensor],
    final: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view in views:
        key = _view_key(view)
        target = records[key]["target"]
        mask = footprints[key]
        _, original_psnr, pixels, _ = _masked_metric(original[key], target, mask)
        _, initial_psnr, _, _ = _masked_metric(initial[key], target, mask)
        _, final_psnr, _, _ = _masked_metric(final[key], target, mask)
        full = np.ones(mask.shape, dtype=np.bool_)
        _, original_full, _, _ = _masked_metric(original[key], target, full)
        _, initial_full, _, _ = _masked_metric(initial[key], target, full)
        _, final_full, _, _ = _masked_metric(final[key], target, full)
        rows.append(
            {
                "split": split,
                "frame": int(view["frame"]),
                "camera": int(view["camera"]),
                "footprint_pixels": pixels,
                "original_actor_psnr_db": original_psnr,
                "initial_actor_psnr_db": initial_psnr,
                "final_actor_psnr_db": final_psnr,
                "final_minus_initial_actor_psnr_db": (
                    final_psnr - initial_psnr
                    if final_psnr is not None and initial_psnr is not None
                    else None
                ),
                "original_full_psnr_db": original_full,
                "initial_full_psnr_db": initial_full,
                "final_full_psnr_db": final_full,
            }
        )
    return rows


def _save_heldout_images(
    run_dir: Path,
    views: list[Mapping[str, Any]],
    records: Mapping[str, Any],
    footprints: Mapping[str, np.ndarray],
    original: Mapping[str, torch.Tensor],
    initial: Mapping[str, torch.Tensor],
    final: Mapping[str, torch.Tensor],
) -> None:
    for view in views:
        key = _view_key(view)
        imageio.imwrite(run_dir / f"{key}_target.png", m24._uint8_rgb(records[key]["target"]))
        imageio.imwrite(run_dir / f"{key}_original.png", m24._uint8_rgb(original[key]))
        imageio.imwrite(run_dir / f"{key}_initial.png", m24._uint8_rgb(initial[key]))
        imageio.imwrite(run_dir / f"{key}_final.png", m24._uint8_rgb(final[key]))
        imageio.imwrite(
            run_dir / f"{key}_footprint.png",
            footprints[key].astype(np.uint8) * 255,
        )


def run(
    config_path: Path,
    run_id: str,
    carrier_transform: Any = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M25 requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed = int(config["optimization"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.monotonic()

    try:
        carrier = m24._load_actor_carrier(config)
        carrier_metadata: dict[str, Any] = {}
        if carrier_transform is not None:
            carrier, carrier_metadata = carrier_transform(carrier, config)
        dataset, trainer = m24._load_runtime(config, device)
        train_views = list(config["train_views"])
        heldout_views = list(config["heldout_views"])
        all_views = train_views + heldout_views
        records = _load_views(dataset, all_views)

        original = _render_records(trainer, records, device)
        m24._hide_actor(trainer.models["RigidNodes"], int(config["rigid_model_index"]))
        hidden = _render_records(trainer, records, device)
        footprints = _make_footprints(
            original,
            hidden,
            int(config["evaluation"]["footprint_threshold_uint8"]),
            int(config["evaluation"]["footprint_dilation_radius"]),
        )

        m24._restore_checkpoint(trainer, Path(config["streetgs_checkpoint"]), device)
        rigid = trainer.models["RigidNodes"]
        removed, inserted = m24._replace_actor(
            rigid, int(config["rigid_model_index"]), carrier
        )
        # DriveStudio keeps its child modules in a plain dict, so they are not
        # reached by trainer.parameters().  Freeze that model tree explicitly
        # before reopening only the target actor's appearance tensors.
        for parameter in trainer.parameters():
            parameter.requires_grad_(False)
        for model in trainer.models.values():
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        for parameter in (rigid._features_dc, rigid._features_rest, rigid._opacities):
            parameter.requires_grad_(True)

        actor_mask = rigid.point_ids[..., 0] == int(config["rigid_model_index"])
        expected_render_count = int(
            config.get(
                "render_carrier_gaussian_count",
                config["physical_carrier_gaussian_count"],
            )
        )
        if int(actor_mask.sum()) != expected_render_count:
            raise RuntimeError("render carrier count changed before optimization")
        rigid._features_dc.register_hook(
            lambda gradient: gradient * actor_mask[:, None].to(gradient.dtype)
        )
        rigid._features_rest.register_hook(
            lambda gradient: gradient * actor_mask[:, None, None].to(gradient.dtype)
        )
        rigid._opacities.register_hook(
            lambda gradient: gradient * actor_mask[:, None].to(gradient.dtype)
        )
        initial_opacity = torch.sigmoid(
            rigid._opacities.detach()[actor_mask]
        ).clone()
        initial = _render_records(trainer, records, device)

        optim = config["optimization"]
        optimizer = torch.optim.Adam(
            [
                {"params": [rigid._features_dc], "lr": float(optim["sh_dc_lr"])},
                {"params": [rigid._features_rest], "lr": float(optim["sh_rest_lr"])},
                {"params": [rigid._opacities], "lr": float(optim["opacity_lr"])},
            ]
        )
        training_rows: list[dict[str, Any]] = []
        _write_json(
            run_dir / "status.json",
            {"status": "running", "phase": "attribute_optimization", "step": 0},
        )
        for step in range(int(optim["steps"])):
            view = train_views[step % len(train_views)]
            key = _view_key(view)
            record = records[key]
            target = record["target"].to(device)
            hidden_image = hidden[key].to(device)
            footprint = torch.from_numpy(footprints[key]).to(device=device)
            if not bool(footprint.any()):
                raise RuntimeError(f"zero training footprint at {key}")
            optimizer.zero_grad(set_to_none=True)
            prediction = _render(trainer, record, device, require_grad=True)
            actor_loss = F.smooth_l1_loss(
                prediction[footprint], target[footprint], beta=1.0 / 255.0
            )
            outside = ~footprint
            spill_loss = F.l1_loss(
                prediction[outside], hidden_image[outside]
            )
            opacity_anchor = F.l1_loss(
                torch.sigmoid(rigid._opacities[actor_mask]), initial_opacity
            )
            total = (
                actor_loss
                + float(optim["spill_weight"]) * spill_loss
                + float(optim["opacity_anchor_weight"]) * opacity_anchor
            )
            total.backward()
            optimizer.step()
            if (step + 1) % int(optim["log_every"]) == 0 or step == 0:
                row = {
                    "step": step + 1,
                    "view": key,
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "spill_loss": float(spill_loss.detach().cpu()),
                    "opacity_anchor_loss": float(opacity_anchor.detach().cpu()),
                    "total_loss": float(total.detach().cpu()),
                }
                training_rows.append(row)
                _write_json(
                    run_dir / "status.json",
                    {
                        "status": "running",
                        "phase": "attribute_optimization",
                        "step": step + 1,
                    },
                )

        final = _render_records(trainer, records, device)
        train_keys = {_view_key(view) for view in train_views}
        heldout_keys = {_view_key(view) for view in heldout_views}
        train_original = {key: original[key] for key in train_keys}
        train_initial = {key: initial[key] for key in train_keys}
        train_final = {key: final[key] for key in train_keys}
        train_records = {key: records[key] for key in train_keys}
        train_footprints = {key: footprints[key] for key in train_keys}
        heldout_original = {key: original[key] for key in heldout_keys}
        heldout_initial = {key: initial[key] for key in heldout_keys}
        heldout_final = {key: final[key] for key in heldout_keys}
        heldout_records = {key: records[key] for key in heldout_keys}
        heldout_footprints = {key: footprints[key] for key in heldout_keys}

        aggregate = {
            "train_original_actor_psnr_db": _pooled_psnr(
                train_original, train_records, train_footprints
            ),
            "train_initial_actor_psnr_db": _pooled_psnr(
                train_initial, train_records, train_footprints
            ),
            "train_final_actor_psnr_db": _pooled_psnr(
                train_final, train_records, train_footprints
            ),
            "heldout_original_actor_psnr_db": _pooled_psnr(
                heldout_original, heldout_records, heldout_footprints
            ),
            "heldout_initial_actor_psnr_db": _pooled_psnr(
                heldout_initial, heldout_records, heldout_footprints
            ),
            "heldout_final_actor_psnr_db": _pooled_psnr(
                heldout_final, heldout_records, heldout_footprints
            ),
        }
        initial_psnr = aggregate["heldout_initial_actor_psnr_db"]
        final_psnr = aggregate["heldout_final_actor_psnr_db"]
        original_psnr = aggregate["heldout_original_actor_psnr_db"]
        aggregate["heldout_final_minus_initial_psnr_db"] = (
            final_psnr - initial_psnr
            if final_psnr is not None and initial_psnr is not None
            else None
        )
        aggregate["heldout_original_minus_final_gap_db"] = (
            original_psnr - final_psnr
            if original_psnr is not None and final_psnr is not None
            else None
        )

        rows = _metric_rows(
            "train", train_views, records, footprints, original, initial, final
        ) + _metric_rows(
            "heldout", heldout_views, records, footprints, original, initial, final
        )
        _write_jsonl(run_dir / "TRAINING_ROWS.jsonl", training_rows)
        _write_jsonl(run_dir / "VIEW_ROWS.jsonl", rows)
        _save_heldout_images(
            run_dir,
            heldout_views,
            records,
            footprints,
            original,
            initial,
            final,
        )
        np.savez_compressed(
            run_dir / "OPTIMIZED_APPEARANCE_SIDECAR.npz",
            centers=rigid._means.detach()[actor_mask].cpu().numpy(),
            scales_xyz=torch.exp(rigid._scales.detach()[actor_mask]).cpu().numpy(),
            quaternions=rigid._quats.detach()[actor_mask].cpu().numpy(),
            features_dc=rigid._features_dc.detach()[actor_mask].cpu().numpy(),
            features_rest=rigid._features_rest.detach()[actor_mask].cpu().numpy(),
            opacity_logits=rigid._opacities.detach()[actor_mask].cpu().numpy(),
        )

        heldout_nonzero = sum(
            int(footprints[_view_key(view)].sum()) > 0 for view in heldout_views
        )
        decisions = {
            "all_heldout_views_have_actor_footprint": heldout_nonzero
            == len(heldout_views),
            "heldout_actor_psnr_improved": (
                final_psnr is not None
                and initial_psnr is not None
                and final_psnr > initial_psnr
            ),
            "physical_geometry_excluded_from_optimizer": not any(
                parameter.requires_grad
                for parameter in (rigid._means, rigid._scales, rigid._quats)
            ),
        }
        reference_psnr = config.get("decision", {}).get(
            "heldout_actor_psnr_reference_db"
        )
        if reference_psnr is not None:
            decisions["heldout_actor_psnr_exceeds_frozen_reference"] = (
                final_psnr is not None and final_psnr > float(reference_psnr)
            )
        passed = all(decisions.values())
        summary = {
            "schema_version": config.get(
                "summary_schema_version",
                "worldsim_v71.m25_geometry_locked_attribute_optimization.v1",
            ),
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": (
                config.get(
                    "success_verdict", "appearance_only_training_signal_supported"
                )
                if passed
                else config.get(
                    "failure_verdict", "appearance_only_capacity_rejected"
                )
            ),
            "scene_name": config["scene_name"],
            "actor_token": config["actor_token"],
            "rigid_model_index": int(config["rigid_model_index"]),
            "hazardous": bool(config["hazardous"]),
            "view_selection": dict(config["view_selection"]),
            "train_views": train_views,
            "heldout_views": heldout_views,
            "train_view_count": len(train_views),
            "heldout_view_count": len(heldout_views),
            "heldout_nonzero_footprint_count": heldout_nonzero,
            "original_visual_gaussian_count": removed,
            "physical_carrier_gaussian_count": int(
                config["physical_carrier_gaussian_count"]
            ),
            "render_carrier_gaussian_count": inserted,
            "effective_trainable_attribute_count": int(actor_mask.sum())
            * (3 + int(rigid._features_rest.shape[1]) * 3 + 1),
            "optimization": dict(optim),
            "aggregate": aggregate,
            "rows": rows,
            "training_rows": training_rows,
            "decisions": decisions,
            "geometry_gradient": False,
            "geometry_optimizer_parameters": 0,
            "other_actor_attribute_gradient_masked": True,
            "trajectory_gradient": False,
            "background_gradient": False,
            "quality_selection": False,
            "checkpoint_written": False,
            "attribute_sidecar_written": True,
            "physical_metric_read": False,
            "external_read": False,
            "carrier_metadata": carrier_metadata,
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
                "phase": "heldout_evaluation",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": "geometry_locked_attribute_optimization",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_id), ensure_ascii=False))


if __name__ == "__main__":
    main()
