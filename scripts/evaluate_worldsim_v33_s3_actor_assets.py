#!/usr/bin/env python3
"""在冻结 development/heldout 视图比较 manual 与 auto 1/2/4-view actor assets。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import imageio.v2 as imageio
import lpips
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.actor_asset_schema import validate_actor_asset
from scripts.evaluate_worldsim_v32_s3_actor_asset import (
    boundary_f1,
    dilate,
    make_panel,
    masked_lpips_inputs,
)
from scripts.lift_worldsim_v32_semantics import build_runtime
from scripts.materialize_worldsim_v3_a3_s_b_sidecar import (
    render_variant as render_native_variant,
)
from scripts.render_worldsim_v32_s3_actor_asset import render_variant


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_file(path: str | Path, expected: str, role: str) -> Path:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{role} 不存在: {source}")
    actual = sha256_file(source)
    if actual != expected:
        raise RuntimeError(f"{role} SHA 漂移: {actual} != {expected}")
    return source


def parse_assets(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"--asset 必须为 ARM=MANIFEST: {value}")
        if name in output:
            raise ValueError(f"重复 asset arm: {name}")
        output[name] = Path(path)
    return output


def find_mask_row(
    manifest: Mapping[str, Any], role: str, frame: int, camera: int
) -> dict[str, Any]:
    rows = [
        row
        for row in manifest["masks"]
        if row["role"] == role
        and int(row["frame"]) == int(frame)
        and int(row["camera_id"]) == int(camera)
        and bool(row["accepted"])
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"S3 eval mask 必须唯一: {role} f{frame:03d} c{camera}, matches={len(rows)}"
        )
    return dict(rows[0])


def largest_component_fraction(mask: np.ndarray) -> tuple[int, float]:
    if not mask.any():
        return 0, 1.0
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.max()) if sizes.size else 0
    return int(count), float(largest / max(1, int(mask.sum())))


def snapshot_sources(run_dir: Path, config_path: Path) -> dict[str, Any]:
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir()
    sources = {
        "config": config_path,
        "evaluator": Path(__file__).resolve(),
        "v32_evaluator": PROJECT / "scripts" / "evaluate_worldsim_v32_s3_actor_asset.py",
        "native_renderer": PROJECT
        / "scripts"
        / "materialize_worldsim_v3_a3_s_b_sidecar.py",
        "v32_renderer": PROJECT / "scripts" / "render_worldsim_v32_s3_actor_asset.py",
        "adapter": PROJECT / "motion_proj" / "worldsim_v32" / "asset_harvester_adapter.py",
        "schema": PROJECT / "motion_proj" / "worldsim_v32" / "actor_asset_schema.py",
    }
    report = {}
    for role, source in sources.items():
        target = snapshot / source.name
        shutil.copy2(source, target)
        report[role] = {
            "path": str(target),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
    return report


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    effect_ratios = [
        row["generated_effect_pixels"] / max(1, row["sam_pixels"]) for row in rows
    ]
    mean_ratio = float(np.mean(effect_ratios))
    consistency = float(
        np.clip(1.0 - np.std(effect_ratios) / max(mean_ratio, 1e-12), 0.0, 1.0)
    )
    return {
        "view_count_evaluated": len(rows),
        "mean_silhouette_iou": float(np.mean([row["silhouette_iou"] for row in rows])),
        "mean_boundary_f1_tolerance_3px": float(
            np.mean([row["boundary_f1_tolerance_3px"] for row in rows])
        ),
        "mean_masked_rgb_psnr": float(np.mean([row["masked_rgb_psnr"] for row in rows])),
        "mean_masked_crop_lpips_alex": float(
            np.mean([row["masked_crop_lpips_alex"] for row in rows])
        ),
        "max_non_target_original_delete_l1_uint8": float(
            max(row["non_target_original_delete_l1_uint8"] for row in rows)
        ),
        "max_lateral_fragmentation": float(max(row["lateral_fragmentation"] for row in rows)),
        "mean_lateral_fragmentation": float(
            np.mean([row["lateral_fragmentation"] for row in rows])
        ),
        "cross_view_effect_ratio_consistency": consistency,
    }


def development_decision(
    aggregate: Mapping[str, Mapping[str, float]], config: Mapping[str, Any]
) -> dict[str, Any]:
    selection = config["selection"]
    baseline_arm = selection.get("baseline_arm", "A0_manual_2view")
    baseline = aggregate[baseline_arm]
    gate = (
        selection["acceptance_vs_baseline"]
        if "acceptance_vs_baseline" in selection
        else selection["acceptance_vs_manual_a0"]
    )
    accepted = {}
    for arm in config["selection"]["auto_arms"]:
        row = aggregate[arm]
        gates = {
            "iou": row["mean_silhouette_iou"]
            >= baseline["mean_silhouette_iou"] - float(gate["maximum_iou_degradation"]),
            "boundary_f1": row["mean_boundary_f1_tolerance_3px"]
            >= baseline["mean_boundary_f1_tolerance_3px"]
            - float(gate["maximum_boundary_f1_degradation"]),
            "psnr": row["mean_masked_rgb_psnr"]
            >= baseline["mean_masked_rgb_psnr"] - float(gate["maximum_psnr_degradation_db"]),
            "lpips": row["mean_masked_crop_lpips_alex"]
            <= baseline["mean_masked_crop_lpips_alex"] + float(gate["maximum_lpips_increase"]),
            "outside_l1": row["max_non_target_original_delete_l1_uint8"]
            <= float(gate["maximum_outside_l1_uint8"]),
            "lateral_fragmentation": row["max_lateral_fragmentation"]
            <= float(gate["maximum_lateral_fragmentation"]),
        }
        accepted[arm] = {"gates": gates, "accepted": bool(all(gates.values()))}
    eligible = [arm for arm, row in accepted.items() if row["accepted"]]
    if eligible:
        selected = sorted(
            eligible,
            key=lambda arm: (
                -aggregate[arm]["mean_silhouette_iou"],
                aggregate[arm]["mean_masked_crop_lpips_alex"],
                -aggregate[arm]["mean_masked_rgb_psnr"],
                -aggregate[arm]["mean_boundary_f1_tolerance_3px"],
                arm,
            ),
        )[0]
        reason = "best_accepted_auto_arm_by_frozen_metric_order"
    else:
        selected = baseline_arm
        reason = "all_auto_arms_failed_development_retention_gates"
    return {
        "protocol": "development_only_arm_selection_then_heldout_confirmation",
        "metric_order": config["selection"]["development_metric_order"],
        "arms": accepted,
        "selected_arm": selected,
        "baseline_arm": baseline_arm,
        "reason": reason,
        "heldout_read": False,
    }


def heldout_confirmation(
    aggregate: Mapping[str, Mapping[str, float]],
    selected_arm: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_arm = config["selection"].get("baseline_arm", "A0_manual_2view")
    baseline = aggregate[baseline_arm]
    selected = aggregate[selected_arm]
    gate = config["selection"]["heldout_confirmation"]
    gates = {
        "iou": selected["mean_silhouette_iou"]
        >= baseline["mean_silhouette_iou"] - float(gate["maximum_iou_degradation"]),
        "boundary_f1": selected["mean_boundary_f1_tolerance_3px"]
        >= baseline["mean_boundary_f1_tolerance_3px"]
        - float(gate["maximum_boundary_f1_degradation"]),
        "psnr": selected["mean_masked_rgb_psnr"]
        >= baseline["mean_masked_rgb_psnr"] - float(gate["maximum_psnr_degradation_db"]),
        "lpips": selected["mean_masked_crop_lpips_alex"]
        <= baseline["mean_masked_crop_lpips_alex"] + float(gate["maximum_lpips_increase"]),
    }
    return {
        "protocol": "frozen_development_winner_single_heldout_confirmation",
        "selected_arm": selected_arm,
        "baseline_arm": baseline_arm,
        "gates": gates,
        "accepted": bool(all(gates.values())),
        "deltas": {
            "iou": selected["mean_silhouette_iou"] - baseline["mean_silhouette_iou"],
            "boundary_f1": selected["mean_boundary_f1_tolerance_3px"]
            - baseline["mean_boundary_f1_tolerance_3px"],
            "psnr_db": selected["mean_masked_rgb_psnr"] - baseline["mean_masked_rgb_psnr"],
            "lpips": selected["mean_masked_crop_lpips_alex"]
            - baseline["mean_masked_crop_lpips_alex"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", choices=("development", "heldout"), required=True)
    parser.add_argument("--asset", action="append", default=[], metavar="ARM=MANIFEST")
    parser.add_argument("--inference-manifest", type=Path)
    parser.add_argument("--inference-manifest-sha256")
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--decision-sha256")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"S3 eval run-dir 非空: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = args.run_dir / "artifacts"
    render_root = artifacts / "renders"
    panel_root = artifacts / "panels"
    render_root.mkdir(parents=True)
    panel_root.mkdir()
    started = time.time()
    atomic_json(args.run_dir / "status.json", {"state": "running", "started_unix": started})

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "worldsim_v33_s3_evaluation_v1":
        raise ValueError("S3 evaluation config schema 漂移")
    inputs = config["inputs"]
    checkpoint = verify_file(inputs["checkpoint"], inputs["checkpoint_sha256"], "checkpoint")
    verify_file(inputs["source_config"], inputs["source_config_sha256"], "source_config")
    if args.phase == "development":
        manifest_path = verify_file(
            inputs["train_mask_manifest"],
            inputs["train_mask_manifest_sha256"],
            "train_mask_manifest",
        )
        views = config["development_views"]
        heldout_read = False
    else:
        manifest_path = verify_file(
            inputs["heldout_mask_manifest"],
            inputs["heldout_mask_manifest_sha256"],
            "heldout_mask_manifest",
        )
        views = config["heldout_views"]
        heldout_read = True
        if not args.decision or not args.decision_sha256:
            raise ValueError("heldout phase 必须冻结 development decision")
        if sha256_file(args.decision) != args.decision_sha256:
            raise RuntimeError("S3 development decision SHA 漂移")
    mask_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actor = config["actor"]
    assets = parse_assets(args.asset)
    selection = config["selection"]
    baseline_arm = selection.get("baseline_arm", "A0_manual_2view")
    native_baseline_arm = selection.get("native_baseline_arm")
    if native_baseline_arm is not None and native_baseline_arm != baseline_arm:
        raise RuntimeError("S3 native baseline arm 必须等于 baseline_arm")
    required_arms = (
        set(selection["arms"])
        if args.phase == "development"
        else {
            baseline_arm,
            json.loads(args.decision.read_text(encoding="utf-8"))["selected_arm"],
        }
    )
    required_asset_arms = required_arms - ({native_baseline_arm} if native_baseline_arm else set())
    if set(assets) != required_asset_arms:
        raise RuntimeError(
            f"S3 {args.phase} asset arms 漂移: "
            f"{sorted(assets)} != {sorted(required_asset_arms)}"
        )
    if "A0_manual_2view" in assets:
        manual = inputs["manual_a0"]
        if assets["A0_manual_2view"].resolve() != Path(manual["manifest"]).resolve():
            raise RuntimeError("A0 manual manifest 路径漂移")
        verify_file(manual["manifest"], manual["manifest_sha256"], "manual manifest")
        verify_file(manual["asset"], manual["asset_sha256"], "manual asset")
        source_summary_path = verify_file(
            manual["source_summary"],
            manual["source_summary_sha256"],
            "manual source summary",
        )
        manual_source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    inference_resources = None
    if args.inference_manifest:
        if not args.inference_manifest_sha256 or sha256_file(
            args.inference_manifest
        ) != args.inference_manifest_sha256:
            raise RuntimeError("S3 inference manifest SHA 漂移")
        inference_resources = json.loads(
            args.inference_manifest.read_text(encoding="utf-8")
        )["runtime"]

    if not torch.cuda.is_available():
        raise RuntimeError("S3 actor evaluation 需要 CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    runtime_config = {
        "inputs": {"checkpoint": str(checkpoint), "source_config": inputs["source_config"]},
        "runtimes": {"drivestudio_checkout": config["runtimes"]["drivestudio_checkout"]},
    }
    dataset, trainer = build_runtime(runtime_config, device)
    perceptual = lpips.LPIPS(net="alex").eval().to(device)
    checkpoint_before = sha256_file(checkpoint)
    rows = []
    panels = []
    asset_summaries = {}
    evaluation_assets: dict[str, Path | None] = dict(assets)
    if native_baseline_arm:
        evaluation_assets[native_baseline_arm] = None
    for arm in sorted(evaluation_assets):
        manifest_file = evaluation_assets[arm]
        is_native = arm == native_baseline_arm
        asset = None
        if is_native:
            if float(config["render"]["lateral_offset_m"]) != 1.0:
                raise RuntimeError("native renderer 当前只冻结 lateral_offset_m=1.0")
            asset_summaries[arm] = {
                "kind": "immutable_native_checkpoint_actor",
                "checkpoint_sha256": inputs["checkpoint_sha256"],
                "rigid_model_index": int(actor["rigid_model_index"]),
            }
        else:
            assert manifest_file is not None
            asset_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if asset_manifest["instance_token"] != actor["instance_token"]:
                raise RuntimeError(f"S3 {arm} actor token 错配")
            asset_path = Path(asset_manifest["asset"]["path"])
            if sha256_file(asset_path) != asset_manifest["asset"]["sha256"]:
                raise RuntimeError(f"S3 {arm} asset SHA 漂移")
            with np.load(asset_path, allow_pickle=False) as payload:
                asset = {name: payload[name].copy() for name in payload.files}
            validate_actor_asset(asset)
            asset_summaries[arm] = {
                "kind": "generated_actor_asset",
                "manifest": str(manifest_file),
                "manifest_sha256": sha256_file(manifest_file),
                "asset_bytes": asset_path.stat().st_size,
                "asset_sha256": sha256_file(asset_path),
                "gaussian_count": int(len(asset["means"])),
            }
            if arm == "A0_manual_2view":
                asset_summaries[arm]["historical_inference_resources"] = (
                    manual_source_summary["inference"]["runtime"]
                )
            elif inference_resources is not None:
                asset_summaries[arm]["shared_auto_inference_resources"] = (
                    inference_resources
                )
        for frame_value, camera_value in views:
            frame, camera = int(frame_value), int(camera_value)
            mask_row = find_mask_row(mask_manifest, actor["role"], frame, camera)
            mask_path = verify_file(
                mask_row["mask"], mask_row["mask_sha256"], "evaluation mask"
            )
            source_path = verify_file(
                mask_row["source_image"],
                mask_row["source_image_sha256"],
                "evaluation image",
            )
            with np.load(mask_path, allow_pickle=False) as payload:
                target_mask = payload["binary"].astype(bool)
            output_dir = render_root / f"{arm}_f{frame:03d}_c{camera}"
            output_dir.mkdir()
            images = {}
            counts = {}
            native_gaussian_count = None
            for variant in ("original", "lateral", "delete"):
                if is_native:
                    output = render_native_variant(
                        trainer=trainer,
                        dataset=dataset,
                        checkpoint=checkpoint,
                        frame=frame,
                        camera=camera,
                        model_index=int(actor["rigid_model_index"]),
                        variant=variant,
                        device=device,
                    )
                    image = output["rgb"]
                    if variant != "delete":
                        rigid = trainer.models["RigidNodes"]
                        native_gaussian_count = int(
                            (rigid.point_ids[..., 0] == int(actor["rigid_model_index"]))
                            .sum()
                            .item()
                        )
                        asset_summaries[arm]["gaussian_count"] = native_gaussian_count
                    if native_gaussian_count is None:
                        raise RuntimeError("native actor Gaussian 计数未在 delete 前冻结")
                    variant_counts = {
                        "native_gaussians": native_gaussian_count,
                        "inserted_gaussians": 0,
                        "removed_gaussians": (
                            native_gaussian_count if variant == "delete" else 0
                        ),
                    }
                else:
                    assert asset is not None
                    image, variant_counts = render_variant(
                        trainer=trainer,
                        dataset=dataset,
                        checkpoint=checkpoint,
                        actor_index=int(actor["rigid_model_index"]),
                        asset=asset,
                        frame=frame,
                        camera=camera,
                        variant=variant,
                        lateral_offset_m=float(config["render"]["lateral_offset_m"]),
                        device=device,
                    )
                path = output_dir / f"{variant}.png"
                imageio.imwrite(path, image)
                images[variant] = image
                counts[variant] = variant_counts
            original_effect = np.max(
                np.abs(images["original"].astype(np.int16) - images["delete"].astype(np.int16)),
                axis=2,
            ) > 2
            lateral_effect = np.max(
                np.abs(images["lateral"].astype(np.int16) - images["delete"].astype(np.int16)),
                axis=2,
            ) > 2
            original_effect = dilate(original_effect, 2)
            lateral_effect = dilate(lateral_effect, 2)
            if int(original_effect.sum()) < int(config["render"]["minimum_effect_pixels"]):
                raise RuntimeError(f"S3 {arm} original effect 太小")
            for name, mask in (("original", original_effect), ("lateral", lateral_effect)):
                path = output_dir / f"{name}_effect.png"
                imageio.imwrite(path, mask.astype(np.uint8) * 255)
            with Image.open(source_path) as handle:
                source = np.asarray(
                    handle.convert("RGB").resize(
                        (images["original"].shape[1], images["original"].shape[0]),
                        Image.Resampling.LANCZOS,
                    )
                )
            intersection = int((original_effect & target_mask).sum())
            union = int((original_effect | target_mask).sum())
            difference = source.astype(np.float32) - images["original"].astype(np.float32)
            mse = float(np.square(difference[target_mask]).mean())
            source_tensor, generated_tensor = masked_lpips_inputs(
                source, images["original"], target_mask
            )
            with torch.inference_mode():
                lpips_value = float(
                    perceptual(
                        source_tensor.to(device), generated_tensor.to(device)
                    ).item()
                )
            outside = ~dilate(target_mask, 8)
            outside_l1 = float(
                np.abs(
                    images["original"].astype(np.int16)
                    - images["delete"].astype(np.int16)
                )[outside].mean()
            )
            component_count, largest_fraction = largest_component_fraction(lateral_effect)
            row = {
                "arm": arm,
                "phase": args.phase,
                "frame": frame,
                "camera_id": camera,
                "camera_name": mask_row["camera_name"],
                "silhouette_iou": float(intersection / union) if union else 0.0,
                "boundary_f1_tolerance_3px": boundary_f1(
                    original_effect,
                    target_mask,
                    int(config["render"]["boundary_tolerance_pixels"]),
                ),
                "masked_rgb_psnr": float(10.0 * np.log10((255.0**2) / max(mse, 1e-12))),
                "masked_crop_lpips_alex": lpips_value,
                "non_target_original_delete_l1_uint8": outside_l1,
                "generated_effect_pixels": int(original_effect.sum()),
                "lateral_effect_pixels": int(lateral_effect.sum()),
                "lateral_component_count": component_count,
                "lateral_largest_component_fraction": largest_fraction,
                "lateral_fragmentation": float(1.0 - largest_fraction),
                "sam_pixels": int(target_mask.sum()),
                "variant_counts": counts,
                "source_image_sha256": mask_row["source_image_sha256"],
                "source_mask_sha256": mask_row["mask_sha256"],
            }
            rows.append(row)
            panel = make_panel(
                source,
                images["original"],
                images["lateral"],
                images["delete"],
                target_mask,
                original_effect,
                f"{args.phase} {arm} f{frame:03d} {mask_row['camera_name']}",
            )
            panel_path = panel_root / f"{arm}_f{frame:03d}_c{camera}.png"
            panel.save(panel_path)
            panels.append({"path": str(panel_path), "sha256": sha256_file(panel_path)})
            print(f"{args.phase} {arm} f{frame:03d} c{camera} IoU={row['silhouette_iou']:.4f}", flush=True)

    aggregate = {
        arm: aggregate_rows([row for row in rows if row["arm"] == arm])
        for arm in sorted(evaluation_assets)
    }
    if args.phase == "development":
        decision = development_decision(aggregate, config)
    else:
        frozen = json.loads(args.decision.read_text(encoding="utf-8"))
        selected_arm = frozen["selected_arm"]
        decision = heldout_confirmation(aggregate, selected_arm, config)
        decision["development_decision_sha256"] = args.decision_sha256
    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("S3 evaluation 修改了 D2 checkpoint")
    source_snapshot = snapshot_sources(args.run_dir, args.config.resolve())
    decision_path = artifacts / "decision.json"
    atomic_json(decision_path, decision)
    summary = {
        "schema_version": "worldsim_v33_s3_actor_asset_evaluation_v1",
        "task_id": config["task_id"],
        "state": "completed",
        "phase": args.phase,
        "views": [[int(a), int(b)] for a, b in views],
        "heldout_read": heldout_read,
        "optimization_performed": False,
        "assets": asset_summaries,
        "rows": rows,
        "aggregate": aggregate,
        "decision": decision,
        "decision_sha256": sha256_file(decision_path),
        "panels": panels,
        "inference_resources": inference_resources,
        "evaluation_runtime": {
            "elapsed_seconds": time.time() - started,
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "source_snapshot": source_snapshot,
        "claims": {
            "observed_views": "quality metrics against RGB/SAM proxy",
            "generated_backside": "completeness/consistency only; no GT correctness",
            "heldout_selection": False,
        },
    }
    summary_path = args.run_dir / "summary.json"
    atomic_json(summary_path, summary)
    terminal = "completed" if args.phase == "development" or decision.get("accepted", True) else "rejected"
    atomic_json(
        args.run_dir / "status.json",
        {
            "state": terminal,
            "started_unix": started,
            "completed_unix": time.time(),
            "summary_sha256": sha256_file(summary_path),
            "decision_sha256": sha256_file(decision_path),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if terminal == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
