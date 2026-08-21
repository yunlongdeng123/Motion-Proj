"""WorldSim V6 R32 将 verified actor layer 绑定到 SceneIR actor_0000。"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
import yaml
from PIL import Image
from scipy.ndimage import binary_dilation
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from motion_proj.worldsim_v6.r26_typed_semantic_multisource import (
    _load_semantic_model,
    _predict_segformer,
    _predict_semantic,
)


TASK_ID = "WS-V6-R32-ACTOR-IDENTITY-FACTOR-VERIFICATION-01"


class R32ExperimentError(RuntimeError):
    """R32 正式合同失败。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R32ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def _rgb(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["rgb"], dtype=np.float32)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8), mode="L").resize(
            (width, height), Image.Resampling.NEAREST
        )
    ) > 0


def _load_depth_model(root: Path) -> tuple[Any, Any]:
    processor = AutoImageProcessor.from_pretrained(root, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    ).cuda().eval()
    return processor, model


def _predict_depth(processor: Any, model: Any, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    inputs = processor(images=Image.fromarray(image, mode="RGB"), return_tensors="pt")
    inputs = {name: value.cuda() for name, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs).predicted_depth[:, None]
        output = torch_functional.interpolate(
            output, size=(height, width), mode="bicubic", align_corners=False
        )[:, 0]
    return output[0].float().cpu().numpy()


def _affine_align(
    predicted: np.ndarray,
    target: np.ndarray,
    calibration_mask: np.ndarray,
    minimum_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = (
        calibration_mask
        & np.isfinite(predicted)
        & np.isfinite(target)
        & (target > 1.0e-6)
    )
    count = int(np.count_nonzero(valid))
    if count < minimum_pixels:
        raise R32ExperimentError("R32 depth calibration evidence 不足")
    x = predicted[valid].astype(np.float64)
    y = target[valid].astype(np.float64)
    design = np.stack([x, np.ones_like(x)], axis=1)
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if int(rank) < 2 or not np.all(np.isfinite(coefficients)):
        raise R32ExperimentError("R32 depth affine alignment 退化")
    aligned = predicted.astype(np.float64) * coefficients[0] + coefficients[1]
    return aligned.astype(np.float32), {
        "pixel_count": count,
        "scale": float(coefficients[0]),
        "offset": float(coefficients[1]),
    }


def _masked_iou(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    union = mask & (first | second)
    if not np.any(union):
        return 1.0
    return float(np.count_nonzero(mask & first & second) / np.count_nonzero(union))


def run_experiment(
    repo_root: Path,
    config_path: Path,
    run_root: Path,
    depth_model_root: Path,
    deeplab_root: Path,
    segformer_root: Path,
) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R32ExperimentError("正式 R32 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R32ExperimentError("R32 task_id 漂移")
    sources = config["sources"]
    r24 = _resolve_runs_uri(sources["r24_run"])
    r30 = _resolve_runs_uri(sources["r30_run"])
    r31 = _resolve_runs_uri(sources["r31_run"])
    cohort_run = _resolve_runs_uri(sources["actor_cohort_run"])
    binding_run = _resolve_runs_uri(sources["sceneir_binding_run"])
    case_id = str(config["cohort"]["case_id"])
    proposal_path = r24 / sources["proposal_directory"] / f"{case_id}__repeat1.npy"
    verifier_input = r24 / "verifier_inputs" / f"{case_id}.npz"
    cohort_logged = cohort_run / sources["actor_cohort_logged_file"]
    cohort_removed = cohort_run / sources["actor_cohort_removed_file"]
    source_files = {
        r24 / "MANIFEST.json": sources["r24_manifest_sha256"],
        proposal_path: sources["proposal_sha256"],
        verifier_input: sources["verifier_input_sha256"],
        r30 / "MANIFEST.json": sources["r30_manifest_sha256"],
        r30 / "R30_GATE.json": sources["r30_gate_sha256"],
        r30 / "package/ACTOR_ASSET_REGISTRY.jsonl": sources["r30_registry_sha256"],
        r31 / "MANIFEST.json": sources["r31_manifest_sha256"],
        r31 / "R31_GATE.json": sources["r31_gate_sha256"],
        cohort_run / "MANIFEST.json": sources["actor_cohort_manifest_sha256"],
        cohort_run / "R13_ACTOR_COHORT_GATE.json": sources["actor_cohort_gate_sha256"],
        cohort_run / "ACTOR_VERDICTS.jsonl": sources["actor_cohort_verdicts_sha256"],
        cohort_run / "ACTOR_CASE_METRICS.jsonl": sources["actor_cohort_metrics_sha256"],
        cohort_logged: sources["actor_cohort_logged_sha256"],
        cohort_removed: sources["actor_cohort_removed_sha256"],
        binding_run / "MANIFEST.json": sources["sceneir_binding_manifest_sha256"],
        binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json": sources[
            "sceneir_binding_gate_sha256"
        ],
        binding_run / "BINDING_AUDIT.json": sources["sceneir_binding_audit_sha256"],
        depth_model_root / config["models"]["depth"]["model_file"]: config["models"][
            "depth"
        ]["model_sha256"],
        deeplab_root / config["models"]["deeplab"]["model_file"]: config["models"][
            "deeplab"
        ]["model_sha256"],
        segformer_root / config["models"]["segformer"]["model_file"]: config[
            "models"
        ]["segformer"]["model_sha256"],
        segformer_root / "config.json": config["models"]["segformer"]["config_sha256"],
        segformer_root / "preprocessor_config.json": config["models"]["segformer"][
            "preprocessor_sha256"
        ],
    }
    for path, expected in source_files.items():
        if _sha256(path) != expected:
            raise R32ExperimentError(f"冻结输入漂移：{path}")
    if not json.loads((r30 / "R30_GATE.json").read_text(encoding="utf-8"))["checks"][
        "passed"
    ]:
        raise R32ExperimentError("R30 gate 未通过")
    if not json.loads((r31 / "R31_GATE.json").read_text(encoding="utf-8"))["checks"][
        "passed"
    ]:
        raise R32ExperimentError("R31 gate 未通过")
    cohort_gate = json.loads(
        (cohort_run / "R13_ACTOR_COHORT_GATE.json").read_text(encoding="utf-8")
    )
    binding_gate = json.loads(
        (binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json").read_text(encoding="utf-8")
    )
    if not cohort_gate["checks"]["passed"] or not binding_gate["checks"]["passed"]:
        raise R32ExperimentError("既有 actor identity authority 未通过")
    verdicts = _read_jsonl(cohort_run / "ACTOR_VERDICTS.jsonl")
    verdict = next(row for row in verdicts if row["model_index"] == 0)
    binding = json.loads((binding_run / "BINDING_AUDIT.json").read_text(encoding="utf-8"))
    if not verdict["v6_accepted"]:
        raise R32ExperimentError("model index 0 既有 verdict 漂移")
    if (
        binding["frontend_model_index"] != 0
        or binding["sceneir_actor_id"] != "actor_0000"
        or binding["sceneir_chunk_id"] != "streetgs_actor_0000"
    ):
        raise R32ExperimentError("SceneIR actor binding 漂移")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R32ExperimentError("R32 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__identity-factor-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.manual_seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        proposal = np.load(proposal_path, allow_pickle=False).astype(np.uint8)
        with np.load(verifier_input, allow_pickle=False) as archive:
            input_image = np.asarray(archive["input_image"], dtype=np.uint8)
            target_rgb = np.asarray(archive["target_rgb"], dtype=np.uint8)
            target_depth = np.asarray(archive["target_depth"], dtype=np.float32)
            target_depth_valid = np.asarray(archive["target_depth_valid"], dtype=bool)
            target_dynamic = np.asarray(archive["target_dynamic"], dtype=bool)
            layer_mask = np.asarray(archive["mask"], dtype=bool)
        logged_rgb = _rgb(cohort_logged)
        removed_rgb = _rgb(cohort_removed)
        pixel_difference = np.mean(np.abs(logged_rgb - removed_rgb), axis=-1)
        identity_highres = binary_dilation(
            pixel_difference > float(config["identity_mask"]["rgb_change_threshold"]),
            iterations=int(config["identity_mask"]["effect_mask_dilation_px"]),
        )
        original_identity_pixels = int(np.count_nonzero(identity_highres))
        identity_resized = _resize_mask(
            identity_highres, proposal.shape[1], proposal.shape[0]
        )
        identity_mask = identity_resized & layer_mask
        resized_pixels = int(np.count_nonzero(identity_resized))
        identity_pixels = int(np.count_nonzero(identity_mask))
        support_recall = identity_pixels / max(resized_pixels, 1)
        layer_fraction = identity_pixels / int(np.count_nonzero(layer_mask))
        np.save(run_dir / "ACTOR_0000_IDENTITY_MASK.npy", identity_mask, allow_pickle=False)

        rgb_error = np.mean(
            np.abs(proposal.astype(np.float32) - target_rgb.astype(np.float32)) / 255.0,
            axis=2,
        )
        photo_mae = float(np.mean(rgb_error[identity_mask]))
        photo_usable = float(
            np.mean(
                rgb_error[identity_mask]
                <= float(config["factors"]["photo"]["truth_pixel_absolute_error"])
            )
        )
        photo_accept = photo_mae <= float(
            config["factors"]["photo"]["maximum_masked_rgb_mae"]
        )
        photo_truth = photo_usable >= float(
            config["factors"]["photo"]["truth_minimum_usable_fraction"]
        )

        depth_processor, depth_model = _load_depth_model(depth_model_root)
        predicted_depth = _predict_depth(depth_processor, depth_model, proposal)
        aligned_depth, alignment = _affine_align(
            predicted_depth,
            target_depth,
            (~layer_mask) & target_depth_valid,
            int(config["factors"]["geometry"]["minimum_alignment_pixels"]),
        )
        valid_depth = identity_mask & target_depth_valid & np.isfinite(aligned_depth)
        relative = np.abs(aligned_depth[valid_depth] - target_depth[valid_depth]) / np.maximum(
            np.abs(target_depth[valid_depth]), 1.0e-3
        )
        depth_mean_error = float(np.mean(relative))
        depth_usable = float(
            np.mean(
                relative
                <= float(config["factors"]["geometry"]["truth_pixel_relative_error"])
            )
        )
        geometry_accept = depth_mean_error <= float(
            config["factors"]["geometry"]["maximum_masked_mean_relative_depth_error"]
        )
        geometry_truth = depth_usable >= float(
            config["factors"]["geometry"]["truth_minimum_usable_fraction"]
        )
        del depth_model, depth_processor, predicted_depth, aligned_depth
        gc.collect()
        torch.cuda.empty_cache()

        deeplab = _load_semantic_model(deeplab_root)
        segformer_processor = SegformerImageProcessor.from_pretrained(
            segformer_root, local_files_only=True
        )
        segformer = SegformerForSemanticSegmentation.from_pretrained(
            segformer_root, local_files_only=True
        ).cuda().eval()
        deeplab_labels = _predict_semantic(deeplab, proposal)
        segformer_labels = _predict_segformer(segformer_processor, segformer, proposal)
        dynamic_ids = [int(value) for value in config["factors"]["semantic"]["dynamic_class_ids"]]
        deeplab_dynamic = np.isin(deeplab_labels, dynamic_ids)
        segformer_dynamic = np.isin(segformer_labels, dynamic_ids)
        consensus_iou = _masked_iou(deeplab_dynamic, segformer_dynamic, identity_mask)
        target_iou = _masked_iou(deeplab_dynamic, target_dynamic, identity_mask)
        semantic_accept = consensus_iou >= float(
            config["factors"]["semantic"]["minimum_dynamic_iou"]
        )
        semantic_truth = target_iou >= float(
            config["factors"]["semantic"]["truth_minimum_dynamic_iou"]
        )
        factors = {
            "photo": {
                "decision": "ACCEPT" if photo_accept else "REJECT",
                "truth_safe": photo_truth,
                "masked_rgb_mae": photo_mae,
                "truth_usable_fraction": photo_usable,
            },
            "geometry": {
                "decision": "ACCEPT" if geometry_accept else "REJECT",
                "truth_safe": geometry_truth,
                "masked_mean_relative_depth_error": depth_mean_error,
                "truth_usable_fraction": depth_usable,
                "alignment": alignment,
            },
            "semantic": {
                "decision": "ACCEPT" if semantic_accept else "REJECT",
                "truth_safe": semantic_truth,
                "architecture_consensus_dynamic_iou": consensus_iou,
                "target_dynamic_iou_evaluation_only": target_iou,
            },
            "dynamics": {
                "decision": "ABSTAIN",
                "reason": "no_independent_trajectory_verifier",
            },
        }
        joint_accept = all(factors[name]["decision"] == "ACCEPT" for name in ("photo", "geometry", "semantic"))
        joint_truth = all(factors[name]["truth_safe"] for name in ("photo", "geometry", "semantic"))
        identity = {
            "schema_version": "worldsim_v6.r32_identity_factor.v1",
            "case_id": case_id,
            "frontend_model_index": 0,
            "sceneir_actor_id": "actor_0000",
            "sceneir_chunk_id": "streetgs_actor_0000",
            "primitive_count": int(binding["primitive_count"]),
            "original_identity_effect_pixels": original_identity_pixels,
            "resized_identity_effect_pixels": resized_pixels,
            "identity_pixels_within_verified_layer": identity_pixels,
            "identity_support_recall": support_recall,
            "verified_layer_fraction": layer_fraction,
            "factors": factors,
            "overall_decision": "ACCEPT" if joint_accept else "ABSTAIN",
            "joint_truth_safe": joint_truth,
            "false_safe": bool(joint_accept and not joint_truth),
        }
        _write_json(run_dir / "IDENTITY_FACTOR_RESULT.json", identity)
        wall_seconds = time.monotonic() - started
        checks = {
            "existing_model_index_verdict_accepted": verdict["v6_accepted"],
            "sceneir_identity_binding_exact": binding["frontend_model_index"] == 0
            and binding["sceneir_actor_id"] == "actor_0000",
            "original_effect_denominator_exact": original_identity_pixels
            == int(config["identity_mask"]["expected_original_effect_pixels"]),
            "minimum_identity_pixels": identity_pixels
            >= int(config["identity_mask"]["minimum_resized_identity_pixels"]),
            "minimum_identity_support_recall": support_recall
            >= float(config["identity_mask"]["minimum_layer_support_recall"]),
            "photo_geometry_semantic_all_accept": joint_accept,
            "every_factor_truth_safe": joint_truth,
            "false_safe_zero": not identity["false_safe"],
            "outside_verified_layer_exact": np.array_equal(
                proposal[~layer_mask], input_image[~layer_mask]
            ),
            "dynamics_abstain": factors["dynamics"]["decision"] == "ABSTAIN",
            "source_immutable": all(
                _sha256(path) == expected for path, expected in source_files.items()
            ),
            "gpu_within_budget": torch.cuda.max_memory_reserved() / (1024**2)
            <= float(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds
            <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
            "bake_not_started": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(
            run_dir / "R32_GATE.json",
            {
                "schema_version": "worldsim_v6.r32_gate.v1",
                "checks": checks,
                "decision": "proceed_to_identity_bound_actor_bake"
                if checks["passed"]
                else "reject_or_pivot_actor_identity_binding",
            },
        )
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r32_resource_audit.v1",
                "gpu": config["resources"]["gpu"],
                "peak_gpu_memory_mib": float(torch.cuda.max_memory_reserved() / (1024**2)),
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r32_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development" if checks["passed"] else "rejected",
            "source_commit": source_commit,
            "identity": "streetgs_model_index_0_to_sceneir_actor_0000",
            "overall_decision": identity["overall_decision"],
            "false_safe": identity["false_safe"],
            "dynamics_status": "ABSTAIN",
            "bake_started": False,
            "training_started": False,
            "confirmation_content_read": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "ACTOR_0000_IDENTITY_MASK.npy",
            "IDENTITY_FACTOR_RESULT.json",
            "R32_GATE.json",
            "RESOURCE_AUDIT.json",
            "SUMMARY.json",
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r32_manifest.v1",
                "files": {
                    name: {
                        "bytes": (run_dir / name).stat().st_size,
                        "sha256": _sha256(run_dir / name),
                    }
                    for name in tracked
                },
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
            },
        )
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r32_actor_identity_factor_verification_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    parser.add_argument(
        "--depth-model-root",
        type=Path,
        default=Path("/root/autodl-tmp/models/worldsim_v6/r9_depth_anything_v2_small"),
    )
    parser.add_argument(
        "--deeplab-root",
        type=Path,
        default=Path("/root/autodl-tmp/models/worldsim_v6/r9_semantic_deeplab_cityscapes"),
    )
    parser.add_argument(
        "--segformer-root",
        type=Path,
        default=Path("/root/autodl-tmp/models/worldsim_v6/r20_semantic_segformer_cityscapes"),
    )
    args = parser.parse_args()
    run_experiment(
        args.repo_root,
        args.config,
        args.run_root,
        args.depth_model_root,
        args.deeplab_root,
        args.segformer_root,
    )
    return 0
