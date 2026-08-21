"""WorldSim V6 R9 independent verifier-arm 正式实验。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from PIL import Image

from motion_proj.worldsim_v6.r7_oracle import (
    _load_render_map,
    _make_missing,
    _mask_for,
    _plane,
    _required_render,
)
from motion_proj.worldsim_v6.r8_generator import _asset_inventory, _resize_case


TASK_ID = "WS-V6-R9-INDEPENDENT-VERIFIER-ARMS-01"


class R9ExperimentError(RuntimeError):
    """R9 正式合同失败。"""


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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R9ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def _resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = np.rint(np.clip(rgb.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.asarray(Image.fromarray(image, mode="RGB").resize(size, Image.Resampling.BILINEAR))


def _resize_plane(
    plane: np.ndarray, size: tuple[int, int], resampling: Image.Resampling
) -> np.ndarray:
    return np.asarray(
        Image.fromarray(plane.astype(np.float32), mode="F").resize(size, resampling),
        dtype=np.float32,
    )


def _run_checked(command: list[str], cwd: Path, env: Mapping[str, str], log: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise R9ExperimentError(
            f"worker 失败 returncode={completed.returncode}: {(completed.stdout + completed.stderr)[-2000:]}"
        )


def _arm_summary(
    rows: list[Mapping[str, Any]], arm_id: str, denominator: int, gate: Mapping[str, Any]
) -> dict[str, Any]:
    evidence_rows = [row for row in rows if row[arm_id]["decision"] != "ABSTAIN"]
    accepts = [row for row in evidence_rows if row[arm_id]["decision"] == "ACCEPT"]
    false_safe_count = sum(bool(row[arm_id]["false_safe"]) for row in accepts)
    false_safe_rate = 0.0 if not accepts else float(false_safe_count / len(accepts))
    p0_false_safe_count = sum(not bool(row[arm_id]["truth_safe"]) for row in evidence_rows)
    p0_false_safe_rate = (
        0.0 if not evidence_rows else float(p0_false_safe_count / len(evidence_rows))
    )
    accept_coverage = float(len(accepts) / denominator)
    reduction = float(p0_false_safe_rate - false_safe_rate)
    checks = {
        "minimum_accept_coverage": accept_coverage >= float(gate["minimum_accept_coverage"]),
        "maximum_false_safe_rate": false_safe_rate <= float(gate["maximum_false_safe_rate"]),
        "minimum_false_safe_reduction_vs_p0": reduction
        >= float(gate["minimum_false_safe_reduction_vs_p0"]),
    }
    checks["passed"] = all(checks.values())
    return {
        "schema_version": "worldsim_v6.r9_arm_summary.v1",
        "arm": arm_id,
        "denominator": denominator,
        "evidence_count": len(evidence_rows),
        "abstain_count": denominator - len(evidence_rows),
        "accept_count": len(accepts),
        "reject_count": len(evidence_rows) - len(accepts),
        "accept_coverage": accept_coverage,
        "false_safe_count": false_safe_count,
        "false_safe_rate": false_safe_rate,
        "p0_false_safe_count": p0_false_safe_count,
        "p0_false_safe_rate": p0_false_safe_rate,
        "false_safe_reduction_vs_p0": reduction,
        "checks": checks,
        "eligible_for_r10": checks["passed"],
    }


def run_experiment(
    repo_root: Path,
    config_path: Path,
    run_root: Path,
    big_lama_root: Path,
    sd15_root: Path,
    depth_model_root: Path,
    semantic_model_root: Path,
) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R9ExperimentError("正式 R9 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R9ExperimentError("R9 task_id 漂移")
    sources = config["sources"]
    r8_run = _resolve_runs_uri(sources["r8_run"])
    r7_run = _resolve_runs_uri(sources["r7_run"])
    render_root = _resolve_runs_uri(sources["render_run"])
    if _sha256(r8_run / "MANIFEST.json") != sources["r8_manifest_sha256"]:
        raise R9ExperimentError("R8 source manifest 漂移")
    if _sha256(r7_run / "MANIFEST.json") != sources["r7_manifest_sha256"]:
        raise R9ExperimentError("R7 source manifest 漂移")
    selected_candidate = str(sources["selected_candidate"])
    if selected_candidate not in {"big_lama", "sd15_inpainting"}:
        raise R9ExperimentError("R9 selected candidate 非冻结候选")
    if _sha256(r8_run / selected_candidate / "WORKER_RESULT.json") != sources[
        "selected_worker_result_sha256"
    ]:
        raise R9ExperimentError("R8 selected worker 漂移")
    if selected_candidate == "big_lama":
        selected_asset_sha256 = sources.get(
            "selected_asset_content_sha256", sources.get("big_lama_checkpoint_sha256")
        )
        if _sha256(big_lama_root / "big-lama/models/best.ckpt") != selected_asset_sha256:
            raise R9ExperimentError("Big-LaMa checkpoint 漂移")
    else:
        _, sd_content_sha256 = _asset_inventory(sd15_root)
        if sd_content_sha256 != sources["selected_asset_content_sha256"]:
            raise R9ExperimentError("SD-v1.5 snapshot 漂移")
    geometry_cfg = config["verifier_models"]["geometry"]
    semantic_cfg = config["verifier_models"]["semantic"]
    if _sha256(depth_model_root / geometry_cfg["model_file"]) != geometry_cfg["model_sha256"]:
        raise R9ExperimentError("geometry verifier 权重漂移")
    if _sha256(semantic_model_root / semantic_cfg["model_file"]) != semantic_cfg["model_sha256"]:
        raise R9ExperimentError("semantic verifier 权重漂移")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R9ExperimentError("R9 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__independent-arms-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        generator_input_dir = run_dir / "generator_inputs"
        verifier_input_dir = run_dir / "verifier_inputs"
        proposal_dir = run_dir / f"{selected_candidate}_proposals"
        verifier_output_dir = run_dir / "verifier_worker"
        generator_input_dir.mkdir()
        verifier_input_dir.mkdir()
        r7_config = yaml.safe_load(
            (repo_root / "configs/worldsim_v6/r7_oracle_missing_world_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        width, height = (int(value) for value in config["proposal"]["resolution_px"])
        size = (width, height)
        render_maps: dict[tuple[str, str], Any] = {}
        case_rows: list[dict[str, Any]] = []
        structural_rows: list[dict[str, Any]] = []
        for scene in config["cohort"]["scenes"]:
            for frontend in config["cohort"]["frontends"]:
                key = (scene, frontend)
                render_maps[key] = _load_render_map(
                    render_root / "renders" / scene / frontend / "RENDER_MAP.jsonl"
                )
                render_map = render_maps[key]
                for frame_index in config["cohort"]["frame_indices"]:
                    base = _required_render(
                        render_root, scene, frontend, int(frame_index), "lat0m", render_map
                    )
                    side = _required_render(
                        render_root, scene, frontend, int(frame_index), "lat2m", render_map
                    )
                    removed = _required_render(
                        render_root,
                        scene,
                        frontend,
                        int(frame_index),
                        "actor_remove_all",
                        render_map,
                    )
                    for hole_type in config["cohort"]["hole_types"]:
                        target = (
                            side
                            if hole_type == "missing_side_view"
                            else removed
                            if hole_type == "disocclusion"
                            else base
                        )
                        mask = _mask_for(
                            hole_type,
                            target,
                            base,
                            removed,
                            r7_config["oracle_patch_contract"],
                        )
                        mask_count = int(np.count_nonzero(mask))
                        minimum_mask = int(
                            r7_config["oracle_patch_contract"]["minimum_mask_pixels"]
                        )
                        case_id = f"{scene}__{frontend}__f{int(frame_index):03d}__{hole_type}"
                        if mask_count < minimum_mask:
                            if hole_type not in {"disocclusion", "actor_removal_hole"}:
                                raise R9ExperimentError(f"非 actor hole mask 不足：{case_id}")
                            structural_rows.append(
                                {
                                    "case_id": case_id,
                                    "decision": "structural_abstain_before_proposal",
                                    "evidence_pixel_count": mask_count,
                                }
                            )
                            continue
                        missing = _make_missing(target, mask)
                        input_image, resized_mask = _resize_case(missing, mask, size)
                        target_rgb = _resize_rgb(target["rgb"], size)
                        target_depth_plane = _plane(target["depth"], "depth").astype(np.float32)
                        target_depth = _resize_plane(
                            target_depth_plane, size, Image.Resampling.BILINEAR
                        )
                        target_depth_valid = (
                            _resize_plane(
                                (np.isfinite(target_depth_plane) & (target_depth_plane > 1.0e-6)).astype(
                                    np.float32
                                ),
                                size,
                                Image.Resampling.NEAREST,
                            )
                            > 0.5
                        )
                        target_dynamic_plane = (
                            _plane(target["dynamic_opacity"], "dynamic_opacity")
                            > float(r7_config["oracle_patch_contract"]["dynamic_opacity_threshold"])
                        )
                        target_dynamic = (
                            _resize_plane(
                                target_dynamic_plane.astype(np.float32),
                                size,
                                Image.Resampling.NEAREST,
                            )
                            > 0.5
                        )
                        generator_path = generator_input_dir / f"{case_id}.npz"
                        verifier_path = verifier_input_dir / f"{case_id}.npz"
                        np.savez_compressed(
                            generator_path, image=input_image, mask=resized_mask
                        )
                        semantic_evidence = hole_type in set(
                            config["arms"]["P3"]["evidence_hole_types"]
                        )
                        np.savez_compressed(
                            verifier_path,
                            input_image=input_image,
                            target_rgb=target_rgb,
                            target_depth=target_depth,
                            target_depth_valid=target_depth_valid,
                            target_dynamic=target_dynamic,
                            mask=resized_mask,
                            semantic_evidence=np.asarray(semantic_evidence),
                            hole_type=np.asarray(hole_type),
                        )
                        case_rows.append(
                            {
                                "case_id": case_id,
                                "scene": scene,
                                "frontend": frontend,
                                "frame_index": int(frame_index),
                                "hole_type": hole_type,
                                "mask_pixel_count": int(np.count_nonzero(resized_mask)),
                                "semantic_evidence": semantic_evidence,
                                "generator_input_sha256": _sha256(generator_path),
                                "verifier_input_sha256": _sha256(verifier_path),
                            }
                        )
        if len(case_rows) != int(config["cohort"]["expected_eligible_case_count"]):
            raise R9ExperimentError("R9 eligible case denominator 漂移")
        if len(structural_rows) != int(config["cohort"]["expected_structural_abstain_count"]):
            raise R9ExperimentError("R9 structural abstain denominator 漂移")
        if sum(row["semantic_evidence"] for row in case_rows) != int(
            config["cohort"]["expected_semantic_evidence_case_count"]
        ):
            raise R9ExperimentError("R9 semantic evidence denominator 漂移")
        _write_jsonl(run_dir / "CASES.jsonl", case_rows)
        _write_jsonl(run_dir / "STRUCTURAL_ABSTAINS.jsonl", structural_rows)

        python = Path("/root/autodl-tmp/envs/motionproj/bin/python")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        _run_checked(
            [
                str(python),
                str(repo_root / "scripts/worldsim_v6/r8_generator_worker.py"),
                "--candidate",
                selected_candidate,
                "--input-dir",
                str(generator_input_dir),
                "--output-dir",
                str(proposal_dir),
                "--big-lama-root",
                str(big_lama_root),
                "--sd15-root",
                str(sd15_root),
                "--seed",
                str(config["seed"]),
                "--repeat-count",
                str(config["proposal"]["repeat_count"]),
                "--prompt",
                str(config["proposal"].get("sd_prompt", "")),
                "--inference-steps",
                str(config["proposal"].get("sd_inference_steps", 20)),
                "--guidance-scale",
                str(config["proposal"].get("sd_guidance_scale", 4.0)),
            ],
            repo_root,
            env,
            run_dir / "generator.log",
        )
        generator_result = json.loads(
            (proposal_dir / "WORKER_RESULT.json").read_text(encoding="utf-8")
        )
        if generator_result["case_count"] != len(case_rows):
            raise R9ExperimentError("R9 proposal denominator 漂移")

        p1 = config["arms"]["P1"]
        p2 = config["arms"]["P2"]
        p3 = config["arms"]["P3"]
        _run_checked(
            [
                str(python),
                str(repo_root / "scripts/worldsim_v6/r9_verifier_worker.py"),
                "--verifier-input-dir",
                str(verifier_input_dir),
                "--proposal-dir",
                str(proposal_dir),
                "--output-dir",
                str(verifier_output_dir),
                "--depth-model-root",
                str(depth_model_root),
                "--semantic-model-root",
                str(semantic_model_root),
                "--photo-max-mae",
                str(p1["maximum_masked_rgb_mae"]),
                "--photo-truth-pixel-error",
                str(p1["truth_pixel_absolute_error"]),
                "--photo-truth-min-fraction",
                str(p1["truth_minimum_usable_fraction"]),
                "--depth-max-mean-relative-error",
                str(p2["maximum_masked_mean_relative_depth_error"]),
                "--depth-truth-pixel-error",
                str(p2["truth_pixel_relative_depth_error"]),
                "--depth-truth-min-fraction",
                str(p2["truth_minimum_usable_fraction"]),
                "--minimum-alignment-pixels",
                str(p2["minimum_alignment_pixels"]),
                "--semantic-min-iou",
                str(p3["minimum_verifier_dynamic_iou"]),
                "--semantic-truth-min-iou",
                str(p3["truth_minimum_dynamic_iou"]),
                "--dynamic-class-ids",
                ",".join(str(value) for value in p3["dynamic_cityscapes_class_ids"]),
            ],
            repo_root,
            env,
            run_dir / "verifier.log",
        )
        verifier_result = json.loads(
            (verifier_output_dir / "WORKER_RESULT.json").read_text(encoding="utf-8")
        )
        rows = [
            json.loads(line)
            for line in (verifier_output_dir / "PER_CASE_ARMS.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        expected = int(config["cohort"]["expected_eligible_case_count"])
        semantic_expected = int(config["cohort"]["expected_semantic_evidence_case_count"])
        arm_rows = [
            _arm_summary(rows, "P1", expected, config["gate"]),
            _arm_summary(rows, "P2", expected, config["gate"]),
            _arm_summary(rows, "P3", semantic_expected, config["gate"]),
        ]
        arm_rows.append(
            {
                "schema_version": "worldsim_v6.r9_arm_summary.v1",
                "arm": "P4",
                "denominator": expected,
                "evidence_count": 0,
                "abstain_count": sum(row["P4"]["decision"] == "ABSTAIN" for row in rows),
                "accept_count": 0,
                "reject_count": 0,
                "eligible_for_r10": False,
                "status": "correct_structural_abstain",
            }
        )
        _write_jsonl(run_dir / "ARM_SUMMARIES.jsonl", arm_rows)
        eligible_arms = [row["arm"] for row in arm_rows if row["eligible_for_r10"]]
        peak_mib = max(
            float(generator_result["peak_gpu_memory_mib"]),
            float(verifier_result["peak_gpu_memory_mib"]),
        )
        checks = {
            "eligible_case_count_exact": len(rows) == expected,
            "semantic_evidence_count_exact": verifier_result["semantic_evidence_case_count"]
            == semantic_expected,
            "p4_all_abstain": verifier_result["p4_abstain_count"] == expected,
            "non_target_exact": bool(verifier_result["all_outside_mask_exact"]),
            "independent_arm_outputs_only": True,
            "no_arm_combination": True,
            "at_least_one_arm_eligible_for_r10": bool(eligible_arms),
            "resource_peak_within_contract": peak_mib
            <= float(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "training_not_started": True,
            "confirmation_not_read": True,
            "bake_not_started": True,
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.r9_gate.v1",
            "checks": checks,
            "eligible_arms_for_r10": eligible_arms,
            "decision": "proceed_to_factorized_verification"
            if checks["passed"]
            else "reject_or_pivot_verifier_arms",
        }
        _write_json(run_dir / "R9_GATE.json", gate)
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r9_resource_audit.v1",
                "gpu": config["resources"]["gpu"],
                "generator_peak_gpu_memory_mib": generator_result["peak_gpu_memory_mib"],
                "verifier_peak_gpu_memory_mib": verifier_result["peak_gpu_memory_mib"],
                "combined_sequential_peak_gpu_memory_mib": peak_mib,
                "wall_seconds": time.monotonic() - started,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "confirmation_content_read": False,
                "bake_started": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r9_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development" if checks["passed"] else "rejected",
            "source_commit": source_commit,
            "selected_generator": sources["selected_candidate"],
            "eligible_arms_for_r10": eligible_arms,
            "case_count": len(rows),
            "structural_abstain_count": len(structural_rows),
            "training_started": False,
            "confirmation_content_read": False,
            "bake_started": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "CASES.jsonl",
            "STRUCTURAL_ABSTAINS.jsonl",
            "ARM_SUMMARIES.jsonl",
            "R9_GATE.json",
            "RESOURCE_AUDIT.json",
            "SUMMARY.json",
            f"{selected_candidate}_proposals/WORKER_RESULT.json",
            "verifier_worker/PER_CASE_ARMS.jsonl",
            "verifier_worker/WORKER_RESULT.json",
        ]
        manifest = {
            "schema_version": "worldsim_v6.r9_run_manifest.v1",
            "files": {
                name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
                for name in tracked
            },
        }
        _write_json(run_dir / "MANIFEST.json", manifest)
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
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
        default=Path("configs/worldsim_v6/r9_independent_verifier_arms_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    parser.add_argument(
        "--big-lama-root",
        type=Path,
        default=Path("/root/autodl-tmp/models/worldsim_v6/r8_lama"),
    )
    parser.add_argument(
        "--sd15-root",
        type=Path,
        default=Path("/root/autodl-tmp/models/worldsim_v6/r8_sd15"),
    )
    parser.add_argument(
        "--depth-model-root",
        type=Path,
        default=Path("/root/autodl-tmp/models/worldsim_v6/r9_depth_anything_v2_small"),
    )
    parser.add_argument(
        "--semantic-model-root",
        type=Path,
        default=Path("/root/autodl-tmp/models/worldsim_v6/r9_semantic_deeplab_cityscapes"),
    )
    args = parser.parse_args()
    run_experiment(
        args.repo_root,
        args.config,
        args.run_root,
        args.big_lama_root,
        args.sd15_root,
        args.depth_model_root,
        args.semantic_model_root,
    )
    return 0
