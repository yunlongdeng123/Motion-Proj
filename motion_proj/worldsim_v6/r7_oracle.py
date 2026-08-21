"""WorldSim V6 R7 oracle missing-world extension 正式实验。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


TASK_ID = "WS-V6-R7-ORACLE-MISSING-WORLD-01"


class R7ExperimentError(RuntimeError):
    """R7 正式合同失败。"""


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


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes())
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
        raise R7ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def _load_render_map(path: Path) -> dict[tuple[int, str], Mapping[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = {}
    for row in rows:
        key = (int(row["frame_index"]), str(row["path"]))
        if key in result:
            raise R7ExperimentError(f"重复 render map key：{key}")
        result[key] = row
    return result


def _required_render(
    render_root: Path,
    scene: str,
    frontend: str,
    frame_index: int,
    suffix: str,
    render_map: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    name = f"frame{frame_index:03d}_{suffix}.npz"
    row = render_map.get((frame_index, name))
    if row is None:
        raise R7ExperimentError(f"render map 缺少 {scene}/{frontend}/{name}")
    path = render_root / "renders" / scene / frontend / name
    if _sha256(path) != row["sha256"]:
        raise R7ExperimentError(f"render hash 漂移：{path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"rgb", "depth", "dynamic_opacity"}
        if not required <= set(archive.files):
            raise R7ExperimentError(f"render 字段缺失：{path}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for offset_y in range(2 * radius + 1):
        for offset_x in range(2 * radius + 1):
            result |= padded[offset_y : offset_y + height, offset_x : offset_x + width]
    return result


def _plane(array: np.ndarray, name: str) -> np.ndarray:
    """把 frontend 的 H×W 或 singleton-channel 输出归一为显式二维平面。"""
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    raise R7ExperimentError(f"{name} 不是二维 singleton-channel 平面：{array.shape}")


def _mask_for(
    hole_type: str,
    target: Mapping[str, np.ndarray],
    base: Mapping[str, np.ndarray],
    removed: Mapping[str, np.ndarray],
    patch_config: Mapping[str, Any],
) -> np.ndarray:
    height, width = target["rgb"].shape[:2]
    yy, xx = np.indices((height, width))
    dynamic = _plane(np.asarray(base["dynamic_opacity"]), "dynamic_opacity") > float(
        patch_config["dynamic_opacity_threshold"]
    )
    base_rgb = np.clip(base["rgb"].astype(np.float32), 0.0, 1.0)
    removed_rgb = np.clip(removed["rgb"].astype(np.float32), 0.0, 1.0)
    actor_edit_change = np.mean(np.abs(base_rgb - removed_rgb), axis=2) > float(
        patch_config["actor_edit_rgb_change_threshold"]
    )
    base_depth = _plane(base["depth"], "depth").astype(np.float32)
    removed_depth = _plane(removed["depth"], "depth").astype(np.float32)
    comparable_depth = (base_depth > 1.0e-6) & (removed_depth > 1.0e-6)
    actor_edit_change |= comparable_depth & (
        np.abs(base_depth - removed_depth) / np.maximum(np.abs(base_depth), 1.0e-3)
        > float(patch_config["actor_edit_relative_depth_change_threshold"])
    )
    actor_evidence = dynamic | actor_edit_change
    if hole_type == "missing_route_support":
        y = yy / max(height - 1, 1)
        x = xx / max(width - 1, 1)
        half_width = 0.10 + 0.34 * np.clip((y - 0.52) / 0.48, 0.0, 1.0)
        return (y >= 0.52) & (np.abs(x - 0.5) <= half_width) & ~dynamic
    if hole_type == "missing_side_view":
        return (xx >= int(round(width * 0.70))) & ~dynamic
    if hole_type == "disocclusion":
        return _dilate(actor_evidence, radius=5)
    if hole_type == "actor_removal_hole":
        return actor_evidence
    raise R7ExperimentError(f"未知 hole type：{hole_type}")


def _copy_modalities(source: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "rgb": np.array(source["rgb"], copy=True),
        "depth": np.array(source["depth"], copy=True),
        "dynamic_opacity": np.array(source["dynamic_opacity"], copy=True),
    }


def _make_missing(target: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    result = _copy_modalities(target)
    result["rgb"][mask] = np.asarray(0.0, dtype=result["rgb"].dtype)
    result["depth"][mask] = np.asarray(0.0, dtype=result["depth"].dtype)
    result["dynamic_opacity"][mask] = np.asarray(
        0.0, dtype=result["dynamic_opacity"].dtype
    )
    return result


def _proposal(
    target: Mapping[str, np.ndarray], mask: np.ndarray, *, decoy: bool
) -> dict[str, np.ndarray]:
    proposal = _copy_modalities(target)
    if decoy:
        rgb = np.clip(proposal["rgb"].astype(np.float32), 0.0, 1.0)
        proposal["rgb"] = (1.0 - rgb)[..., [2, 0, 1]].astype(
            proposal["rgb"].dtype, copy=False
        )
        proposal["depth"] = (
            proposal["depth"].astype(np.float32) * np.float32(1.35)
        ).astype(proposal["depth"].dtype, copy=False)
        dynamic = proposal["dynamic_opacity"].astype(np.float32)
        dynamic[mask] = 1.0 - np.clip(dynamic[mask], 0.0, 1.0)
        proposal["dynamic_opacity"] = dynamic.astype(
            proposal["dynamic_opacity"].dtype, copy=False
        )
    return proposal


def _verify(
    proposal: Mapping[str, np.ndarray],
    observation: Mapping[str, np.ndarray],
    mask: np.ndarray,
    semantic_available: bool,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    verifier = config["verifier"]
    proposal_rgb = np.clip(proposal["rgb"].astype(np.float32), 0.0, 1.0)
    target_rgb = np.clip(observation["rgb"].astype(np.float32), 0.0, 1.0)
    rgb_mae = float(np.mean(np.abs(proposal_rgb[mask] - target_rgb[mask])))
    target_depth = _plane(observation["depth"], "depth").astype(np.float32)
    proposal_depth = _plane(proposal["depth"], "depth").astype(np.float32)
    depth_mask = mask & np.isfinite(target_depth) & (target_depth > 1.0e-6)
    depth_count = int(np.count_nonzero(depth_mask))
    depth_error = None
    if depth_count:
        depth_error = float(
            np.mean(
                np.abs(proposal_depth[depth_mask] - target_depth[depth_mask])
                / np.maximum(np.abs(target_depth[depth_mask]), 1.0e-3)
            )
        )
    dynamic_iou = None
    if semantic_available:
        threshold = float(config["oracle_patch_contract"]["dynamic_opacity_threshold"])
        predicted = _plane(proposal["dynamic_opacity"], "dynamic_opacity") > threshold
        observed = _plane(observation["dynamic_opacity"], "dynamic_opacity") > threshold
        union = mask & (predicted | observed)
        intersection = mask & predicted & observed
        dynamic_iou = (
            1.0 if not np.any(union) else float(np.count_nonzero(intersection) / np.count_nonzero(union))
        )
    factor_pass = {
        "photo": rgb_mae <= float(verifier["maximum_rgb_mae"]),
        "geometry": depth_error is None
        or depth_error <= float(verifier["maximum_relative_depth_error"]),
        "semantic": dynamic_iou is None
        or dynamic_iou >= float(verifier["minimum_dynamic_iou"]),
    }
    return {
        "rgb_mae": rgb_mae,
        "relative_depth_error": depth_error,
        "depth_evidence_pixel_count": depth_count,
        "dynamic_iou": dynamic_iou,
        "factor_pass": factor_pass,
        "accepted": all(factor_pass.values()),
    }


def _bake(
    missing: Mapping[str, np.ndarray],
    proposal: Mapping[str, np.ndarray],
    mask: np.ndarray,
    accepted: bool,
) -> dict[str, np.ndarray]:
    result = _copy_modalities(missing)
    if accepted:
        for name in result:
            result[name][mask] = proposal[name][mask]
    return result


def _relative_depth_error(
    value: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> tuple[float | None, np.ndarray]:
    target_depth = _plane(target, "depth").astype(np.float32)
    value_depth = _plane(value, "depth").astype(np.float32)
    valid = mask & np.isfinite(target_depth) & (target_depth > 1.0e-6)
    if not np.any(valid):
        return None, valid
    per_pixel = np.zeros_like(target_depth, dtype=np.float32)
    per_pixel[valid] = np.abs(value_depth[valid] - target_depth[valid]) / np.maximum(
        np.abs(target_depth[valid]), 1.0e-3
    )
    return float(np.mean(per_pixel[valid])), valid


def _evaluate(
    value: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    mask: np.ndarray,
    semantic_available: bool,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    value_rgb = np.clip(value["rgb"].astype(np.float32), 0.0, 1.0)
    target_rgb = np.clip(target["rgb"].astype(np.float32), 0.0, 1.0)
    rgb_per_pixel = np.mean(np.abs(value_rgb - target_rgb), axis=2)
    rgb_mae = float(np.mean(rgb_per_pixel[mask]))
    depth_error, depth_valid = _relative_depth_error(value["depth"], target["depth"], mask)
    rgb_ok = rgb_per_pixel <= float(evaluation["usable_rgb_absolute_error"])
    depth_ok = np.ones_like(mask, dtype=bool)
    if np.any(depth_valid):
        target_depth = _plane(target["depth"], "depth").astype(np.float32)
        value_depth = _plane(value["depth"], "depth").astype(np.float32)
        relative = np.zeros_like(target_depth, dtype=np.float32)
        relative[depth_valid] = np.abs(value_depth[depth_valid] - target_depth[depth_valid]) / np.maximum(
            np.abs(target_depth[depth_valid]), 1.0e-3
        )
        depth_ok[depth_valid] = relative[depth_valid] <= float(
            evaluation["usable_depth_relative_error"]
        )
    semantic_ok = np.ones_like(mask, dtype=bool)
    if semantic_available:
        threshold = float(config["oracle_patch_contract"]["dynamic_opacity_threshold"])
        semantic_ok = (
            (_plane(value["dynamic_opacity"], "dynamic_opacity") > threshold)
            == (_plane(target["dynamic_opacity"], "dynamic_opacity") > threshold)
        )
    target_content = (
        np.max(np.abs(target_rgb), axis=2)
        > float(config["oracle_patch_contract"]["content_rgb_threshold"])
    ) | depth_valid
    if semantic_available:
        target_content |= mask
    denominator = mask & target_content
    if not np.any(denominator):
        raise R7ExperimentError("usable coverage denominator 为空")
    usable = denominator & rgb_ok & depth_ok & semantic_ok
    return {
        "rgb_mae": rgb_mae,
        "relative_depth_error": depth_error,
        "usable_pixel_count": int(np.count_nonzero(usable)),
        "usable_denominator": int(np.count_nonzero(denominator)),
        "usable_coverage": float(np.count_nonzero(usable) / np.count_nonzero(denominator)),
    }


def _outside_exact(
    baked: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray], mask: np.ndarray
) -> bool:
    outside = ~mask
    return all(np.array_equal(baked[name][outside], target[name][outside]) for name in baked)


def _improvement(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    if before <= 1.0e-12:
        return 1.0 if after <= 1.0e-12 else 0.0
    return float((before - after) / before)


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R7ExperimentError("正式 R7 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R7ExperimentError("R7 task_id 漂移")
    recovery = _resolve_runs_uri(config["sources"]["recovery_run"])
    render_root = _resolve_runs_uri(config["sources"]["render_run"])
    if _sha256(recovery / "MANIFEST.json") != config["sources"]["recovery_manifest_sha256"]:
        raise R7ExperimentError("R3 recovery manifest 漂移")
    if _sha256(recovery / "REUSED_RENDER_EVIDENCE.json") != config["sources"]["reused_render_evidence_sha256"]:
        raise R7ExperimentError("R3 reused render evidence 漂移")
    reused = json.loads((recovery / "REUSED_RENDER_EVIDENCE.json").read_text(encoding="utf-8"))
    if not reused["all_render_hashes_reverified"] or reused["source_run"] != str(render_root):
        raise R7ExperimentError("R3 render recovery 证据不满足 R7")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R7ExperimentError("R7 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__oracle-missing-world-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        worker_index = {(row["scene"], row["frontend"]): row for row in reused["workers"]}
        render_maps = {}
        for scene in config["cohort"]["scenes"]:
            for frontend in config["cohort"]["frontends"]:
                map_path = render_root / "renders" / scene / frontend / "RENDER_MAP.jsonl"
                worker = worker_index[(scene, frontend)]
                if _sha256(map_path) != worker["render_map_sha256"]:
                    raise R7ExperimentError(f"render map 漂移：{scene}/{frontend}")
                render_maps[(scene, frontend)] = _load_render_map(map_path)
        oracle_rows: list[dict[str, Any]] = []
        decoy_rows: list[dict[str, Any]] = []
        provenance_rows: list[dict[str, Any]] = []
        structural_rows: list[dict[str, Any]] = []
        minimum_mask = int(config["oracle_patch_contract"]["minimum_mask_pixels"])
        for scene in config["cohort"]["scenes"]:
            for frontend in config["cohort"]["frontends"]:
                render_map = render_maps[(scene, frontend)]
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
                        target = side if hole_type == "missing_side_view" else removed if hole_type == "disocclusion" else base
                        mask = _mask_for(
                            hole_type,
                            target,
                            base,
                            removed,
                            config["oracle_patch_contract"],
                        )
                        mask_count = int(np.count_nonzero(mask))
                        if mask_count < minimum_mask:
                            if hole_type in {"disocclusion", "actor_removal_hole"}:
                                structural_rows.append(
                                    {
                                        "schema_version": "worldsim_v6.r7_structural_abstain.v1",
                                        "case_id": f"{scene}__{frontend}__f{int(frame_index):03d}__{hole_type}",
                                        "scene": scene,
                                        "frontend": frontend,
                                        "frame_index": int(frame_index),
                                        "hole_type": hole_type,
                                        "evidence_pixel_count": mask_count,
                                        "minimum_required_pixels": minimum_mask,
                                        "decision": "structural_abstain_before_proposal",
                                        "reason": "no_nonempty_actor_edit_evidence_denominator",
                                    }
                                )
                                continue
                            raise R7ExperimentError(
                                f"hole mask 太小：{scene}/{frontend}/{frame_index}/{hole_type}={mask_count}"
                            )
                        missing = _make_missing(target, mask)
                        semantic_available = hole_type in {"disocclusion", "actor_removal_hole"}
                        case_id = f"{scene}__{frontend}__f{int(frame_index):03d}__{hole_type}"
                        baseline_metrics = _evaluate(
                            missing, target, mask, semantic_available, config
                        )
                        for proposal_kind, is_decoy in (("oracle", False), ("corrupted_control", True)):
                            candidate = _proposal(target, mask, decoy=is_decoy)
                            verdict = _verify(
                                candidate, target, mask, semantic_available, config
                            )
                            baked = _bake(missing, candidate, mask, bool(verdict["accepted"]))
                            baked_metrics = _evaluate(
                                baked, target, mask, semantic_available, config
                            )
                            row = {
                                "schema_version": "worldsim_v6.r7_case.v1",
                                "case_id": case_id,
                                "proposal_kind": proposal_kind,
                                "scene": scene,
                                "frontend": frontend,
                                "frame_index": int(frame_index),
                                "hole_type": hole_type,
                                "mask_pixel_count": mask_count,
                                "proposal_content_sha256": _array_sha256(
                                    candidate["rgb"][mask],
                                    candidate["depth"][mask],
                                    candidate["dynamic_opacity"][mask],
                                ),
                                "verifier": verdict,
                                "bake_applied": bool(verdict["accepted"]),
                                "outside_mask_exact": _outside_exact(baked, target, mask),
                                "baseline": baseline_metrics,
                                "baked": baked_metrics,
                                "photo_error_reduction": _improvement(
                                    baseline_metrics["rgb_mae"], baked_metrics["rgb_mae"]
                                ),
                                "depth_error_reduction": _improvement(
                                    baseline_metrics["relative_depth_error"],
                                    baked_metrics["relative_depth_error"],
                                ),
                                "usable_coverage_gain": float(
                                    baked_metrics["usable_coverage"]
                                    - baseline_metrics["usable_coverage"]
                                ),
                            }
                            provenance_rows.append(
                                {
                                    "schema_version": config["oracle_patch_contract"]["schema"],
                                    "case_id": case_id,
                                    "proposal_kind": proposal_kind,
                                    "source_type": "observed" if not is_decoy else "generated_control",
                                    "sensor_support": "heldout_render_observation",
                                    "time_support": int(frame_index),
                                    "view_support": frontend,
                                    "reconstruction_source": "r3_hash_verified_frozen_render",
                                    "generation_source": "oracle_heldout_observation"
                                    if not is_decoy
                                    else "deterministic_corruption_control",
                                    "content_sha256": row["proposal_content_sha256"],
                                    "bake_decision": "accept" if verdict["accepted"] else "reject",
                                }
                            )
                            if is_decoy:
                                decoy_rows.append(row)
                            else:
                                oracle_rows.append(row)
        expected_cases = int(config["cohort"]["expected_eligible_case_count"])
        expected_abstains = int(config["cohort"]["expected_structural_abstain_count"])
        if len(oracle_rows) != expected_cases or len(decoy_rows) != expected_cases:
            raise R7ExperimentError("R7 case denominator 漂移")
        if len(structural_rows) != expected_abstains:
            raise R7ExperimentError("R7 structural abstain denominator 漂移")
        hole_rows = []
        for hole_type in config["cohort"]["hole_types"]:
            subset = [row for row in oracle_rows if row["hole_type"] == hole_type]
            hole_rows.append(
                {
                    "schema_version": "worldsim_v6.r7_hole_metrics.v1",
                    "hole_type": hole_type,
                    "case_count": len(subset),
                    "oracle_acceptance": float(
                        np.mean([row["verifier"]["accepted"] for row in subset])
                    ),
                    "median_photo_error_reduction": float(
                        np.median([row["photo_error_reduction"] for row in subset])
                    ),
                    "median_depth_error_reduction": float(
                        np.median(
                            [
                                row["depth_error_reduction"]
                                for row in subset
                                if row["depth_error_reduction"] is not None
                            ]
                        )
                    ),
                    "mean_usable_coverage_gain": float(
                        np.mean([row["usable_coverage_gain"] for row in subset])
                    ),
                    "minimum_usable_coverage_gain": float(
                        np.min([row["usable_coverage_gain"] for row in subset])
                    ),
                }
            )
        oracle_acceptance = float(
            np.mean([row["verifier"]["accepted"] for row in oracle_rows])
        )
        decoy_acceptance = float(
            np.mean([row["verifier"]["accepted"] for row in decoy_rows])
        )
        photo_reductions = [row["photo_error_reduction"] for row in oracle_rows]
        depth_reductions = [
            row["depth_error_reduction"]
            for row in oracle_rows
            if row["depth_error_reduction"] is not None
        ]
        aggregate = {
            "oracle_acceptance": oracle_acceptance,
            "decoy_acceptance": decoy_acceptance,
            "median_photo_error_reduction": float(np.median(photo_reductions)),
            "median_eligible_depth_error_reduction": float(np.median(depth_reductions)),
            "minimum_per_hole_usable_coverage_gain": float(
                min(row["minimum_usable_coverage_gain"] for row in hole_rows)
            ),
            "outside_mask_exact": all(row["outside_mask_exact"] for row in oracle_rows + decoy_rows),
            "provenance_row_count": len(provenance_rows),
            "provenance_exact": len(provenance_rows) == expected_cases * 2
            and all(row["content_sha256"] for row in provenance_rows),
            "structural_abstain_count": len(structural_rows),
        }
        gate_cfg = config["gate"]
        checks = {
            "oracle_acceptance_passed": oracle_acceptance
            >= float(gate_cfg["minimum_oracle_acceptance"]),
            "decoy_rejection_passed": decoy_acceptance
            <= float(gate_cfg["maximum_decoy_acceptance"]),
            "photo_recovery_passed": aggregate["median_photo_error_reduction"]
            >= float(gate_cfg["minimum_median_photo_error_reduction"]),
            "depth_recovery_passed": aggregate["median_eligible_depth_error_reduction"]
            >= float(gate_cfg["minimum_median_eligible_depth_error_reduction"]),
            "usable_region_passed": aggregate["minimum_per_hole_usable_coverage_gain"]
            >= float(gate_cfg["minimum_per_hole_usable_coverage_gain"]),
            "outside_mask_exact_passed": aggregate["outside_mask_exact"],
            "provenance_exact_passed": bool(aggregate["provenance_exact"]),
            "structural_abstain_count_passed": len(structural_rows) == expected_abstains,
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.r7_oracle_gate.v1",
            "aggregate": aggregate,
            "checks": checks,
            "decision": "oracle_path_supports_frozen_generator_study"
            if checks["passed"]
            else "reject_generator_and_repair_representation_verifier_or_bake",
        }
        _write_jsonl(run_dir / "ORACLE_CASES.jsonl", oracle_rows)
        _write_jsonl(run_dir / "DECOY_CONTROLS.jsonl", decoy_rows)
        _write_jsonl(run_dir / "PROPOSAL_PROVENANCE.jsonl", provenance_rows)
        _write_jsonl(run_dir / "HOLE_TYPE_METRICS.jsonl", hole_rows)
        _write_jsonl(run_dir / "STRUCTURAL_ABSTAINS.jsonl", structural_rows)
        _write_json(run_dir / "ORACLE_GATE.json", gate)
        elapsed = time.monotonic() - started
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r7_resource_audit.v1",
                "gpu_used": False,
                "training_started": False,
                "confirmation_content_read": False,
                "wall_seconds": elapsed,
                "disk_free_gib_at_start": free_gib,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r7_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development" if checks["passed"] else "rejected",
            "source_commit": source_commit,
            "oracle_case_count": len(oracle_rows),
            "decoy_case_count": len(decoy_rows),
            "structural_abstain_count": len(structural_rows),
            "gate_passed": checks["passed"],
            "training_started": False,
            "confirmation_content_read": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "ORACLE_CASES.jsonl",
            "DECOY_CONTROLS.jsonl",
            "PROPOSAL_PROVENANCE.jsonl",
            "HOLE_TYPE_METRICS.jsonl",
            "STRUCTURAL_ABSTAINS.jsonl",
            "ORACLE_GATE.json",
            "RESOURCE_AUDIT.json",
            "SUMMARY.json",
        ]
        manifest = {
            "schema_version": "worldsim_v6.r7_run_manifest.v1",
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
        default=Path("configs/worldsim_v6/r7_oracle_missing_world_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
