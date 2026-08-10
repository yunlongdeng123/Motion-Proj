#!/usr/bin/env python3
"""构建 V3.3 S2 RoadPatch-Lite 的原生 Background 静态索引。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v33.roadpatch import (  # noqa: E402
    BEV_AXES,
    EXCLUDE_ACTOR_SEMANTIC,
    EXCLUDE_GENERATED,
    EXCLUDE_GEOMETRY_OUTLIER,
    EXCLUDE_LOW_VIEW_SUPPORT,
    EXCLUDE_LOW_VISIBILITY,
    EXCLUDE_SCALE_OUTLIER,
    EXCLUDE_SPARSE,
    FEATURE_NAMES,
    SCHEMA_VERSION,
    VERTICAL_AXIS,
    atomic_save_patch_index,
    build_patch_index,
    native_row_eligibility,
    sha256_arrays,
)


FLAG_NAMES = {
    int(EXCLUDE_SPARSE): "sparse",
    int(EXCLUDE_ACTOR_SEMANTIC): "actor_semantic",
    int(EXCLUDE_LOW_VISIBILITY): "low_visibility",
    int(EXCLUDE_LOW_VIEW_SUPPORT): "low_view_support",
    int(EXCLUDE_GEOMETRY_OUTLIER): "geometry_outlier",
    int(EXCLUDE_SCALE_OUTLIER): "scale_outlier",
    int(EXCLUDE_GENERATED): "generated",
}


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


def verify_file(path: str | Path, expected_sha256: str, role: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{role} 不存在: {source}")
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{role} SHA 漂移: expected={expected_sha256} actual={actual}"
        )
    return {"path": str(source), "sha256": actual, "bytes": source.stat().st_size}


def iter_frozen_files(config: dict[str, Any]) -> Iterable[tuple[str, str, str]]:
    inputs = config["inputs"]
    for role, path_key, sha_key in (
        ("checkpoint", "checkpoint", "checkpoint_sha256"),
        ("source_config", "source_config", "source_config_sha256"),
        ("s1_config", "s1_config", "s1_config_sha256"),
        ("s1_instance_field", "s1_instance_field", "s1_instance_field_sha256"),
        ("s1_summary", "s1_summary", "s1_summary_sha256"),
        ("v31_chunk_manifest", "v31_chunk_manifest", "v31_chunk_manifest_sha256"),
    ):
        yield role, inputs[path_key], inputs[sha_key]
    for role, row in inputs["semantic_sidecars"].items():
        yield f"semantic_sidecar:{role}", row["path"], row["sha256"]
    for role, row in inputs["v32_telea"].items():
        if role.endswith("sha256"):
            continue
        yield f"v32_telea:{role}", row, inputs["v32_telea"][f"{role}_sha256"]
    for role, target in config["targets"].items():
        for name in ("semantic_mask", "cross_view_observed_mask", "completion_reference"):
            yield f"target:{role}:{name}", target[name], target[f"{name}_sha256"]


def preflight(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("schema_version") != "worldsim_v33_s2_roadpatch_v1":
        raise ValueError("S2 config schema_version 漂移")
    scene = config["scene"]
    if tuple(scene["bev_axes"]) != BEV_AXES or int(scene["vertical_axis"]) != VERTICAL_AXIS:
        raise ValueError("scene 坐标轴契约漂移；RoadPatch 必须使用 x/z BEV 与 y 竖直轴")
    if tuple(float(value) for value in config["index"]["patch_sizes_m"]) != (1.0, 2.0, 4.0):
        raise ValueError("patch size 必须冻结为 1/2/4 m")
    verified = {
        role: verify_file(path, digest, role)
        for role, path, digest in iter_frozen_files(config)
    }

    background_count = int(config["inputs"]["checkpoint_background_count"])
    provenance_path = config["inputs"]["v32_telea"]["generated_provenance"]
    with np.load(provenance_path, allow_pickle=False) as payload:
        generated_rows = payload["background_row_index"].astype(np.int64)
        generated_codes = payload["provenance_code"].astype(np.uint8)
    expected_rows = np.arange(
        background_count, background_count + generated_rows.size, dtype=np.int64
    )
    if generated_rows.size == 0 or not np.array_equal(generated_rows, expected_rows):
        raise RuntimeError("V3.2 生成 Background 行不是 D2 尾部连续追加，拒绝建立 donor 索引")
    if not np.all(generated_codes == 1):
        raise RuntimeError("V3.2 生成 provenance code 漂移")

    manifest = json.loads(Path(config["inputs"]["v31_chunk_manifest"]).read_text(encoding="utf-8"))
    p3_inventory = manifest.get("static_grid", {}).get("expected_source_inventory", {})
    if int(p3_inventory.get("background_count", -1)) != background_count:
        raise RuntimeError("V3.1 P3 manifest 的 Background 库存行数与 D2 漂移")
    package_root = Path(config["inputs"]["v31_chunk_manifest"]).parent
    payload_candidates = [path for path in package_root.rglob("*") if path.is_file()]
    p3_payload_state = {
        "manifest_present": True,
        "files_present_including_manifest": len(payload_candidates),
        "payload_files_present": max(len(payload_candidates) - 1, 0),
        "source_checkpoint": manifest.get("source_checkpoint"),
        "legacy_static_axes": manifest.get("static_grid", {}).get("axes"),
        "coordinate_contract_compatible": False,
        "incompatibility": "V3.1 P3 使用 P2 FP16 派生 checkpoint 与 x/y 网格；V3.3 场景真值是 D2 FP32 与 x/z BEV",
        "policy": "P3 仅作历史 schema 参考；从已冻结 D2 checkpoint 精确重建 x/z 1/2/4m 索引",
    }
    generated_audit = {
        "background_count": background_count,
        "generated_count": int(generated_rows.size),
        "first_generated_row": int(generated_rows[0]),
        "last_generated_row": int(generated_rows[-1]),
        "contiguous_tail_append": True,
        "donor_source": "D2 native Background only",
        "generated_rows_admitted_to_donor_source": 0,
    }
    return verified, {"v32_generated_rows": generated_audit, "v31_p3": p3_payload_state}


def tensor_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def load_background_state(config: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    payload = torch.load(
        config["inputs"]["checkpoint"], map_location="cpu", weights_only=False
    )
    background = payload["models"]["Background"]
    keys = ("_means", "_scales", "_quats", "_features_dc", "_features_rest", "_opacities")
    state = {name: tensor_numpy(background[name]) for name in keys}
    count = int(state["_means"].shape[0])
    expected = int(config["inputs"]["checkpoint_background_count"])
    if count != expected:
        raise RuntimeError(f"D2 Background 行数漂移: expected={expected} actual={count}")
    ancestry = background.get("worldsim_a2_ancestry")
    if not isinstance(ancestry, dict) or not isinstance(ancestry.get("fields"), dict):
        raise RuntimeError("D2 Background 缺少 worldsim_a2_ancestry.fields")
    gaussian_ids = tensor_numpy(ancestry["fields"]["gaussian_id"]).astype(np.int64)
    if gaussian_ids.shape != (count,) or np.unique(gaussian_ids).size != count:
        raise RuntimeError("D2 Background Gaussian ID 非一一映射")
    return state, gaussian_ids


def load_semantic_features(config: dict[str, Any], count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actor_scores: list[np.ndarray] = []
    view_counts: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    for row in config["inputs"]["semantic_sidecars"].values():
        with np.load(row["path"], allow_pickle=False) as payload:
            if int(payload["background_count"].item()) != count:
                raise RuntimeError("semantic sidecar Background 行数漂移")
            actor_scores.append(payload["semantic_score"][:count].astype(np.float32))
            view_counts.append(payload["visible_view_count"][:count].astype(np.float32))
            masses.append(payload["visible_mass"][:count].astype(np.float32))
    return (
        np.maximum.reduce(actor_scores),
        np.maximum.reduce(view_counts),
        np.maximum.reduce(masses),
    )


def camera_intrinsics(path: Path, scale_x: float, scale_y: float) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if values.size not in (4, 9):
        raise ValueError(f"intrinsics schema 漂移: {path}")
    fx, fy, cx, cy = values[:4].tolist()
    return np.array(
        [[fx * scale_x, 0.0, cx * scale_x], [0.0, fy * scale_y, cy * scale_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def aligned_camera_to_world(processed_root: Path, frame: int, camera_id: int) -> np.ndarray:
    first_front = np.loadtxt(processed_root / "extrinsics" / "000_0.txt", dtype=np.float64)
    raw = np.loadtxt(
        processed_root / "extrinsics" / f"{int(frame):03d}_{int(camera_id)}.txt",
        dtype=np.float64,
    )
    if first_front.shape != (4, 4) or raw.shape != (4, 4):
        raise ValueError("extrinsics schema 漂移")
    return np.linalg.inv(first_front) @ raw


def frustum_multi_camera_count(
    means: np.ndarray, config: dict[str, Any], chunk_rows: int = 200_000
) -> tuple[np.ndarray, dict[str, Any]]:
    """仅用 development frames 统计每个点曾进入多少个相机视锥；不宣称遮挡可见。"""

    scene = config["scene"]
    root = Path(scene["processed_root"])
    width, height = int(scene["model_native_width"]), int(scene["model_native_height"])
    scale_x = width / float(scene["source_width"])
    scale_y = height / float(scene["source_height"])
    counts = np.zeros(means.shape[0], dtype=np.uint8)
    camera_stats: dict[str, Any] = {}
    for camera_id in (int(value) for value in scene["cameras"]):
        intrinsics = camera_intrinsics(root / "intrinsics" / f"{camera_id}.txt", scale_x, scale_y)
        transforms = [
            np.linalg.inv(aligned_camera_to_world(root, int(frame), camera_id))
            for frame in scene["development_frames"]
        ]
        ever_inside = np.zeros(means.shape[0], dtype=bool)
        for start in range(0, means.shape[0], int(chunk_rows)):
            end = min(start + int(chunk_rows), means.shape[0])
            points = means[start:end].astype(np.float64)
            homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=np.float64)])
            inside = np.zeros(points.shape[0], dtype=bool)
            for world_to_camera in transforms:
                camera = homogeneous @ world_to_camera.T
                depth = camera[:, 2]
                valid = depth > 1e-4
                safe_depth = np.where(valid, depth, 1.0)
                pixel_x = intrinsics[0, 0] * camera[:, 0] / safe_depth + intrinsics[0, 2]
                pixel_y = intrinsics[1, 1] * camera[:, 1] / safe_depth + intrinsics[1, 2]
                inside |= valid & (pixel_x >= 0.0) & (pixel_x < width) & (pixel_y >= 0.0) & (pixel_y < height)
            ever_inside[start:end] = inside
        counts += ever_inside.astype(np.uint8)
        camera_stats[str(camera_id)] = {
            "development_frames": [int(value) for value in scene["development_frames"]],
            "ever_inside_count": int(ever_inside.sum()),
            "criterion": "positive camera depth and inside native 800x450 image bounds in at least one development frame",
        }
    return counts, camera_stats


def summarize_index(index: Any) -> dict[str, Any]:
    by_size: dict[str, Any] = {}
    for size in (1.0, 2.0, 4.0):
        selected = index.patch_sizes_m == size
        flags = index.exclusion_flags[selected]
        by_size[f"{int(size)}m"] = {
            "total": int(selected.sum()),
            "valid": int(np.sum(flags == 0)),
            "excluded_any": int(np.sum(flags != 0)),
            "flag_counts_nonexclusive": {
                name: int(np.sum((flags & flag) != 0)) for flag, name in FLAG_NAMES.items()
            },
        }
    valid = index.exclusion_flags == 0
    return {
        "patch_count": int(index.patch_ids.size),
        "flat_membership_count": int(index.flat_indices.size),
        "valid_patch_count": int(valid.sum()),
        "by_size": by_size,
        "feature_names": list(FEATURE_NAMES),
        "coordinate_contract": {
            "frame": "first_CAM_FRONT_aligned_OpenCV_x_right_y_down_z_forward",
            "bev_axes": list(BEV_AXES),
            "vertical_axis": VERTICAL_AXIS,
        },
    }


def snapshot_sources(run_dir: Path, config_path: Path) -> dict[str, Any]:
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    sources = {
        "config": config_path,
        "builder": Path(__file__).resolve(),
        "roadpatch": PROJECT / "motion_proj" / "worldsim_v33" / "roadpatch.py",
    }
    report: dict[str, Any] = {}
    for role, source in sources.items():
        target = snapshot / source.name
        shutil.copy2(source, target)
        report[role] = {
            "source_path": str(source),
            "snapshot_path": str(target),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.run_dir.exists() and any(
        (args.run_dir / name).exists() for name in ("status.json", "summary.json", "artifacts")
    ):
        raise FileExistsError(f"run-dir 已含结构化输出，拒绝覆盖: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = args.run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    started = time.time()
    atomic_json(args.run_dir / "status.json", {"state": "running", "started_unix": started})

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    verified_inputs, lineage_audit = preflight(config)
    checkpoint_before = sha256_file(config["inputs"]["checkpoint"])
    state, gaussian_ids = load_background_state(config)
    count = int(state["_means"].shape[0])
    actor_score, view_count, visibility_mass = load_semantic_features(config, count)
    multi_camera_count, camera_stats = frustum_multi_camera_count(state["_means"], config)
    native_mask = np.ones(count, dtype=bool)
    eligible_rows, row_exclusions = native_row_eligibility(
        raw_scales=state["_scales"],
        actor_semantic_score=actor_score,
        train_view_observation_count=view_count,
        visibility_mass=visibility_mass,
        native_donor_mask=native_mask,
        thresholds=config["index"]["thresholds"],
    )
    row_filter_report = {
        "protocol": "fail_closed_before_patch_grouping",
        "source_rows": count,
        "eligible_native_rows": int(eligible_rows.sum()),
        "excluded_rows_any": int((~eligible_rows).sum()),
        "reason_counts_nonexclusive": {
            name: int(mask.sum()) for name, mask in row_exclusions.items()
        },
    }

    index = build_patch_index(
        means=state["_means"],
        raw_scales=state["_scales"],
        raw_opacities=state["_opacities"],
        features_dc=state["_features_dc"],
        actor_semantic_score=actor_score,
        train_view_observation_count=view_count,
        visibility_mass=visibility_mass,
        multi_camera_count=multi_camera_count,
        native_donor_mask=native_mask,
        patch_sizes_m=config["index"]["patch_sizes_m"],
        thresholds=config["index"]["thresholds"],
    )
    index.validate(background_count=count)
    index_path = artifacts / "static_patch_index.npz"
    atomic_save_patch_index(index_path, index)
    repeat_path = artifacts / "static_patch_index.repeat.npz"
    atomic_save_patch_index(repeat_path, index)
    deterministic = sha256_file(index_path) == sha256_file(repeat_path)
    repeat_path.unlink()
    if not deterministic:
        raise RuntimeError("patch index 重序列化字节不确定")

    gaussian_id_report = {
        "count": int(gaussian_ids.size),
        "minimum": int(gaussian_ids.min()),
        "maximum": int(gaussian_ids.max()),
        "unique": True,
        "sha256_arrays": sha256_arrays({"gaussian_id": gaussian_ids}),
    }
    index_report = summarize_index(index)
    checkpoint_after = sha256_file(config["inputs"]["checkpoint"])
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("source checkpoint 被意外修改")
    source_snapshot = snapshot_sources(args.run_dir, args.config.resolve())
    elapsed = time.time() - started
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": config["task_id"],
        "config": {
            "path": str(args.config.resolve()),
            "sha256": sha256_file(args.config),
            "bytes": args.config.stat().st_size,
        },
        "verified_inputs": verified_inputs,
        "lineage_audit": lineage_audit,
        "source_checkpoint_sha256_before": checkpoint_before,
        "source_checkpoint_sha256_after": checkpoint_after,
        "source_checkpoint_mutated": False,
        "background_count": count,
        "gaussian_ids": gaussian_id_report,
        "multi_camera_frustum_support": {
            "protocol": "development_frames_only_no_heldout_no_occlusion_claim",
            "distribution": {str(value): int(np.sum(multi_camera_count == value)) for value in range(4)},
            "cameras": camera_stats,
        },
        "row_filter": row_filter_report,
        "patch_index": {
            "path": str(index_path),
            "sha256": sha256_file(index_path),
            "bytes": index_path.stat().st_size,
            "arrays_sha256": sha256_arrays(index.as_arrays()),
            "deterministic_reserialization": deterministic,
            **index_report,
        },
        "source_snapshot": source_snapshot,
        "elapsed_seconds": elapsed,
    }
    atomic_json(artifacts / "patch_index_manifest.json", manifest)
    summary = {
        "task_id": config["task_id"],
        "state": "completed",
        "elapsed_seconds": elapsed,
        "checkpoint_immutable": True,
        "generated_donors_admitted": 0,
        "eligible_native_rows": int(eligible_rows.sum()),
        "heldout_frames_used": [],
        "index_sha256": manifest["patch_index"]["sha256"],
        "index_bytes": manifest["patch_index"]["bytes"],
        **index_report,
    }
    atomic_json(args.run_dir / "summary.json", summary)
    atomic_json(
        args.run_dir / "status.json",
        {
            "state": "completed",
            "started_unix": started,
            "completed_unix": time.time(),
            "elapsed_seconds": elapsed,
            "summary_sha256": sha256_file(args.run_dir / "summary.json"),
            "manifest_sha256": sha256_file(artifacts / "patch_index_manifest.json"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
