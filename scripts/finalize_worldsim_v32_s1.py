#!/usr/bin/env python
"""严格验收 S1 的 mask、Gaussian sidecar、smoke 与冻结资产不变量。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import CORE_POSITIVE, sha256_file


REQUIRED_POSTERIOR_FIELDS = {
    "labels",
    "semantic_score",
    "weighted_positive",
    "weighted_total",
    "num_positive_views",
    "num_negative_views",
    "depth_consistency_rate",
    "boundary_score",
    "background_count",
    "rigid_point_ids",
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层不是 mapping: {path}")
    return payload


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_root = Path("/root/autodl-tmp/runs/worldsim_v32").resolve()
    run_dir = args.run_dir.resolve()
    if run_root not in run_dir.parents:
        raise ValueError(f"S1 run 不在冻结根目录下: {run_dir}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config_sha = sha256_file(args.config)
    mask_path = run_dir / "artifacts/sam2/mask_manifest.json"
    semantic_path = run_dir / "artifacts/semantic_sidecar/semantic_manifest.json"
    masks = load_json(mask_path)
    semantics = load_json(semantic_path)

    if masks["config_sha256"] != config_sha:
        raise RuntimeError("mask manifest config SHA 漂移")
    if masks["checkpoint_sha256_before"] != masks["checkpoint_sha256_after"]:
        raise RuntimeError("SAM2 checkpoint 在推理中发生 mutation")
    if not masks.get("heldout_excluded"):
        raise RuntimeError("mask manifest 未声明 heldout 排除")
    prompt_path = Path(masks["prompt_manifest"])
    if sha256_file(prompt_path) != masks["prompt_manifest_sha256"]:
        raise RuntimeError("prompt manifest SHA 漂移")
    prompts = load_json(prompt_path)
    if prompts.get("actor_identity_contract") != "validated":
        raise RuntimeError("prompt manifest 未通过 actor identity 合同")
    for role, actor_config in config["actors"].items():
        prompt_actor = prompts["actors"].get(role)
        if prompt_actor is None:
            raise RuntimeError(f"prompt manifest 缺 actor role: {role}")
        for field in (
            "instance_token",
            "dataset_instance_id",
            "rigid_model_index",
            "class_name",
        ):
            if prompt_actor[field] != actor_config[field]:
                raise RuntimeError(f"prompt/config actor identity 漂移: {role}/{field}")
    heldout = {int(value) for value in masks["heldout_frames"]}
    expected_shape = (
        int(config["outputs"]["model_native_height"]),
        int(config["outputs"]["model_native_width"]),
    )
    seen: set[tuple[str, int, int]] = set()
    accepted = 0
    rejected = 0
    for row in masks["masks"]:
        key = (row["role"], int(row["frame"]), int(row["camera_id"]))
        if key in seen:
            raise RuntimeError(f"重复 mask row: {key}")
        seen.add(key)
        if key[1] in heldout:
            raise RuntimeError(f"heldout mask 泄漏: {key}")
        for field in (
            "instance_token",
            "source_image_sha256",
            "prompt_frame",
            "prompt_box_xyxy",
            "prompt_source",
            "timestamp_source",
        ):
            if field not in row:
                raise RuntimeError(f"mask provenance 缺字段 {field}: {key}")
        mask_file = Path(row["mask"])
        if sha256_file(mask_file) != row["mask_sha256"]:
            raise RuntimeError(f"mask SHA 漂移: {mask_file}")
        with np.load(mask_file, allow_pickle=False) as arrays:
            if set(arrays.files) != {"logits", "raw_binary", "binary"}:
                raise RuntimeError(f"mask NPZ schema 漂移: {mask_file}")
            if any(arrays[name].shape != expected_shape for name in arrays.files):
                raise RuntimeError(f"mask shape 漂移: {mask_file}")
            positive = int(arrays["binary"].sum())
        if positive != int(row["positive_pixels"]):
            raise RuntimeError(f"mask positive_pixels 漂移: {mask_file}")
        if bool(row["accepted"]):
            accepted += 1
            if positive <= 0:
                raise RuntimeError(f"accepted mask 为空: {mask_file}")
        else:
            rejected += 1
            if positive != 0 or not row["rejection_reasons"]:
                raise RuntimeError(f"fail-closed mask 不合法: {mask_file}")
    if len(seen) != int(masks["mask_count"]):
        raise RuntimeError("mask_count 与唯一 row 数不一致")
    if accepted != int(masks["accepted_mask_count"]) or rejected != int(
        masks["rejected_mask_count"]
    ):
        raise RuntimeError("mask QC 计数漂移")
    if accepted == 0:
        raise RuntimeError("S1 没有任何可用 mask")

    source = Path(config["inputs"]["checkpoint"])
    source_sha = sha256_file(source)
    expected_source_sha = config["inputs"]["checkpoint_sha256"]
    if source_sha != expected_source_sha:
        raise RuntimeError("D2 checkpoint SHA 漂移")
    if not semantics.get("sidecar_only") or not semantics.get("heldout_excluded"):
        raise RuntimeError("semantic sidecar 违反只读或 split 合同")
    if semantics["checkpoint_sha256_before"] != expected_source_sha:
        raise RuntimeError("semantic sidecar 输入 checkpoint 不匹配")
    if semantics["checkpoint_sha256_after"] != expected_source_sha:
        raise RuntimeError("semantic lift 修改了 D2 checkpoint")
    total_gaussians = sum(int(value) for value in semantics["gaussian_counts"].values())
    actor_summaries: dict[str, Any] = {}
    for role in config["actors"]:
        actor = semantics["actors"][role]
        sidecar = Path(actor["path"])
        if sha256_file(sidecar) != actor["sha256"]:
            raise RuntimeError(f"semantic sidecar SHA 漂移: {role}")
        with np.load(sidecar, allow_pickle=False) as arrays:
            missing = REQUIRED_POSTERIOR_FIELDS - set(arrays.files)
            if missing:
                raise RuntimeError(f"semantic sidecar 缺字段 {role}: {sorted(missing)}")
            for name in REQUIRED_POSTERIOR_FIELDS - {"background_count", "rigid_point_ids"}:
                if arrays[name].shape != (total_gaussians,):
                    raise RuntimeError(f"semantic field shape 漂移 {role}/{name}")
            labels = arrays["labels"]
            core_count = int((labels == CORE_POSITIVE).sum())
            if core_count != int(actor["core_count"]):
                raise RuntimeError(f"registry core 未 exact 保留: {role}")
            for name in ("semantic_score", "depth_consistency_rate", "boundary_score"):
                values = arrays[name]
                if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                    raise RuntimeError(f"posterior 比率非法 {role}/{name}")
        actor_summaries[role] = {
            "sidecar": str(sidecar),
            "sha256": actor["sha256"],
            "core_count": core_count,
            "label_counts": actor["label_counts"],
        }

    smokes = semantics["smoke"]
    expected_smokes = len(config["actors"]) * 3
    if len(smokes) != expected_smokes:
        raise RuntimeError(f"smoke 数量错误: {len(smokes)} != {expected_smokes}")
    for role in config["actors"]:
        role_rows = [row for row in smokes if row["role"] == role]
        variants = {row["variant"] for row in role_rows}
        if variants != {"original", "delete", "lateral"}:
            raise RuntimeError(f"smoke variant 缺失: {role}/{variants}")
        hashes = set()
        for row in role_rows:
            path = Path(row["path"])
            actual = sha256_file(path)
            if actual != row["sha256"] or path.stat().st_size == 0:
                raise RuntimeError(f"smoke artifact 漂移: {path}")
            hashes.add(actual)
        if len(hashes) != 3:
            raise RuntimeError(f"smoke 编辑输出未发生变化: {role}")

    summary = {
        "schema_version": "worldsim_v32_s1_summary_v1",
        "task_id": config["task_id"],
        "status": "done",
        "selected": "M3_prior_guided_semantic_sidecar",
        "config": str(args.config.resolve()),
        "config_sha256": config_sha,
        "prompt_manifest": masks["prompt_manifest"],
        "prompt_manifest_sha256": masks["prompt_manifest_sha256"],
        "mask_manifest": str(mask_path),
        "mask_manifest_sha256": sha256_file(mask_path),
        "semantic_manifest": str(semantic_path),
        "semantic_manifest_sha256": sha256_file(semantic_path),
        "checkpoint_sha256_before": expected_source_sha,
        "checkpoint_sha256_after": source_sha,
        "heldout_leaks": 0,
        "actor_identity_contract": "validated",
        "mask_count": len(seen),
        "accepted_mask_count": accepted,
        "rejected_mask_count": rejected,
        "actors": actor_summaries,
        "smoke_count": len(smokes),
        "runtime": {
            "sam2": masks["runtime"],
            "semantic_lift": semantics["runtime"],
        },
    }
    summary_path = run_dir / "summary.json"
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
