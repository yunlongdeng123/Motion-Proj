#!/usr/bin/env python
"""验收 S1 pseudo target，证明其只用于冻结的 formal evaluation。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import sha256_file
from motion_proj.worldsim_v33.evaluation_partition import (
    manifest_evaluation_partition,
    resolve_evaluation_frames,
)


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
    parser.add_argument(
        "--partition", choices=("development", "heldout"), default="heldout"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prompt_path = args.run_dir / "artifacts/prompts/prompt_manifest.json"
    mask_path = args.run_dir / "artifacts/masks/mask_manifest.json"
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    masks = json.loads(mask_path.read_text(encoding="utf-8"))
    config_sha = sha256_file(args.config)
    if prompts.get("config_sha256") != config_sha or masks.get("config_sha256") != config_sha:
        raise RuntimeError("evaluation target config SHA 漂移")
    if not prompts.get("optimization_forbidden") or not masks.get("optimization_forbidden"):
        raise RuntimeError("evaluation target 未声明 optimization_forbidden")
    expected_frames = set(resolve_evaluation_frames(config, args.partition))
    for name, payload in (("prompt", prompts), ("mask", masks)):
        if manifest_evaluation_partition(payload) != args.partition:
            raise RuntimeError(f"{name} evaluation partition 漂移")
    if set(prompts["evaluation_frames"]) != expected_frames:
        raise RuntimeError("evaluation prompt frame 集漂移")
    if set(masks["evaluation_frames"]) != expected_frames:
        raise RuntimeError("evaluation mask frame 集漂移")
    if masks["checkpoint_sha256_before"] != masks["checkpoint_sha256_after"]:
        raise RuntimeError("SAM2 checkpoint 在 evaluation target 生成中发生 mutation")
    if masks["checkpoint_sha256_after"] != config["sam2_fallback"]["checkpoint_sha256"]:
        raise RuntimeError("SAM2 checkpoint SHA 不等于冻结值")
    runtime_contract = config["sam2_fallback"]["runtime"]
    for name in ("environment", "python", "torch", "torchvision", "numpy"):
        if str(masks["runtime"][name]) != str(runtime_contract[name]):
            raise RuntimeError(f"SAM2 runtime {name} 不等于冻结值")
    if masks["runtime"]["conda_explicit_sha256"] != runtime_contract["conda_explicit_sha256"]:
        raise RuntimeError("SAM2 conda explicit SHA 漂移")
    if masks["runtime"]["pip_freeze_sha256"] != runtime_contract["pip_freeze_sha256"]:
        raise RuntimeError("SAM2 pip freeze SHA 漂移")
    accepted = 0
    seen = set()
    for row in masks["masks"]:
        key = (str(row["role"]), int(row["frame"]), int(row["camera_id"]))
        if key in seen or key[1] not in expected_frames:
            raise RuntimeError(f"evaluation target row 重复或越权: {key}")
        seen.add(key)
        if sha256_file(row["source_image"]) != row["source_image_sha256"]:
            raise RuntimeError(f"evaluation source image SHA 漂移: {key}")
        if sha256_file(row["mask"]) != row["mask_sha256"]:
            raise RuntimeError(f"evaluation mask SHA 漂移: {key}")
        accepted += int(bool(row["accepted"]) and int(row["positive_pixels"]) > 0)
    if accepted != int(masks["accepted_mask_count"]) or accepted == 0:
        raise RuntimeError("evaluation accepted mask 计数不合法")
    train = json.loads(
        Path(config["inputs"]["train_mask_manifest"]).read_text(encoding="utf-8")
    )
    if any(int(row["frame"]) in expected_frames for row in train["masks"]):
        raise RuntimeError("train mask 混入 evaluation partition")

    snapshot_dir = args.run_dir / "artifacts/source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    for source in (
        args.config,
        PROJECT / "scripts/prepare_worldsim_v33_s1_eval_prompts.py",
        PROJECT / "scripts/build_worldsim_v33_s1_eval_masks.py",
        PROJECT / "scripts/finalize_worldsim_v33_s1_eval_targets.py",
    ):
        shutil.copy2(source, snapshot_dir / source.name)
    summary = {
        "schema_version": "worldsim_v33_s1_eval_targets_summary_v1",
        "task_id": config["task_id"],
        "status": "done",
        "config_sha256": config_sha,
        "prompt_manifest": str(prompt_path),
        "prompt_manifest_sha256": sha256_file(prompt_path),
        "mask_manifest": str(mask_path),
        "mask_manifest_sha256": sha256_file(mask_path),
        "evaluation_frames": sorted(expected_frames),
        "evaluation_partition": args.partition,
        "optimization_forbidden": True,
        "mask_count": len(seen),
        "accepted_mask_count": accepted,
        "rejected_mask_count": int(masks["rejected_mask_count"]),
        "sam2_source_commit": masks["source_commit"],
        "sam2_checkpoint_sha256": masks["checkpoint_sha256_after"],
        "runtime": masks["runtime"],
    }
    atomic_json(args.run_dir / "summary.json", summary)
    entries = []
    for path in sorted(args.run_dir.rglob("*")):
        if path.is_file() and path.name not in {"run_manifest.json", "status.json"}:
            entries.append(
                {
                    "path": str(path.relative_to(args.run_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    atomic_json(
        args.run_dir / "run_manifest.json",
        {
            "schema_version": "worldsim_v33_s1_eval_targets_manifest_v1",
            "task_id": config["task_id"],
            "file_count": len(entries),
            "files": entries,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
