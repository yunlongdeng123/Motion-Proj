#!/usr/bin/env python3
"""为已完成的 M5 场景 checkpoint 创建不可变的修正合同。"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    (args.run_dir / "stages").mkdir(parents=True)
    source_terminal = json.loads(
        (args.source_run / "terminal.json").read_text(encoding="utf-8")
    )
    source_summary = json.loads(
        (args.source_run / "summary.json").read_text(encoding="utf-8")
    )
    if source_terminal["status"] != "done" or source_summary["status"] != "done":
        raise RuntimeError("source M5 training run 未完成")
    checkpoint = Path(source_summary["checkpoint"])
    registry = Path(source_summary["registry"])
    if not checkpoint.is_file() or not registry.is_file():
        raise RuntimeError("source checkpoint/registry 缺失")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    step = int(checkpoint_payload.get("step", -1))
    del checkpoint_payload
    if step != 30000:
        raise RuntimeError(f"checkpoint step {step} != 30000")
    config = OmegaConf.load(checkpoint.parent / "config.yaml")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    scene = source_summary["scene_name"]
    stride = int(config.data.pixel_source.test_image_stride)
    if stride != int(protocol["heldout"]["test_image_stride"]):
        raise RuntimeError("checkpoint held-out stride 与协议不一致")
    expected_frames = list(range(stride, int(config.data.end_timestep if int(config.data.end_timestep) > 0 else 196), stride))
    if expected_frames != protocol["heldout"]["frames"]:
        raise RuntimeError("协议 held-out frame 列表与 DriveStudio split 实现不一致")
    source_sha = sha256_file(checkpoint)
    if source_sha != source_summary["checkpoint_sha256"]:
        raise RuntimeError("source checkpoint SHA 已变化")
    selected = source_summary["selected_actors"]
    summary = {
        "status": "done",
        "scene_name": scene,
        "scene_index": source_summary["scene_index"],
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": source_sha,
        "registry": str(registry),
        "registry_sha256": sha256_file(registry),
        "selected_actors": selected,
        "test_image_stride": stride,
        "heldout_frames": expected_frames,
        "truth_tier_a_images": len(expected_frames) * 3,
        "recovery_mode": "correct held-out count contract; no checkpoint bytes changed",
        "source_run": str(args.source_run),
        "source_terminal_sha256": sha256_file(args.source_run / "terminal.json"),
        "source_summary_sha256": sha256_file(args.source_run / "summary.json"),
    }
    write_new(
        args.run_dir / "manifest.json",
        {
            "schema_version": 1,
            "task_id": "DR-V2-M5-STRESS-3SCENE-01",
            "component": "held-out training contract recovery",
            "source_run": str(args.source_run),
            "created_at": now(),
        },
    )
    write_new(args.run_dir / "summary.json", summary)
    write_new(
        args.run_dir / "stages/training_recovery.json",
        {"stage": "training_recovery", "status": "done", **summary},
    )
    write_new(
        args.run_dir / "terminal.json",
        {"status": "done", "updated_at": now(), "failure": None},
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
